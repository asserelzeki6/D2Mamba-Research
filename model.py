"""
D2Mamba Research — Branch 2: exp/2-physics-attenuation-scan
Underwater Physics-Anchored Single-Scan Mamba:
  - Micro-AT GPU physics head (<0.15M params): predicts per-pixel relative optical depth
    proxy d_hat in [0, 1], decoupled direct/backscatter attenuation (beta_D, beta_B) with
    smooth softplus/tanh bounds (NO hard beta_R > beta_G > beta_B clamp so green coastal
    waters remain valid), and far-field veiling light B_inf anchored from top-5% farthest pixels.
  - Near-to-far GPU token ordering by d_hat with telescoping Beer-Lambert dt modulation via
    non-negative incremental depth differences delta_d_k = clamp(d_sorted[k] - d_sorted[k-1], min=0)
    multiplied by (beta_D, beta_B).
  - 100% pure PyTorch, zero CPU heapq / .cpu() / .numpy(), torch.jit.trace verified,
    compatible with official D2Mamba checkpoints via --init_ckpt.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Pure-PyTorch Exact Chunked SSD Scan & GPU Physics Helpers
# ---------------------------------------------------------------------------
@torch.jit.script
def _invert_permutation(order: torch.Tensor) -> torch.Tensor:
    """Compute exact inverse permutation on GPU in O(L) with zero CPU sync."""
    B, L = order.shape
    arange = torch.arange(L, device=order.device, dtype=order.dtype).unsqueeze(0).expand(B, L)
    inv_order = torch.empty_like(order)
    inv_order.scatter_(1, order, arange)
    return inv_order


@torch.jit.script
def _anchor_veiling_light(x_rgb: torch.Tensor, d_map: torch.Tensor) -> torch.Tensor:
    """Anchor far-field veiling light B_inf from the top-5% farthest d_hat pixels on GPU.
    100% dynamic-resolution safe inside torch.jit.trace.

    Args:
        x_rgb: (B, 3, H, W) in [0, 1]
        d_map: (B, 1, H, W) relative optical depth proxy in [0, 1]
    Returns:
        B_inf: (B, 3, 1, 1) float32 in [0, 1]
    """
    B, C, H, W = x_rgb.shape
    L = H * W
    k = max(1, (L * 5) // 100)
    d_flat = d_map.float().reshape(B, L)
    x_flat = x_rgb.float().clamp(0.0, 1.0).reshape(B, C, L).transpose(1, 2)  # (B, L, 3)

    top_d, top_idx = torch.topk(d_flat, k=k, dim=1, largest=True, sorted=False)
    top_rgb = torch.gather(x_flat, dim=1, index=top_idx.unsqueeze(-1).expand(B, k, C))
    weights = F.softmax(top_d * 10.0, dim=1).unsqueeze(-1)  # (B, k, 1)
    b_inf = (top_rgb * weights).sum(dim=1).reshape(B, C, 1, 1)
    return b_inf.clamp(0.0, 1.0)


@torch.jit.script
def _inter_chunk_scan(chunk_decay: torch.Tensor, chunk_states: torch.Tensor) -> torch.Tensor:
    """Exact Hillis-Steele associative prefix scan over chunks (FP32, zero overflow)."""
    M = chunk_decay.shape[2]
    a = chunk_decay
    u = chunk_states
    stride = 1
    while stride < M:
        a_prev = a[:, :, :-stride]
        a_curr = a[:, :, stride:]
        u_prev = u[:, :, :-stride]
        u_curr = u[:, :, stride:]
        u_upd = a_curr.unsqueeze(-1).unsqueeze(-1) * u_prev + u_curr
        a_upd = a_curr * a_prev
        u = torch.cat([u[:, :, :stride], u_upd], dim=2)
        a = torch.cat([a[:, :, :stride], a_upd], dim=2)
        stride = stride * 2

    zeros = torch.zeros_like(u[:, :, :1])
    if M > 1:
        return torch.cat([zeros, u[:, :, :-1]], dim=2)
    return zeros


@torch.jit.script
def chunked_ssd_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B_param: torch.Tensor,
    C_param: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Exact pure-PyTorch multi-head scalar-decay Structured State Space Duality (SSD) scan."""
    orig_dtype = x.dtype
    B_sz, L, H, P = x.shape
    N = B_param.shape[-1]

    rem = L % chunk_size
    pad_len = (chunk_size - rem) if rem != 0 else 0
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len))
        B_param = F.pad(B_param, (0, 0, 0, 0, 0, pad_len))
        C_param = F.pad(C_param, (0, 0, 0, 0, 0, pad_len))

    L_pad = x.shape[1]
    M = L_pad // chunk_size
    Q = chunk_size

    dt_f = dt.float().permute(0, 2, 1).reshape(B_sz, H, M, Q)
    x_f = x.float().permute(0, 2, 1, 3).reshape(B_sz, H, M, Q, P)
    B_f = B_param.float().permute(0, 2, 1, 3).reshape(B_sz, H, M, Q, N)
    C_f = C_param.float().permute(0, 2, 1, 3).reshape(B_sz, H, M, Q, N)

    log_a = torch.clamp(dt_f * A.float().view(1, H, 1, 1), max=0.0)
    cum_log_a = torch.cumsum(log_a, dim=-1)

    diff = cum_log_a.unsqueeze(-1) - cum_log_a.unsqueeze(-2)
    tril_mask = torch.tril(torch.ones((Q, Q), device=x.device, dtype=torch.bool))
    diff_safe = torch.where(tril_mask, diff.clamp(max=0.0), torch.full_like(diff, -1e9))
    L_mat = torch.exp(diff_safe) * tril_mask.to(diff.dtype)

    x_scaled = x_f * dt_f.unsqueeze(-1)
    CB = torch.matmul(C_f, B_f.transpose(-1, -2))
    Y_diag = torch.matmul(L_mat * CB, x_scaled)

    decay_to_end = torch.exp((cum_log_a[:, :, :, -1:] - cum_log_a).clamp(max=0.0))
    B_decayed = B_f * decay_to_end.unsqueeze(-1)
    H_local = torch.matmul(x_scaled.transpose(-1, -2), B_decayed)

    chunk_decay = torch.exp(cum_log_a[:, :, :, -1].clamp(max=0.0))
    H_prev = _inter_chunk_scan(chunk_decay, H_local)

    decay_from_start = torch.exp(cum_log_a.clamp(max=0.0))
    C_decayed = C_f * decay_from_start.unsqueeze(-1)
    Y_off = torch.matmul(C_decayed, H_prev.transpose(-1, -2))

    Y = (Y_diag + Y_off).reshape(B_sz, H, L_pad, P).permute(0, 2, 1, 3)
    if pad_len > 0:
        Y = Y[:, :L]
    return Y.to(orig_dtype)


# ---------------------------------------------------------------------------
# 2. Micro-AT Underwater Physics Head (< 0.15M params, 100% GPU-Native)
# ---------------------------------------------------------------------------
class MicroATHead(nn.Module):
    """
    Lightweight Akkaynak-Treibitz physics estimation head (< 0.15M params):
      - d_hat: per-pixel relative optical depth proxy in [0, 1]
      - beta_D: decoupled 3-channel direct attenuation (smoothly bounded, NO R>G>B clamp)
      - beta_B: decoupled 3-channel backscatter attenuation (smoothly bounded, NO R>G>B clamp)
      - B_inf: far-field veiling light anchored from top-5% farthest d_hat pixels
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, beta_max: float = 4.0):
        super().__init__()
        self.beta_max = float(beta_max)

        self.feat_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        rgb_dim = hidden_dim // 2
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, rgb_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(rgb_dim, rgb_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim + rgb_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        # Decoupled prediction heads
        self.depth_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.beta_D_head = nn.Conv2d(hidden_dim, 3, kernel_size=1)
        self.beta_B_head = nn.Conv2d(hidden_dim, 3, kernel_size=1)

        # Initialize beta_D and beta_B with slightly distinct biases so they start decoupled
        # (free to learn any channel ordering, including green coastal water where G/B swap)
        with torch.no_grad():
            self.beta_D_head.bias.copy_(torch.tensor([0.30, 0.00, -0.20]))
            self.beta_B_head.bias.copy_(torch.tensor([-0.20, 0.15, 0.30]))

    def _smooth_bound_beta(self, raw: torch.Tensor) -> torch.Tensor:
        """Smoothly positive & upper-bounded attenuation via softplus + tanh (FP32, no channel clamp)."""
        sp = F.softplus(raw.float())
        return self.beta_max * torch.tanh(sp / self.beta_max) + 1e-3

    def forward(
        self, x_rgb: torch.Tensor, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_rgb: (B, 3, H, W) input underwater image in [0, 1]
            features: (B, C, H_f, W_f) bottleneck feature map
        Returns:
            d_hat: (B, 1, H_f, W_f) float32 in [0, 1]
            beta_D: (B, 3, H_f, W_f) float32 > 0
            beta_B: (B, 3, H_f, W_f) float32 > 0
            B_inf: (B, 3, 1, 1) float32 in [0, 1]
        """
        H_f, W_f = features.shape[-2:]
        x_low = F.interpolate(x_rgb, size=(H_f, W_f), mode="bilinear", align_corners=False)

        h_feat = self.feat_proj(features)
        h_rgb = self.rgb_stem(x_low)
        h = self.fuse(torch.cat([h_feat, h_rgb], dim=1))

        # Strict FP32 for physics outputs (safe under --amp; cast h to float32 before FP32 head convs)
        with torch.amp.autocast(device_type=x_rgb.device.type, enabled=False):
            h_f = h.float()
            d_hat = torch.sigmoid(self.depth_head(h_f))
            beta_D = self._smooth_bound_beta(self.beta_D_head(h_f))
            beta_B = self._smooth_bound_beta(self.beta_B_head(h_f))
            B_inf = _anchor_veiling_light(x_low.float(), d_hat)

        return d_hat, beta_D, beta_B, B_inf


# ---------------------------------------------------------------------------
# 3. Physics-Modulated Attentive State-Space Block (ASE + Telescoping dt)
# ---------------------------------------------------------------------------
class ASEMambaBlock(nn.Module):
    """
    Attentive State-Space Block with:
      - Prompt-Augmented C-Projection (ASE)
      - Optional Telescoping Beer-Lambert dt modulation from optical-depth increments
        opt_delta = delta_d_k * [beta_D, beta_B] >= 0 (strictly zero-bias so delta_d=0 => dt_phys=0).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        num_heads: int = 4,
        num_prompts: int = 16,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.num_heads = num_heads
        assert self.d_inner % num_heads == 0
        self.head_dim = self.d_inner // num_heads
        self.num_prompts = num_prompts
        self.chunk_size = chunk_size

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.ssm_proj = nn.Linear(self.d_inner, num_heads + 2 * num_heads * d_state)

        dt_init = torch.exp(torch.linspace(math.log(0.01), math.log(0.1), num_heads))
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_bias = nn.Parameter(inv_dt)

        self.A_log = nn.Parameter(torch.log(torch.linspace(0.5, 4.0, num_heads)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ASE Prompt Pool
        self.prompt_pool = nn.Parameter(torch.randn(num_prompts, num_heads * d_state) * 0.02)
        self.prompt_scale = nn.Parameter(torch.tensor(0.5))

        # Zero-bias positive physics projection for telescoping Beer-Lambert dt modulation
        # Maps 6 optical-depth increments (delta_d * [beta_D(3), beta_B(3)]) -> num_heads >= 0
        self.phys_dt_weight = nn.Parameter(torch.full((6, num_heads), -1.0))

        self.out_proj = nn.Linear(self.d_inner, d_model)

    def compute_physics_dt(self, opt_delta: torch.Tensor) -> torch.Tensor:
        """Compute non-negative telescoping Beer-Lambert dt increment (zero when delta_d == 0)."""
        w_pos = F.softplus(self.phys_dt_weight.float())  # (6, H) > 0
        return torch.matmul(opt_delta.float(), w_pos)  # (B, L, H) >= 0

    def forward(
        self,
        x: torch.Tensor,
        prompt_weights: Optional[torch.Tensor] = None,
        opt_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        B_sz, L, _ = x_norm.shape

        xz = self.in_proj(x_norm)
        u, z = xz.chunk(2, dim=-1)
        u = F.silu(u)

        proj = self.ssm_proj(u)
        hn = self.num_heads * self.d_state
        dt_raw, B_flat, C_flat = proj.split([self.num_heads, hn, hn], dim=-1)

        # Positive discretization step strictly in float32
        dt = F.softplus(dt_raw.float() + self.dt_bias.float())  # (B, L, H)
        if opt_delta is not None:
            # Telescoping Beer-Lambert modulation: adds >= 0 strictly proportional to delta_d_k * (beta_D, beta_B)
            dt = dt + self.compute_physics_dt(opt_delta)

        A = -torch.exp(self.A_log.float())

        B_param = B_flat.view(B_sz, L, self.num_heads, self.d_state)
        C_param = C_flat.view(B_sz, L, self.num_heads, self.d_state)

        if prompt_weights is not None:
            prompt_c = torch.matmul(prompt_weights.to(self.prompt_pool.dtype), self.prompt_pool)
            prompt_c = prompt_c.view(B_sz, L, self.num_heads, self.d_state)
            C_param = C_param + self.prompt_scale * prompt_c.to(C_param.dtype)

        u_heads = u.view(B_sz, L, self.num_heads, self.head_dim)
        y_heads = chunked_ssd_scan(u_heads, dt, A, B_param, C_param, self.chunk_size)
        y = y_heads.reshape(B_sz, L, self.d_inner)

        y = y + u * self.D
        y = y * F.silu(z)
        out = self.out_proj(y)
        return residual + out


# ---------------------------------------------------------------------------
# 4. Core Encoder & Decoder (Compatible with official D2Mamba checkpoints)
# ---------------------------------------------------------------------------
class ConvAttentionBlock(nn.Module):
    """Standard block combining convolution with spatial attention (JIT-safe functional upsample)."""

    def __init__(self, dim: int, heads: int = 8, attention_downsample_size: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.AdaptiveAvgPool2d(attention_downsample_size)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.upsample = nn.Upsample(scale_factor=None, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.relu(self.norm(self.conv(x)))
        B, C, H, W = x_conv.shape
        identity = x_conv
        x_down = self.downsample(x_conv)
        S = x_down.shape[-1]
        x_flat = x_down.flatten(2).transpose(1, 2)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).reshape(B, C, S, S)
        attn_out_up = F.interpolate(attn_out, size=(H, W), mode="bilinear", align_corners=False)
        return identity + attn_out_up


class DualStreamEncoder(nn.Module):
    """Spatial + FP32-safe FFT encoder (exact state_dict compatibility with official D2Mamba)."""

    def __init__(self, in_channels: int = 3, base_dim: int = 64):
        super().__init__()
        self.spatial_enc1 = nn.Sequential(nn.Conv2d(in_channels, base_dim, 3, padding=1), ConvAttentionBlock(base_dim))
        self.spatial_enc2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, 3, padding=1, stride=2), ConvAttentionBlock(base_dim * 2)
        )
        self.freq_enc1 = nn.Sequential(nn.Conv2d(2, base_dim, 3, padding=1), nn.ReLU())
        self.freq_enc2 = nn.Sequential(nn.Conv2d(base_dim, base_dim * 2, 3, padding=1, stride=2), nn.ReLU())
        self.fusion1 = nn.Conv2d(base_dim * 2, base_dim, 1)
        self.fusion2 = nn.Conv2d(base_dim * 4, base_dim * 2, 1)
        self.final_conv = nn.Conv2d(base_dim * 2, base_dim * 4, 3, padding=1, stride=2)
        self.feature_dim = base_dim * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            x_fft = torch.fft.fft2(x_f, norm="ortho")
            x_fft_input = torch.stack([torch.abs(x_fft).mean(1), torch.angle(x_fft).mean(1)], dim=1)
        x_fft_input = x_fft_input.to(x.dtype)

        s1 = self.spatial_enc1(x)
        f1 = self.freq_enc1(x_fft_input)
        fused1 = self.fusion1(torch.cat([s1, f1], dim=1))
        s2 = self.spatial_enc2(fused1)
        f2 = self.freq_enc2(f1)
        fused2 = self.fusion2(torch.cat([s2, f2], dim=1))
        return self.final_conv(fused2)


class GatedSynthesisDecoder(nn.Module):
    """Reconstructs the final image from original and Mamba-corrected features."""

    def __init__(self, feature_dim: int = 256, base_dim: int = 64):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1), nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, base_dim * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_dim, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, original_features: torch.Tensor, mamba_corrections: torch.Tensor) -> torch.Tensor:
        fused_input = torch.cat([original_features, mamba_corrections], dim=1)
        gate_map = self.gate(fused_input)
        fused_features = (1.0 - gate_map) * original_features + gate_map * mamba_corrections
        return self.decoder(fused_features)


# ---------------------------------------------------------------------------
# 5. Scan Strategies (Branch 2 Default: PhysicsAttenuationScanStrategy)
# ---------------------------------------------------------------------------
class SemanticGuidedSingleScanStrategy(nn.Module):
    """Branch 1 MambaIRv2 Semantic-Guided Single Scan (SGN + ASE)."""

    def __init__(self, feature_dim: int, mamba_corrector: nn.ModuleList, num_prompts: int = 16):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prompts = num_prompts
        self.mamba_corrector = mamba_corrector

        self.pre_dwconv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )
        hidden_r = max(16, feature_dim // 2)
        self.semantic_router = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_r, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_r, num_prompts, kernel_size=1),
        )
        self.post_dwconv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )

    def compute_semantic_order(self, feat_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = feat_2d.shape
        L = H * W
        logits = self.semantic_router(feat_2d)
        p_soft = F.softmax(logits.float(), dim=1).to(feat_2d.dtype)
        cluster_idx = torch.argmax(p_soft, dim=1)
        conf = p_soft.max(dim=1).values
        sort_key = (cluster_idx.float() + (1.0 - conf.float()) * 0.99).reshape(B, L)
        order = torch.argsort(sort_key, dim=1)
        inv_order = _invert_permutation(order)
        p_flat = p_soft.flatten(2).transpose(1, 2)
        return order, inv_order, p_flat

    def forward(self, features: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, int, Optional[dict]]:
        B, C, H, W = features.shape
        L = H * W
        local_feat = features + self.pre_dwconv(features)
        order, inv_order, p_flat = self.compute_semantic_order(local_feat)

        x_flat = local_feat.flatten(2).transpose(1, 2)
        x_seq = torch.gather(x_flat, dim=1, index=order.unsqueeze(-1).expand(B, L, C))
        p_seq = torch.gather(p_flat, dim=1, index=order.unsqueeze(-1).expand(B, L, self.num_prompts))

        y_seq = x_seq
        for block in self.mamba_corrector:
            y_seq = block(y_seq, prompt_weights=p_seq)

        y_restored = torch.gather(y_seq, dim=1, index=inv_order.unsqueeze(-1).expand(B, L, C))
        y_2d = y_restored.transpose(1, 2).reshape(B, C, H, W)
        y_2d = y_2d + self.post_dwconv(y_2d)
        return y_2d, L, None


class PhysicsAttenuationScanStrategy(nn.Module):
    """
    Branch 2 Physics-Anchored Attenuation Single Scan:
      1. 2D depthwise pre-conv preserving local spatial neighborhoods before reordering.
      2. Orders bottleneck tokens near (clear, small d_hat) -> far (hazy, large d_hat) on GPU.
      3. Computes non-negative incremental depth differences:
             delta_d_k = clamp(d_sorted[k] - d_sorted[k-1], min=0)
         and multiplies by decoupled (beta_D, beta_B) to modulate Mamba dt with exact
         telescoping Beer-Lambert invariance (constant depth => delta_d_k == 0 => no state collapse).
      4. Runs prompt-augmented ASE Mamba blocks + scatters back to 2D + 2D local post-refinement.
    """

    def __init__(
        self,
        feature_dim: int,
        mamba_corrector: nn.ModuleList,
        physics_head: MicroATHead,
        num_prompts: int = 16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prompts = num_prompts
        self.mamba_corrector = mamba_corrector
        self.physics_head = physics_head

        self.pre_dwconv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )
        hidden_r = max(16, feature_dim // 2)
        self.semantic_router = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_r, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_r, num_prompts, kernel_size=1),
        )
        self.post_dwconv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )

    @staticmethod
    def compute_depth_order_and_opt_delta(
        d_hat: torch.Tensor,
        beta_D: torch.Tensor,
        beta_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sort tokens near->far on GPU and compute telescoping opt_delta = delta_d * [beta_D, beta_B]."""
        B, _, H, W = d_hat.shape
        L = H * W
        d_flat = d_hat.float().reshape(B, L)
        order = torch.argsort(d_flat, dim=1)  # ascending: near (clear) -> far (hazy)
        inv_order = _invert_permutation(order)

        d_sorted = torch.gather(d_flat, dim=1, index=order)  # (B, L)
        # Non-negative incremental depth difference: delta_d_0 = 0, delta_d_k = clamp(d[k] - d[k-1], min=0)
        diffs = torch.clamp(d_sorted[:, 1:] - d_sorted[:, :-1], min=0.0)
        delta_d = F.pad(diffs, (1, 0), value=0.0)  # (B, L) >= 0

        beta_cat = torch.cat([beta_D.float(), beta_B.float()], dim=1)  # (B, 6, H, W)
        beta_flat = beta_cat.flatten(2).transpose(1, 2)  # (B, L, 6)
        beta_sorted = torch.gather(beta_flat, dim=1, index=order.unsqueeze(-1).expand(B, L, 6))
        opt_delta = delta_d.unsqueeze(-1) * beta_sorted  # (B, L, 6) >= 0

        return order, inv_order, delta_d, opt_delta

    def forward(
        self,
        features: torch.Tensor,
        x_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
        B, C, H, W = features.shape
        L = H * W

        # 1. Estimate underwater physics (Micro-AT head) on GPU
        d_hat, beta_D, beta_B, B_inf = self.physics_head(x_rgb, features)

        # 2. 2D local pre-conv before reordering
        local_feat = features + self.pre_dwconv(features)
        p_soft = F.softmax(self.semantic_router(local_feat).float(), dim=1).to(local_feat.dtype)
        p_flat = p_soft.flatten(2).transpose(1, 2)  # (B, L, K)

        # 3. Near-to-far depth ordering & telescoping optical-depth increments
        order, inv_order, _, opt_delta = self.compute_depth_order_and_opt_delta(d_hat, beta_D, beta_B)

        x_flat = local_feat.flatten(2).transpose(1, 2)
        x_seq = torch.gather(x_flat, dim=1, index=order.unsqueeze(-1).expand(B, L, C))
        p_seq = torch.gather(p_flat, dim=1, index=order.unsqueeze(-1).expand(B, L, self.num_prompts))

        # 4. Physics-modulated single-scan ASE Mamba
        y_seq = x_seq
        for block in self.mamba_corrector:
            y_seq = block(y_seq, prompt_weights=p_seq, opt_delta=opt_delta)

        # 5. Scatter back to 2D + 2D local refinement
        y_restored = torch.gather(y_seq, dim=1, index=inv_order.unsqueeze(-1).expand(B, L, C))
        y_2d = y_restored.transpose(1, 2).reshape(B, C, H, W)
        y_2d = y_2d + self.post_dwconv(y_2d)

        physics = {
            "d_hat": d_hat,
            "beta_D": beta_D,
            "beta_B": beta_B,
            "B_inf": B_inf,
        }
        return y_2d, L, physics


class RasterScanStrategy(nn.Module):
    def __init__(self, mamba_corrector: nn.ModuleList, **kwargs):
        super().__init__()
        self.mamba_corrector = mamba_corrector

    def forward(self, features: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, int, Optional[dict]]:
        B, C, H, W = features.shape
        y_seq = features.flatten(2).transpose(1, 2)
        for block in self.mamba_corrector:
            y_seq = block(y_seq)
        return y_seq.transpose(1, 2).reshape(B, C, H, W), H * W, None


# ---------------------------------------------------------------------------
# 6. Main D2Mamba Model (Branch 2: Underwater Physics-Anchored Single Scan)
# ---------------------------------------------------------------------------
class D2Mamba(nn.Module):
    """
    D2Mamba with Micro-AT Physics Head + Near-to-Far Telescoping Beer-Lambert Single Scan.
    Compatible with official D2Mamba encoder/decoder checkpoints via --init_ckpt.
    """

    def __init__(
        self,
        scan_strategy_name: str = "physics_attenuation",
        in_channels: int = 3,
        base_dim: int = 24,
        mamba_layers: int = 4,
        num_paths: int = 4,
        num_prompts: int = 16,
        d_state: int = 16,
        num_heads: int = 4,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.scan_strategy_name = scan_strategy_name
        self.supports_return_aux = True

        self.encoder = DualStreamEncoder(in_channels, base_dim)
        feature_dim = self.encoder.feature_dim

        self.mamba_corrector = nn.ModuleList(
            [
                ASEMambaBlock(
                    d_model=feature_dim,
                    d_state=d_state,
                    expand=2,
                    num_heads=num_heads,
                    num_prompts=num_prompts,
                    chunk_size=chunk_size,
                )
                for _ in range(mamba_layers)
            ]
        )

        self.decoder = GatedSynthesisDecoder(feature_dim, base_dim)

        if scan_strategy_name == "physics_attenuation":
            self.physics_head = MicroATHead(feature_dim=feature_dim, hidden_dim=32, beta_max=4.0)
            self.scan_strategy = PhysicsAttenuationScanStrategy(
                feature_dim=feature_dim,
                mamba_corrector=self.mamba_corrector,
                physics_head=self.physics_head,
                num_prompts=num_prompts,
            )
        elif scan_strategy_name == "mambairv2":
            self.physics_head = None
            self.scan_strategy = SemanticGuidedSingleScanStrategy(
                feature_dim=feature_dim,
                mamba_corrector=self.mamba_corrector,
                num_prompts=num_prompts,
            )
        elif scan_strategy_name == "raster":
            self.physics_head = None
            self.scan_strategy = RasterScanStrategy(mamba_corrector=self.mamba_corrector)
        else:
            raise ValueError(
                f"Unknown scan strategy '{scan_strategy_name}'. "
                f"Supported on exp/2-physics-attenuation-scan: ['physics_attenuation', 'mambairv2', 'raster']"
            )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = self.encoder(x)
        if self.scan_strategy_name == "physics_attenuation":
            mamba_corrections, path_len, physics = self.scan_strategy(features, x_rgb=x)
        else:
            mamba_corrections, path_len, physics = self.scan_strategy(features)

        output_image = self.decoder(features, mamba_corrections)

        if self.training:
            if return_aux and physics is not None:
                return output_image, path_len, {"physics": physics}
            return output_image, path_len

        return output_image, path_len, physics
