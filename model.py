"""
D2Mamba Research — Branch 1: exp/1-mambairv2-gpu-scan
100% GPU-native MambaIRv2-inspired Semantic-Guided Single Scan (SGN + prompt-augmented ASE)
with exact pure-PyTorch chunked SSD scan (zero CPU heapq, zero .cpu()/.numpy(), zero mamba-ssm).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Pure-PyTorch Exact Chunked SSD Scan (Mamba-2 Duality, 100% GPU & JIT-Safe)
# ---------------------------------------------------------------------------
@torch.jit.script
def _invert_permutation(order: torch.Tensor) -> torch.Tensor:
    """Compute exact inverse permutation on GPU in O(L) with zero CPU sync.
    Args:
        order: (B, L) int64 permutation indices.
    Returns:
        inv_order: (B, L) int64 such that gather(gather(x, order), inv_order) == x.
    """
    B, L = order.shape
    arange = torch.arange(L, device=order.device, dtype=order.dtype).unsqueeze(0).expand(B, L)
    inv_order = torch.empty_like(order)
    inv_order.scatter_(1, order, arange)
    return inv_order


@torch.jit.script
def _inter_chunk_scan(chunk_decay: torch.Tensor, chunk_states: torch.Tensor) -> torch.Tensor:
    """Exact Hillis-Steele associative prefix scan over chunks.
    Computes exclusive carried states H_{m-1} where H_m = chunk_decay[m] * H_{m-1} + chunk_states[m].
    Since chunk_decay in (0, 1], this never overflows in FP32.

    Args:
        chunk_decay: (B, H, M) float32 in (0, 1]
        chunk_states: (B, H, M, P, N) float32
    Returns:
        prev_states: (B, H, M, P, N) float32 with prev_states[:, :, 0] == 0
    """
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
    """Exact pure-PyTorch multi-head scalar-decay Structured State Space Duality (SSD) scan.
    Solves h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t (x) x_t, y_t = <h_t, C_t> exactly.
    All exponentials & cumulative sums are computed strictly in float32 for AMP safety.

    Args:
        x: (B, L, H, P)
        dt: (B, L, H) strictly positive
        A: (H,) strictly negative
        B_param: (B, L, H, N)
        C_param: (B, L, H, N)
        chunk_size: int (default 64)
    Returns:
        y: (B, L, H, P) in x.dtype
    """
    orig_dtype = x.dtype
    B_sz, L, H, P = x.shape
    N = B_param.shape[-1]

    # Dynamic padding to multiple of chunk_size
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

    # Reshape to (B, H, M, Q, ...) in float32 for numerical stability
    dt_f = dt.float().permute(0, 2, 1).reshape(B_sz, H, M, Q)
    x_f = x.float().permute(0, 2, 1, 3).reshape(B_sz, H, M, Q, P)
    B_f = B_param.float().permute(0, 2, 1, 3).reshape(B_sz, H, M, Q, N)
    C_f = C_param.float().permute(0, 2, 1, 3).reshape(B_sz, H, M, Q, N)

    # Log-decay per step: log_a <= 0 ALWAYS since dt > 0 and A < 0
    log_a = dt_f * A.float().view(1, H, 1, 1)
    log_a = torch.clamp(log_a, max=0.0)
    cum_log_a = torch.cumsum(log_a, dim=-1)  # (B, H, M, Q) <= 0

    # 1. Intra-chunk 1-semiseparable lower-triangular decay matrix L_mat
    # For i >= j: cum_log_a[i] - cum_log_a[j] <= 0 -> exp(...) in (0, 1], zero overflow!
    diff = cum_log_a.unsqueeze(-1) - cum_log_a.unsqueeze(-2)  # (B, H, M, Q, Q)
    tril_mask = torch.tril(torch.ones((Q, Q), device=x.device, dtype=torch.bool))
    diff_safe = torch.where(tril_mask, diff.clamp(max=0.0), torch.full_like(diff, -1e9))
    L_mat = torch.exp(diff_safe) * tril_mask.to(diff.dtype)

    x_scaled = x_f * dt_f.unsqueeze(-1)  # (B, H, M, Q, P)
    CB = torch.matmul(C_f, B_f.transpose(-1, -2))  # (B, H, M, Q, Q)
    Y_diag = torch.matmul(L_mat * CB, x_scaled)  # (B, H, M, Q, P)

    # 2. Local chunk end-states H_local: (B, H, M, P, N)
    decay_to_end = torch.exp((cum_log_a[:, :, :, -1:] - cum_log_a).clamp(max=0.0))
    B_decayed = B_f * decay_to_end.unsqueeze(-1)
    H_local = torch.matmul(x_scaled.transpose(-1, -2), B_decayed)

    # 3. Inter-chunk associative scan across M chunks
    chunk_decay = torch.exp(cum_log_a[:, :, :, -1].clamp(max=0.0))  # (B, H, M)
    H_prev = _inter_chunk_scan(chunk_decay, H_local)  # (B, H, M, P, N)

    # 4. Cross-chunk off-diagonal contribution
    decay_from_start = torch.exp(cum_log_a.clamp(max=0.0))  # (B, H, M, Q)
    C_decayed = C_f * decay_from_start.unsqueeze(-1)
    Y_off = torch.matmul(C_decayed, H_prev.transpose(-1, -2))  # (B, H, M, Q, P)

    Y = (Y_diag + Y_off).reshape(B_sz, H, L_pad, P).permute(0, 2, 1, 3)  # (B, L_pad, H, P)
    if pad_len > 0:
        Y = Y[:, :L]
    return Y.to(orig_dtype)


# ---------------------------------------------------------------------------
# 2. MambaIRv2-Inspired Attentive State-Space Block (ASE)
# ---------------------------------------------------------------------------
class ASEMambaBlock(nn.Module):
    """
    Attentive State-Space Block with Prompt-Augmented C-Projection (ASE).
    Uses exact pure-PyTorch chunked SSD scan without 1D conv1d across reordered tokens
    (2D local neighborhoods are preserved beforehand by a 2D depthwise pre-conv).
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
        assert self.d_inner % num_heads == 0, f"d_inner ({self.d_inner}) must be divisible by num_heads ({num_heads})"
        self.head_dim = self.d_inner // num_heads
        self.num_prompts = num_prompts
        self.chunk_size = chunk_size

        self.norm = nn.LayerNorm(d_model)
        # Keep in_proj, D, out_proj names & shapes compatible with official D2Mamba ckpt!
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Dynamic SSM projections: dt (H), B (H*N), C_base (H*N)
        self.ssm_proj = nn.Linear(self.d_inner, num_heads + 2 * num_heads * d_state)

        # Well-conditioned dt bias (inverse softplus of [0.01 .. 0.1])
        dt_init = torch.exp(torch.linspace(math.log(0.01), math.log(0.1), num_heads))
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_bias = nn.Parameter(inv_dt)

        # Per-head log decay rate A = -exp(A_log)
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.5, 4.0, num_heads)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ASE Prompt Pool augmenting C-projection for non-causal semantic retrieval in 1 scan
        self.prompt_pool = nn.Parameter(torch.randn(num_prompts, num_heads * d_state) * 0.02)
        self.prompt_scale = nn.Parameter(torch.tensor(0.5))

        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(
        self,
        x: torch.Tensor,
        prompt_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            prompt_weights: optional (B, L, num_prompts) soft semantic routing distribution
        """
        residual = x
        x_norm = self.norm(x)
        B_sz, L, _ = x_norm.shape

        xz = self.in_proj(x_norm)
        u, z = xz.chunk(2, dim=-1)
        u = F.silu(u)

        proj = self.ssm_proj(u)
        hn = self.num_heads * self.d_state
        dt_raw, B_flat, C_flat = proj.split([self.num_heads, hn, hn], dim=-1)

        # Compute positive dt strictly in float32
        dt = F.softplus(dt_raw.float() + self.dt_bias.float())  # (B, L, H)
        A = -torch.exp(self.A_log.float())  # (H,) < 0

        B_param = B_flat.view(B_sz, L, self.num_heads, self.d_state)
        C_param = C_flat.view(B_sz, L, self.num_heads, self.d_state)

        # Prompt-augmented C-projection (ASE)
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
# 3. Core Encoder & Decoder (Compatible with official D2Mamba checkpoints)
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
        # Functional interpolate avoids mutating self.upsample.size (100% JIT-trace safe!)
        attn_out_up = F.interpolate(attn_out, size=(H, W), mode="bilinear", align_corners=False)
        return identity + attn_out_up


class DualStreamEncoder(nn.Module):
    """
    Encodes the image using two parallel streams: spatial (pixel) and frequency (FFT).
    Identical state_dict to official D2Mamba; FFT runs in float32 for FP16/AMP safety at 720p.
    """

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
        # Run FFT in FP32 so cuFFT never crashes on non-power-of-two FP16 resolutions like 720x1280
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            x_fft = torch.fft.fft2(x_f, norm="ortho")
            x_fft_mag = torch.abs(x_fft)
            x_fft_phase = torch.angle(x_fft)
            x_fft_input = torch.stack([x_fft_mag.mean(1), x_fft_phase.mean(1)], dim=1)
        x_fft_input = x_fft_input.to(x.dtype)

        s1 = self.spatial_enc1(x)
        f1 = self.freq_enc1(x_fft_input)
        fused1 = self.fusion1(torch.cat([s1, f1], dim=1))
        s2 = self.spatial_enc2(fused1)
        f2 = self.freq_enc2(f1)
        fused2 = self.fusion2(torch.cat([s2, f2], dim=1))
        final_features = self.final_conv(fused2)
        return final_features


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
# 4. GPU-Native Scan Strategies (MambaIRv2 SGN+ASE Default + Pure-GPU Baselines)
# ---------------------------------------------------------------------------
class SemanticGuidedSingleScanStrategy(nn.Module):
    """
    100% GPU-native MambaIRv2-inspired Semantic-Guided Single Scan (SGN + ASE):
      1. Lightweight 2D depthwise pre-conv BEFORE token reordering (preserves local 2D neighborhoods).
      2. Soft/argmax semantic routing on GPU + torch.argsort token grouping (zero CPU sync).
      3. Single-direction prompt-augmented ASE Mamba blocks.
      4. O(L) GPU scatter back to 2D + lightweight 2D depthwise post-refinement.
    """

    def __init__(
        self,
        feature_dim: int,
        mamba_corrector: nn.ModuleList,
        num_prompts: int = 16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prompts = num_prompts
        self.mamba_corrector = mamba_corrector

        # 2D depthwise local pre-conv BEFORE token reordering
        self.pre_dwconv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )

        # Learnable semantic router predicting K degradation/semantic group scores
        hidden_r = max(16, feature_dim // 2)
        self.semantic_router = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_r, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_r, num_prompts, kernel_size=1),
        )

        # Lightweight 2D depthwise post-conv smoothing semantic cluster boundaries
        self.post_dwconv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )

    def compute_semantic_order(
        self, feat_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute GPU semantic grouping permutation, inverse permutation, and soft routing probs."""
        B, C, H, W = feat_2d.shape
        L = H * W
        logits = self.semantic_router(feat_2d)  # (B, K, H, W)
        p_soft = F.softmax(logits.float(), dim=1).to(feat_2d.dtype)  # (B, K, H, W)

        # Group by argmax semantic cluster, ordered smoothly by confidence within each cluster
        cluster_idx = torch.argmax(p_soft, dim=1)  # (B, H, W)
        conf = p_soft.max(dim=1).values  # (B, H, W) in [1/K, 1]
        sort_key = (cluster_idx.float() + (1.0 - conf.float()) * 0.99).reshape(B, L)

        order = torch.argsort(sort_key, dim=1)  # 100% GPU-native, zero CPU sync
        inv_order = _invert_permutation(order)
        p_flat = p_soft.flatten(2).transpose(1, 2)  # (B, L, K)
        return order, inv_order, p_flat

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, int, Optional[dict]]:
        B, C, H, W = features.shape
        L = H * W

        # 1. Preserve 2D local neighborhoods BEFORE token reordering
        local_feat = features + self.pre_dwconv(features)

        # 2. Semantic-Guided Neighboring (SGN) reordering on GPU
        order, inv_order, p_flat = self.compute_semantic_order(local_feat)

        x_flat = local_feat.flatten(2).transpose(1, 2)  # (B, L, C)
        x_seq = torch.gather(x_flat, dim=1, index=order.unsqueeze(-1).expand(B, L, C))
        p_seq = torch.gather(p_flat, dim=1, index=order.unsqueeze(-1).expand(B, L, self.num_prompts))

        # 3. Single-direction prompt-augmented ASE Mamba scan
        y_seq = x_seq
        for block in self.mamba_corrector:
            y_seq = block(y_seq, prompt_weights=p_seq)

        # 4. Inverse permutation back to 2D spatial layout + 2D local refinement
        y_restored = torch.gather(y_seq, dim=1, index=inv_order.unsqueeze(-1).expand(B, L, C))
        y_2d = y_restored.transpose(1, 2).reshape(B, C, H, W)
        y_2d = y_2d + self.post_dwconv(y_2d)

        return y_2d, L, None


class RasterScanStrategy(nn.Module):
    """Fast GPU-native single-direction raster scan ablation using ASEMambaBlock."""

    def __init__(self, mamba_corrector: nn.ModuleList, **kwargs):
        super().__init__()
        self.mamba_corrector = mamba_corrector

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, int, Optional[dict]]:
        B, C, H, W = features.shape
        L = H * W
        y_seq = features.flatten(2).transpose(1, 2)
        for block in self.mamba_corrector:
            y_seq = block(y_seq, prompt_weights=None)
        return y_seq.transpose(1, 2).reshape(B, C, H, W), L, None


class BidirectionalScanStrategy(nn.Module):
    """Fast GPU-native bidirectional raster scan ablation using ASEMambaBlock."""

    def __init__(self, mamba_corrector: nn.ModuleList, **kwargs):
        super().__init__()
        self.mamba_corrector = mamba_corrector

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, int, Optional[dict]]:
        B, C, H, W = features.shape
        L = H * W
        flat = features.flatten(2).transpose(1, 2)
        y_fwd = flat
        y_bwd = torch.flip(flat, dims=[1])
        for block in self.mamba_corrector:
            y_fwd = block(y_fwd, prompt_weights=None)
            y_bwd = block(y_bwd, prompt_weights=None)
        y_out = 0.5 * (y_fwd + torch.flip(y_bwd, dims=[1]))
        return y_out.transpose(1, 2).reshape(B, C, H, W), L, None


# ---------------------------------------------------------------------------
# 5. Main D2Mamba Model (Branch 1: MambaIRv2 GPU-Native Semantic-Guided Scan)
# ---------------------------------------------------------------------------
class D2Mamba(nn.Module):
    """
    D2Mamba with GPU-native MambaIRv2 Semantic-Guided Single Scan (SGN + ASE).
    Compatible with official D2Mamba encoder/decoder checkpoints via --init_ckpt.
    """

    def __init__(
        self,
        scan_strategy_name: str = "mambairv2",
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

        if scan_strategy_name == "mambairv2":
            self.scan_strategy = SemanticGuidedSingleScanStrategy(
                feature_dim=feature_dim,
                mamba_corrector=self.mamba_corrector,
                num_prompts=num_prompts,
            )
        elif scan_strategy_name == "raster":
            self.scan_strategy = RasterScanStrategy(mamba_corrector=self.mamba_corrector)
        elif scan_strategy_name == "bidirectional":
            self.scan_strategy = BidirectionalScanStrategy(mamba_corrector=self.mamba_corrector)
        else:
            raise ValueError(
                f"Unknown scan strategy '{scan_strategy_name}'. "
                f"Supported on exp/1-mambairv2-gpu-scan: ['mambairv2', 'raster', 'bidirectional']"
            )

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        mamba_corrections, path_len, visualizations = self.scan_strategy(features)
        output_image = self.decoder(features, mamba_corrections)

        if self.training:
            return output_image, path_len
        return output_image, path_len, visualizations
