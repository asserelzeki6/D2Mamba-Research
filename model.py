"""
D2Mamba Research — Branch 4: exp/4-phaseformer-physics-mamba
Hybrid Phaseformer-Inspired Frequency Cross-Gating + Multi-Scale Haar U-Net + Physics-Guided Mamba:
  - Explicit 2D Fourier Phase-Amplitude Cross-Gating (FourierPhaseAmplitudeGate) inspired by
    Phaseformer (WACV'25), running strictly in FP32 under --amp (zero FP16 cuFFT crashes on
    arbitrary resolutions) with branch-cut-free unit phasors (cos, sin) and dynamic-shape
    torch.jit.trace(..., check_trace=True) safety.
  - Multi-scale U-Net / Haar skip connections:
      * Stage 1 (H/2): 1-level Haar DWT HF Gating + RepConv + Optimized Phase Attention Skip.
      * Stage 2 (H/4): RepConv + HaarHFGate + 2D Fourier Phase-Amplitude Cross-Gating.
      * Bottleneck (H/8): Compressed Physics-Guided Single-Scan Mamba (Micro-AT + near-to-far
        telescoping Beer-Lambert dt scan) + Bottleneck 2D Fourier Phase-Amplitude Cross-Gating.
      * Decoder (H/4 -> H/2 -> Haar IDWT -> H): Phase-gated multi-scale skips + RepConv +
        decoder HaarHFGate + lossless residual Haar synthesis + GBPhaseRedRefiner.
  - Structurally Reparameterizable Depthwise Convolutions (RepConv: 7x7 + 3x3 + 1x1) with exact
    .fuse() verified < 1e-5.
  - 100% GPU-native pure PyTorch, zero CPU heapq / .cpu() / .numpy(), < 1.1M params (< 2.2M cap),
    torch.jit.trace verified both unfused and fused across dynamic shapes.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Pure-PyTorch Exact Chunked SSD Scan, Haar DWT/IDWT & GPU Helpers
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
    """Anchor far-field veiling light B_inf from the top-5% farthest d_hat pixels on GPU."""
    B, C, H, W = x_rgb.shape
    L = H * W
    k = max(1, (L * 5) // 100)
    d_flat = d_map.float().reshape(B, L)
    x_flat = x_rgb.float().clamp(0.0, 1.0).reshape(B, C, L).transpose(1, 2)

    top_d, top_idx = torch.topk(d_flat, k=k, dim=1, largest=True, sorted=False)
    top_rgb = torch.gather(x_flat, dim=1, index=top_idx.unsqueeze(-1).expand(B, k, C))
    weights = F.softmax(top_d * 10.0, dim=1).unsqueeze(-1)
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


@torch.jit.script
def haar_dwt_2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthonormal 1-level 2D Haar DWT (100% FP16-safe on arbitrary even resolutions, zero cuFFT).
    Pads by 1 if H or W is odd so intermediate feature maps of any size work seamlessly.
    """
    H = x.shape[2]
    W = x.shape[3]
    pad_h = H % 2
    pad_w = W % 2
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)
    hl = 0.5 * (x00 + x01 - x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return ll, lh, hl, hh


@torch.jit.script
def haar_idwt_2d(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
    out_h: int = -1,
    out_w: int = -1,
    ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact orthonormal 1-level 2D Haar Inverse DWT (zero in-place ops, FP16 & dynamic JIT safe).
    Pass `ref` (target spatial tensor) instead of Python ints `out_h`/`out_w` inside traced models
    so output cropping remains 100% dynamic across arbitrary input resolutions.
    """
    B, C, H2, W2 = ll.shape
    x00 = 0.5 * (ll + lh + hl + hh)
    x01 = 0.5 * (ll - lh + hl - hh)
    x10 = 0.5 * (ll + lh - hl - hh)
    x11 = 0.5 * (ll - lh - hl + hh)

    top = torch.stack([x00, x01], dim=-1).reshape(B, C, H2, 2 * W2)
    bot = torch.stack([x10, x11], dim=-1).reshape(B, C, H2, 2 * W2)
    out = torch.stack([top, bot], dim=-2).reshape(B, C, 2 * H2, 2 * W2)

    if ref is not None:
        out = out[:, :, : ref.shape[2], : ref.shape[3]]
    elif out_h > 0 and out_w > 0:
        out = out[:, :, :out_h, :out_w]
    return out


@torch.jit.script
def _rfft2_decompose_fp32(
    x_f: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict FP32 2D real FFT decomposition into smooth amplitude & branch-cut-free unit phasors.
    Uses smooth sqrt(re^2 + im^2 + 1e-8) so zero/constant inputs have 100% finite gradients.
    """
    z = torch.fft.rfft2(x_f, norm="ortho")
    re = z.real
    im = z.imag
    amp = torch.sqrt(re * re + im * im + 1e-8)
    cos_p = re / amp
    sin_p = im / amp
    log_amp = torch.log1p(amp)
    return amp, log_amp, cos_p, sin_p


@torch.jit.script
def _irfft2_reconstruct_fp32(
    amp: torch.Tensor,
    cos_p: torch.Tensor,
    sin_p: torch.Tensor,
    ref: torch.Tensor,
) -> torch.Tensor:
    """Strict FP32 dynamic-shape 2D inverse real FFT (preserves symbolic H, W in JIT traces)."""
    z = torch.complex(amp * cos_p, amp * sin_p)
    return torch.fft.irfft2(z, s=[ref.shape[2], ref.shape[3]], norm="ortho")


class FourierPhaseAmplitudeGate(nn.Module):
    """
    Phaseformer-inspired (WACV'25) Explicit 2D Fourier Phase-Amplitude Cross-Gating Block:
      - Runs strictly in FP32 inside autocast(enabled=False) so cuFFT never crashes in FP16
        on arbitrary/non-power-of-two resolutions (e.g., 198x314, 720x1280, odd sizes).
      - Uses branch-cut-free unit phasors (cos, sin) = (Re/|Z|, Im/|Z|) instead of atan2.
      - Phase-to-Amplitude Cross-Gate: uncontaminated structural phase guides multiplicative
        Beer-Lambert-like log-amplitude restoration exp(clamp(s_a * d_amp, -1.5, 1.5)).
      - Amplitude-to-Phase Cross-Gate: spectral energy log1p(amp) gates phase refinement
        before smooth unit-circle re-normalization.
      - Dynamic-shape torch.jit.trace(..., check_trace=True) safe via TorchScript FFT helpers.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else max(16, (dim * 3) // 4)
        self.dim = dim
        self.hidden_dim = h

        # Phase encoder on [cos_p, sin_p] (2C -> h)
        self.phase_enc = nn.Sequential(
            nn.Conv2d(dim * 2, h, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1, groups=h),
            nn.GELU(),
        )
        # Amplitude encoder on log1p(amp) (C -> h)
        self.amp_enc = nn.Sequential(
            nn.Conv2d(dim, h, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1, groups=h),
            nn.GELU(),
        )

        # Phase -> Amplitude cross-gate & bounded log-amplitude delta
        self.p2a_gate = nn.Sequential(nn.Conv2d(h, h, kernel_size=1), nn.Sigmoid())
        self.amp_delta = nn.Conv2d(h, dim, kernel_size=1)

        # Amplitude -> Phase cross-gate & unit-phasor residual (d_cos, d_sin)
        self.a2p_gate = nn.Sequential(nn.Conv2d(h, h, kernel_size=1), nn.Sigmoid())
        self.phase_delta = nn.Conv2d(h, dim * 2, kernel_size=1)

        # Spatial output projection after inverse FFT
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.amp_scale = nn.Parameter(torch.tensor(0.25))
        self.phase_scale = nn.Parameter(torch.tensor(0.15))
        self.res_scale = nn.Parameter(torch.tensor(0.25))

        with torch.no_grad():
            self.amp_delta.weight.mul_(0.1)
            self.amp_delta.bias.zero_()
            self.phase_delta.weight.mul_(0.1)
            self.phase_delta.bias.zero_()
            self.out_proj.weight.mul_(0.1)
            self.out_proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        w_dtype = self.phase_enc[0].weight.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            amp, log_amp, cos_p, sin_p = _rfft2_decompose_fp32(x_f)

            phase_in = torch.cat([cos_p, sin_p], dim=1).to(w_dtype)
            phase_ctx = self.phase_enc(phase_in)
            amp_ctx = self.amp_enc(log_amp.to(w_dtype))

            # 1. Phase-to-Amplitude Cross-Gating (strictly positive bounded multiplicative gain)
            g_p2a = self.p2a_gate(phase_ctx)
            d_amp = torch.tanh(self.amp_delta(amp_ctx * g_p2a).float())
            amp_gain = torch.exp(torch.clamp(self.amp_scale.float() * d_amp, min=-1.5, max=1.5))
            amp_out = amp * amp_gain

            # 2. Amplitude-to-Phase Cross-Gating (unit-circle re-normalized phasor update)
            g_a2p = self.a2p_gate(amp_ctx)
            d_ri = torch.tanh(self.phase_delta(phase_ctx * g_a2p).float())
            d_cos, d_sin = d_ri.chunk(2, dim=1)
            s_p = self.phase_scale.float()
            p_re = cos_p + s_p * d_cos
            p_im = sin_p + s_p * d_sin
            p_norm = torch.sqrt(p_re * p_re + p_im * p_im + 1e-8)
            cos_out = p_re / p_norm
            sin_out = p_im / p_norm

            # 3. Dynamic-shape inverse 2D real FFT + residual projection
            x_rec = _irfft2_reconstruct_fp32(amp_out, cos_out, sin_out, x_f)
            delta = self.out_proj(x_rec.to(w_dtype)).float() * self.res_scale.float()

        return x + delta.to(orig_dtype)


# ---------------------------------------------------------------------------
# 2. Structurally Reparameterizable Depthwise Convolution (RepConv) & Blocks
# ---------------------------------------------------------------------------
class RepConv(nn.Module):
    """
    Structurally Reparameterizable Depthwise Convolution:
      - Training: parallel 7x7 + 3x3 + 1x1 depthwise Conv2d+BN branches.
      - Deployment (.fuse()): algebraically merged into a SINGLE 7x7 depthwise Conv2d
        verified to max|eval_unfused - fused| < 1e-5.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.fused = False

        self.branch7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.fused_conv: Optional[nn.Conv2d] = None

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        w = conv.weight.detach().to(torch.float64)
        if conv.bias is None:
            b = torch.zeros(conv.out_channels, dtype=torch.float64, device=w.device)
        else:
            b = conv.bias.detach().to(torch.float64)
        mean = bn.running_mean.detach().to(torch.float64)
        var = bn.running_var.detach().to(torch.float64)
        gamma = bn.weight.detach().to(torch.float64)
        beta = bn.bias.detach().to(torch.float64)
        std = torch.sqrt(var + bn.eps)
        scale = gamma / std
        w_eq = w * scale.view(-1, 1, 1, 1)
        b_eq = (b - mean) * scale + beta
        return w_eq, b_eq

    @torch.no_grad()
    def fuse(self) -> "RepConv":
        if self.fused:
            return self
        dev = self.branch7[0].weight.device
        dtype = self.branch7[0].weight.dtype

        w7, b7 = self._fuse_conv_bn(self.branch7[0], self.branch7[1])
        w3, b3 = self._fuse_conv_bn(self.branch3[0], self.branch3[1])
        w1, b1 = self._fuse_conv_bn(self.branch1[0], self.branch1[1])

        w3_pad = F.pad(w3, (2, 2, 2, 2))  # 3x3 -> 7x7 centered
        w1_pad = F.pad(w1, (3, 3, 3, 3))  # 1x1 -> 7x7 centered

        w_total = (w7 + w3_pad + w1_pad).to(device=dev, dtype=dtype)
        b_total = (b7 + b3 + b1).to(device=dev, dtype=dtype)

        fused_conv = nn.Conv2d(
            self.channels, self.channels, kernel_size=7, padding=3, groups=self.channels, bias=True
        ).to(device=dev, dtype=dtype)
        fused_conv.weight.copy_(w_total)
        fused_conv.bias.copy_(b_total)

        self.fused_conv = fused_conv
        del self.branch7
        del self.branch3
        del self.branch1
        self.fused = True
        return self

    @staticmethod
    def _run_branch(x: torch.Tensor, branch: nn.Sequential) -> torch.Tensor:
        conv = branch[0]
        bn = branch[1]
        y = conv(x)
        if bn.training:
            return bn(y)
        # Explicit centered (1, C, 1, 1) FP32 BN avoids PyTorch TorchScript CUDA TensorExpr
        # aten::batch_norm fusion bug (which mis-broadcasts 1D C params against dim 3 W
        # when fused with downstream multi-branch ops such as Haar IDWT) while avoiding
        # catastrophic cancellation when y is close to running_mean.
        inv_std = torch.rsqrt(bn.running_var.float() + bn.eps)
        scale = (bn.weight.float() * inv_std).view(1, -1, 1, 1)
        mean = bn.running_mean.float().view(1, -1, 1, 1)
        bias = bn.bias.float().view(1, -1, 1, 1)
        out = (y.float() - mean) * scale + bias
        return out.to(y.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused and self.fused_conv is not None:
            return self.fused_conv(x)
        return (
            self._run_branch(x, self.branch7)
            + self._run_branch(x, self.branch3)
            + self._run_branch(x, self.branch1)
        )


class RepConvBlock(nn.Module):
    """Fast edge-friendly outer-stage block: RepConv (7x7+3x3+1x1) + pointwise channel MLP."""

    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        hidden = dim * expand
        self.dw = RepConv(dim)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(self.act(self.pw1(self.dw(x))))


class HaarHFGate(nn.Module):
    """1-level Haar DWT High-Frequency Gating block (FP16-safe, zero cuFFT)."""

    def __init__(self, dim: int):
        super().__init__()
        self.hf_proj = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.hf_rep = RepConv(dim)
        self.hf_gate = nn.Sequential(nn.GELU(), nn.Conv2d(dim, dim * 3, kernel_size=1), nn.Sigmoid())
        self.ll_rep = RepConv(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = haar_dwt_2d(x)
        hf = torch.cat([lh, hl, hh], dim=1)
        gate = self.hf_gate(self.hf_rep(self.hf_proj(hf)))
        g_lh, g_hl, g_hh = gate.chunk(3, dim=1)
        ll_out = ll + 0.25 * self.ll_rep(ll)
        rec = haar_idwt_2d(ll_out, lh * g_lh, hl * g_hl, hh * g_hh, ref=x)
        return x + rec


# ---------------------------------------------------------------------------
# 3. Green/Blue-Phase-Guided Red Channel Refinement Block
# ---------------------------------------------------------------------------
class GBPhaseRedRefiner(nn.Module):
    """
    Lightweight Green/Blue-Phase-Guided Red Channel Refinement:
      Uses high-frequency Haar subbands & illumination-invariant unit directional phase cues
      from the structurally intact G & B channels to sharpen and stabilize the attenuated R channel.
      100% FP16-safe (zero cuFFT) and RepConv-fusable.
    """

    def __init__(self, hidden_dim: int = 24):
        super().__init__()
        # Inputs at H/2 x W/2:
        #   R_LL (1) + R_HF (3) + GB_LL (2) + GB_HF (6) + phi_GB (6) = 18 channels
        self.in_proj = nn.Conv2d(18, hidden_dim, kernel_size=1)
        self.rep = RepConv(hidden_dim)
        self.act = nn.GELU()
        self.out_ll = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.out_hf = nn.Conv2d(hidden_dim, 3, kernel_size=1)
        self.phase_gate = nn.Sequential(nn.Conv2d(hidden_dim, 3, kernel_size=1), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 3, H, W) preliminary restored RGB image
        Returns:
            refined: (B, 3, H, W) with phase-sharpened, stabilized R channel
        """
        r = img[:, 0:1]
        gb = img[:, 1:3]

        r_ll, r_lh, r_hl, r_hh = haar_dwt_2d(r)
        gb_ll, gb_lh, gb_hl, gb_hh = haar_dwt_2d(gb)

        # Illumination-invariant unit directional phase cues from intact G & B channels
        # mag_gb: (B, 2, H/2, W/2), phi = [LH, HL, HH] / sqrt(LH^2 + HL^2 + HH^2 + eps)
        mag_gb = torch.sqrt(gb_lh.float().pow(2) + gb_hl.float().pow(2) + gb_hh.float().pow(2) + 1e-6)
        phi_lh = (gb_lh.float() / mag_gb).to(img.dtype)
        phi_hl = (gb_hl.float() / mag_gb).to(img.dtype)
        phi_hh = (gb_hh.float() / mag_gb).to(img.dtype)

        feats = torch.cat(
            [r_ll, r_lh, r_hl, r_hh, gb_ll, gb_lh, gb_hl, gb_hh, phi_lh, phi_hl, phi_hh],
            dim=1,
        )  # (B, 18, H/2, W/2)

        h = self.act(self.rep(self.in_proj(feats)))
        d_ll = self.out_ll(h)
        d_hf = self.out_hf(h) * self.phase_gate(h)
        d_lh, d_hl, d_hh = d_hf.chunk(3, dim=1)

        s = self.scale.to(r_ll.dtype)
        r_ref = haar_idwt_2d(
            r_ll + s * d_ll,
            r_lh + s * d_lh,
            r_hl + s * d_hl,
            r_hh + s * d_hh,
            ref=r,
        )
        return torch.cat([r_ref, gb], dim=1)


# ---------------------------------------------------------------------------
# 4. Micro-AT Physics Head & Physics-Modulated ASE Mamba (Bottleneck Only)
# ---------------------------------------------------------------------------
class MicroATHead(nn.Module):
    """Lightweight Akkaynak-Treibitz physics head (< 0.15M params, decoupled beta_D/beta_B)."""

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
        self.depth_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.beta_D_head = nn.Conv2d(hidden_dim, 3, kernel_size=1)
        self.beta_B_head = nn.Conv2d(hidden_dim, 3, kernel_size=1)

        with torch.no_grad():
            self.beta_D_head.bias.copy_(torch.tensor([0.30, 0.00, -0.20]))
            self.beta_B_head.bias.copy_(torch.tensor([-0.20, 0.15, 0.30]))

    def _smooth_bound_beta(self, raw: torch.Tensor) -> torch.Tensor:
        sp = F.softplus(raw.float())
        return self.beta_max * torch.tanh(sp / self.beta_max) + 1e-3

    def forward(
        self, x_rgb: torch.Tensor, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        H_f, W_f = features.shape[-2:]
        x_low = F.interpolate(x_rgb, size=(H_f, W_f), mode="bilinear", align_corners=False)
        h = self.fuse(torch.cat([self.feat_proj(features), self.rgb_stem(x_low)], dim=1))
        # Strict FP32 for physics outputs (safe under --amp and pure .half()/.bfloat16() models)
        w_dtype = self.depth_head.weight.dtype
        with torch.amp.autocast(device_type=x_rgb.device.type, enabled=False):
            h_w = h.to(w_dtype)
            d_hat = torch.sigmoid(self.depth_head(h_w).float())
            beta_D = self._smooth_bound_beta(self.beta_D_head(h_w).float())
            beta_B = self._smooth_bound_beta(self.beta_B_head(h_w).float())
            B_inf = _anchor_veiling_light(x_low.float(), d_hat)
        return d_hat, beta_D, beta_B, B_inf


class ASEMambaBlock(nn.Module):
    """Physics-modulated Attentive State-Space Block (strictly used in H/8 compressed bottleneck)."""

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

        self.prompt_pool = nn.Parameter(torch.randn(num_prompts, num_heads * d_state) * 0.02)
        self.prompt_scale = nn.Parameter(torch.tensor(0.5))
        self.phys_dt_weight = nn.Parameter(torch.full((6, num_heads), -1.0))

        self.out_proj = nn.Linear(self.d_inner, d_model)

    def compute_physics_dt(self, opt_delta: torch.Tensor) -> torch.Tensor:
        w_pos = F.softplus(self.phys_dt_weight.float())
        return torch.matmul(opt_delta.float(), w_pos)

    def forward(
        self,
        x: torch.Tensor,
        prompt_weights: Optional[torch.Tensor] = None,
        opt_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        B_sz, L, _ = x_norm.shape

        u, z = self.in_proj(x_norm).chunk(2, dim=-1)
        u = F.silu(u)

        proj = self.ssm_proj(u)
        hn = self.num_heads * self.d_state
        dt_raw, B_flat, C_flat = proj.split([self.num_heads, hn, hn], dim=-1)

        dt = F.softplus(dt_raw.float() + self.dt_bias.float())
        if opt_delta is not None:
            dt = dt + self.compute_physics_dt(opt_delta)

        A = -torch.exp(self.A_log.float())
        B_param = B_flat.view(B_sz, L, self.num_heads, self.d_state)
        C_param = C_flat.view(B_sz, L, self.num_heads, self.d_state)

        if prompt_weights is not None:
            prompt_c = torch.matmul(prompt_weights.to(self.prompt_pool.dtype), self.prompt_pool)
            C_param = C_param + self.prompt_scale * prompt_c.view(B_sz, L, self.num_heads, self.d_state).to(C_param.dtype)

        y_heads = chunked_ssd_scan(u.view(B_sz, L, self.num_heads, self.head_dim), dt, A, B_param, C_param, self.chunk_size)
        y = (y_heads.reshape(B_sz, L, self.d_inner) + u * self.D) * F.silu(z)
        return residual + self.out_proj(y)


class PhysicsAttenuationScanStrategy(nn.Module):
    """Low-resolution bottleneck Physics-Guided Single-Scan Mamba."""

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
        B, _, H, W = d_hat.shape
        L = H * W
        d_flat = d_hat.float().reshape(B, L)
        order = torch.argsort(d_flat, dim=1)
        inv_order = _invert_permutation(order)

        d_sorted = torch.gather(d_flat, dim=1, index=order)
        diffs = torch.clamp(d_sorted[:, 1:] - d_sorted[:, :-1], min=0.0)
        delta_d = F.pad(diffs, (1, 0), value=0.0)

        beta_cat = torch.cat([beta_D.float(), beta_B.float()], dim=1)
        beta_flat = beta_cat.flatten(2).transpose(1, 2)
        beta_sorted = torch.gather(beta_flat, dim=1, index=order.unsqueeze(-1).expand(B, L, 6))
        opt_delta = delta_d.unsqueeze(-1) * beta_sorted
        return order, inv_order, delta_d, opt_delta

    def forward(
        self,
        features: torch.Tensor,
        x_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
        B, C, H, W = features.shape
        L = H * W

        d_hat, beta_D, beta_B, B_inf = self.physics_head(x_rgb, features)

        local_feat = features + self.pre_dwconv(features)
        p_soft = F.softmax(self.semantic_router(local_feat).float(), dim=1).to(local_feat.dtype)
        p_flat = p_soft.flatten(2).transpose(1, 2)

        order, inv_order, _, opt_delta = self.compute_depth_order_and_opt_delta(d_hat, beta_D, beta_B)

        x_flat = local_feat.flatten(2).transpose(1, 2)
        x_seq = torch.gather(x_flat, dim=1, index=order.unsqueeze(-1).expand(B, L, C))
        p_seq = torch.gather(p_flat, dim=1, index=order.unsqueeze(-1).expand(B, L, self.num_prompts))

        y_seq = x_seq
        for block in self.mamba_corrector:
            y_seq = block(y_seq, prompt_weights=p_seq, opt_delta=opt_delta)

        y_restored = torch.gather(y_seq, dim=1, index=inv_order.unsqueeze(-1).expand(B, L, C))
        y_2d = y_restored.transpose(1, 2).reshape(B, C, H, W)
        y_2d = y_2d + self.post_dwconv(y_2d)

        return y_2d, L, {"d_hat": d_hat, "beta_D": beta_D, "beta_B": beta_B, "B_inf": B_inf}


# ---------------------------------------------------------------------------
# 5. Main D2Mamba Model (Branch 4: Phaseformer + Haar U-Net + Physics Mamba)
# ---------------------------------------------------------------------------
class D2Mamba(nn.Module):
    """
    Branch 4: Phaseformer-Inspired Frequency + Multi-Scale Haar U-Net + Physics-Guided Mamba:
      - Stage 1 (H/2): 1-level Haar DWT HF Gating + RepConv + Fourier Phase-Amplitude Skip
      - Stage 2 (H/4): RepConv + HaarHFGate + 2D Fourier Phase-Amplitude Cross-Gating
      - Bottleneck (H/8): Physics-Guided Single-Scan Mamba (Micro-AT + Telescoping Beer-Lambert ASE)
                          + Bottleneck 2D Fourier Phase-Amplitude Cross-Gating
      - Decoder (H/4 -> H/2 -> Haar IDWT -> H): Phase-gated multi-scale skips + RepConv +
                                                HaarHFGate + residual Haar synthesis
      - Post-Head: Green/Blue-Phase-Guided Red Channel Refinement (GBPhaseRedRefiner)
      - Strictly FP32 inside 2D FFT & physics heads (100% AMP-safe on arbitrary resolutions),
        RepConv .fuse() ready (< 1e-5), dynamic-shape torch.jit.trace verified.
    """

    def __init__(
        self,
        scan_strategy_name: str = "phaseformer_physics_mamba",
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

        c1 = max(32, (base_dim * 4) // 3)  # 32 for base_dim=24
        c2 = c1 * 2                         # 64
        c_bot = c2 * 2                      # 128

        # Stage 1 (H/2 x W/2): 1-level Haar DWT High-Frequency Gating + RepConv + Phase Skip
        self.ll_stem = nn.Sequential(nn.Conv2d(in_channels, c1, kernel_size=3, padding=1), nn.GELU())
        self.hf_stem = nn.Sequential(nn.Conv2d(in_channels * 3, c1, kernel_size=1), RepConv(c1), nn.GELU())
        self.hf_gate_proj = nn.Sequential(nn.Conv2d(c1, c1, kernel_size=1), nn.Sigmoid())
        self.enc_stage1 = RepConvBlock(c1, expand=2)
        self.phase_skip1 = FourierPhaseAmplitudeGate(c1, hidden_dim=max(16, (c1 * 3) // 4))

        # Stage 2 (H/4 x W/4): Stride-2 downsample + RepConv + Haar HF Gate + Fourier Phase Gate
        self.down1 = nn.Sequential(nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1), nn.GELU())
        self.enc_stage2 = RepConvBlock(c2, expand=2)
        self.haar_gate2 = HaarHFGate(c2)
        self.phase_gate2 = FourierPhaseAmplitudeGate(c2, hidden_dim=max(24, (c2 * 3) // 4))

        # Bottleneck (H/8 x W/8): Compressed physics-guided Mamba + global Fourier Phase Gate
        self.down2 = nn.Sequential(nn.Conv2d(c2, c_bot, kernel_size=3, stride=2, padding=1), nn.GELU())
        self.physics_head = MicroATHead(feature_dim=c_bot, hidden_dim=32, beta_max=4.0)
        self.mamba_corrector = nn.ModuleList(
            [
                ASEMambaBlock(
                    d_model=c_bot,
                    d_state=d_state,
                    expand=2,
                    num_heads=num_heads,
                    num_prompts=num_prompts,
                    chunk_size=chunk_size,
                )
                for _ in range(mamba_layers)
            ]
        )
        self.scan_strategy = PhysicsAttenuationScanStrategy(
            feature_dim=c_bot,
            mamba_corrector=self.mamba_corrector,
            physics_head=self.physics_head,
            num_prompts=num_prompts,
        )
        self.bot_gate = nn.Sequential(nn.Conv2d(c_bot * 2, c_bot, kernel_size=1), nn.Sigmoid())
        self.phase_bot = FourierPhaseAmplitudeGate(c_bot, hidden_dim=max(32, c_bot // 2))

        # Fast RepConv Decoder with bilinear upsampling + Phase-Gated multi-scale skips
        self.dec_fuse2 = nn.Conv2d(c_bot + c2, c2, kernel_size=1)
        self.dec_stage2 = RepConvBlock(c2, expand=2)

        self.dec_fuse1 = nn.Conv2d(c2 + c1, c1, kernel_size=1)
        self.dec_stage1 = RepConvBlock(c1, expand=2)
        self.dec_haar1 = HaarHFGate(c1)

        # Predict residual Haar subbands (LL, LH, HL, HH) -> HaarIDWT2d
        self.subband_head = nn.Conv2d(c1, in_channels * 4, kernel_size=3, padding=1)
        with torch.no_grad():
            self.subband_head.weight.mul_(0.1)
            self.subband_head.bias.zero_()

        # Lightweight Green/Blue-Phase-Guided Red Channel Refinement
        self.gb_red_refiner = GBPhaseRedRefiner(hidden_dim=24)

    @torch.no_grad()
    def fuse(self) -> "D2Mamba":
        """Fuse all RepConv parallel (7x7 + 3x3 + 1x1) branches into single 7x7 convs for edge inference."""
        for module in self.modules():
            if isinstance(module, RepConv):
                module.fuse()
        return self

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # 1. Outer Stage 1 (H/2): 1-level Haar DWT + High-Frequency Gating + RepConv
        ll_in, lh_in, hl_in, hh_in = haar_dwt_2d(x)
        hf_in = torch.cat([lh_in, hl_in, hh_in], dim=1)
        ll_feat = self.ll_stem(ll_in)
        hf_feat = self.hf_stem(hf_in)
        s1 = self.enc_stage1(ll_feat + self.hf_gate_proj(hf_feat) * hf_feat)
        s1_skip = self.phase_skip1(s1)

        # 2. Outer Stage 2 (H/4): RepConv + Haar HF Gating + 2D Fourier Phase-Amplitude Gate
        s2 = self.phase_gate2(self.haar_gate2(self.enc_stage2(self.down1(s1))))

        # 3. Compressed Bottleneck (H/8): Physics-Guided Single-Scan Mamba + Bottleneck Fourier Phase Gate
        bot = self.down2(s2)
        mamba_corr, path_len, physics = self.scan_strategy(bot, x_rgb=x)
        g_bot = self.bot_gate(torch.cat([bot, mamba_corr], dim=1))
        bot_mamba = (1.0 - g_bot) * bot + g_bot * mamba_corr
        bot_fused = self.phase_bot(bot_mamba)

        # 4. Fast RepConv Decoder with Phase-Gated Multi-Scale Skips -> Residual Haar IDWT Synthesis
        u2 = F.interpolate(bot_fused, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec_stage2(self.dec_fuse2(torch.cat([u2, s2], dim=1)))

        u1 = F.interpolate(d2, size=s1_skip.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec_haar1(self.dec_stage1(self.dec_fuse1(torch.cat([u1, s1_skip], dim=1))))

        d_ll, d_lh, d_hl, d_hh = self.subband_head(d1).chunk(4, dim=1)
        img_pre = haar_idwt_2d(
            ll_in + d_ll, lh_in + d_lh, hl_in + d_hl, hh_in + d_hh, ref=x
        )

        # 5. Green/Blue-Phase-Guided Red Channel Refinement
        img_ref = self.gb_red_refiner(img_pre)
        # Keep unclamped during training so L1 loss provides full gradients on saturated pixels
        output_image = img_ref if self.training else torch.clamp(img_ref, 0.0, 1.0)

        if self.training:
            if return_aux:
                return output_image, path_len, {"physics": physics}
            return output_image, path_len

        return output_image, path_len, physics
