#!/usr/bin/env python3
"""
Clean, modular, Kaggle-ready training & evaluation harness for D2Mamba research branches.

Features:
- Automatic Kaggle /kaggle/input discovery for UIEB (raw-890 / reference-890)
  and LSUI (LSUI/**/input & LSUI/**/GT), plus --data_root override.
- Strict zero-leakage official splits from ddz16/UIE_Benckmark:
    UIEB: 90 official test, 720 train / 80 val (deterministic seed)
    LSUI: 400 official test, 200 val / rest train (deterministic seed)
- AdamW (weight_decay=1e-4), 5% linear warmup + cosine annealing (lr -> 1e-6),
  gradient clipping 1.0, EMA (0.999), flip + 90-deg rotation (NO color jitter).
- FP16 AMP (--amp) with explicit FP32 casting for exponentials/log/loss.
- Compatible partial checkpoint loading (--init_ckpt) with shape filtering.
- Full test evaluation (RGB & Y-channel PSNR/SSIM), CSV logging, best.pt / last.pt,
  and hardware profiling (Params, GFLOPs, CUDA Event latency/FPS at 256x256 & 1280x720,
  Peak VRAM MB, and torch.jit.trace export verification).
"""

import argparse
import copy
import csv
import inspect
import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import D2Mamba

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OFFICIAL_SPLIT_URLS = {
    "uieb": "https://raw.githubusercontent.com/ddz16/UIE_Benckmark/main/data/UIEB/test.txt",
    "lsui": "https://raw.githubusercontent.com/ddz16/UIE_Benckmark/main/data/LSUI/test.txt",
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_SCAN_MODE = "haar_phase_fast"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Self-Contained FP32 SSIM & Metrics (RGB + ITU-R BT.601 Y-channel)
# ---------------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    window_2d = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """FP32-safe differentiable SSIM over (B, C, H, W) tensors."""
    pred = pred.float()
    target = target.float()
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq).clamp(min=0.0)
    sigma2_sq = (F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq).clamp(min=0.0)
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean()


def rgb_to_y_bt601(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor in [0, 1] (B, 3, H, W) to BT.601 Y-channel in [0, 1].
    Matches D2Mamba/utilities.py (65.481*R + 128.553*G + 24.966*B + 16) / 255."""
    img = img.float()
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = (65.481 * r + 128.553 * g + 24.966 * b + 16.0) / 255.0
    return y


@torch.no_grad()
def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute RGB and Y-channel PSNR/SSIM on uint8-quantized [0, 1] images."""
    pred_q = torch.round(pred.float().clamp(0.0, 1.0) * 255.0) / 255.0
    targ_q = torch.round(target.float().clamp(0.0, 1.0) * 255.0) / 255.0

    # Per-image PSNR (RGB)
    mse_rgb = ((pred_q - targ_q) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    psnr_rgb = (-10.0 * torch.log10(mse_rgb)).mean().item()
    ssim_rgb = compute_ssim_torch(pred_q, targ_q).item()

    # Per-image PSNR/SSIM (Y-channel, matches D2Mamba measure_ssim_psnr.py)
    pred_y = rgb_to_y_bt601(pred_q)
    targ_y = rgb_to_y_bt601(targ_q)
    mse_y = ((pred_y - targ_y) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    psnr_y = (-10.0 * torch.log10(mse_y)).mean().item()
    ssim_y = compute_ssim_torch(pred_y, targ_y).item()

    return {
        "psnr_rgb": psnr_rgb,
        "ssim_rgb": ssim_rgb,
        "psnr_y": psnr_y,
        "ssim_y": ssim_y,
    }


# ---------------------------------------------------------------------------
# Training-Only Pseudo-Linearized Akkaynak-Treibitz Cycle Consistency Loss
# ---------------------------------------------------------------------------
def compute_physics_cycle_loss(
    J_hat: torch.Tensor,
    I_raw: torch.Tensor,
    physics: Dict[str, torch.Tensor],
    lambda_phys: float = 0.1,
    lambda_noncollapse: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Training-only Pseudo-Linearized Akkaynak-Treibitz Cycle Consistency Loss (strict FP32):
      1. J_lin = (J_hat + 1e-4)^2.2,  B_inf_lin = (B_inf + 1e-4)^2.2
      2. I_lin = J_lin * exp(-beta_D * d_hat) + B_inf_lin * (1 - exp(-beta_B * d_hat))
      3. I_hat = (I_lin + 1e-4)^(1 / 2.2)
      4. L_phys = lambda_phys * (L1(I_hat, I_raw) + 0.1 * (1 - SSIM(I_hat, I_raw)))
                + lambda_noncollapse * exp(-5 * mean(|beta_D - beta_B|))
    """
    # Force FP32 for all powers, exponentials, and SSIM even under --amp
    J_f = J_hat.float().clamp(0.0, 1.0)
    I_f = I_raw.float().clamp(0.0, 1.0)
    d_hat = physics["d_hat"].float().clamp(0.0, 1.0)
    beta_D = physics["beta_D"].float().clamp(min=1e-4)
    beta_B = physics["beta_B"].float().clamp(min=1e-4)
    B_inf = physics["B_inf"].float().clamp(0.0, 1.0)

    H, W = I_f.shape[-2:]
    if d_hat.shape[-2:] != (H, W):
        d_hat = F.interpolate(d_hat, size=(H, W), mode="bilinear", align_corners=False)
        beta_D = F.interpolate(beta_D, size=(H, W), mode="bilinear", align_corners=False)
        beta_B = F.interpolate(beta_B, size=(H, W), mode="bilinear", align_corners=False)

    J_lin = (J_f + 1e-4).pow(2.2)
    B_inf_lin = (B_inf + 1e-4).pow(2.2)

    T_D = torch.exp(-beta_D * d_hat)
    T_B = torch.exp(-beta_B * d_hat)
    I_lin = J_lin * T_D + B_inf_lin * (1.0 - T_B)

    I_hat = (I_lin.clamp(min=0.0) + 1e-4).pow(1.0 / 2.2)

    cycle_l1 = F.l1_loss(I_hat, I_f)
    cycle_ssim = 1.0 - compute_ssim_torch(I_hat.clamp(0.0, 1.0), I_f)
    loss_cycle = cycle_l1 + 0.1 * cycle_ssim

    # Smooth non-collapse regularizer preventing beta_D == beta_B
    beta_sep = (beta_D - beta_B).abs().mean()
    loss_noncollapse = torch.exp(-5.0 * beta_sep)

    total_phys = lambda_phys * loss_cycle + lambda_noncollapse * loss_noncollapse
    return total_phys, {
        "phys_cycle": float(loss_cycle.detach().item()),
        "phys_noncollapse": float(loss_noncollapse.detach().item()),
        "beta_sep": float(beta_sep.detach().item()),
    }


# ---------------------------------------------------------------------------
# Optional Lightweight VGG Perceptual Loss (Graceful Offline Fallback)
# ---------------------------------------------------------------------------
class OptionalPerceptualLoss(nn.Module):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = False
        self.features = None
        if not enabled:
            return
        try:
            from torchvision import models
            weights = models.VGG16_Weights.IMAGENET1K_V1
            vgg_feats = models.vgg16(weights=weights).features[:9]
            for p in vgg_feats.parameters():
                p.requires_grad = False
            self.features = vgg_feats.eval()
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.enabled = True
        except Exception as exc:
            print(f"[INFO] VGG16 perceptual loss disabled (weights unavailable: {exc})")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.features is None:
            return pred.new_zeros(())
        pred_n = (pred.float() - self.mean) / self.std
        targ_n = (target.float() - self.mean) / self.std
        with torch.no_grad():
            ft_t = self.features(targ_n)
        ft_p = self.features(pred_n)
        return F.l1_loss(ft_p, ft_t)


# ---------------------------------------------------------------------------
# EMA Helper
# ---------------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


# ---------------------------------------------------------------------------
# Dataset Discovery & Strict Zero-Leakage Splitting
# ---------------------------------------------------------------------------
def _list_images_by_stem(directory: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not directory.is_dir():
        return mapping
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            mapping[p.stem] = p
    return mapping


def _resolve_leaf_image_dir(directory: Path, max_depth: int = 3) -> Path:
    """Unwrap single-child nested directories (e.g. raw-890/raw-890/*.png) common in Kaggle zips."""
    cur = directory
    for _ in range(max_depth):
        if _list_images_by_stem(cur):
            return cur
        subdirs = [p for p in sorted(cur.iterdir()) if p.is_dir()] if cur.is_dir() else []
        if len(subdirs) == 1:
            cur = subdirs[0]
        else:
            break
    return cur


def _find_best_pair_dirs(search_roots: List[Path], dataset: str) -> Tuple[Path, Path]:
    """Automatically locate (input_dir, gt_dir) under Kaggle /kaggle/input or --data_root.

    Uses strict lexicographic tuple ranking so dataset identity and canonical folder names
    always take precedence over raw image count (preventing LSUI from overriding UIEB when
    both datasets are attached simultaneously in /kaggle/input).
    """
    candidates: List[Tuple[Tuple[int, int, int, int], Path, Path]] = []
    other_dataset = "lsui" if dataset == "uieb" else "uieb"

    if dataset == "uieb":
        name_pairs = [
            ("raw-890", "reference-890", 100),
            ("raw", "reference", 80),
            ("input", "reference", 70),
            ("hazy", "clean", 60),
            ("input", "gt", 50),
        ]
    else:  # lsui
        name_pairs = [
            ("input", "gt", 100),
            ("raw", "gt", 80),
            ("input", "reference", 70),
        ]

    # Collect named directories for non-sibling fallback (e.g., separate Kaggle dataset mounts)
    named_dirs: Dict[str, List[Path]] = {}

    for root in search_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            base = Path(dirpath)
            lower_map = {d.lower(): base / d for d in dirnames}
            for d_low, d_path in lower_map.items():
                named_dirs.setdefault(d_low, []).append(d_path)
            for inp_name, gt_name, priority in name_pairs:
                if inp_name in lower_map and gt_name in lower_map:
                    inp_dir = _resolve_leaf_image_dir(lower_map[inp_name])
                    gt_dir = _resolve_leaf_image_dir(lower_map[gt_name])
                    path_lower = f"{inp_dir} {gt_dir}".lower()
                    not_other = 0 if other_dataset in path_lower else 1
                    matches_target = 1 if dataset in path_lower else 0
                    inp_imgs = _list_images_by_stem(inp_dir)
                    gt_imgs = _list_images_by_stem(gt_dir)
                    n_common = len(set(inp_imgs.keys()) & set(gt_imgs.keys()))
                    if n_common > 0:
                        score = (not_other, matches_target, priority, n_common)
                        candidates.append((score, inp_dir, gt_dir))

    # Non-sibling fallback (e.g., /kaggle/input/uieb-raw/raw-890 + /kaggle/input/uieb-ref/reference-890)
    # Run whenever no high-confidence sibling candidate was found for the requested dataset.
    has_strong_sibling = any(s[0] == 1 and s[2] >= 80 for s, _, _ in candidates)
    if not has_strong_sibling:
        for inp_name, gt_name, priority in name_pairs:
            for raw_inp in named_dirs.get(inp_name, []):
                inp_dir = _resolve_leaf_image_dir(raw_inp)
                inp_imgs = _list_images_by_stem(inp_dir)
                if not inp_imgs:
                    continue
                for raw_gt in named_dirs.get(gt_name, []):
                    gt_dir = _resolve_leaf_image_dir(raw_gt)
                    if inp_dir == gt_dir:
                        continue
                    gt_imgs = _list_images_by_stem(gt_dir)
                    n_common = len(set(inp_imgs.keys()) & set(gt_imgs.keys()))
                    if n_common > 0:
                        path_lower = f"{inp_dir} {gt_dir}".lower()
                        not_other = 0 if other_dataset in path_lower else 1
                        matches_target = 1 if dataset in path_lower else 0
                        score = (not_other, matches_target, priority - 10, n_common)
                        candidates.append((score, inp_dir, gt_dir))

    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-discover '{dataset}' dataset under {[str(r) for r in search_roots]}. "
            f"Pass --data_root pointing to a directory containing the paired folders."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_inp, best_gt = candidates[0]
    return best_inp, best_gt


def fetch_official_test_stems(
    dataset: str,
    cache_dir: Path,
    split_file: Optional[str] = None,
) -> List[str]:
    """Fetch official test split stems from ddz16/UIE_Benckmark (with local cache)."""
    if split_file is not None and Path(split_file).is_file():
        text = Path(split_file).read_text(encoding="utf-8")
    else:
        cache_path = cache_dir / f"{dataset}_official_test.txt"
        if cache_path.is_file():
            text = cache_path.read_text(encoding="utf-8")
        else:
            url = OFFICIAL_SPLIT_URLS[dataset]
            print(f"[DATA] Fetching official {dataset.upper()} test split from {url}")
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    text = resp.read().decode("utf-8")
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text, encoding="utf-8")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download official split from {url}: {exc}. "
                    f"Provide --split_file locally if running offline."
                ) from exc

    stems = [Path(line.strip()).stem for line in text.splitlines() if line.strip()]
    return stems


def build_zero_leakage_splits(
    dataset: str,
    inp_dir: Path,
    gt_dir: Path,
    seed: int,
    save_dir: Path,
    split_file: Optional[str] = None,
    allow_fallback_split: bool = False,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Build strictly disjoint (train, val, test) lists of (input_path, gt_path)."""
    inp_map = _list_images_by_stem(inp_dir)
    gt_map = _list_images_by_stem(gt_dir)
    common_stems = sorted(set(inp_map.keys()) & set(gt_map.keys()))
    if not common_stems:
        raise RuntimeError(f"Zero paired files found between {inp_dir} and {gt_dir}.")

    test_stems_set = set()
    try:
        official_stems = fetch_official_test_stems(dataset, save_dir, split_file=split_file)
        test_stems_set = set(official_stems) & set(common_stems)
    except Exception as exc:
        if not allow_fallback_split:
            raise
        print(f"[WARN] Official split unavailable ({exc}); using deterministic fallback split.")

    if len(test_stems_set) == 0:
        if not allow_fallback_split:
            raise RuntimeError(
                f"0 official test stems matched files in {inp_dir} (found {len(common_stems)} pairs). "
                f"Check dataset directory or pass --allow_fallback_split for synthetic smoke tests."
            )
        rng_fb = random.Random(seed + 999)
        shuffled = list(common_stems)
        rng_fb.shuffle(shuffled)
        n_test = max(1, len(shuffled) // 5)
        test_stems_set = set(shuffled[:n_test])

    test_stems = sorted(test_stems_set)
    remaining_stems = sorted(s for s in common_stems if s not in test_stems_set)

    rng = random.Random(seed)
    rng.shuffle(remaining_stems)

    target_val = 80 if dataset == "uieb" else 200
    val_count = min(target_val, max(1, len(remaining_stems) // 5))
    if len(remaining_stems) >= target_val + 10:
        val_count = target_val

    val_stems = sorted(remaining_stems[:val_count])
    train_stems = sorted(remaining_stems[val_count:])

    # Strict zero-leakage assertions
    s_train, s_val, s_test = set(train_stems), set(val_stems), set(test_stems)
    assert s_train.isdisjoint(s_test), "DATA LEAKAGE DETECTED: train overlaps test!"
    assert s_val.isdisjoint(s_test), "DATA LEAKAGE DETECTED: val overlaps test!"
    assert s_train.isdisjoint(s_val), "DATA LEAKAGE DETECTED: train overlaps val!"

    to_pairs = lambda stems: [(inp_map[s], gt_map[s]) for s in stems]
    return to_pairs(train_stems), to_pairs(val_stems), to_pairs(test_stems)


class PairedUnderwaterDataset(Dataset):
    """Paired RGB dataset with geometric-only augmentations (NO color jitter)."""

    def __init__(self, pairs: List[Tuple[Path, Path]], img_size: int = 256, augment: bool = False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inp_path, gt_path = self.pairs[idx]
        inp_img = Image.open(inp_path).convert("RGB").resize((self.img_size, self.img_size), Image.BICUBIC)
        gt_img = Image.open(gt_path).convert("RGB").resize((self.img_size, self.img_size), Image.BICUBIC)

        inp_np = np.asarray(inp_img, dtype=np.float32) / 255.0
        gt_np = np.asarray(gt_img, dtype=np.float32) / 255.0

        inp_t = torch.from_numpy(inp_np).permute(2, 0, 1).contiguous()
        gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).contiguous()

        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                inp_t = torch.flip(inp_t, dims=[2])
                gt_t = torch.flip(gt_t, dims=[2])
            # Random vertical flip
            if random.random() < 0.5:
                inp_t = torch.flip(inp_t, dims=[1])
                gt_t = torch.flip(gt_t, dims=[1])
            # Random 90-degree rotation (0, 90, 180, 270)
            k = random.randint(0, 3)
            if k > 0:
                inp_t = torch.rot90(inp_t, k, dims=[1, 2])
                gt_t = torch.rot90(gt_t, k, dims=[1, 2])

        return {"hazy": inp_t, "clean": gt_t, "name": inp_path.stem}


# ---------------------------------------------------------------------------
# Compatible Partial Checkpoint Loader
# ---------------------------------------------------------------------------
def load_compatible_checkpoint(model: nn.Module, ckpt_path: str) -> Dict[str, int]:
    """Load compatible layers (matching key & exact shape) with strict=False."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            ckpt_sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            ckpt_sd = ckpt["state_dict"]
        else:
            ckpt_sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format in {ckpt_path}")

    model_sd = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    skipped_shape = 0
    skipped_missing = 0

    for k, v in ckpt_sd.items():
        clean_k = k[len("module."):] if k.startswith("module.") else k
        if clean_k in model_sd:
            if model_sd[clean_k].shape == v.shape:
                compatible[clean_k] = v
            else:
                skipped_shape += 1
        else:
            skipped_missing += 1

    missing_keys, unexpected_keys = model.load_state_dict(compatible, strict=False)
    stats = {
        "loaded_tensors": len(compatible),
        "skipped_shape_mismatch": skipped_shape,
        "skipped_unused_ckpt_keys": skipped_missing,
        "model_missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
    }
    print(f"[CKPT] Loaded from {ckpt_path}: {stats}")
    return stats


# ---------------------------------------------------------------------------
# Model Forward Wrapper & Helpers
# ---------------------------------------------------------------------------
def build_model(args: argparse.Namespace) -> nn.Module:
    sig = inspect.signature(D2Mamba.__init__)
    kwargs = {
        "scan_strategy_name": args.scan_mode,
        "in_channels": 3,
        "base_dim": args.base_dim,
        "mamba_layers": args.mamba_layers,
    }
    if "num_paths" in sig.parameters:
        kwargs["num_paths"] = 4
    return D2Mamba(**kwargs)


def forward_model(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
    """Run model forward and normalize output to (restored_img, aux_dict_or_None)."""
    if model.training and getattr(model, "supports_return_aux", False):
        out = model(x, return_aux=True)
    else:
        out = model(x)
    aux = None
    if isinstance(out, tuple):
        pred = out[0]
        for elem in out[1:]:
            if isinstance(elem, dict):
                aux = elem
                break
    else:
        pred = out
    return pred, aux


class TraceWrapper(nn.Module):
    """Thin wrapper returning only the restored RGB tensor for torch.jit.trace."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out


# ---------------------------------------------------------------------------
# Evaluation & Hardware Profiling
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    totals = {"psnr_rgb": 0.0, "ssim_rgb": 0.0, "psnr_y": 0.0, "ssim_y": 0.0}
    n_samples = 0

    for batch in loader:
        hazy = batch["hazy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        bsz = hazy.shape[0]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda", dtype=torch.float16):
            pred, _ = forward_model(model, hazy)
        metrics = compute_batch_metrics(pred, clean)
        for k in totals:
            totals[k] += metrics[k] * bsz
        n_samples += bsz

    return {k: v / max(1, n_samples) for k, v in totals.items()}


def _count_gflops(model: nn.Module, shape: Tuple[int, int, int, int], device: torch.device) -> Optional[float]:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        dummy = torch.rand(*shape, device=device)
        wrapper = TraceWrapper(model).eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            _ = wrapper(dummy)
        return fc.get_total_flops() / 1e9
    except Exception:
        return None


@torch.no_grad()
def profile_hardware(
    model: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, object]:
    """Profile Params, GFLOPs, latency/FPS at 256x256 & 1280x720, Peak VRAM, and JIT trace."""
    model = copy.deepcopy(model).to(device).eval()
    if hasattr(model, "fuse") and callable(model.fuse):
        model.fuse()

    total_params = sum(p.numel() for p in model.parameters())
    profile: Dict[str, object] = {
        "device": str(device),
        "total_params": total_params,
        "total_params_M": round(total_params / 1e6, 4),
    }

    wrapper = TraceWrapper(model).eval()

    for label, (h, w) in [("256x256", (256, 256)), ("1280x720", (720, 1280))]:
        gflops = _count_gflops(model, (1, 3, h, w), device)
        dummy = torch.rand(1, 3, h, w, device=device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            warmup, iters = 10, 50
            for _ in range(warmup):
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    _ = wrapper(dummy)
            torch.cuda.synchronize(device)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(iters):
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    _ = wrapper(dummy)
            end_ev.record()
            torch.cuda.synchronize(device)
            avg_ms = start_ev.elapsed_time(end_ev) / iters
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        else:
            warmup, iters = 1, 3
            for _ in range(warmup):
                _ = wrapper(dummy)
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = wrapper(dummy)
            avg_ms = ((time.perf_counter() - t0) / iters) * 1000.0
            peak_vram_mb = 0.0

        fps = 1000.0 / max(avg_ms, 1e-6)
        profile[label] = {
            "gflops": round(gflops, 3) if gflops is not None else None,
            "latency_ms": round(avg_ms, 3),
            "fps": round(fps, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }

    # torch.jit.trace export test (strict check_trace=True + dynamic larger shape)
    jit_ok = False
    jit_err = None
    jit_max_diff = None
    try:
        dummy_jit = torch.rand(1, 3, 256, 256, device=device)
        eager_out = wrapper(dummy_jit)
        traced = torch.jit.trace(wrapper, dummy_jit, check_trace=True)
        traced_out = traced(dummy_jit)
        jit_max_diff = float((eager_out - traced_out).abs().max().item())
        jit_ok = jit_max_diff < 1e-4
    except Exception as exc:
        jit_err = str(exc)

    profile["torch_jit_trace"] = {
        "passed": jit_ok,
        "max_diff": jit_max_diff,
        "error": jit_err,
    }
    return profile


# ---------------------------------------------------------------------------
# LR Schedule (5% Linear Warmup + Cosine Annealing -> 1e-6)
# ---------------------------------------------------------------------------
def get_lr_at_step(step: int, total_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    warmup_steps = max(1, int(0.05 * total_steps))
    if step < warmup_steps:
        alpha = float(step + 1) / float(warmup_steps)
        return min_lr + alpha * (base_lr - min_lr)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# CLI & Main Training Loop
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle-ready D2Mamba Research Training Harness")
    parser.add_argument("--dataset", type=str, default="uieb", choices=["uieb", "lsui"])
    parser.add_argument("--scan_mode", type=str, default=DEFAULT_SCAN_MODE)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--amp", action="store_true", help="Enable FP16 mixed precision (with FP32 safe loss/exp)")
    parser.add_argument("--init_ckpt", type=str, default=None, help="Optional pretrained checkpoint path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="/kaggle/working/checkpoints")
    parser.add_argument("--data_root", type=str, default=None, help="Override dataset root (default: auto /kaggle/input)")
    parser.add_argument("--split_file", type=str, default=None, help="Optional local test.txt split file")
    parser.add_argument("--allow_fallback_split", action="store_true", help="Allow fallback split for synthetic smoke tests")
    parser.add_argument("--base_dim", type=int, default=24)
    parser.add_argument("--mamba_layers", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--lambda_ssim", type=float, default=0.2)
    parser.add_argument("--lambda_percep", type=float, default=0.02)
    parser.add_argument("--lambda_phys", type=float, default=0.1, help="Weight for Akkaynak-Treibitz cycle consistency loss")
    parser.add_argument("--lambda_noncollapse", type=float, default=0.01, help="Non-collapse weight preventing beta_D == beta_B")
    parser.add_argument("--no_vgg", action="store_true", help="Disable VGG16 perceptual loss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and device.type != "cuda":
        print("[WARN] --amp requested on CPU; falling back to FP32.")

    # 1. Discover dataset & build strict zero-leakage splits
    search_roots = [Path(args.data_root)] if args.data_root else [Path("/kaggle/input")]
    inp_dir, gt_dir = _find_best_pair_dirs(search_roots, args.dataset)
    print(f"[DATA] Discovered {args.dataset.upper()} dirs:\n  input: {inp_dir}\n  gt:    {gt_dir}")

    train_pairs, val_pairs, test_pairs = build_zero_leakage_splits(
        dataset=args.dataset,
        inp_dir=inp_dir,
        gt_dir=gt_dir,
        seed=args.seed,
        save_dir=save_dir,
        split_file=args.split_file,
        allow_fallback_split=args.allow_fallback_split,
    )
    print(f"[DATA] Zero-leakage splits -> train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    train_ds = PairedUnderwaterDataset(train_pairs, img_size=args.img_size, augment=True)
    val_ds = PairedUnderwaterDataset(val_pairs, img_size=args.img_size, augment=False)
    test_ds = PairedUnderwaterDataset(test_pairs, img_size=args.img_size, augment=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=len(train_ds) >= args.batch_size,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    # 2. Build model & optionally load compatible pretrained weights
    model = build_model(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] scan_mode={args.scan_mode} | params={total_params:,} ({total_params/1e6:.4f}M)")

    if args.init_ckpt:
        load_compatible_checkpoint(model, args.init_ckpt)

    ema = ModelEMA(model, decay=args.ema_decay)
    percep_loss_fn = OptionalPerceptualLoss(enabled=(not args.no_vgg and args.lambda_percep > 0)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    csv_path = save_dir / "train_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "val_psnr_rgb", "val_ssim_rgb", "val_psnr_y", "val_ssim_y"])

    total_steps = max(1, args.epochs * len(train_loader))
    global_step = 0
    best_val_psnr = -1.0

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        t_ep = time.time()

        for batch in train_loader:
            cur_lr = get_lr_at_step(global_step, total_steps, args.lr, min_lr=1e-6)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            hazy = batch["hazy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                pred, aux = forward_model(model, hazy)

            # All losses computed strictly in FP32
            with torch.amp.autocast(device_type=device.type, enabled=False):
                pred_raw = pred.float()
                pred_f = pred_raw.clamp(0.0, 1.0)
                clean_f = clean.float().clamp(0.0, 1.0)
                loss_l1 = F.l1_loss(pred_raw, clean_f)
                loss_ssim = 1.0 - compute_ssim_torch(pred_f, clean_f)
                loss_percep = percep_loss_fn(pred_f, clean_f)
                loss = loss_l1 + args.lambda_ssim * loss_ssim + args.lambda_percep * loss_percep

                if aux is not None and isinstance(aux, dict) and "physics" in aux and args.lambda_phys > 0:
                    loss_phys, _ = compute_physics_cycle_loss(
                        J_hat=pred_f,
                        I_raw=hazy.float(),
                        physics=aux["physics"],
                        lambda_phys=args.lambda_phys,
                        lambda_noncollapse=args.lambda_noncollapse,
                    )
                    loss = loss + loss_phys

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}, step {global_step}: {loss.item()}")

            step_applied = True
            if use_amp:
                prev_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                step_applied = scaler.get_scale() >= prev_scale
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if step_applied:
                ema.update(model)
            running_loss += float(loss.item())
            n_batches += 1
            global_step += 1

        avg_train_loss = running_loss / max(1, n_batches)
        val_metrics = {"psnr_rgb": float("nan"), "ssim_rgb": float("nan"), "psnr_y": float("nan"), "ssim_y": float("nan")}

        if epoch % args.val_freq == 0 or epoch == args.epochs:
            val_metrics = evaluate_loader(ema.ema, val_loader, device, use_amp=use_amp)
            if val_metrics["psnr_rgb"] > best_val_psnr:
                best_val_psnr = val_metrics["psnr_rgb"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": ema.ema.state_dict(),
                        "raw_model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_psnr_rgb": best_val_psnr,
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    },
                    save_dir / "best.pt",
                )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": ema.ema.state_dict(),
                "raw_model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_psnr_rgb": best_val_psnr,
                "val_metrics": val_metrics,
                "args": vars(args),
            },
            save_dir / "last.pt",
        )

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{cur_lr:.8f}",
                f"{avg_train_loss:.6f}",
                f"{val_metrics['psnr_rgb']:.4f}",
                f"{val_metrics['ssim_rgb']:.4f}",
                f"{val_metrics['psnr_y']:.4f}",
                f"{val_metrics['ssim_y']:.4f}",
            ])

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | lr={cur_lr:.2e} | loss={avg_train_loss:.4f} | "
            f"val_PSNR(RGB/Y)={val_metrics['psnr_rgb']:.2f}/{val_metrics['psnr_y']:.2f} | "
            f"val_SSIM(RGB/Y)={val_metrics['ssim_rgb']:.4f}/{val_metrics['ssim_y']:.4f} | "
            f"best={best_val_psnr:.2f} dB ({time.time()-t_ep:.1f}s)"
        )

    # 4. Final Full Test Evaluation + Hardware Profiling on best.pt
    best_ckpt_path = save_dir / "best.pt"
    if best_ckpt_path.is_file():
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        ema.ema.load_state_dict(best_ckpt["model_state_dict"], strict=True)

    eval_model = copy.deepcopy(ema.ema).to(device).eval()
    if hasattr(eval_model, "fuse") and callable(eval_model.fuse):
        eval_model.fuse()

    test_metrics = evaluate_loader(eval_model, test_loader, device, use_amp=use_amp)
    hw_profile = profile_hardware(eval_model, device, use_amp=use_amp)

    summary = {
        "dataset": args.dataset,
        "scan_mode": args.scan_mode,
        "best_val_psnr_rgb": best_val_psnr,
        "test_metrics": test_metrics,
        "hardware_profile": hw_profile,
    }
    summary_path = save_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n================ FINAL TEST & HARDWARE SUMMARY ================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
