# Kaggle T4 Runbook — Branch 4 (`exp/4-phaseformer-physics-mamba`) & Dual-Notebook Plan (Branches 1–4)

## Branch 4 Architecture Summary (`exp/4-phaseformer-physics-mamba`)
Hybrid **Phaseformer-Inspired Frequency Cross-Gating + Multi-Scale Haar U-Net + Physics-Guided Mamba** building on `exp/3-haar-phase-fast`:
- **Explicit 2D Fourier Phase-Amplitude Cross-Gating (`FourierPhaseAmplitudeGate`)**:
  - Inspired by **Phaseformer (WACV'25)**, operates strictly in **FP32** for `rfft2`/`irfft2`, `exp`, and unit-circle normalization inside `autocast(enabled=False)` so `cuFFT` never crashes in FP16 on non-power-of-two or odd resolutions — and works cleanly under both `--amp` and pure `.half()`/`.bfloat16()` deployment.
  - Uses branch-cut-free unit phasors `(cos φ, sin φ) = (Re/|Z|, Im/|Z|)` (zero `atan2`/`abs` NaN gradients at `0+0j`) and log-compressed amplitude `log1p(|Z|)`.
  - **Phase → Amplitude Cross-Gate**: Uncontaminated structural phase guides bounded multiplicative Beer–Lambert-like amplitude restoration (`exp(clamp(s * d_amp, -1.5, 1.5))`).
  - **Amplitude → Phase Cross-Gate**: Spectral energy gates phase residual updates before unit-circle re-normalization.
  - Placed at **Stage 1 Skip (`H/2` Optimized Phase Attention Skip)**, **Stage 2 (`H/4`)**, and **Bottleneck (`H/8`)**.
- **Multi-Scale Haar U-Net + `RepConv` (`7x7 + 3x3 + 1x1`)**: Exact `.fuse()` verified `< 1e-5`, lossless 1-level Haar DWT/IDWT synthesis + decoder `HaarHFGate`.
- **Compressed Physics-Guided Single-Scan Mamba Bottleneck (`H/8`)**: `Micro-AT` (`22.4K` params) + near-to-far telescoping Beer–Lambert `dt` modulation + training-only FP32 Akkaynak–Treibitz cycle consistency loss.
- **Green/Blue-Phase-Guided Red Channel Refinement (`GBPhaseRedRefiner`)**: Sharpens attenuated R channel using intact G/B Haar directional phase cues.
- **Footprint**: `0.965M` params (`0.959M` fused), **`3.525 GFLOPs` @ 256×256**, 100% GPU-native, dynamic-shape `torch.jit.trace(..., check_trace=True)` + `torch.jit.save`/`load` verified (`max_diff = 0.0`).

---

## Prerequisite (Before Launching Notebook B)
Push Branch 4 to GitHub so Kaggle Notebook B can clone it:
```bash
git -C "D2Mamba-Research" push -u origin exp/4-phaseformer-physics-mamba
```

---

## Step-by-Step Kaggle Setup (Do This For BOTH Notebooks)

1. **Create 2 Kaggle Notebooks** (or upload `kaggle_notebooks/notebook_A_exp1_exp2.ipynb` and `kaggle_notebooks/notebook_B_exp3_exp4.ipynb` via **File → Import Notebook**):
   - **Notebook A**: Runs **Branch 1 (`exp/1-mambairv2-gpu-scan`)** + **Branch 2 (`exp/2-physics-attenuation-scan`)**
   - **Notebook B**: Runs **Branch 3 (`exp/3-haar-phase-fast`)** + **Branch 4 (`exp/4-phaseformer-physics-mamba`)**
2. **Right Sidebar → Session Options**:
   - **Accelerator**: `GPU T4 x2` (or `GPU P100` — `train_kaggle.py` runs sequentially on `cuda:0` so hardware FPS profiling has zero cross-process CPU/PCIe contention).
   - **Internet**: **ON** (required to `git clone` from GitHub, download VGG16 weights, and fetch the official zero-leakage `ddz16/UIE_Benckmark` test splits).
   - **Persistence**: `Files only` (optional, protects outputs).
3. **Right Sidebar → Add Input (Datasets)**:
   - Attach **UIEB** (containing `raw-890` and `reference-890`).
   - Attach **LSUI** (containing `input` and `GT`).
   - `train_kaggle.py` automatically discovers both datasets under `/kaggle/input` without overriding each other.
4. **Run Both Notebooks in Parallel**:
   - Click **Save Version → Save & Run All (Commit)** on **Notebook A** and **Notebook B** simultaneously.
   - Each notebook runs **both UIEB runs FIRST** (~45–60 min total) before the multi-hour LSUI runs, so both UIEB checkpoints & metrics are safely saved early. LSUI uses `--val_freq 2` to cut validation disk I/O in half while still evaluating epoch 200.
   - All bash cells use `set -euo pipefail` + `python3 -u` so any clone/verify/training failure immediately stops the notebook instead of silently continuing.
5. **Important Training Protocol Note (From Scratch vs. `--init_ckpt`)**:
   - **Recommended for Fair, Zero-Leakage Comparison (Default below)**: Train **ALL 4 branches from scratch for 200 epochs** (NO `--init_ckpt`). This guarantees identical random initialization, zero risk of test-split leakage from the original D2Mamba checkpoint, and a true apples-to-apples comparison across Branches 1, 2, 3, and 4.
   - *Optional fast milestone for Notebook A only*: Branches 1 & 2 share D2Mamba's `DualStreamEncoder`/`GatedSynthesisDecoder`, so you *can* optionally add `--init_ckpt ckpt/netG_UIEB.pt --epochs 60` (or `ckpt/netG_LSUI.pt`) for a quick warm-start sanity run — but do **not** mix warm-started 1/2 numbers with from-scratch 3/4 numbers in your final table.

---

## Notebook A — Branches 1 & 2 (`exp/1-mambairv2-gpu-scan` & `exp/2-physics-attenuation-scan`)

### Cell 1 — Clone & Verify Branch 1 and Branch 2 (Fail-Fast)
```bash
%%bash
set -euo pipefail
REPO="https://github.com/asserelzeki6/D2Mamba-Research.git"

rm -rf /kaggle/working/exp1 /kaggle/working/exp2
git clone --depth 1 -b exp/1-mambairv2-gpu-scan "$REPO" /kaggle/working/exp1
git clone --depth 1 -b exp/2-physics-attenuation-scan "$REPO" /kaggle/working/exp2

(cd /kaggle/working/exp1 && python3 -u verify_branch.py --device cuda)
(cd /kaggle/working/exp2 && python3 -u verify_branch.py --device cuda)
```

### Cell 2 — Train & Evaluate Branch 1 on UIEB (From Scratch, 200 Epochs)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/exp1
python3 -u train_kaggle.py \
    --dataset uieb \
    --scan_mode mambairv2 \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --val_freq 1 \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp1_uieb
```

### Cell 3 — Train & Evaluate Branch 2 on UIEB (From Scratch, 200 Epochs + Physics Cycle Loss)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/exp2
python3 -u train_kaggle.py \
    --dataset uieb \
    --scan_mode physics_attenuation \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --lambda_phys 0.1 \
    --val_freq 1 \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp2_uieb
```

### Cell 4 — Train & Evaluate Branch 1 on LSUI (From Scratch, 200 Epochs)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/exp1
python3 -u train_kaggle.py \
    --dataset lsui \
    --scan_mode mambairv2 \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --val_freq 2 \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp1_lsui
```

### Cell 5 — Train & Evaluate Branch 2 on LSUI (From Scratch, 200 Epochs + Physics Cycle Loss)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/exp2
python3 -u train_kaggle.py \
    --dataset lsui \
    --scan_mode physics_attenuation \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --lambda_phys 0.1 \
    --val_freq 2 \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp2_lsui
```

### Cell 6 — Print & Save Summary Table (Notebook A)
```python
import json
from pathlib import Path

combined = {}
for run in ["exp1_uieb", "exp2_uieb", "exp1_lsui", "exp2_lsui"]:
    p = Path("/kaggle/working/checkpoints") / run / "eval_summary.json"
    if not p.is_file():
        print(f"{run}: NOT FOUND")
        continue
    d = json.loads(p.read_text())
    combined[run] = d
    m = d["test_metrics"]
    hw = d["hardware_profile"]
    p256 = hw["256x256"]
    p720 = hw["1280x720"]
    print(
        f"{run:10s} | PSNR(RGB/Y)={m['psnr_rgb']:.2f}/{m['psnr_y']:.2f} dB | "
        f"SSIM(RGB/Y)={m['ssim_rgb']:.4f}/{m['ssim_y']:.4f} | "
        f"Params={hw['total_params_M']:.3f}M | GFLOPs={p256['gflops']} | "
        f"256={p256['fps']:.1f} FPS ({p256['latency_ms']:.1f} ms) | "
        f"720p={p720['fps']:.1f} FPS ({p720['latency_ms']:.1f} ms) | "
        f"VRAM={p256['peak_vram_mb']:.0f} MB | JIT={hw['torch_jit_trace']['passed']}"
    )

out_path = Path("/kaggle/working/checkpoints/notebook_A_summary.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"Saved combined summary -> {out_path}")
```

---

## Notebook B — Branches 3 & 4 (`exp/3-haar-phase-fast` & `exp/4-phaseformer-physics-mamba`)

### Cell 1 — Clone & Verify Branch 3 and Branch 4 (Fail-Fast)
```bash
%%bash
set -euo pipefail
REPO="https://github.com/asserelzeki6/D2Mamba-Research.git"

rm -rf /kaggle/working/exp3 /kaggle/working/exp4
git clone --depth 1 -b exp/3-haar-phase-fast "$REPO" /kaggle/working/exp3
git clone --depth 1 -b exp/4-phaseformer-physics-mamba "$REPO" /kaggle/working/exp4

(cd /kaggle/working/exp3 && python3 -u verify_branch.py --device cuda)
(cd /kaggle/working/exp4 && python3 -u verify_branch.py --device cuda)
```

### Cell 2 — Train & Evaluate Branch 3 on UIEB (From Scratch, 200 Epochs + Auto RepConv `.fuse()`)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/exp3
python3 -u train_kaggle.py \
    --dataset uieb \
    --scan_mode haar_phase_fast \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --lambda_phys 0.1 \
    --val_freq 1 \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp3_uieb
```

### Cell 3 — Train & Evaluate Branch 4 on UIEB (From Scratch, 200 Epochs + Auto RepConv `.fuse()`)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/exp4
python3 -u train_kaggle.py \
    --dataset uieb \
    --scan_mode phaseformer_physics_mamba \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --lambda_phys 0.1 \
    --val_freq 1 \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp4_uieb
```

### Cell 4 — Train & Evaluate Branch 3 on LSUI (From Scratch, 200 Epochs + Auto RepConv `.fuse()`)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/exp3
python3 -u train_kaggle.py \
    --dataset lsui \
    --scan_mode haar_phase_fast \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --lambda_phys 0.1 \
    --val_freq 2 \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp3_lsui
```

### Cell 5 — Train & Evaluate Branch 4 on LSUI (From Scratch, 200 Epochs + Auto RepConv `.fuse()`)
```bash
%%bash
set -euo pipefail
cd /kaggle/working/exp4
python3 -u train_kaggle.py \
    --dataset lsui \
    --scan_mode phaseformer_physics_mamba \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --lambda_phys 0.1 \
    --val_freq 2 \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp4_lsui
```

### Cell 6 — Print & Save Summary Table (Notebook B)
```python
import json
from pathlib import Path

combined = {}
for run in ["exp3_uieb", "exp4_uieb", "exp3_lsui", "exp4_lsui"]:
    p = Path("/kaggle/working/checkpoints") / run / "eval_summary.json"
    if not p.is_file():
        print(f"{run}: NOT FOUND")
        continue
    d = json.loads(p.read_text())
    combined[run] = d
    m = d["test_metrics"]
    hw = d["hardware_profile"]
    p256 = hw["256x256"]
    p720 = hw["1280x720"]
    print(
        f"{run:10s} | PSNR(RGB/Y)={m['psnr_rgb']:.2f}/{m['psnr_y']:.2f} dB | "
        f"SSIM(RGB/Y)={m['ssim_rgb']:.4f}/{m['ssim_y']:.4f} | "
        f"Params={hw['total_params_M']:.3f}M | GFLOPs={p256['gflops']} | "
        f"256={p256['fps']:.1f} FPS ({p256['latency_ms']:.1f} ms) | "
        f"720p={p720['fps']:.1f} FPS ({p720['latency_ms']:.1f} ms) | "
        f"VRAM={p256['peak_vram_mb']:.0f} MB | JIT={hw['torch_jit_trace']['passed']}"
    )

out_path = Path("/kaggle/working/checkpoints/notebook_B_summary.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"Saved combined summary -> {out_path}")
```
