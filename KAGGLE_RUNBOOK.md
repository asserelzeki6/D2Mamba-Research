# Kaggle T4 Runbook — Branch 3: `exp/3-haar-phase-fast`

Maximum edge-speed (>45–50 FPS on T4) + Frequency & Underwater Physics architecture building on `exp/2-physics-attenuation-scan`:
- **Outer Stages (`H/2`, `H/4`)**: Structurally Reparameterizable Depthwise Convolutions (`RepConv`: parallel `7x7 + 3x3 + 1x1` branches with exact `.fuse()` verified `< 1e-5`) + **1-level Haar DWT High-Frequency Gating** (100% FP16-safe on arbitrary even resolutions, zero `cuFFT`).
- **Compressed Bottleneck (`H/8`)**: Physics-guided single-scan Mamba (`Micro-AT` + telescoping Beer–Lambert `dt` modulation) strictly at low resolution (4x fewer tokens than `H/4`).
- **Green/Blue-Phase-Guided Red Channel Refinement (`GBPhaseRedRefiner`)**: Uses Haar high-frequency & illumination-invariant unit directional phase cues from intact G & B channels to sharpen and stabilize the attenuated R channel.
- Automatically fuses all `RepConv` branches during final test evaluation & hardware profiling.

> **Kaggle Setup**: Attach UIEB (`raw-890`/`reference-890`) and/or LSUI (`LSUI/**/input`, `LSUI/**/GT`) datasets with GPU T4 enabled and Internet ON.

---

### Cell 1 — Clone Branch & Smoke Verify (includes `RepConv.fuse()` `< 1e-5` check)
```python
!git clone -b exp/3-haar-phase-fast <YOUR_FORK_URL> D2Mamba-Research
%cd D2Mamba-Research
!python3 verify_branch.py
```

### Cell 2 — Train & Evaluate on UIEB (200 Epochs, FP16 AMP + Physics Cycle Loss)
```python
!python3 train_kaggle.py \
    --dataset uieb \
    --scan_mode haar_phase_fast \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --lambda_phys 0.1 \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp3_uieb
```

### Cell 3 — Train & Evaluate on LSUI (200 Epochs, FP16 AMP + Physics Cycle Loss)
```python
!python3 train_kaggle.py \
    --dataset lsui \
    --scan_mode haar_phase_fast \
    --epochs 200 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --lambda_phys 0.1 \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp3_lsui
```
