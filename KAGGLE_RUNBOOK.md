# Kaggle T4 Runbook — Branch 2: `exp/2-physics-attenuation-scan`

Underwater physics-anchored single-scan Mamba building on `exp/1-mambairv2-gpu-scan`:
- **Micro-AT GPU physics head** (`<0.15M` params) predicting relative depth proxy `d_hat` in `[0, 1]`, decoupled `(beta_D, beta_B)` (no hard `beta_R > beta_G > beta_B` clamp so green coastal waters remain valid), and top-5% far-field veiling light `B_inf`.
- **Near-to-far depth-ordered scan** with **telescoping Beer–Lambert `dt` modulation** (`delta_d_k = clamp(d_sorted[k] - d_sorted[k-1], min=0)` multiplied by `(beta_D, beta_B)`).
- **Training-only Pseudo-Linearized Akkaynak–Treibitz Cycle Consistency Loss** (strictly FP32 even under `--amp`) + non-collapse regularizer preventing `beta_D == beta_B`.

> **Kaggle Setup**: Attach UIEB (`raw-890`/`reference-890`) and/or LSUI (`LSUI/**/input`, `LSUI/**/GT`) datasets with GPU T4 enabled and Internet ON.

---

### Cell 1 — Clone Branch & Smoke Verify
```python
!git clone -b exp/2-physics-attenuation-scan <YOUR_FORK_URL> D2Mamba-Research
%cd D2Mamba-Research
!python3 verify_branch.py
```

### Cell 2 — Train & Evaluate on UIEB (with Akkaynak–Treibitz Cycle Consistency Loss)
```python
!python3 train_kaggle.py \
    --dataset uieb \
    --scan_mode physics_attenuation \
    --epochs 60 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --lambda_phys 0.1 \
    --init_ckpt ckpt/netG_UIEB.pt \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp2_uieb
```
*(For full 200-epoch training from scratch, remove `--init_ckpt ckpt/netG_UIEB.pt` and set `--epochs 200`.)*

### Cell 3 — Train & Evaluate on LSUI
```python
!python3 train_kaggle.py \
    --dataset lsui \
    --scan_mode physics_attenuation \
    --epochs 60 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --lambda_phys 0.1 \
    --init_ckpt ckpt/netG_LSUI.pt \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp2_lsui
```
