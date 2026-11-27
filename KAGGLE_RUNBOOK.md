# Kaggle T4 Runbook — Branch 1: `exp/1-mambairv2-gpu-scan`

Replaces `D2Mamba`'s 84.85 ms CPU `heapq` A* scan with a 100% GPU-native `MambaIRv2`-inspired Semantic-Guided Single Scan (`SGN` + 2D depthwise pre-conv + prompt-augmented `ASE` + exact pure-PyTorch chunked SSD scan). Fully compatible with official `D2Mamba` checkpoints (`ckpt/netG_UIEB.pt`, `ckpt/netG_LSUI.pt`).

> **Kaggle Setup**: Attach the UIEB (`raw-890`/`reference-890`) and/or LSUI (`LSUI/**/input`, `LSUI/**/GT`) datasets to your Kaggle Notebook with GPU T4 enabled and Internet ON (to fetch official zero-leakage test splits).

---

### Cell 1 — Clone Branch & Smoke Verify
```python
!git clone -b exp/1-mambairv2-gpu-scan <YOUR_FORK_URL> D2Mamba-Research
%cd D2Mamba-Research
!python3 verify_branch.py
```

### Cell 2 — Train & Evaluate on UIEB (60-Epoch Warm-Start Milestone or 200-Epoch Full)
```python
# Fast 60-epoch milestone fine-tuning from official D2Mamba UIEB checkpoint:
!python3 train_kaggle.py \
    --dataset uieb \
    --scan_mode mambairv2 \
    --epochs 60 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --init_ckpt ckpt/netG_UIEB.pt \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp1_uieb
```
*(For full 200-epoch training from scratch, remove `--init_ckpt ckpt/netG_UIEB.pt` and set `--epochs 200`.)*

### Cell 3 — Train & Evaluate on LSUI
```python
!python3 train_kaggle.py \
    --dataset lsui \
    --scan_mode mambairv2 \
    --epochs 60 \
    --batch_size 8 \
    --lr 3e-4 \
    --img_size 256 \
    --amp \
    --init_ckpt ckpt/netG_LSUI.pt \
    --seed 42 \
    --save_dir /kaggle/working/checkpoints/exp1_lsui
```
