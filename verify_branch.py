#!/usr/bin/env python3
"""
Local CPU/GPU smoke-test & invariant verification script for D2Mamba research branches.

Verifies:
  (a) Forward + backward pass at (1, 3, 256, 256) and (1, 3, 720, 1280) with zero NaNs/Infs.
  (b) Total parameter count < 2.2M.
  (c) torch.jit.trace succeeds cleanly and matches eager output (< 1e-4 max abs diff).
  (d) Zero .cpu() / .numpy() / heapq calls inside forward() (runtime guard + static AST check).
  (e) Branch-specific mathematical invariants & --init_ckpt ckpt/netG_UIEB.pt compatibility.
"""

import argparse
import ast
import contextlib
import heapq
import inspect
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from model import D2Mamba
from train_kaggle import DEFAULT_SCAN_MODE, TraceWrapper, forward_model, load_compatible_checkpoint

MAX_PARAMS = 2_200_000


def _build_model(scan_mode: str, base_dim: int = 24, mamba_layers: int = 4) -> nn.Module:
    sig = inspect.signature(D2Mamba.__init__)
    kwargs = {
        "scan_strategy_name": scan_mode,
        "in_channels": 3,
        "base_dim": base_dim,
        "mamba_layers": mamba_layers,
    }
    if "num_paths" in sig.parameters:
        kwargs["num_paths"] = 4
    return D2Mamba(**kwargs)


@contextlib.contextmanager
def forbid_cpu_sync_and_heapq():
    """Runtime guard that raises RuntimeError if Tensor.cpu, Tensor.numpy, or heapq is called."""
    orig_cpu = torch.Tensor.cpu
    orig_numpy = torch.Tensor.numpy
    orig_heappush = heapq.heappush
    orig_heappop = heapq.heappop

    def _blocked_cpu(self, *args, **kwargs):
        raise RuntimeError("Forbidden Tensor.cpu() call detected inside forward()!")

    def _blocked_numpy(self, *args, **kwargs):
        raise RuntimeError("Forbidden Tensor.numpy() call detected inside forward()!")

    def _blocked_heappush(*args, **kwargs):
        raise RuntimeError("Forbidden heapq.heappush() call detected inside forward()!")

    def _blocked_heappop(*args, **kwargs):
        raise RuntimeError("Forbidden heapq.heappop() call detected inside forward()!")

    torch.Tensor.cpu = _blocked_cpu
    torch.Tensor.numpy = _blocked_numpy
    heapq.heappush = _blocked_heappush
    heapq.heappop = _blocked_heappop
    try:
        yield
    finally:
        torch.Tensor.cpu = orig_cpu
        torch.Tensor.numpy = orig_numpy
        heapq.heappush = orig_heappush
        heapq.heappop = orig_heappop


def check_static_ast_no_cpu_heapq(model: nn.Module) -> Tuple[bool, List[str]]:
    """Inspect active module forward() source files via AST for forbidden .cpu()/.numpy()/heapq calls."""
    violations: List[str] = []
    checked_classes = set()
    for mod in model.modules():
        cls = type(mod)
        if cls in checked_classes:
            continue
        checked_classes.add(cls)
        if cls.__module__.startswith("torch."):
            continue
        try:
            tree = ast.parse(inspect.cleandoc(inspect.getsource(cls.forward)))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in {"cpu", "numpy", "heappush", "heappop"}:
                    violations.append(f"{cls.__name__}.forward calls .{func.attr}()")
                elif isinstance(func, ast.Name) and func.id in {"heappush", "heappop"}:
                    violations.append(f"{cls.__name__}.forward calls {func.id}()")
    return len(violations) == 0, violations


def check_param_budget(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    print(f"[CHECK (b)] Total params: {total:,} ({total / 1e6:.4f}M) | limit: {MAX_PARAMS:,} (2.2000M)")
    for name, child in model.named_children():
        c_params = sum(p.numel() for p in child.parameters())
        print(f"            - {name}: {c_params:,}")
    assert total < MAX_PARAMS, f"Parameter count {total:,} exceeds 2.2M limit!"
    print("            -> PASS")
    return total


def check_forward_backward(model: nn.Module, device: torch.device, shapes: List[Tuple[int, int, int, int]]) -> None:
    print("[CHECK (a) & (d)] Forward + backward with runtime CPU-sync/heapq guard:")
    model.train()
    for shape in shapes:
        t0 = time.perf_counter()
        x = torch.rand(*shape, device=device, requires_grad=True)
        model.zero_grad(set_to_none=True)
        with forbid_cpu_sync_and_heapq():
            pred, aux = forward_model(model, x)

        assert pred.shape == shape, f"Expected output shape {shape}, got {tuple(pred.shape)}"
        assert torch.isfinite(pred).all(), f"Non-finite values in output for shape {shape}"

        loss = pred.mean()
        if aux is not None and isinstance(aux, dict):
            phys = aux.get("physics", aux) if isinstance(aux.get("physics", None), dict) else None
            if phys is not None:
                for k, v in phys.items():
                    if isinstance(v, torch.Tensor):
                        assert torch.isfinite(v).all(), f"Non-finite physics tensor '{k}' at {shape}"
                        loss = loss + 0.01 * v.float().mean()

        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), f"Non-finite input grad at {shape}"

        non_zero_grads = 0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in param '{name}' at {shape}"
                if p.grad.abs().sum().item() > 0:
                    non_zero_grads += 1

        elapsed = time.perf_counter() - t0
        print(
            f"            - shape={shape}: out=[{pred.min().item():.4f}, {pred.max().item():.4f}], "
            f"nonzero_grad_params={non_zero_grads}, time={elapsed:.2f}s -> PASS"
        )

    ok_ast, violations = check_static_ast_no_cpu_heapq(model)
    assert ok_ast, f"Static AST check failed: {violations}"
    print("            - Static AST check (zero .cpu()/.numpy()/heapq in forward): PASS")


def check_mixed_precision_autocast(model: nn.Module, device: torch.device) -> None:
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    print(f"[CHECK (a-amp)] Mixed-precision autocast ({device.type}, {amp_dtype}) forward + backward:")
    model.train()
    x = torch.rand(2, 3, 128, 128, device=device, requires_grad=True)
    model.zero_grad(set_to_none=True)
    with forbid_cpu_sync_and_heapq(), torch.amp.autocast(device_type=device.type, enabled=True, dtype=amp_dtype):
        pred, aux = forward_model(model, x)
    with torch.amp.autocast(device_type=device.type, enabled=False):
        assert torch.isfinite(pred.float()).all(), "Non-finite output under mixed-precision autocast!"
        loss = pred.float().mean()
        if aux is not None and isinstance(aux, dict):
            phys = aux.get("physics", aux) if isinstance(aux.get("physics", None), dict) else None
            if phys is not None:
                for k, v in phys.items():
                    if isinstance(v, torch.Tensor):
                        assert v.dtype == torch.float32, f"Physics tensor '{k}' must stay FP32 under AMP, got {v.dtype}"
                        assert torch.isfinite(v).all(), f"Non-finite physics tensor '{k}' under AMP!"
                        loss = loss + 0.01 * v.mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "Non-finite input grad under autocast!"
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite param grad '{name}' under autocast!"
    print(f"            - autocast fwd+bwd finite (out dtype={pred.dtype}) -> PASS")


def check_jit_trace(model: nn.Module, device: torch.device) -> None:
    print("[CHECK (c)] torch.jit.trace export verification (strict check_trace=True + dynamic shapes):")
    model.eval()
    wrapper = TraceWrapper(model).eval()
    x_256 = torch.rand(1, 3, 256, 256, device=device)
    with torch.no_grad(), forbid_cpu_sync_and_heapq():
        eager_256 = wrapper(x_256)
        traced = torch.jit.trace(wrapper, x_256, check_trace=True)
        traced_256 = traced(x_256)
        diff_256 = (eager_256 - traced_256).abs().max().item()
        assert torch.isfinite(traced_256).all(), "Non-finite values in traced output!"
        assert diff_256 < 1e-4, f"JIT trace max diff too large: {diff_256:.3e}"

        # Dynamic batch (B=2) and BOTH smaller (128x128) & larger (320x320) resolutions
        for label, shape in [("B=2 256x256", (2, 3, 256, 256)), ("128x128", (1, 3, 128, 128)), ("320x320", (1, 3, 320, 320))]:
            xd = torch.rand(*shape, device=device)
            ed = wrapper(xd)
            td = traced(xd)
            dd = (ed - td).abs().max().item()
            assert td.shape == ed.shape and torch.isfinite(td).all(), f"Dynamic traced output invalid for {label}"
            assert dd < 1e-4, f"Dynamic JIT trace mismatch at {label}: {dd:.3e}"
            print(f"            - dynamic {label} max|eager - traced| = {dd:.3e} -> PASS")
    print(f"            - (1, 3, 256, 256) check_trace=True max|eager - traced| = {diff_256:.3e} -> PASS")


def check_init_ckpt_compat(scan_mode: str, device: torch.device, base_dim: int, mamba_layers: int) -> None:
    ckpt_path = Path(__file__).resolve().parent / "ckpt" / "netG_UIEB.pt"
    if not ckpt_path.is_file():
        print("[CHECK (e-ckpt)] ckpt/netG_UIEB.pt not found, skipping init_ckpt test.")
        return
    m = _build_model(scan_mode, base_dim=base_dim, mamba_layers=mamba_layers).to(device)
    stats = load_compatible_checkpoint(m, str(ckpt_path))
    x = torch.rand(1, 3, 256, 256, device=device)
    m.eval()
    with torch.no_grad():
        pred, _ = forward_model(m, x)
    assert torch.isfinite(pred).all(), "Non-finite output after loading netG_UIEB.pt!"
    print(f"[CHECK (e-ckpt)] --init_ckpt ckpt/netG_UIEB.pt compatibility ({stats['loaded_tensors']} tensors loaded) -> PASS")


def check_branch_invariants(model: nn.Module, device: torch.device) -> None:
    """Branch 4 invariants:
      - RepConv .fuse() <1e-5 (unit & full end-to-end model)
      - Explicit 2D Fourier Phase-Amplitude Cross-Gating: FP32 round-trip exactness,
        AMP/FP16 safety on arbitrary non-pow2 & odd shapes, zero/constant input gradient stability
      - Haar DWT/IDWT exactness & FP16 safety
      - H/8 bottleneck Mamba & Micro-AT (<0.15M), GBPhaseRedRefiner, Telescoping Beer-Lambert & AT cycle loss
      - Fused model torch.jit.trace (check_trace=True) across dynamic batch, smaller, non-pow2, odd & larger shapes
      - FlopCounterMode GFLOPs report at 256x256
    """
    import copy
    from model import (
        FourierPhaseAmplitudeGate,
        GBPhaseRedRefiner,
        MicroATHead,
        PhysicsAttenuationScanStrategy,
        RepConv,
        _irfft2_reconstruct_fp32,
        _rfft2_decompose_fp32,
        haar_dwt_2d,
        haar_idwt_2d,
    )
    from train_kaggle import _count_gflops, compute_physics_cycle_loss

    print("[CHECK (e)] Branch 4 (Phaseformer-Physics-Mamba + Haar U-Net + RepConv) invariants:")

    # 1. Unit RepConv .fuse() exactness (< 1e-5) with non-trivial BN running stats
    rc = RepConv(32).to(device).train()
    for _ in range(3):
        _ = rc(torch.randn(2, 32, 32, 32, device=device) * 2.5 + 0.7)
    rc.eval()
    x_rc = torch.randn(1, 32, 32, 32, device=device)
    with torch.no_grad():
        y_train_branch = rc(x_rc)
        rc.fuse()
        y_fused = rc(x_rc)
    unit_fuse_err = (y_train_branch - y_fused).abs().max().item()
    assert rc.fused and rc.fused_conv is not None
    assert unit_fuse_err < 1e-5, f"Unit RepConv .fuse() error too large: {unit_fuse_err:.3e}"
    print(f"            - Unit RepConv (7x7+3x3+1x1) .fuse(): max|train - fused| = {unit_fuse_err:.3e} (< 1e-5) -> PASS")

    # 2. Full end-to-end model.fuse() exactness (< 1e-5)
    m_copy = copy.deepcopy(model).to(device).train()
    for _ in range(2):
        _ = m_copy(torch.rand(1, 3, 128, 128, device=device))
    m_copy.eval()
    x_full = torch.rand(1, 3, 256, 256, device=device)
    with torch.no_grad():
        y_unfused = m_copy(x_full)[0]
        m_copy.fuse()
        y_model_fused = m_copy(x_full)[0]
    full_fuse_err = (y_unfused - y_model_fused).abs().max().item()
    assert full_fuse_err < 1e-5, f"Full model.fuse() error too large: {full_fuse_err:.3e}"
    assert all(m.fused for m in m_copy.modules() if isinstance(m, RepConv)), "Some RepConv layers were not fused!"
    print(f"            - Full model.fuse() end-to-end: max|train - fused| = {full_fuse_err:.3e} (< 1e-5) -> PASS")

    # 3. Explicit 2D Fourier Phase-Amplitude Cross-Gating invariants
    phase_gates = [m for m in model.modules() if isinstance(m, FourierPhaseAmplitudeGate)]
    assert len(phase_gates) >= 3, f"Expected >=3 FourierPhaseAmplitudeGate blocks, found {len(phase_gates)}"
    for res in [(256, 256), (198, 314), (127, 129), (720, 1280)]:
        x_f = torch.randn(1, 3, res[0], res[1], device=device, dtype=torch.float32)
        amp, _, cos_p, sin_p = _rfft2_decompose_fp32(x_f)
        x_rec = _irfft2_reconstruct_fp32(amp, cos_p, sin_p, x_f)
        f_err = (x_f - x_rec).abs().max().item()
        assert f_err < 1e-4, f"2D Fourier decompose/reconstruct error at {res}: {f_err:.3e}"

    # Zero-input and constant-input gradient safety (no atan2/abs NaN at 0+0j)
    fg = FourierPhaseAmplitudeGate(16, hidden_dim=12).to(device).train()
    for label, x_edge in [
        ("zeros", torch.zeros(1, 16, 33, 47, device=device, requires_grad=True)),
        ("const", torch.full((1, 16, 32, 32), 0.5, device=device, requires_grad=True)),
    ]:
        fg.zero_grad(set_to_none=True)
        y_edge = fg(x_edge)
        assert y_edge.shape == x_edge.shape and torch.isfinite(y_edge).all(), f"Non-finite FFT gate output on {label}"
        y_edge.mean().backward()
        assert x_edge.grad is not None and torch.isfinite(x_edge.grad).all(), f"Non-finite input grad on {label}"
        for n, p in fg.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"Non-finite param grad {n} on {label}"

    # FP16/BF16 autocast + direct half-input safety on non-power-of-two & odd resolutions
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    fg.eval()
    with torch.no_grad():
        for res in [(64, 64), (55, 79), (99, 157)]:
            x_amp = torch.randn(1, 16, res[0], res[1], device=device)
            with torch.amp.autocast(device_type=device.type, enabled=True, dtype=amp_dtype):
                y_amp = fg(x_amp)
            assert y_amp.shape == x_amp.shape and torch.isfinite(y_amp.float()).all()
            y_half = fg(x_amp.half())
            assert y_half.dtype == torch.float16 and torch.isfinite(y_half.float()).all()
    print(f"            - FourierPhaseAmplitudeGate (n={len(phase_gates)}, zero/const grad finite, FP16/non-pow2/odd safe) -> PASS")

    # 4. Haar DWT/IDWT exact orthonormal reconstruction + FP16 safety on arbitrary resolutions
    for res in [(256, 256), (198, 314), (127, 129), (720, 1280)]:
        x_w = torch.randn(1, 3, res[0], res[1], device=device)
        ll, lh, hl, hh = haar_dwt_2d(x_w)
        x_rec = haar_idwt_2d(ll, lh, hl, hh, out_h=res[0], out_w=res[1])
        w_err = (x_w - x_rec).abs().max().item()
        assert w_err < 1e-5, f"Haar DWT/IDWT reconstruction error at {res}: {w_err:.3e}"
        x_h = x_w.half()
        ll_h, lh_h, hl_h, hh_h = haar_dwt_2d(x_h)
        rec_h = haar_idwt_2d(ll_h, lh_h, hl_h, hh_h, out_h=res[0], out_w=res[1])
        assert rec_h.dtype == torch.float16 and torch.isfinite(rec_h).all()
    with torch.no_grad():
        for res in [(198, 314), (127, 129)]:
            y_res = m_copy(torch.rand(1, 3, res[0], res[1], device=device))[0]
            assert y_res.shape == (1, 3, res[0], res[1]) and torch.isfinite(y_res).all()
    print("            - Haar DWT/IDWT exactness (<1e-5) & arbitrary even/odd res (198x314, 127x129, 720x1280) -> PASS")

    # 5. Physics Mamba strictly in compressed H/8 bottleneck (256/8 = 32 -> L = 1024)
    with torch.no_grad():
        _, path_len, _ = m_copy(x_full)
    assert path_len == (256 // 8) * (256 // 8), f"Expected H/8 bottleneck L=1024, got {path_len}"
    micro_at_params = sum(p.numel() for p in model.physics_head.parameters())
    assert isinstance(model.physics_head, MicroATHead) and micro_at_params < 150_000
    print(f"            - Compressed H/8 bottleneck Mamba (L={path_len}) & Micro-AT ({micro_at_params:,} < 0.15M) -> PASS")

    # 6. Green/Blue-Phase-Guided Red Channel Refinement invariant
    refiner = model.gb_red_refiner
    assert isinstance(refiner, GBPhaseRedRefiner)
    refiner.train()
    refiner.zero_grad(set_to_none=True)
    x_rgb = torch.rand(1, 3, 64, 64, device=device, requires_grad=True)
    out_rgb = refiner(x_rgb)
    gb_diff = (out_rgb[:, 1:3] - x_rgb[:, 1:3]).abs().max().item()
    assert gb_diff == 0.0, f"GBPhaseRedRefiner unexpectedly modified G/B channels: {gb_diff}"
    out_rgb[:, 0:1].mean().backward()
    ref_grad = sum(p.grad.abs().sum().item() for p in refiner.parameters() if p.grad is not None)
    assert ref_grad > 0.0, "GBPhaseRedRefiner received zero gradient!"
    print(f"            - GBPhaseRedRefiner (G/B preserved exact=0.0, R grad={ref_grad:.2e}) -> PASS")

    # 7. Telescoping Beer-Lambert & Akkaynak-Treibitz cycle loss check
    d_const = torch.full((1, 1, 16, 16), 0.75, device=device)
    bD_c = torch.full((1, 3, 16, 16), 1.2, device=device)
    bB_c = torch.full((1, 3, 16, 16), 0.6, device=device)
    _, _, dd, od = PhysicsAttenuationScanStrategy.compute_depth_order_and_opt_delta(d_const, bD_c, bB_c)
    assert dd.abs().max().item() == 0.0 and model.mamba_corrector[0].compute_physics_dt(od).abs().max().item() == 0.0
    model.train()
    model.zero_grad(set_to_none=True)
    inp = torch.rand(2, 3, 128, 128, device=device)
    pred, aux = forward_model(model, inp)
    loss_phys, _ = compute_physics_cycle_loss(pred, inp, aux["physics"])
    assert torch.isfinite(loss_phys).all()
    loss_phys.backward()
    print("            - Telescoping Beer-Lambert & Akkaynak-Treibitz cycle loss -> PASS")

    # 8. Fused model torch.jit.trace (check_trace=True) + dynamic smaller/larger/non-pow2/odd resolutions
    wrapper = TraceWrapper(m_copy).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, x_full, check_trace=True)
        diff_256 = (wrapper(x_full) - traced(x_full)).abs().max().item()
        assert diff_256 < 1e-4, f"Fused JIT trace mismatch at 256x256: {diff_256:.3e}"
        dyn_msgs = []
        for lbl, shp in [
            ("B=2 256x256", (2, 3, 256, 256)),
            ("128x128", (1, 3, 128, 128)),
            ("198x314", (1, 3, 198, 314)),
            ("127x129", (1, 3, 127, 129)),
            ("320x384", (1, 3, 320, 384)),
        ]:
            xd = torch.rand(*shp, device=device)
            ed = wrapper(xd)
            td = traced(xd)
            dd = (ed - td).abs().max().item()
            assert td.shape == ed.shape and dd < 1e-4, f"Fused dynamic JIT mismatch at {lbl}: {dd:.3e}"
            dyn_msgs.append(f"{lbl}={dd:.2e}")
    print(f"            - Fused model JIT trace (check_trace=True, 256={diff_256:.2e}, {', '.join(dyn_msgs)}) -> PASS")

    # 9. GFLOPs budget report (fused model at 256x256)
    gflops_256 = _count_gflops(m_copy, (1, 3, 256, 256), device)
    if gflops_256 is not None:
        print(f"            - Fused GFLOPs @ 256x256: {gflops_256:.3f} GFLOPs -> PASS")
    else:
        print("            - Fused GFLOPs @ 256x256: unavailable (FlopCounterMode skipped)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify D2Mamba branch constraints & invariants")
    parser.add_argument("--scan_mode", type=str, default=DEFAULT_SCAN_MODE)
    parser.add_argument("--base_dim", type=int, default=24)
    parser.add_argument("--mamba_layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Verifying branch | scan_mode={args.scan_mode} | device={device} ===")

    model = _build_model(args.scan_mode, base_dim=args.base_dim, mamba_layers=args.mamba_layers).to(device)

    check_param_budget(model)
    check_forward_backward(model, device, shapes=[(1, 3, 256, 256), (1, 3, 720, 1280)])
    check_mixed_precision_autocast(model, device)
    check_jit_trace(model, device)
    check_init_ckpt_compat(args.scan_mode, device, args.base_dim, args.mamba_layers)
    check_branch_invariants(model, device)

    print("=== ALL VERIFICATION CHECKS PASSED ===")


if __name__ == "__main__":
    main()
