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
    ckpt_path = Path("ckpt/netG_UIEB.pt")
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
    """Branch 2 invariants: Micro-AT <0.15M, decoupled beta_D/beta_B, top-5% B_inf,
    telescoping Beer-Lambert invariance, Akkaynak-Treibitz cycle loss, and dynamic JIT."""
    from model import (
        MicroATHead,
        PhysicsAttenuationScanStrategy,
        _anchor_veiling_light,
    )
    from train_kaggle import compute_physics_cycle_loss

    print("[CHECK (e)] Branch 2 (Physics-Anchored Attenuation Scan) mathematical invariants:")

    # 1. Micro-AT parameter count strictly < 0.15M
    assert isinstance(model.physics_head, MicroATHead), "Expected MicroATHead on Branch 2"
    micro_at_params = sum(p.numel() for p in model.physics_head.parameters())
    assert micro_at_params < 150_000, f"Micro-AT params ({micro_at_params:,}) exceeds 0.15M!"
    print(f"            - Micro-AT params: {micro_at_params:,} ({micro_at_params/1e6:.4f}M < 0.15M) -> PASS")

    # 2. Verify NO hard beta_R > beta_G > beta_B clamp (green coastal water validity)
    raw_green = torch.tensor([[[[-1.0]], [[2.0]], [[0.5]]]], device=device)  # G > B > R
    bounded = model.physics_head._smooth_bound_beta(raw_green)
    assert bounded[0, 1, 0, 0] > bounded[0, 2, 0, 0] > bounded[0, 0, 0, 0] > 0.0, (
        "Smooth beta bound unexpectedly clamped channel ordering!"
    )
    print("            - Decoupled beta_D/beta_B free of R>G>B clamp (green coastal valid) -> PASS")

    # 3. Verify top-5% farthest pixel veiling light anchoring
    H_t, W_t = 20, 20  # L = 400 -> top 5% = 20 pixels
    d_syn = torch.full((1, 1, H_t, W_t), 0.1, device=device)
    x_syn = torch.tensor([0.9, 0.1, 0.1], device=device).view(1, 3, 1, 1).expand(1, 3, H_t, W_t).clone()
    # Set 20 farthest pixels (5%) to known far-field veiling color [0.12, 0.68, 0.42]
    d_syn[0, 0, 0, :] = 0.99
    target_binf = torch.tensor([0.12, 0.68, 0.42], device=device)
    x_syn[0, :, 0, :] = target_binf.view(3, 1)
    est_binf = _anchor_veiling_light(x_syn, d_syn).view(3)
    binf_err = (est_binf - target_binf).abs().max().item()
    assert binf_err < 1e-4, f"Top-5% B_inf anchoring mismatch: {binf_err:.3e}"
    print(f"            - Top-5% farthest pixel B_inf anchoring: max_err={binf_err:.3e} -> PASS")

    # 4. Telescoping Beer-Lambert invariance
    # 4a) Constant depth => delta_d == 0 and dt_phys == 0 (no hidden-state collapse!)
    d_const = torch.full((1, 1, 16, 16), 0.75, device=device)
    bD_const = torch.full((1, 3, 16, 16), 1.2, device=device)
    bB_const = torch.full((1, 3, 16, 16), 0.6, device=device)
    _, _, delta_d_c, opt_delta_c = PhysicsAttenuationScanStrategy.compute_depth_order_and_opt_delta(
        d_const, bD_const, bB_const
    )
    dt_phys_c = model.mamba_corrector[0].compute_physics_dt(opt_delta_c)
    assert delta_d_c.abs().max().item() == 0.0, "Constant depth produced non-zero delta_d!"
    assert dt_phys_c.abs().max().item() == 0.0, "Constant depth produced non-zero dt_phys!"

    # 4b) Resolution invariance: linear ramp 0.1 -> 0.9 at 8x8 (L=64) vs 16x16 (L=256)
    def _total_dt_phys_for_size(s: int) -> torch.Tensor:
        ramp = torch.linspace(0.1, 0.9, s * s, device=device).view(1, 1, s, s)
        bD = torch.tensor([1.4, 0.7, 0.5], device=device).view(1, 3, 1, 1).expand(1, 3, s, s)
        bB = torch.tensor([0.5, 0.9, 1.1], device=device).view(1, 3, 1, 1).expand(1, 3, s, s)
        _, _, _, opt_d = PhysicsAttenuationScanStrategy.compute_depth_order_and_opt_delta(ramp, bD, bB)
        return model.mamba_corrector[0].compute_physics_dt(opt_d).sum(dim=1)  # (1, H)

    tot_64 = _total_dt_phys_for_size(8)
    tot_256 = _total_dt_phys_for_size(16)
    tele_err = (tot_64 - tot_256).abs().max().item()
    assert tele_err < 1e-5, f"Telescoping Beer-Lambert invariance violated: {tele_err:.3e}"
    print(f"            - Telescoping Beer-Lambert invariance (const=0, ramp diff={tele_err:.2e}) -> PASS")

    # 5. Pseudo-Linearized Akkaynak-Treibitz Cycle Consistency Loss + non-collapse regularizer
    model.train()
    model.zero_grad(set_to_none=True)
    inp = torch.rand(2, 3, 128, 128, device=device)
    pred, aux = forward_model(model, inp)
    assert aux is not None and "physics" in aux, "Training forward did not expose physics dict!"
    phys = aux["physics"]
    assert 0.0 <= phys["d_hat"].min().item() and phys["d_hat"].max().item() <= 1.0
    assert (phys["beta_D"] > 0).all() and (phys["beta_B"] > 0).all()
    assert 0.0 <= phys["B_inf"].min().item() and phys["B_inf"].max().item() <= 1.0

    loss_phys, pstats = compute_physics_cycle_loss(pred, inp, phys, lambda_phys=0.1, lambda_noncollapse=0.01)
    assert torch.isfinite(loss_phys).all(), "Non-finite physics cycle loss!"
    loss_phys.backward()
    phys_head_grad = sum(p.grad.abs().sum().item() for p in model.physics_head.parameters() if p.grad is not None)
    assert phys_head_grad > 0.0, "MicroATHead received zero gradient from physics cycle loss!"

    # Verify non-collapse regularizer penalizes beta_D == beta_B
    eq_phys = {**{k: v.detach() for k, v in phys.items()}, "beta_B": phys["beta_D"].detach()}
    _, eq_stats = compute_physics_cycle_loss(pred.detach(), inp, eq_phys)
    assert eq_stats["phys_noncollapse"] > pstats["phys_noncollapse"], (
        "Non-collapse regularizer failed to penalize beta_D == beta_B!"
    )
    print(
        f"            - Akkaynak-Treibitz cycle loss + non-collapse grad={phys_head_grad:.2e} "
        f"(sep={pstats['beta_sep']:.3f}) -> PASS"
    )

    # 6. Dynamic-shape JIT trace (traced@256, eval@128 & eval@320) & deepcopy safety post-train-forward
    import copy
    _ = copy.deepcopy(model)  # Must succeed cleanly without non-leaf tensors attached to self!
    model.eval()
    wrapper = TraceWrapper(model).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, torch.rand(1, 3, 256, 256, device=device), check_trace=True)
        x_128 = torch.rand(1, 3, 128, 128, device=device)
        x_320 = torch.rand(1, 3, 320, 320, device=device)
        diff_128 = (wrapper(x_128) - traced(x_128)).abs().max().item()
        diff_320 = (wrapper(x_320) - traced(x_320)).abs().max().item()
    assert diff_128 < 1e-4 and diff_320 < 1e-4, f"Dynamic-shape JIT trace mismatch: 128={diff_128:.3e}, 320={diff_320:.3e}"
    print(f"            - Dynamic-shape JIT trace (traced@256, eval@128={diff_128:.2e}, eval@320={diff_320:.2e}) -> PASS")


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
