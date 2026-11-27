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
    """Branch 1 invariants: exact chunked SSD scan, SGN bijection, ASE prompt grads, dynamic JIT."""
    from model import SemanticGuidedSingleScanStrategy, chunked_ssd_scan

    print("[CHECK (e)] Branch 1 (MambaIRv2 GPU Scan) mathematical invariants:")

    # 1. Exact equivalence of chunked_ssd_scan vs sequential recurrence on non-chunk-multiple L=135
    torch.manual_seed(0)
    B, L, H, P, N = 2, 135, 4, 8, 16
    x = torch.randn(B, L, H, P, device=device)
    dt = torch.rand(B, L, H, device=device) * 0.2 + 0.01
    A = -torch.linspace(0.5, 4.0, H, device=device)
    Bp = torch.randn(B, L, H, N, device=device) * 0.2
    Cp = torch.randn(B, L, H, N, device=device) * 0.2

    y_ssd = chunked_ssd_scan(x, dt, A, Bp, Cp, chunk_size=32)
    # Brute-force reference recurrence
    h = torch.zeros(B, H, P, N, device=device)
    ys = []
    for t in range(L):
        a_t = torch.exp(dt[:, t, :] * A.view(1, H))  # (B, H)
        b_t = dt[:, t, :].unsqueeze(-1) * Bp[:, t]    # (B, H, N)
        h = a_t.unsqueeze(-1).unsqueeze(-1) * h + x[:, t].unsqueeze(-1) * b_t.unsqueeze(-2)
        y_t = (h * Cp[:, t].unsqueeze(-2)).sum(dim=-1)
        ys.append(y_t)
    y_ref = torch.stack(ys, dim=1)
    ssd_err = (y_ssd - y_ref).abs().max().item()
    assert ssd_err < 1e-4, f"chunked_ssd_scan mismatch vs sequential recurrence: {ssd_err:.3e}"
    print(f"            - Exact chunked_ssd_scan vs sequential recurrence (L={L}): max_err={ssd_err:.3e} -> PASS")

    # 2. SGN permutation bijection & pre_dwconv presence
    assert isinstance(model.scan_strategy, SemanticGuidedSingleScanStrategy), "Expected SemanticGuidedSingleScanStrategy"
    assert hasattr(model.scan_strategy, "pre_dwconv"), "Missing 2D depthwise pre_dwconv before token reordering"
    feat = torch.randn(2, model.encoder.feature_dim, 16, 16, device=device)
    order, inv_order, p_flat = model.scan_strategy.compute_semantic_order(feat)
    x_flat = feat.flatten(2).transpose(1, 2)
    C = x_flat.shape[-1]
    L_f = x_flat.shape[1]
    x_perm = torch.gather(x_flat, 1, order.unsqueeze(-1).expand(2, L_f, C))
    x_back = torch.gather(x_perm, 1, inv_order.unsqueeze(-1).expand(2, L_f, C))
    bij_err = (x_flat - x_back).abs().max().item()
    assert bij_err == 0.0, f"SGN permutation is not an exact bijection: {bij_err}"
    assert torch.allclose(p_flat.sum(dim=-1), torch.ones(2, L_f, device=device), atol=1e-5)
    print(f"            - SGN GPU permutation bijection & softmax probs: max_err={bij_err:.1e} -> PASS")

    # 3. Verify ASE prompt_pool & semantic_router receive non-zero gradients
    model.train()
    model.zero_grad(set_to_none=True)
    inp = torch.rand(1, 3, 128, 128, device=device)
    pred, _ = forward_model(model, inp)
    pred.mean().backward()
    router_grad = sum(p.grad.abs().sum().item() for p in model.scan_strategy.semantic_router.parameters() if p.grad is not None)
    prompt_grad = sum(b.prompt_pool.grad.abs().sum().item() for b in model.mamba_corrector if b.prompt_pool.grad is not None)
    assert router_grad > 0.0, "semantic_router received zero gradient!"
    assert prompt_grad > 0.0, "ASE prompt_pool received zero gradient!"
    print(f"            - End-to-end ASE gradients (router={router_grad:.2e}, prompt_pool={prompt_grad:.2e}) -> PASS")

    # 4. Verify traced JIT model works dynamically at smaller (128x128) & larger (320x320) resolutions
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
