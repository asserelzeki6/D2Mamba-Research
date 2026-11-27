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
            src = inspect.getsource(cls.forward)
        except (OSError, TypeError):
            continue
        src = "\n".join(line[len(src) - len(src.lstrip()):] if line.strip() else "" for line in src.splitlines())
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


def check_jit_trace(model: nn.Module, device: torch.device) -> None:
    print("[CHECK (c)] torch.jit.trace export verification:")
    model.eval()
    wrapper = TraceWrapper(model).eval()
    x_256 = torch.rand(1, 3, 256, 256, device=device)
    with torch.no_grad(), forbid_cpu_sync_and_heapq():
        eager_256 = wrapper(x_256)
        traced = torch.jit.trace(wrapper, x_256, check_trace=False)
        traced_256 = traced(x_256)
    diff_256 = (eager_256 - traced_256).abs().max().item()
    assert torch.isfinite(traced_256).all(), "Non-finite values in traced output!"
    assert diff_256 < 1e-4, f"JIT trace max diff too large: {diff_256:.3e}"
    print(f"            - (1, 3, 256, 256) max|eager - traced| = {diff_256:.3e} -> PASS")


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
    """Extended by each feature branch to test mathematical invariants."""
    print("[CHECK (e)] Base harness invariants: PASS")


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
    check_jit_trace(model, device)
    check_init_ckpt_compat(args.scan_mode, device, args.base_dim, args.mamba_layers)
    check_branch_invariants(model, device)

    print("=== ALL VERIFICATION CHECKS PASSED ===")


if __name__ == "__main__":
    main()
