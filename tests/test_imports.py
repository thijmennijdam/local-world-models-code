"""Smoke tests: every module imports, and released checkpoints load and run on CPU.

Run with ``pytest tests/`` or ``python tests/test_imports.py`` (no pytest needed).

Checkpoint tests look for ``$LWM_CKPT_DIR/<name>/best.ckpt`` (with ``.hydra/config.yaml``
next to it) for the names in ``CHECKPOINTS``; missing checkpoints are reported as skipped.
"""

import importlib
import os
import pathlib
import pkgutil
import sys
import traceback

import torch

import lwm

CHECKPOINTS = ["LWM-100M_d1024_L8", "LWM-200M_d1024_L16"]


def all_modules():
    root = pathlib.Path(lwm.__file__).parent
    names = ["lwm"]
    for info in pkgutil.walk_packages([str(root)], prefix="lwm."):
        names.append(info.name)
    return sorted(names)


def test_imports():
    failed = []
    for name in all_modules():
        try:
            importlib.import_module(name)
        except Exception:
            failed.append(name)
            traceback.print_exc()
    assert not failed, f"failed to import: {failed}"


def _forward_dummy(path):
    from lwm.world_model.analysis.checkpoint import load_checkpoint
    from lwm.world_model.model import LWMv3

    model, ckpt, norm_stats, cfg = load_checkpoint(str(path), device="cpu")
    assert isinstance(model, LWMv3)
    assert norm_stats is not None

    torch.manual_seed(0)
    batch, n_points, horizon = 2, 512, cfg.model.n_future_steps
    obs = {
        "agent_state": torch.randn(batch, cfg.model.agent_state_dim),
        "pointcloud": torch.randn(batch, n_points, 3) * 0.1,
        "static_affordance": torch.randn(batch, 3) * 0.1,
        "action_horizon": torch.randn(batch, horizon, cfg.model.action_dim) * 0.01,
    }
    with torch.no_grad():
        out = model(obs)
    tensors = list(out.values()) if isinstance(out, dict) else list(out) if isinstance(out, (tuple, list)) else [out]
    for t in tensors:
        if torch.is_tensor(t):
            assert t.shape[0] == batch, t.shape
            assert torch.isfinite(t).all()
    return out


def test_checkpoints_forward():
    ckpt_dir = os.environ.get("LWM_CKPT_DIR")
    for name in CHECKPOINTS:
        path = pathlib.Path(ckpt_dir) / name / "best.ckpt" if ckpt_dir else None
        if path is None or not path.exists():
            print(f"  skip {name} (set LWM_CKPT_DIR to the directory holding it)")
            continue
        out = _forward_dummy(path)
        keys = list(out.keys()) if isinstance(out, dict) else type(out).__name__
        print(f"  {name}: forward OK ({keys})")


if __name__ == "__main__":
    ok = True
    for fn in (test_imports, test_checkpoints_forward):
        print(f"== {fn.__name__}")
        try:
            fn()
            print("   PASS")
        except Exception:
            traceback.print_exc()
            ok = False
            print("   FAIL")
    sys.exit(0 if ok else 1)
