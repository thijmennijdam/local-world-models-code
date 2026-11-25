"""Shared helpers for the experiment-switch tests (tests/test_*.py).

Data and checkpoints default to ``<repo>/data/...`` and can be overridden:
  LWM_EXPERT_DATA_DIR       expert episodes (e.g. the lwm-expert-data subset)
  LWM_INTERACTION_DATA_DIR  interaction episodes (e.g. the lwm-interaction-data subset)
  LWM_CKPT_DIR              directory holding <name>/best.ckpt (+ .hydra/config.yaml)

The tests run on CPU. Norm-stat caches are written to a temporary HOME, never to ~/.cache.
"""

import contextlib
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_DATA = REPO / "data"
EXPERT = os.environ.get("LWM_EXPERT_DATA_DIR", str(_DATA / "lwm-expert-data-subset"))
INTERACTION = os.environ.get("LWM_INTERACTION_DATA_DIR", str(_DATA / "lwm-interaction-data-subset"))
CKPT_DIR = pathlib.Path(os.environ.get("LWM_CKPT_DIR", str(_DATA / "lwm-checkpoints")))

SMALL_SPLITS = {
    "expert": {"enabled": True, "step_budget": 1500},
    "new_config": {"enabled": True, "step_budget": 1500},
    "replay": {"enabled": False, "step_budget": 0},
}


@contextlib.contextmanager
def temp_home():
    """Point HOME (used for ~/.cache/lwm_norm_stats) at a fresh temp dir."""
    old = os.environ.get("HOME")
    with tempfile.TemporaryDirectory(prefix="lwm_test_home_") as d:
        os.environ["HOME"] = d
        try:
            yield d
        finally:
            if old is not None:
                os.environ["HOME"] = old


def dataset_kwargs(**overrides):
    kw = dict(
        expert_dataset_path=EXPERT,
        rollout_dataset_path=INTERACTION,
        horizon=6,
        seed=42,
        val_ratio=0.1,
        n_obs_steps=1,
        data_splits=SMALL_SPLITS,
        use_part_pointcloud=True,
        use_pcd_transform_target=True,
        skip_rotation_normalization=True,
        load_unseen_val=False,
    )
    kw.update(overrides)
    return kw


def build_dataset(module=None, **overrides):
    """Build an LWMDataset (from ``module``, default the current one) with a fresh cache."""
    if module is None:
        from lwm.world_model import dataset as module
    with temp_home():
        return module.LWMDataset(**dataset_kwargs(**overrides))


def load_main_module(rel_path, name):
    """Import the version of ``rel_path`` on git branch ``main`` as module ``name``."""
    src = subprocess.check_output(["git", "-C", str(REPO), "show", f"main:{rel_path}"], text=True)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", prefix=name + "_", delete=False)
    tmp.write(src)
    tmp.close()
    try:
        spec = importlib.util.spec_from_file_location(name, tmp.name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        os.unlink(tmp.name)
    return mod


def run_tests(fns):
    import traceback
    ok = True
    for fn in fns:
        print(f"== {fn.__name__}")
        try:
            fn()
            print("   PASS")
        except Exception:
            traceback.print_exc()
            ok = False
            print("   FAIL")
    sys.exit(0 if ok else 1)
