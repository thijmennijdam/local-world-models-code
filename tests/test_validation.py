"""Held-out validation and the windowed rollout metric logged during training.

1. Every split/unseen validation dataset iterates held-out episodes only
   (val split of episode_splits, or unseen objects), never training episodes.
2. The rollout metric logged at validation epochs (validation_utils.
   compute_val_rollout_metrics on fixed windows) equals what analysis.core.
   evaluate_windowed computes for the same windows, on a released checkpoint (CPU).

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/test_validation.py
"""

import _common as C

import numpy as np

_cache = {}


def _ds():
    if "ds" not in _cache:
        _cache["ds"] = C.build_dataset()
    return _cache["ds"]


def _episodes_of(vds):
    """Episode indices whose sequences a (validation) dataset iterates."""
    ends = vds.replay_buffer.episode_ends[:]
    starts = vds.sampler.indices[:, 0]
    return set(np.searchsorted(ends, starts, side="right").tolist())


def test_split_validation_uses_heldout_episodes():
    ds = _ds()
    train_eps = set(np.where(ds.train_mask)[0].tolist())
    val_eps = set(np.where(ds.val_mask)[0].tolist())
    for name, getter, split in (("expert", ds.get_expert_validation_dataset, ds.expert_episode_mask),
                                ("new_config", ds.get_new_config_validation_dataset, ds.new_config_episode_mask)):
        vds = getter()
        eps = _episodes_of(vds)
        expected = val_eps & set(np.where(split)[0].tolist())
        print(f"   {name}: iterates {len(eps)} episodes, {len(vds)} pairs; held-out {name} episodes {len(expected)}")
        assert eps == expected, (eps, expected)
        assert not (eps & train_eps)
        assert vds.augment_yaw == 0 and vds.augment_translation == 0


def test_rollout_metric_matches_core():
    from lwm.world_model.analysis.checkpoint import load_checkpoint
    from lwm.world_model.analysis import core
    from lwm.world_model.evaluation_utils import compose_affordance_with_9dof_delta
    from lwm.world_model.validation_utils import (
        compute_val_rollout_metrics, select_val_windows, heldout_episode_groups,
    )
    ds = _ds()
    model, _, stats, _ = load_checkpoint(str(C.CKPT_DIR / "LWM-100M_d1024_L8" / "best.ckpt"), device="cpu")

    groups = heldout_episode_groups(ds)
    assert set(groups) <= {"seen_expert", "seen_interaction", "unseen_expert", "unseen_interaction"}
    sel = select_val_windows(ds, n_windows=16, window_steps=6, seed=0)
    assert sel == select_val_windows(ds, n_windows=16, window_steps=6, seed=0)  # fixed
    for name, wins in sel.items():
        assert all(ep in groups[name] for ep, _ in wins)

    # core.evaluate_windowed in its normal (episodes + seed) mode
    eps = groups["seen_expert"][:3]
    ref = core.evaluate_windowed(model, ds.replay_buffer, eps, "cpu", stats, window_steps=6,
                                 windows_per_episode=2, compose_affordance_fn=compose_affordance_with_9dof_delta,
                                 seed=7)
    wins = core.collect_windows(ds.replay_buffer, eps, 6, model.n_future_steps, 2, seed=7)
    logged = compute_val_rollout_metrics(model, ds, {"g": wins}, stats, "cpu", window_steps=6)
    for key, field, step in (("aff_err_step1_cm", "aff_errors", 0), ("aff_err_step6_cm", "aff_errors", 5),
                             ("ee_err_step1_cm", "agent_errors", 0), ("ee_err_step6_cm", "agent_errors", 5),
                             ("finger_err_step6_mm", "finger_errors_mm", 5),
                             ("part_rot_err_step6_deg", "rot_errors_deg", 5)):
        expect = float(np.mean([r[field][step] for r in ref]))
        assert abs(logged[f"rollout_g/{key}"] - expect) < 1e-9, (key, logged[f"rollout_g/{key}"], expect)
    print("   logged metric == core.evaluate_windowed for the same windows: "
          + ", ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in logged.items()))

    # all groups on the fixed subset (what training logs)
    all_m = compute_val_rollout_metrics(model, ds, sel, stats, "cpu", window_steps=6)
    print("   fixed val subset: " + ", ".join(f"{k}={v:.3f}" for k, v in all_m.items() if "step6" in k))
    assert all(np.isfinite(v) for v in all_m.values())


if __name__ == "__main__":
    C.run_tests([test_split_validation_uses_heldout_episodes, test_rollout_metric_matches_core])
