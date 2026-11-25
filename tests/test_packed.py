"""Packed datasets (scripts/pack_dataset.py) are equivalent to the pickle episode dirs.

Packs the expert and interaction data into a temp dir (or uses LWM_PACKED_EXPERT_DIR /
LWM_PACKED_INTERACTION_DIR), then checks for LWMDataset built from packed vs pickle dirs:
  - identical episode lists and order, step counts, split masks and train/val split,
    episode_splits.json content (by episode key);
  - normalisation stats equal up to float32 rounding (the pack stores float32);
  - __getitem__ equal up to float32 rounding for 50 random samples;
and for the evaluation side (analysis.episode_discovery) with LWM-100M's
episode_splits.json: identical test-episode selection and identical episode data.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/test_packed.py
"""

import _common as C

import atexit
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

_cache = {}
FILTERS = dict(
    include_object_ids=["45526", "35059", "46092", "45166", "44853", "44817", "45092", "45132"],
    include_object_ids_new_config=["45526", "35059", "46092", "45166", "44853", "44817", "45092", "45132"],
    allowed_goal_percentages_door=[0.04, 0.08, 0.16, 0.20, 0.28, 0.32, 0.40],
    allowed_goal_percentages_drawer=[0.08, 0.16, 0.32, 0.40, 0.56, 0.64, 0.80],
    load_unseen_val=True,
)


def _packed_dirs():
    if "dirs" not in _cache:
        e, i = os.environ.get("LWM_PACKED_EXPERT_DIR"), os.environ.get("LWM_PACKED_INTERACTION_DIR")
        if not (e and i):
            tmp = tempfile.mkdtemp(prefix="lwm_packed_")
            atexit.register(shutil.rmtree, tmp, ignore_errors=True)
            e, i = os.path.join(tmp, "expert"), os.path.join(tmp, "interaction")
            env = dict(os.environ)
            env.pop("TMPDIR", None)  # multiprocessing sockets break with a long TMPDIR
            for src, out in ((C.EXPERT, e), (C.INTERACTION, i)):
                subprocess.run([sys.executable, str(C.REPO / "scripts" / "pack_dataset.py"),
                                "--src", src, "--out", out, "--workers", "4"], check=True, env=env,
                               capture_output=True)
        _cache["dirs"] = (e, i)
    return _cache["dirs"]


def _pair(**kw):
    e, i = _packed_dirs()
    pk = C.build_dataset(expert_dataset_path=e, rollout_dataset_path=i, **kw)
    pl = C.build_dataset(**kw)
    return pl, pk


def _rel(path, root):
    return os.path.relpath(path, root)


def _compare(pl, pk, label):
    from lwm.world_model.analysis.episode_discovery import episode_key
    e, i = _packed_dirs()
    keys_pl = [episode_key(p) for p in pl.replay_buffer.all_path_list]
    keys_pk = [episode_key(p) for p in pk.replay_buffer.all_path_list]
    assert keys_pl == keys_pk, "episode order differs"
    rel_pl = [_rel(p, C.EXPERT) if p.startswith(C.EXPERT) else _rel(p, C.INTERACTION) for p in pl.replay_buffer.all_path_list]
    rel_pk = [_rel(p, e) if p.startswith(e) else _rel(p, i) for p in pk.replay_buffer.all_path_list]
    assert rel_pl == rel_pk
    assert np.array_equal(pl.replay_buffer.episode_lengths, pk.replay_buffer.episode_lengths)
    for m in ("train_mask", "val_mask", "expert_episode_mask", "new_config_episode_mask",
              "replay_episode_mask", "unseen_episode_mask"):
        assert np.array_equal(getattr(pl, m), getattr(pk, m)), m
    assert len(pl) == len(pk)
    sp_pl, sp_pk = pl.get_episode_splits_metadata(), pk.get_episode_splits_metadata()
    for k in ("train_episodes", "val_episodes"):
        assert [episode_key(x["episode_path"]) for x in sp_pl[k]] == [episode_key(x["episode_path"]) for x in sp_pk[k]]
    stat_diff = 0.0
    for k in ("action_mean", "action_std", "agent_state_delta_mean", "agent_state_delta_std",
              "affordance_delta_mean", "affordance_delta_std"):
        a, b = getattr(pl, k), getattr(pk, k)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-7, err_msg=k)
        stat_diff = max(stat_diff, float(np.abs(a.astype(np.float64) - b).max()))
    rng = np.random.RandomState(0)
    item_diff = 0.0
    for idx in rng.choice(len(pl), 50, replace=False):
        a, b = pl[int(idx)], pk[int(idx)]
        for part in ("observation", "target"):
            for k in a[part]:
                d = float((a[part][k].double() - b[part][k].double()).abs().max())
                item_diff = max(item_diff, d)
                assert d < 1e-4, (label, idx, part, k, d)
    n_unseen = int(pl.unseen_episode_mask.sum())
    print(f"   [{label}] {len(keys_pl)} episodes (train {int(pl.train_mask.sum())}, val {int(pl.val_mask.sum())}, "
          f"unseen {n_unseen}), {len(pl)} pairs identical in order/splits; "
          f"norm stats max abs diff {stat_diff:.2e}; 50 samples max abs diff {item_diff:.2e}")


def test_dataset_equivalence_filtered_budget():
    pl, pk = _pair(**FILTERS)
    _compare(pl, pk, "filters + budgets 1500")


def test_dataset_equivalence_all_episodes():
    splits = {"expert": {"enabled": True, "step_budget": None},
              "new_config": {"enabled": True, "step_budget": None},
              "replay": {"enabled": True, "step_budget": None}}
    pl, pk = _pair(data_splits=splits)
    _compare(pl, pk, "all episodes")


def test_pickle_dtypes():
    """Which dtypes the pickles hold (explains the float32-rounding differences)."""
    import pickle
    from lwm.world_model.dataset import LWMDataset
    p = LWMDataset._collect_episode_paths(None, C.EXPERT, True)[0]
    with open(os.path.join(p, "0.pkl"), "rb") as f:
        d = pickle.load(f)
    print("   pickle dtypes: " + ", ".join(f"{k}={np.asarray(d[k]).dtype}" for k in
                                         ("state", "action", "part_point_cloud", "static_affordance",
                                          "pcd_rigid_body_delta_9dof")))


def test_eval_selection_equivalence():
    from lwm.world_model.analysis.checkpoint import load_episode_splits
    from lwm.world_model.analysis.episode_discovery import (
        find_test_episodes, load_eval_replay_buffer, episode_key)
    e, i = _packed_dirs()
    splits = load_episode_splits(str(C.CKPT_DIR / "LWM-100M_d1024_L8" / "best.ckpt"))
    t_pl, g_pl = find_test_episodes(splits, C.EXPERT, C.INTERACTION)
    t_pk, g_pk = find_test_episodes(splits, e, i)
    for s in t_pl:
        assert [episode_key(p) for p in t_pl[s]] == [episode_key(p) for p in t_pk[s]], s
    counts = {}
    for src in g_pl:
        for cat in g_pl[src]:
            assert [episode_key(p) for p in g_pl[src][cat]] == [episode_key(p) for p in g_pk[src][cat]]
            counts[f"{src}/{cat}"] = len(g_pl[src][cat])
    print(f"   test episodes per split: { {s: len(v) for s, v in t_pl.items()} }; generalization: {counts}")
    rb_pl, si_pl, gi_pl = load_eval_replay_buffer(t_pl, g_pl)
    rb_pk, si_pk, gi_pk = load_eval_replay_buffer(t_pk, g_pk)
    assert si_pl == si_pk and gi_pl == gi_pk
    assert np.array_equal(rb_pl.episode_lengths, rb_pk.episode_lengths)
    from lwm.world_model.evaluation_utils import load_episode_data
    diff = 0.0
    for ep in range(0, len(rb_pl.all_path_list), max(1, len(rb_pl.all_path_list) // 10)):
        for a, b in zip(load_episode_data(rb_pl, ep, "cpu")[:4], load_episode_data(rb_pk, ep, "cpu")[:4]):
            diff = max(diff, float((a.double() - b.double()).abs().max()))
    assert diff < 1e-5
    print(f"   eval replay buffers: {len(rb_pl.all_path_list)} episodes, data max abs diff {diff:.2e}")


def test_point_cloud_not_packed():
    from lwm.world_model.replay_buffer import ReplayBuffer
    from lwm.world_model import packed
    e, _ = _packed_dirs()
    p = packed.episode_paths(e)[0]
    rb = ReplayBuffer.copy_from_multiple_path([p], keys=["point_cloud"])
    try:
        rb._get_data_single_file(p, 0, 1)
    except KeyError as err:
        assert "not packed" in str(err)
        print("   requesting point_cloud raises: " + str(err)[:110] + "...")
    else:
        raise AssertionError("point_cloud must raise for packed data")


if __name__ == "__main__":
    C.run_tests([test_pickle_dtypes, test_dataset_equivalence_filtered_budget, test_dataset_equivalence_all_episodes,
                 test_eval_selection_equivalence, test_point_cloud_not_packed])
