"""With every experiment switch at its default, the code behaves exactly as the committed ``main``
(the file versions from ``git show main:...``), so edits to the switches cannot change the paper models.

1. LWMDataset.__getitem__ returns identical tensors for 20 random samples, and the
   normalisation stats are identical.
2. An old checkpoint (LWM-100M) gives identical predictions through the windowed eval
   (analysis.core.evaluate_windowed), the single-step eval helper
   (evaluation_utils.predict_next_state_and_affordance) and the MPC predictor
   (WorldModelPredictor.predict_trajectory, two chained 6-step calls).

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/test_defaults_unchanged.py
"""

import _common as C

import numpy as np
import torch

STAT_KEYS = [
    "action_mean", "action_std", "action_min", "action_max",
    "agent_state_delta_mean", "agent_state_delta_std",
    "affordance_delta_mean", "affordance_delta_std",
]

_cache = {}


def _datasets():
    if not _cache:
        main_ds_mod = C.load_main_module("src/lwm/world_model/dataset.py", "lwm_main_dataset")
        _cache["main"] = C.build_dataset(module=main_ds_mod)
        _cache["new"] = C.build_dataset()
    return _cache["main"], _cache["new"]


def _assert_same(a, b, what):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (what, a.keys(), b.keys())
        for k in a:
            _assert_same(a[k], b[k], f"{what}.{k}")
    else:
        assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
        assert torch.equal(a, b), f"{what}: max diff {(a - b).abs().max().item()}"


def test_dataset_getitem_and_stats_identical():
    main_ds, new_ds = _datasets()
    assert len(main_ds) == len(new_ds)
    for k in STAT_KEYS:
        assert np.array_equal(getattr(main_ds, k), getattr(new_ds, k)), k
    rng = np.random.RandomState(0)
    idxs = rng.choice(len(new_ds), size=20, replace=False)
    for i in idxs:
        _assert_same(main_ds[int(i)], new_ds[int(i)], f"sample[{i}]")
    # validation dataset too
    main_val, new_val = main_ds.get_validation_dataset(), new_ds.get_validation_dataset()
    assert len(main_val) == len(new_val)
    for i in range(0, len(new_val), max(1, len(new_val) // 5)):
        _assert_same(main_val[i], new_val[i], f"val[{i}]")
    print(f"   20 train samples + {len(range(0, len(new_val), max(1, len(new_val) // 5)))} val samples identical; stats identical")


def _ckpt(name="LWM-100M_d1024_L8"):
    p = C.CKPT_DIR / name / "best.ckpt"
    assert p.exists(), f"missing {p} (set LWM_CKPT_DIR)"
    return str(p)


def test_old_checkpoint_eval_paths_identical():
    from lwm.world_model.analysis.checkpoint import load_checkpoint
    from lwm.world_model.analysis import core as new_core
    from lwm.world_model import evaluation_utils as new_eu

    main_core = C.load_main_module("src/lwm/world_model/analysis/core.py", "lwm_main_core")
    main_eu = C.load_main_module("src/lwm/world_model/evaluation_utils.py", "lwm_main_eu")
    _, ds = _datasets()
    model, _, stats, cfg = load_checkpoint(_ckpt(), device="cpu")
    assert model.affordance_target_frame == "world" and model.input_frame == "world"

    rb = ds.replay_buffer
    eps = [int(i) for i in np.where(ds.val_mask)[0][:2]] + [int(i) for i in np.where(ds.train_mask)[0][:1]]
    kw = dict(window_steps=6, windows_per_episode=2, compose_affordance_fn=new_eu.compose_affordance_with_9dof_delta,
              seed=0, inference_batch_size=8)
    torch.manual_seed(0)
    r_new = new_core.evaluate_windowed(model, rb, eps, "cpu", stats, **kw)
    torch.manual_seed(0)
    r_main = main_core.evaluate_windowed(model, rb, eps, "cpu", stats, **kw)
    assert len(r_new) == len(r_main) > 0
    for a, b in zip(r_new, r_main):
        assert a["aff_errors"] == b["aff_errors"] and a["agent_errors"] == b["agent_errors"]
    print(f"   windowed eval: {len(r_new)} windows identical "
          f"(mean step-6 aff err {np.mean([r['aff_errors'][-1] for r in r_new]):.3f} cm)")

    # single-step helper (used by validation rollouts)
    norm = [torch.from_numpy(stats[k]).float() for k in
            ("action_mean", "action_std", "agent_state_delta_mean", "agent_state_delta_std",
             "affordance_delta_mean", "affordance_delta_std")]
    states, affs, pcds, acts, _ = new_eu.load_episode_data(rb, eps[0], "cpu")
    obs = new_eu.build_observation(3, states[3], affs[3], states, affs, pcds,
                                   (acts[3:9] - norm[0]) / norm[1], 1, 6)
    with torch.no_grad():
        o_new = new_eu.predict_next_state_and_affordance(model, obs, states[3], affs[3], *norm[2:], True,
                                                         new_eu.compose_state_with_delta,
                                                         compose_affordance_fn=new_eu.compose_affordance_with_9dof_delta)
        o_main = main_eu.predict_next_state_and_affordance(model, obs, states[3], affs[3], *norm[2:], True,
                                                           main_eu.compose_state_with_delta,
                                                           compose_affordance_fn=main_eu.compose_affordance_with_9dof_delta)
    for a, b in zip(o_new, o_main):
        assert torch.equal(a, b)
    print("   predict_next_state_and_affordance identical")


def test_old_checkpoint_mpc_predictor_identical():
    from lwm.planning.wm_predictor import WorldModelPredictor as NewWMP
    main_wmp = C.load_main_module("src/lwm/planning/wm_predictor.py", "lwm_main_wmp")
    from lwm.world_model import evaluation_utils as eu
    _, ds = _datasets()
    ep = int(np.where(ds.val_mask)[0][0])
    states, affs, pcds, acts, _ = eu.load_episode_data(ds.replay_buffer, ep, "cpu")
    t = 2
    B = 3
    actions = acts[t:t + 12].unsqueeze(0).repeat(B, 1, 1)
    actions = actions + 1e-3 * torch.randn_like(actions) * torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0.0])
    args = (states[t].repeat(B, 1), affs[t].repeat(B, 1), pcds[t].repeat(B, 1, 1), actions)
    p_new = NewWMP(_ckpt(), device="cpu")
    p_main = main_wmp.WorldModelPredictor(_ckpt(), device="cpu")
    r_new = p_new.predict_trajectory(*args)
    r_main = p_main.predict_trajectory(*args)
    for k in r_main:
        assert torch.equal(r_new[k], r_main[k]), k
    gt = affs[t + 1:t + 13]
    err = (r_new["predicted_affordances"][0] - gt).norm(dim=-1) * 100
    print(f"   predict_trajectory (12 steps, 2 chained calls) identical; aff err step 6/12: "
          f"{err[5]:.2f}/{err[11]:.2f} cm")


if __name__ == "__main__":
    C.run_tests([
        test_dataset_getitem_and_stats_identical,
        test_old_checkpoint_eval_paths_identical,
        test_old_checkpoint_mpc_predictor_identical,
    ])
