"""E7: the part point cloud as K local patch tokens (LWMv3 pcd_tokens > 1).

  - pcd_tokens=1 gives the model of branch main (parameters, state_dict keys, outputs).
  - Forward + backward with K=32 on a real batch; output shapes; gradients reach the
    patch encoder and the centre embedding.
  - The output is invariant to the order of the input points.
  - With input_frame=handle_rel (affordance_target_frame=centroid) the output is
    invariant to translating the whole scene.
  - Parameter count for K=1 vs K=32 at d_model=512, n_layers=8.
  - Save/load round trip through analysis.checkpoint.load_checkpoint (with pcd_tokens in
    the config, and inferred from the state_dict) and the MPC predictor.
  - Two optimizer steps with K=32; CPU time per training step for K=1 vs K=32.
  - pcd_tokens > 1 with n_obs_steps > 1 raises.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/test_patch_tokens.py
"""

import _common as C

import os
import tempfile
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from lwm.world_model.model import LWMv3, group_points

H = 6
_cache = {}


def _ds(frame="world"):
    if frame not in _cache:
        _cache[frame] = C.build_dataset(affordance_target_frame=frame)
    return _cache[frame]


def _batch(ds, n=4, with_target=False):
    items = [ds[i] for i in range(0, len(ds), len(ds) // n)][:n]
    obs = {k: torch.stack([it["observation"][k] for it in items]) for k in items[0]["observation"]}
    if not with_target:
        return obs
    tgt = {k: torch.stack([it["target"][k] for it in items]) for k in items[0]["target"]}
    return obs, tgt


def _model_kwargs(**kw):
    out = dict(agent_state_dim=10, action_dim=10, predict_agent_state=True, n_future_steps=H,
               affordance_output_dim=9, d_model=64, n_heads=4, n_layers=2, use_pointnet2=False)
    out.update(kw)
    return out


def _model(**kw):
    torch.manual_seed(0)
    return LWMv3(**_model_kwargs(**kw)).eval()


def test_k1_identical_to_main():
    """pcd_tokens=1 (default) builds the model of branch main: same parameters, same outputs."""
    main_model = C.load_main_module("src/lwm/world_model/model.py", "lwm_main_model")
    obs = _batch(_ds())
    for kw in (dict(use_pointnet2=False), dict(use_pointnet2=True), dict(use_pointnet2=False, n_obs_steps=2)):
        torch.manual_seed(0)
        a = main_model.LWMv3(**_model_kwargs(**kw)).eval()
        torch.manual_seed(0)
        b = LWMv3(**_model_kwargs(**kw)).eval()
        sa, sb = a.state_dict(), b.state_dict()
        assert list(sa) == list(sb), kw
        assert all(torch.equal(sa[k], sb[k]) for k in sa), kw
        o = obs if kw.get("n_obs_steps", 1) == 1 else {
            k: (v.unsqueeze(1).repeat(1, 2, *([1] * (v.dim() - 1))) if k != "action_horizon" else v)
            for k, v in obs.items()}
        with torch.no_grad():
            oa, ob = a(o), b(o)
        assert all(torch.equal(oa[k], ob[k]) for k in oa), kw
    print("   K=1: state_dict keys/values and outputs identical to main (PointNet, PointNet++, n_obs_steps=2)")


def test_forward_backward_shapes():
    obs = _batch(_ds())
    B, N = obs["pointcloud"].shape[:2]
    centers, patches = group_points(obs["pointcloud"], 32, 32)
    assert centers.shape == (B, 32, 3) and patches.shape == (B, 32, 32, 3)
    # every patch contains its centre (at the origin) and the centres are distinct
    assert torch.allclose(patches.norm(dim=-1).min(dim=-1).values, torch.zeros(B, 32))
    assert all(torch.unique(centers[b], dim=0).shape[0] == 32 for b in range(B))

    m = _model(pcd_tokens=32, pcd_group_size=32).train()
    assert m.n_obs_tokens == 34 and m.attn_mask.shape == (34 + H, 34 + H)
    # obs tokens attend to each other only; action i sees all obs + actions 0..i
    assert not m.attn_mask[:34, :34].any() and m.attn_mask[:34, 34:].all()
    assert not m.attn_mask[34:, :34].any()
    assert torch.equal(m.attn_mask[34:, 34:], ~torch.tril(torch.ones(H, H, dtype=torch.bool)))
    out = m(obs)
    assert out["affordance_delta"].shape == (B, H, 9) and out["agent_state_delta"].shape == (B, H, 10)
    (out["affordance_delta"].square().mean() + out["agent_state_delta"].square().mean()).backward()
    for name in ("pcd_encoder.mlp.0.weight", "pcd_proj.weight", "pcd_center_embed.0.weight", "obs_type_embed"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name
    print(f"   K=32 x G=32 on {B} x {N} points: affordance_delta {tuple(out['affordance_delta'].shape)}, "
          f"agent_state_delta {tuple(out['agent_state_delta'].shape)}; gradients reach patch encoder + centre embedding")


def test_point_order_invariance():
    obs = _batch(_ds())
    m = _model(pcd_tokens=32, pcd_group_size=32)
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(obs["pointcloud"].shape[1], generator=g)
    obs_p = dict(obs, pointcloud=obs["pointcloud"][:, perm])
    with torch.no_grad():
        a, b = m(obs), m(obs_p)
    d = max((a[k] - b[k]).abs().max().item() for k in ("affordance_delta", "agent_state_delta"))
    print(f"   max output change under a random point permutation: {d:.2e}")
    assert d < 1e-5


def test_handle_rel_translation_invariance():
    obs = _batch(_ds("centroid"))
    v = torch.tensor([[0.3, -0.2, 0.1]]).expand(obs["pointcloud"].shape[0], 3)
    s = dict(obs)
    s["pointcloud"] = obs["pointcloud"] + v[:, None, :]
    st = obs["agent_state"].clone()
    st[:, :3] += v
    s["agent_state"] = st
    s["static_affordance"] = obs["static_affordance"] + v
    m = _model(pcd_tokens=32, pcd_group_size=32, input_frame="handle_rel", affordance_target_frame="centroid")
    with torch.no_grad():
        a, b = m(obs), m(s)
    d = max((a[k] - b[k]).abs().max().item() for k in ("affordance_delta", "agent_state_delta"))
    print(f"   handle_rel, K=32: max output change under a {v[0].tolist()} m scene shift: {d:.2e}")
    assert d < 1e-5


def test_param_counts():
    counts = {}
    for k in (1, 32):
        m = LWMv3(**_model_kwargs(d_model=512, n_layers=8, pcd_tokens=k))
        counts[k] = sum(p.numel() for p in m.parameters())
    print(f"   params at d_model=512, n_layers=8: K=1 {counts[1]:,} ({counts[1] / 1e6:.2f}M), "
          f"K=32 {counts[32]:,} ({counts[32] / 1e6:.2f}M), +{counts[32] - counts[1]:,}")
    assert 0 < counts[32] - counts[1] < 100_000


def _save_run(run_dir, model, model_cfg, ds):
    os.makedirs(os.path.join(run_dir, ".hydra"))
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    OmegaConf.save(OmegaConf.create({"model": model_cfg}), os.path.join(run_dir, ".hydra", "config.yaml"))
    stats = {k: getattr(ds, k) for k in ("action_mean", "action_std", "action_min", "action_max",
                                          "agent_state_delta_mean", "agent_state_delta_std",
                                          "affordance_delta_mean", "affordance_delta_std")}
    path = os.path.join(run_dir, "checkpoints", "best.ckpt")
    torch.save({"model_state_dict": model.state_dict(), "normalization_stats": stats, "epoch": 0,
                "best_val_loss": 0.0}, path)
    return path


def test_checkpoint_and_mpc_round_trip():
    from lwm.world_model.analysis.checkpoint import load_checkpoint
    from lwm.planning.wm_predictor import WorldModelPredictor
    from lwm.world_model import evaluation_utils as eu

    ds = _ds("centroid")
    kw = _model_kwargs(pcd_tokens=16, pcd_group_size=24, input_frame="handle",
                       affordance_target_frame="centroid")
    m = _model(**{k: v for k, v in kw.items()})
    obs = _batch(ds)
    with torch.no_grad():
        ref = m(obs)

    with tempfile.TemporaryDirectory(prefix="lwm_patch_ckpt_") as d:
        cfg_full = dict(kw, _target_="lwm.world_model.model.LWMv3")
        p_full = _save_run(os.path.join(d, "full"), m, cfg_full, ds)
        # config without pcd_tokens (and a wrong d_model): both inferred from the state_dict
        cfg_min = {k: v for k, v in cfg_full.items() if k != "pcd_tokens"}
        cfg_min["d_model"] = 128
        p_min = _save_run(os.path.join(d, "min"), m, cfg_min, ds)

        for p in (p_full, p_min):
            lm, _, _, _ = load_checkpoint(p, device="cpu")
            assert lm.pcd_tokens == 16 and lm.pcd_group_size == 24 and lm.input_frame == "handle"
            with torch.no_grad():
                out = lm(obs)
            for k in ref:
                assert torch.equal(out[k], ref[k]), k

        ep = int(np.where(ds.val_mask)[0][0])
        states, affs, pcds, acts, _ = eu.load_episode_data(ds.replay_buffer, ep, "cpu")
        t, B = 0, 3
        actions = acts[t:t + 12].unsqueeze(0).repeat(B, 1, 1)
        args = (states[t].repeat(B, 1), affs[t].repeat(B, 1), pcds[t].repeat(B, 1, 1), actions)
        r_full = WorldModelPredictor(p_full, device="cpu").predict_trajectory(*args)
        r_min = WorldModelPredictor(p_min, device="cpu").predict_trajectory(*args)
        assert r_full["predicted_states"].shape == (B, 12, 10) and r_full["predicted_affordances"].shape == (B, 12, 3)
        for k in r_full:
            assert torch.equal(r_full[k], r_min[k]), k
            assert torch.isfinite(r_full[k]).all(), k
        # the predictor's first chunk is the original model's forward on the normalized actions
        wmp = WorldModelPredictor(p_full, device="cpu")
        o = {"agent_state": args[0], "static_affordance": args[1], "pointcloud": args[2],
             "action_horizon": wmp.normalize_actions(actions[:, :H])}
        with torch.no_grad():
            a, b = m(o), wmp.model(o)
        assert all(torch.equal(a[k], b[k]) for k in a)
    print("   load_checkpoint (config / inferred pcd_tokens) and WorldModelPredictor.predict_trajectory "
          "(12 steps) give identical predictions")


def _train_steps(model, obs, tgt, n):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses, times = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        out = model(obs)
        loss = ((out["affordance_delta"] - tgt["affordance_delta"]) ** 2).mean() + \
               ((out["agent_state_delta"] - tgt["agent_state_delta"]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
    return losses, times


def test_train_steps_and_cpu_cost():
    obs, tgt = _batch(_ds(), n=32, with_target=True)
    m = _model(pcd_tokens=32, pcd_group_size=32)
    losses, _ = _train_steps(m, obs, tgt, 2)
    assert all(np.isfinite(losses)), losses
    print(f"   2 AdamW steps with K=32: loss {losses[0]:.4f} -> {losses[1]:.4f}")

    torch.manual_seed(0)
    res = {}
    for k in (1, 32):
        mk = _model(d_model=128, n_layers=3, pcd_tokens=k)
        _train_steps(mk, obs, tgt, 2)                                  # warm-up
        _, times = _train_steps(mk, obs, tgt, 6)
        res[k] = float(np.median(times))
    print(f"   CPU time per training step (batch 32, d_model 128, 3 layers, {torch.get_num_threads()} threads): "
          f"K=1 {res[1] * 1e3:.1f} ms, K=32 {res[32] * 1e3:.1f} ms, ratio {res[32] / res[1]:.2f}x")


def test_multi_obs_rejected():
    try:
        LWMv3(**_model_kwargs(pcd_tokens=32, n_obs_steps=2))
    except ValueError:
        pass
    else:
        raise AssertionError("pcd_tokens > 1 with n_obs_steps > 1 must raise")
    try:
        LWMv3(**_model_kwargs(pcd_tokens=32, use_pointnet2=True))
    except ValueError:
        pass
    else:
        raise AssertionError("pcd_tokens > 1 with use_pointnet2 must raise")


if __name__ == "__main__":
    C.run_tests([
        test_k1_identical_to_main,
        test_forward_backward_shapes,
        test_point_order_invariance,
        test_handle_rel_translation_invariance,
        test_param_counts,
        test_checkpoint_and_mpc_round_trip,
        test_train_steps_and_cpu_cost,
        test_multi_obs_rejected,
    ])
