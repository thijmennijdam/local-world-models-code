"""E1 (affordance_target_frame=centroid) and E2 (input_frame=handle) on real episodes.

E1:
  - encode(world -> centroid) then decode gives the world form back (max abs err < 1e-5).
  - Decoding the ground-truth centroid targets and applying them step by step to the
    real part cloud reproduces pcd_{t+h+1} as well as the world-form targets do.
  - The dataset in centroid mode returns targets that decode to the world-mode targets.
E2:
  - The model in handle mode is invariant to translating the whole scene (point cloud,
    EE position and affordance point shifted by the same vector) when the absolute
    affordance token is held fixed; a world-mode model is not.
  - input_frame=handle with affordance_target_frame=world raises.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/test_frames.py
"""

import _common as C

import numpy as np
import torch

from lwm.world_model.targets import (
    decode_affordance_delta, encode_affordance_delta, rot6d_to_matrix, check_frames,
)

H = 6
_cache = {}


def _ds(frame):
    if frame not in _cache:
        _cache[frame] = C.build_dataset(affordance_target_frame=frame)
    return _cache[frame]


def _episodes(ds, n=8):
    rb = ds.replay_buffer
    out = []
    for ep in range(min(n, len(rb.all_path_list))):
        L = int(rb.episode_lengths[ep])
        d = rb._get_data_single_file(rb.all_path_list[ep], 0, L)
        out.append((d["part_point_cloud"].astype(np.float64),
                    d["pcd_rigid_body_delta_9dof"].astype(np.float64),
                    d["static_affordance"].astype(np.float64)))
    return out


def _apply(pcd, delta):
    """Apply world-form deltas [H, 9] step by step to pcd [N, 3] -> [H, N, 3]."""
    R = rot6d_to_matrix(torch.from_numpy(delta[:, :6])).numpy()
    out, x = [], pcd
    for h in range(delta.shape[0]):
        x = x @ R[h].T + delta[h, 6:]
        out.append(x)
    return np.stack(out)


def test_e1_round_trip_and_reconstruction():
    eps = _episodes(_ds("world"))
    rt_err, rt_chain_err = 0.0, 0.0
    rec_world, rec_centroid, rec_diff = [], [], []
    moving = []
    for pcds, deltas, affs in eps:
        T = pcds.shape[0]
        for t in range(0, T - H - 1, 3):
            w = torch.from_numpy(deltas[t:t + H])[None]                 # [1, H, 9]
            cents = torch.from_numpy(pcds[t:t + H].mean(axis=1))[None]  # [1, H, 3] measured
            c0 = cents[:, 0]
            # dataset-style encode (measured centroids) -> decode from c0
            enc = encode_affordance_delta(w, cents, "centroid")
            dec = decode_affordance_delta(enc, c0, "centroid")
            rt_err = max(rt_err, (dec - w).abs().max().item())
            # chain-style encode -> decode (exact inverse by construction)
            dec2 = decode_affordance_delta(encode_affordance_delta(w, c0, "centroid"), c0, "centroid")
            rt_chain_err = max(rt_chain_err, (dec2 - w).abs().max().item())
            # reconstruction of the real future clouds
            gt = pcds[t + 1:t + 1 + H]
            pw = _apply(pcds[t], w[0].numpy())
            pc = _apply(pcds[t], dec[0].numpy())
            ew = np.linalg.norm(pw - gt, axis=-1).mean(axis=-1)          # [H]
            ec = np.linalg.norm(pc - gt, axis=-1).mean(axis=-1)
            rec_world.append(ew)
            rec_centroid.append(ec)
            rec_diff.append(np.abs(pw - pc).max())
            moving.append(np.linalg.norm(affs[t + H] - affs[t]) > 1e-3)
    rec_world, rec_centroid = np.array(rec_world), np.array(rec_centroid)
    moving = np.array(moving)
    print(f"   {len(rec_world)} windows ({moving.sum()} with handle motion > 1 mm)")
    print(f"   round trip (measured centroids) max abs err {rt_err:.2e}; chain encode {rt_chain_err:.2e}")
    print(f"   cloud reconstruction, mean point err per step [mm] world:    {np.round(rec_world.mean(0) * 1e3, 4)}")
    print(f"   cloud reconstruction, mean point err per step [mm] centroid: {np.round(rec_centroid.mean(0) * 1e3, 4)}")
    print(f"   max |world-applied - centroid-applied| point diff: {max(rec_diff):.2e} m")
    assert rt_err < 1e-5, rt_err
    assert rt_chain_err < 1e-9, rt_chain_err
    assert max(rec_diff) < 1e-5
    assert np.abs(rec_world.mean(0) - rec_centroid.mean(0)).max() < 1e-6


def test_e1_dataset_targets_decode_to_world():
    dw, dc = _ds("world"), _ds("centroid")
    assert len(dw) == len(dc)
    # stats: rotation dims identical, translation dims differ (centroid form)
    assert np.array_equal(dw.affordance_delta_mean[:6], dc.affordance_delta_mean[:6])
    print(f"   aff translation std world {dw.affordance_delta_std[6:]}, centroid {dc.affordance_delta_std[6:]}")
    print(f"   aff translation mean world {dw.affordance_delta_mean[6:]}, centroid {dc.affordance_delta_mean[6:]}")
    rng = np.random.RandomState(1)
    err = 0.0
    for i in rng.choice(len(dw), 20, replace=False):
        sw, sc = dw[int(i)], dc[int(i)]
        assert torch.equal(sw["observation"]["pointcloud"], sc["observation"]["pointcloud"])
        tw = sw["target"]["affordance_delta"] * torch.from_numpy(dw.affordance_delta_std) + torch.from_numpy(dw.affordance_delta_mean)
        tc = sc["target"]["affordance_delta"] * torch.from_numpy(dc.affordance_delta_std) + torch.from_numpy(dc.affordance_delta_mean)
        c0 = sc["observation"]["pointcloud"].double().mean(0)[None]
        dec = decode_affordance_delta(tc.double()[None], c0, "centroid")[0]
        err = max(err, (dec - tw.double()).abs().max().item())
        assert torch.equal(sw["target"]["agent_state_delta"], sc["target"]["agent_state_delta"])
    print(f"   dataset centroid targets decode to world targets: max abs err {err:.2e}")
    assert err < 1e-5


def _tiny_model(input_frame, frame):
    from lwm.world_model.model import LWMv3
    torch.manual_seed(0)
    m = LWMv3(agent_state_dim=10, action_dim=10, predict_agent_state=True, n_future_steps=H,
              affordance_output_dim=9, d_model=64, n_heads=4, n_layers=2, use_pointnet2=False,
              input_frame=input_frame, affordance_target_frame=frame)
    return m.eval()


def _batch(ds, n=4):
    items = [ds[i] for i in range(0, len(ds), len(ds) // n)][:n]
    return {k: torch.stack([it["observation"][k] for it in items]) for k in items[0]["observation"]}


def _shift(obs, v):
    o = dict(obs)
    o["pointcloud"] = obs["pointcloud"] + v[:, None, :]
    st = obs["agent_state"].clone()
    st[:, :3] += v
    o["agent_state"] = st
    o["static_affordance"] = obs["static_affordance"] + v
    return o


class _FixedAffToken(torch.nn.Module):
    """Affordance-token encoder that ignores its input (holds the absolute token fixed)."""

    def __init__(self, enc, aff):
        super().__init__()
        self.enc, self.aff = enc, aff

    def forward(self, _):
        return self.enc(self.aff)


def test_e2_translation_invariance():
    obs = _batch(_ds("centroid"))
    v = torch.tensor([[0.3, -0.2, 0.1]]).expand(obs["pointcloud"].shape[0], 3)
    results = {}
    for input_frame in ("handle", "world"):
        m = _tiny_model(input_frame, "centroid")
        m.affordance_encoder = _FixedAffToken(m.affordance_encoder, obs["static_affordance"])
        with torch.no_grad():
            a = m(obs)
            b = m(_shift(obs, v))
        results[input_frame] = max((a[k] - b[k]).abs().max().item() for k in ("affordance_delta", "agent_state_delta"))
    print(f"   max output change under a {v[0].tolist()} m scene shift (affordance token fixed): "
          f"handle {results['handle']:.2e}, world {results['world']:.2e}")
    assert results["handle"] < 1e-5
    assert results["world"] > 1e-3

    # Without holding the token fixed, the absolute affordance still enters (by design)
    m = _tiny_model("handle", "centroid")
    with torch.no_grad():
        d = (m(obs)["affordance_delta"] - m(_shift(obs, v))["affordance_delta"]).abs().max().item()
    print(f"   handle mode with the absolute affordance token free: change {d:.2e} (expected > 0)")

    # Multi-obs inputs ([B, T, ...]) go through the same shift
    m = _tiny_model("handle", "centroid")
    m2 = {k: (val.unsqueeze(1).repeat(1, 2, *([1] * (val.dim() - 1))) if k != "action_horizon" else val)
          for k, val in obs.items()}
    from lwm.world_model.targets import shift_observation_to_handle_frame
    s = shift_observation_to_handle_frame(m2)
    assert torch.allclose(s["pointcloud"][:, 1], obs["pointcloud"] - obs["static_affordance"][:, None])
    assert torch.allclose(s["agent_state"][:, 0, :3], obs["agent_state"][:, :3] - obs["static_affordance"])
    assert torch.equal(s["static_affordance"], m2["static_affordance"])

    try:
        check_frames("world", "handle")
    except ValueError:
        pass
    else:
        raise AssertionError("input_frame=handle with world targets must raise")


def test_e2b_handle_rel_fully_invariant():
    obs = _batch(_ds("centroid"))
    v = torch.tensor([[0.3, -0.2, 0.1]]).expand(obs["pointcloud"].shape[0], 3)
    m = _tiny_model("handle_rel", "centroid")
    with torch.no_grad():
        a, b = m(obs), m(_shift(obs, v))
    d = max((a[k] - b[k]).abs().max().item() for k in ("affordance_delta", "agent_state_delta"))
    print(f"   handle_rel: max output change under a {v[0].tolist()} m scene shift, token free: {d:.2e}")
    assert d < 1e-5

    # Multi-obs: the affordance token is relative to the latest step (zero there)
    from lwm.world_model.targets import shift_observation_to_handle_frame
    m2 = {k: (val.unsqueeze(1).repeat(1, 2, *([1] * (val.dim() - 1))) if k != "action_horizon" else val)
          for k, val in obs.items()}
    s = shift_observation_to_handle_frame(m2, shift_affordance=True)
    assert torch.allclose(s["static_affordance"][:, -1], torch.zeros_like(obs["static_affordance"]))
    assert torch.allclose(s["pointcloud"][:, 1], obs["pointcloud"] - obs["static_affordance"][:, None])

    try:
        check_frames("world", "handle_rel")
    except ValueError:
        pass
    else:
        raise AssertionError("input_frame=handle_rel with world targets must raise")


if __name__ == "__main__":
    C.run_tests([
        test_e1_round_trip_and_reconstruction,
        test_e1_dataset_targets_decode_to_world,
        test_e2_translation_invariance,
        test_e2b_handle_rel_fully_invariant,
    ])
