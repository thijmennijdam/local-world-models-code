"""E3: training-only yaw + translation augmentation is consistent.

For random samples with augment_yaw=30 deg, augment_translation=0.05 m, the sampled
scene map x -> Rz x + b is reproduced (same RNG draws) and checked against the
unaugmented sequence:
  - observation: cloud, affordance, EE pose are the mapped originals; actions have
    their translation rotated and their rotation / finger unchanged;
  - part-motion targets (world and centroid frame), applied to the augmented cloud,
    reproduce the augmented future clouds as well as the unaugmented targets do;
  - agent-state targets composed onto the augmented state give the augmented future
    EE poses;
  - actions vs. the sim (pos' = pos + a[:3], R' = R @ dR): the residuals
    p_{t+1} - p_t - a_t and angle(R_t dR_t, R_{t+1}) have the same size before and
    after augmentation;
  - the validation dataset is never augmented.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src python tests/test_augmentation.py
"""

import _common as C

import numpy as np
import torch

from lwm.world_model.targets import decode_affordance_delta, rot6d_to_matrix, sample_scene_transform
from lwm.world_model.evaluation_utils import compose_state_with_delta

H = 6
YAW, TRANS = 30.0, 0.05


def _denorm(x, mean, std):
    return x.double() * torch.from_numpy(std).double() + torch.from_numpy(mean).double()


def _apply(pcd, delta):
    R = rot6d_to_matrix(delta[:, :6])
    out, x = [], pcd
    for h in range(delta.shape[0]):
        x = x @ R[h].T + delta[h, 6:]
        out.append(x)
    return torch.stack(out)


def _rot_angle(Ra, Rb):
    c = ((torch.einsum("...ij,...ij->...", Ra, Rb)) - 1) / 2
    return torch.arccos(c.clamp(-1, 1))


def _check(frame):
    ds = C.build_dataset(affordance_target_frame=frame, augment_yaw=YAW, augment_translation=TRANS)
    ref = C.build_dataset(affordance_target_frame=frame)  # same data, no augmentation
    # Stats are the un-augmented ones, except that x/y stds of translation-like dims are pooled
    pooled = {"action_std": (0, 1), "agent_state_delta_std": (0, 1), "affordance_delta_std": (6, 7)}
    for k in ("action_mean", "action_std", "agent_state_delta_mean", "agent_state_delta_std",
              "affordance_delta_mean", "affordance_delta_std"):
        a, r = getattr(ds, k), getattr(ref, k)
        if k not in pooled:
            assert np.array_equal(a, r), f"stats must not depend on augmentation: {k}"
            continue
        ix, iy = pooled[k]
        rest = [d for d in range(len(a)) if d not in (ix, iy)]
        assert np.array_equal(a[rest], r[rest]), k
        real = [s for s in (r[ix], r[iy]) if s < 0.5]
        expect = np.sqrt(np.mean(np.square(real)))
        assert np.allclose([a[ix], a[iy]], expect, rtol=1e-6), (k, a[[ix, iy]], expect)
        print(f"   [{frame}] {k} ({ix},{iy}): {r[ix]:.2e},{r[iy]:.2e} -> pooled {a[ix]:.2e}")
    rng = np.random.RandomState(3)
    errs = {k: [] for k in ("obs", "pcd_aug", "pcd_ref", "state", "act_pos", "act_rot", "angles")}
    for i in rng.choice(len(ds), 20, replace=False):
        i = int(i)
        seed = 1000 + i
        np.random.seed(seed)
        s = ds[i]
        np.random.seed(seed)
        seq_idx, t = ds.pair_indices[i]
        seq = ds.sampler.sample_sequence(seq_idx)
        Rz, b = sample_scene_transform(np.random, YAW, TRANS, seq["static_affordance"][t])
        errs["angles"].append(np.degrees(np.arctan2(Rz[1, 0], Rz[0, 0])))
        Rz_t, b_t = torch.from_numpy(Rz), torch.from_numpy(b)
        P = torch.from_numpy(seq["part_point_cloud"]).double()          # [T, N, 3]
        S = torch.from_numpy(seq["state"]).double()                      # [T, 10]
        A = torch.from_numpy(seq["action"]).double()                     # [T, 10]
        P_aug = P @ Rz_t.T + b_t
        S_pos_aug = S[:, :3] @ Rz_t.T + b_t
        R_ee = rot6d_to_matrix(S[:, 3:9])
        R_ee_aug = Rz_t @ R_ee

        obs = s["observation"]
        e = max((obs["pointcloud"].double() - P_aug[t]).abs().max().item(),
                (obs["static_affordance"].double() - (torch.from_numpy(seq["static_affordance"][t]).double() @ Rz_t.T + b_t)).abs().max().item(),
                (obs["agent_state"][:3].double() - S_pos_aug[t]).abs().max().item(),
                (rot6d_to_matrix(obs["agent_state"][3:9].double()) - R_ee_aug[t]).abs().max().item(),
                (obs["agent_state"][9].double() - S[t, 9]).abs().item())
        act = _denorm(obs["action_horizon"], ds.action_mean, ds.action_std)
        e = max(e, (act[:, :3] - A[t:t + H, :3] @ Rz_t.T).abs().max().item(),
                (act[:, 3:] - A[t:t + H, 3:]).abs().max().item())
        errs["obs"].append(e)

        # part motion: augmented targets on augmented cloud vs. reference targets on reference cloud
        tgt = _denorm(s["target"]["affordance_delta"], ds.affordance_delta_mean, ds.affordance_delta_std)
        tgt_ref = _denorm(ref[i]["target"]["affordance_delta"], ref.affordance_delta_mean, ref.affordance_delta_std)
        if frame == "centroid":
            tgt = decode_affordance_delta(tgt[None], obs["pointcloud"].double().mean(0)[None], frame)[0]
            tgt_ref = decode_affordance_delta(tgt_ref[None], P[t].mean(0)[None], frame)[0]
        pred_aug = _apply(obs["pointcloud"].double(), tgt)
        pred_ref = _apply(P[t], tgt_ref)
        errs["pcd_aug"].append((pred_aug - P_aug[t + 1:t + 1 + H]).norm(dim=-1).max().item())
        errs["pcd_ref"].append((pred_ref - P[t + 1:t + 1 + H]).norm(dim=-1).max().item())

        # agent-state targets -> future EE poses
        ad = _denorm(s["target"]["agent_state_delta"], ds.agent_state_delta_mean, ds.agent_state_delta_std)
        cur = obs["agent_state"].double()
        e = 0.0
        for h in range(H):
            nxt = compose_state_with_delta(cur, ad[h])
            e = max(e, (nxt[:3] - S_pos_aug[t + 1 + h]).abs().max().item(),
                    (rot6d_to_matrix(nxt[3:9]) - R_ee_aug[t + 1 + h]).abs().max().item(),
                    (nxt[9] - S[t + 1 + h, 9]).abs().item())
        errs["state"].append(e)

        # actions vs. how the sim applies them
        r_ref = S[t + 1:t + 1 + H, :3] - S[t:t + H, :3] - A[t:t + H, :3]
        r_aug = S_pos_aug[t + 1:t + 1 + H] - S_pos_aug[t:t + H] - act[:, :3]
        errs["act_pos"].append(((r_aug.norm(dim=-1) - r_ref.norm(dim=-1)).abs().max().item(),
                                r_ref.norm(dim=-1).mean().item(), A[t:t + H, :3].norm(dim=-1).mean().item()))
        dR = rot6d_to_matrix(A[t:t + H, 3:9])
        ang_ref = _rot_angle(R_ee[t:t + H] @ dR, R_ee[t + 1:t + 1 + H])
        ang_aug = _rot_angle(R_ee_aug[t:t + H] @ rot6d_to_matrix(act[:, 3:9]), R_ee_aug[t + 1:t + 1 + H])
        ang_worldframe = _rot_angle(dR @ R_ee[t:t + H], R_ee[t + 1:t + 1 + H])  # alternative: dR in world frame
        errs["act_rot"].append(((ang_aug - ang_ref).abs().max().item(), ang_ref.mean().item(),
                                _rot_angle(torch.eye(3, dtype=torch.float64).expand_as(dR), dR).mean().item(),
                                ang_worldframe.mean().item()))

    act_pos = np.array(errs["act_pos"])
    act_rot = np.array(errs["act_rot"])
    print(f"   [{frame}] sampled yaw range [{min(errs['angles']):.1f}, {max(errs['angles']):.1f}] deg")
    print(f"   [{frame}] observation vs. mapped original: max err {max(errs['obs']):.2e}")
    print(f"   [{frame}] part-motion targets -> future clouds, max point err: augmented {max(errs['pcd_aug']) * 1e3:.4f} mm, "
          f"unaugmented {max(errs['pcd_ref']) * 1e3:.4f} mm")
    print(f"   [{frame}] agent-state targets -> augmented future EE poses: max err {max(errs['state']):.2e}")
    print(f"   [{frame}] action translation: |p_t+1 - p_t - a_t| unchanged to {act_pos[:, 0].max():.2e} m "
          f"(mean residual {act_pos[:, 1].mean() * 1e3:.3f} mm vs. mean |a| {act_pos[:, 2].mean() * 1e3:.3f} mm)")
    print(f"   [{frame}] action rotation: angle(R_t dR_t, R_t+1) unchanged to {act_rot[:, 0].max():.2e} rad "
          f"(mean residual {np.degrees(act_rot[:, 1].mean()):.3f} deg vs. mean |dR| {np.degrees(act_rot[:, 2].mean()):.3f} deg; "
          f"world-frame hypothesis R_t+1 = dR R_t: {np.degrees(act_rot[:, 3].mean()):.3f} deg)")
    assert max(errs["obs"]) < 1e-5
    assert max(errs["pcd_aug"]) < max(errs["pcd_ref"]) + 1e-5
    assert max(errs["state"]) < 1e-5
    assert act_pos[:, 0].max() < 1e-5 and act_rot[:, 0].max() < 1e-4
    assert act_rot[:, 1].mean() < act_rot[:, 3].mean()  # data agrees with the EE-frame action rotation

    # validation is never augmented
    val = ds.get_validation_dataset()
    assert val.augment_yaw == 0 and val.augment_translation == 0
    np.random.seed(0)
    a = val[0]
    np.random.seed(1)
    b = val[0]
    assert all(torch.equal(a["observation"][k], b["observation"][k]) for k in a["observation"])


def test_e3_augmentation_centroid():
    _check("centroid")


def test_e3_augmentation_world():
    _check("world")


def test_e3_normalized_xy_spread_balanced():
    """With yaw augmentation, normalised part-translation y targets are not inflated beyond x.

    Parts move mostly along x. Without pooling, a yaw turns that x-motion into y, whose
    un-augmented std is several times smaller, so normalised y targets would be several
    times wider than x. With pooling, y stays at most as wide as x (a +-30 deg yaw only
    partly mixes x into y).
    """
    ds = C.build_dataset(affordance_target_frame="centroid", augment_yaw=YAW, augment_translation=TRANS)
    np.random.seed(0)
    idx = np.random.choice(len(ds), min(400, len(ds)), replace=False)
    t = np.concatenate([ds[int(i)]["target"]["affordance_delta"].numpy()[:, 6:8] for i in idx])
    sx, sy = t[:, 0].std(), t[:, 1].std()
    print(f"   normalised centroid-shift std under augmentation: x {sx:.2f}, y {sy:.2f}")
    assert 0.1 < sy / sx < 1.2, (sx, sy)


if __name__ == "__main__":
    C.run_tests([test_e3_augmentation_centroid, test_e3_augmentation_world,
                 test_e3_normalized_xy_spread_balanced])
