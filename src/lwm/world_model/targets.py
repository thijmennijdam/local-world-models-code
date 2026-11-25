"""Frames of the LWM's 9-D part-motion target and of its inputs.

Part-motion target (``affordance_delta``, 9-D per step h, ``[rot6d(6), translation(3)]``):

- ``world`` (the paper models): ``[rot6d(R_h), t_h]`` with
  ``pcd_{t+h+1} ~= R_h pcd_{t+h} + t_h``, i.e. the rotation is about the world origin.
- ``centroid``: ``[rot6d(R_h), dc_h]`` with ``dc_h = c_{h+1} - c_h`` and
  ``c_h = centroid(pcd_{t+h})``, i.e. the rotation is about the part centroid. R_h is the
  same; the translation no longer depends on where the part sits in the world.

The two are related by ``dc_h = t_h + R_h c_h - c_h`` and ``t_h = c_{h+1} - R_h c_h``.
Every consumer that turns denormalised predictions into motion calls
:func:`decode_affordance_delta` right after denormalising, so the rest of the code
always sees the world form.

Model input frame (``input_frame``):

- ``world`` (the paper models): point cloud and EE position in the robot-base frame.
- ``handle``: point cloud and EE position relative to the affordance (handle) point
  ``a_t`` of the observation step. The affordance token keeps the absolute ``a_t``.
  Applied inside ``LWMv3.forward`` (:func:`shift_observation_to_handle_frame`), so every
  consumer gets it.
- ``handle_rel``: as ``handle``, but the affordance token is also expressed in the handle
  frame (it becomes ``a_t - a_t``, i.e. zero for the current step), so no absolute position
  reaches the model.
"""

import numpy as np
import torch

AFFORDANCE_TARGET_FRAMES = ("world", "centroid")
INPUT_FRAMES = ("world", "handle", "handle_rel")


def check_frames(affordance_target_frame="world", input_frame="world"):
    """Validate a frame combination; ``input_frame=handle`` needs ``centroid`` targets."""
    if affordance_target_frame not in AFFORDANCE_TARGET_FRAMES:
        raise ValueError(
            f"affordance_target_frame must be one of {AFFORDANCE_TARGET_FRAMES}, got {affordance_target_frame!r}"
        )
    if input_frame not in INPUT_FRAMES:
        raise ValueError(f"input_frame must be one of {INPUT_FRAMES}, got {input_frame!r}")
    if input_frame in ("handle", "handle_rel") and affordance_target_frame != "centroid":
        raise ValueError(
            f"input_frame={input_frame!r} requires affordance_target_frame='centroid' "
            "(the world-form translation is not translation-invariant)"
        )


def rot6d_to_matrix(rot6d):
    """[..., 6] -> [..., 3, 3]; same Gram-Schmidt as ``rotation_transfer_6D_to_matrix_torch``."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(mat):
    """[..., 3, 3] -> [..., 6] (first two columns)."""
    return torch.cat([mat[..., :, 0], mat[..., :, 1]], dim=-1)


def _matvec(mat, vec):
    return (mat @ vec.unsqueeze(-1)).squeeze(-1)


def encode_affordance_delta(delta_world, centroids, frame="centroid"):
    """World-form 9-D deltas -> ``frame`` form.

    Args:
        delta_world: [B, H, 9] world-form deltas ``[rot6d(R_h), t_h]``.
        centroids: [B, 3] centroid of the part cloud at the chunk start (later centroids
            follow the chain ``c_{h+1} = R_h c_h + t_h``), or [B, H, 3] the measured
            centroids ``c_h = centroid(pcd_{t+h})``.
        frame: ``world`` (returned unchanged) or ``centroid``.
    Returns:
        [B, H, 9]
    """
    if frame == "world":
        return delta_world
    if frame != "centroid":
        raise ValueError(f"unknown affordance_target_frame {frame!r}")
    R = rot6d_to_matrix(delta_world[..., :6])            # [B, H, 3, 3]
    t = delta_world[..., 6:]                             # [B, H, 3]
    H = delta_world.shape[-2]
    out = []
    c = centroids if centroids.dim() == delta_world.dim() - 1 else None
    for h in range(H):
        c_h = c if c is not None else centroids[..., h, :]
        c_next = _matvec(R[..., h, :, :], c_h) + t[..., h, :]
        out.append(torch.cat([delta_world[..., h, :6], c_next - c_h], dim=-1))
        if c is not None:
            c = c_next
    return torch.stack(out, dim=-2)


def decode_affordance_delta(delta, centroid0, frame="world"):
    """``frame``-form 9-D deltas (denormalised) -> world form ``[rot6d(R_h), t_h]``.

    Args:
        delta: [B, H, 9] denormalised model output (or targets).
        centroid0: [B, 3] centroid of the part cloud the deltas start from (the
            model's input cloud for a fresh chunk).
        frame: ``world`` (returned unchanged) or ``centroid``.
    Returns:
        [B, H, 9] world form; applying step h to the cloud as ``R_h x + t_h`` moves its
        centroid from ``c_h`` to ``c_{h+1} = c_h + dc_h``.
    """
    if frame == "world":
        return delta
    if frame != "centroid":
        raise ValueError(f"unknown affordance_target_frame {frame!r}")
    R = rot6d_to_matrix(delta[..., :6])
    dc = delta[..., 6:]
    c = centroid0
    out = []
    for h in range(delta.shape[-2]):
        c_next = c + dc[..., h, :]
        t_h = c_next - _matvec(R[..., h, :, :], c)
        out.append(torch.cat([delta[..., h, :6], t_h], dim=-1))
        c = c_next
    return torch.stack(out, dim=-2)


def observation_centroid(observation):
    """Centroid [B, 3] of the (most recent) input point cloud of an observation dict."""
    pcd = observation["pointcloud"]
    if pcd.dim() == 4:  # [B, n_obs_steps, N, 3]
        pcd = pcd[:, -1]
    return pcd.mean(dim=1)


def shift_observation_to_handle_frame(observation, shift_affordance=False):
    """Express point cloud and EE position relative to the affordance point ``a_t``.

    ``a_t`` is the affordance of the most recent observation step. Rotations, finger and
    actions are unchanged. The affordance token is shifted too if ``shift_affordance``
    (``input_frame=handle_rel``), else kept absolute. Returns a new dict.
    """
    aff = observation["static_affordance"]
    pcd = observation["pointcloud"]
    state = observation["agent_state"]
    ref = aff[:, -1] if aff.dim() == 3 else aff          # [B, 3]
    if pcd.dim() == 4:
        pcd = pcd - ref[:, None, None, :]
    else:
        pcd = pcd - ref[:, None, :]
    pos_ref = ref[:, None, :] if state.dim() == 3 else ref
    state = torch.cat([state[..., :3] - pos_ref, state[..., 3:]], dim=-1)
    out = dict(observation)
    out["pointcloud"] = pcd
    out["agent_state"] = state
    if shift_affordance:
        out["static_affordance"] = aff - (ref[:, None, :] if aff.dim() == 3 else ref)
    return out


# ---------------------------------------------------------------------------
# Training-time augmentation (rotation about the world z-axis + translation)
# ---------------------------------------------------------------------------

def yaw_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sample_scene_transform(rng, max_yaw_deg, max_translation, pivot):
    """Sample ``x -> Rz x + b``: yaw about the vertical axis through ``pivot`` plus an xy shift.

    Returns (Rz [3,3], b [3]) in float64. The z-translation is always 0 (the table /
    gravity direction stays fixed).
    """
    theta = np.deg2rad(rng.uniform(-max_yaw_deg, max_yaw_deg)) if max_yaw_deg > 0 else 0.0
    d = np.zeros(3)
    if max_translation > 0:
        d[:2] = rng.uniform(-max_translation, max_translation, size=2)
    Rz = yaw_matrix(theta)
    pivot = np.asarray(pivot, dtype=np.float64)
    b = pivot - Rz @ pivot + d
    return Rz, b


def transform_world_delta(delta_world, Rz, b):
    """World-form 9-D deltas [..., 9] under the scene map ``x -> Rz x + b`` (numpy).

    ``x_{h+1} = R x_h + t``  =>  ``x'_{h+1} = R' x'_h + t'`` with
    ``R' = Rz R Rz^T`` and ``t' = Rz t + b - R' b``.
    """
    d = torch.from_numpy(np.asarray(delta_world, dtype=np.float64))
    Rz_t = torch.from_numpy(Rz)
    b_t = torch.from_numpy(b)
    R = rot6d_to_matrix(d[..., :6])
    R_new = Rz_t @ R @ Rz_t.T
    t_new = _matvec(Rz_t.expand_as(R), d[..., 6:]) + b_t - _matvec(R_new, b_t.expand_as(d[..., 6:]))
    return torch.cat([matrix_to_rot6d(R_new), t_new], dim=-1).numpy()
