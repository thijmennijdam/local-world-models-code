"""Core evaluation logic: autoregressive rollouts, windowed N-step eval, phase-based eval.

All batch-heavy evaluation uses cross-episode batched inference for GPU efficiency.
"""

import hashlib

import torch
import numpy as np
from termcolor import cprint
from tqdm import tqdm

from lwm.world_model.evaluation_utils import (
    load_episode_data,
    get_episode_data,
    build_observation,
    predict_next_state_and_affordance,
    compose_state_with_delta,
    compose_state_with_delta_batched,
    compose_affordance_with_9dof_delta_batched,
    rotation_transfer_6D_to_matrix_torch,
)
from lwm.world_model.analysis.constants import PHASE_NAMES
from lwm.world_model.targets import decode_affordance_delta, observation_centroid

# Default batch size for batched inference (tune for GPU memory)
_DEFAULT_BATCH_SIZE = 128


# =============================================================================
# Normalization helpers
# =============================================================================

def _make_norm_tensors(normalization_stats, device):
    """Create normalization tensors once, reuse across episodes."""
    return (
        torch.from_numpy(normalization_stats["action_mean"]).float().to(device),
        torch.from_numpy(normalization_stats["action_std"]).float().to(device),
        torch.from_numpy(normalization_stats["agent_state_delta_mean"]).float().to(device),
        torch.from_numpy(normalization_stats["agent_state_delta_std"]).float().to(device),
        torch.from_numpy(normalization_stats["affordance_delta_mean"]).float().to(device),
        torch.from_numpy(normalization_stats["affordance_delta_std"]).float().to(device),
    )


def _batched_step(
    lwm, observation, cur_states, cur_affs,
    agent_delta_mean, agent_delta_std, aff_delta_mean, aff_delta_std,
    predict_agent_state, compose_affordance_fn,
    gt_states_next=None, return_aff_deltas=False,
):
    """Run one batched forward pass and compose next states/affordances.

    With return_aff_deltas=True also returns the denormalised world-form part-motion
    deltas [B, aff_out_dim] of the first step.

    Args:
        observation: dict with batched tensors [B, ...]
        cur_states: [B, state_dim]
        cur_affs: [B, aff_dim]
        gt_states_next: [B, state_dim] fallback when predict_agent_state=False

    Returns:
        next_states: [B, state_dim]
        next_affs: [B, 3]
    """
    output = lwm(observation)

    # Affordance
    pred_aff_deltas = output["affordance_delta"][:, 0]  # [B, aff_out_dim]
    pred_aff_deltas = pred_aff_deltas * aff_delta_std + aff_delta_mean
    # Model frame -> world form (no-op for affordance_target_frame="world")
    frame = getattr(lwm, "affordance_target_frame", "world")
    if frame != "world" and pred_aff_deltas.shape[-1] == 9:
        pred_aff_deltas = decode_affordance_delta(
            pred_aff_deltas.float().unsqueeze(1), observation_centroid(observation).float(), frame
        ).squeeze(1)
    if compose_affordance_fn is not None:
        next_affs = compose_affordance_with_9dof_delta_batched(cur_affs, pred_aff_deltas)
    else:
        next_affs = cur_affs + pred_aff_deltas

    # Agent state
    if predict_agent_state:
        pred_agent_deltas = output["agent_state_delta"][:, 0]  # [B, state_dim]
        pred_agent_deltas = pred_agent_deltas * agent_delta_std + agent_delta_mean
        next_states = compose_state_with_delta_batched(cur_states, pred_agent_deltas)
    else:
        next_states = gt_states_next if gt_states_next is not None else cur_states

    if return_aff_deltas:
        return next_states, next_affs, pred_aff_deltas
    return next_states, next_affs


def kabsch_rotation(pcd_a, pcd_b):
    """Batched Kabsch rotation R [B, 3, 3] with pcd_b ~= R pcd_a + t (float64)."""
    P = pcd_a.double() - pcd_a.double().mean(dim=1, keepdim=True)
    Q = pcd_b.double() - pcd_b.double().mean(dim=1, keepdim=True)
    H = P.transpose(1, 2) @ Q
    U, _, Vt = torch.linalg.svd(H)
    V = Vt.transpose(1, 2)
    d = torch.sign(torch.linalg.det(V @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    return V @ D @ U.transpose(1, 2)


def rotation_angle_deg(R_a, R_b):
    """Geodesic angle (degrees) between batched rotations [B, 3, 3]."""
    cos = ((R_a.transpose(1, 2) @ R_b).diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def collect_windows(replay_buffer, episode_indices, window_steps, horizon, windows_per_episode, seed=42):
    """Sample windows_per_episode random window starts per episode (what evaluate_windowed
    does by default). Returns a list of (ep_idx, t_start) in sampling order."""
    rng = np.random.RandomState(seed)
    out = []
    for ep_idx in episode_indices:
        ep_len = int(replay_buffer.episode_lengths[ep_idx])
        max_start = ep_len - window_steps - horizon
        if max_start <= 0:
            continue
        n_windows = min(windows_per_episode, max_start)
        starts = rng.choice(range(0, max_start), size=n_windows, replace=False)
        out.extend((ep_idx, int(t)) for t in starts)
    return out


# =============================================================================
# Full-Episode Autoregressive Evaluation (single episode — for plots)
# =============================================================================

def evaluate_episode(
    lwm,
    replay_buffer,
    ep_idx,
    device,
    normalization_stats,
    max_steps=256,
    predict_agent_state=True,
    compose_affordance_fn=None,
    preloaded_data=None,
    norm_tensors=None,
):
    """
    Run long-horizon autoregressive evaluation on a single episode.
    Used for individual episode plots (runs only 1-2 times).

    Returns:
        dict with trajectory data and errors, or None if episode too short
    """
    lwm.eval()

    rb = replay_buffer
    horizon = lwm.n_future_steps

    if norm_tensors is not None:
        action_mean, action_std, agent_state_delta_mean, agent_state_delta_std, \
            affordance_delta_mean, affordance_delta_std = norm_tensors
    else:
        action_mean, action_std, agent_state_delta_mean, agent_state_delta_std, \
            affordance_delta_mean, affordance_delta_std = _make_norm_tensors(normalization_stats, device)

    ep_len = int(rb.episode_lengths[ep_idx])
    if ep_len <= horizon + 1:
        return None

    agent_states, static_affordances, point_clouds, actions, _ = get_episode_data(
        rb, ep_idx, device, preloaded_data=preloaded_data
    )

    cur_state = agent_states[0]
    cur_aff = static_affordances[0]

    max_t = min(ep_len - horizon - 1, max_steps)
    if max_t <= 0:
        return None

    pred_affordance_traj = [cur_aff.detach().cpu().numpy()]
    gt_affordance_traj = [static_affordances[0].detach().cpu().numpy()]
    pred_agent_state_traj = [cur_state.detach().cpu().numpy()]
    gt_agent_state_traj = [agent_states[0].detach().cpu().numpy()]

    per_step_agent_errors = []
    per_step_affordance_errors = []
    per_step_gt_displacement_cm = []

    aff_start = static_affordances[0]

    use_amp = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for t in range(max_t):
            action_horizon = actions[t : t + horizon]
            action_horizon_normalized = (action_horizon - action_mean) / action_std

            observation = build_observation(
                t, cur_state, cur_aff, agent_states, static_affordances,
                point_clouds, action_horizon_normalized, lwm.n_obs_steps, horizon
            )

            next_state, next_aff, _, _ = predict_next_state_and_affordance(
                lwm, observation, cur_state, cur_aff,
                agent_state_delta_mean, agent_state_delta_std,
                affordance_delta_mean, affordance_delta_std,
                predict_agent_state, compose_state_with_delta,
                gt_state_fallback=agent_states[t + 1] if not predict_agent_state else None,
                compose_affordance_fn=compose_affordance_fn,
            )

            gt_state = agent_states[t + 1]
            gt_aff = static_affordances[t + 1]

            if predict_agent_state:
                agent_pos_l2_cm = torch.norm(next_state[:3] - gt_state[:3]).item() * 100.0
                per_step_agent_errors.append(agent_pos_l2_cm)
            affordance_l2_cm = torch.norm(next_aff - gt_aff).item() * 100.0
            per_step_affordance_errors.append(affordance_l2_cm)

            gt_disp_cm = torch.norm(gt_aff - aff_start).item() * 100.0
            per_step_gt_displacement_cm.append(gt_disp_cm)

            pred_affordance_traj.append(next_aff.detach().cpu().numpy())
            gt_affordance_traj.append(gt_aff.detach().cpu().numpy())
            pred_agent_state_traj.append(next_state.detach().cpu().numpy())
            gt_agent_state_traj.append(gt_state.detach().cpu().numpy())

            cur_state = next_state.detach()
            cur_aff = next_aff.detach()

    return {
        "pred_affordance": np.array(pred_affordance_traj),
        "gt_affordance": np.array(gt_affordance_traj),
        "pred_agent_state": np.array(pred_agent_state_traj),
        "gt_agent_state": np.array(gt_agent_state_traj),
        "per_step_agent_errors": per_step_agent_errors if predict_agent_state else None,
        "per_step_affordance_errors": per_step_affordance_errors,
        "per_step_gt_displacement_cm": per_step_gt_displacement_cm,
        "episode_idx": ep_idx,
    }


# =============================================================================
# Batched Full-Horizon Rollout (for avg error vs step)
# =============================================================================

def evaluate_episodes_cached(
    lwm, replay_buffer, episode_indices, device, normalization_stats,
    max_steps=256, predict_agent_state=True, compose_affordance_fn=None,
    cache=None, preloaded_data=None, inference_batch_size=_DEFAULT_BATCH_SIZE,
):
    """
    Run full autoregressive rollouts with cross-episode batching.
    Skips already-cached episodes. New results are added to cache in-place.
    """
    if cache is None:
        cache = {}

    to_compute = [idx for idx in episode_indices if idx not in cache]
    if not to_compute:
        return cache

    action_mean, action_std, agent_delta_mean, agent_delta_std, \
        aff_delta_mean, aff_delta_std = _make_norm_tensors(normalization_stats, device)
    horizon = lwm.n_future_steps
    n_obs = lwm.n_obs_steps
    # v1 MLP expects flat [B, ...] inputs; transformer models expect [B, T, ...]
    needs_obs_dim = not hasattr(lwm, 'backbone')

    # Load episode data and compute max_t for each
    # Episode data stays on CPU; moved to GPU per batch during inference.
    ep_data_list = []  # (ep_idx, agent_states, static_affordances, point_clouds, actions, max_t)
    for ep_idx in to_compute:
        ep_len = int(replay_buffer.episode_lengths[ep_idx])
        if ep_len <= horizon + 1:
            continue
        if preloaded_data is not None and ep_idx in preloaded_data:
            agent_states, static_affordances, point_clouds, actions, _ = preloaded_data[ep_idx]
        else:
            agent_states, static_affordances, point_clouds, actions, _ = \
                load_episode_data(replay_buffer, ep_idx, device="cpu")
        max_t = min(ep_len - horizon - 1, max_steps)
        if max_t <= 0:
            continue
        ep_data_list.append((ep_idx, agent_states, static_affordances, point_clouds, actions, max_t))

    if not ep_data_list:
        return cache

    N = len(ep_data_list)
    state_dim = ep_data_list[0][1].shape[1]
    aff_dim = ep_data_list[0][2].shape[1]
    global_max_t = max(d[5] for d in ep_data_list)

    # Per-episode tracked state (on GPU for inference)
    cur_states = torch.stack([d[1][0] for d in ep_data_list]).to(device)
    cur_affs = torch.stack([d[2][0] for d in ep_data_list]).to(device)
    aff_starts = cur_affs.clone()

    # Per-episode, per-step error storage (lists of lists)
    ep_aff_errors = [[] for _ in range(N)]
    ep_agent_errors = [[] for _ in range(N)]
    ep_gt_displacements = [[] for _ in range(N)]

    # Per-episode trajectory storage (for cache results)
    ep_pred_aff = [[cur_affs[i].detach().cpu().numpy()] for i in range(N)]
    ep_gt_aff = [[ep_data_list[i][2][0].detach().cpu().numpy()] for i in range(N)]
    ep_pred_state = [[cur_states[i].detach().cpu().numpy()] for i in range(N)]
    ep_gt_state = [[ep_data_list[i][1][0].detach().cpu().numpy()] for i in range(N)]

    # Active mask: which episodes still have steps remaining
    max_ts = torch.tensor([d[5] for d in ep_data_list], device=device)

    lwm.eval()
    use_amp = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for t in tqdm(range(global_max_t), desc="Full-horizon rollouts (batched)"):
            active_mask = (t < max_ts)
            active_idx = torch.where(active_mask)[0]
            if len(active_idx) == 0:
                break

            # Process in batches
            for b_start in range(0, len(active_idx), inference_batch_size):
                b_end = min(b_start + inference_batch_size, len(active_idx))
                batch_idx = active_idx[b_start:b_end]  # indices into ep_data_list
                B = len(batch_idx)

                # Build batched observation
                batch_cur_states = cur_states[batch_idx]
                batch_cur_affs = cur_affs[batch_idx]

                if n_obs == 1:
                    # Collect per-episode point clouds and actions at timestep t
                    pcds = torch.stack([ep_data_list[i][3][t] for i in batch_idx.tolist()]).to(device)
                    act_horizons = torch.stack([
                        ep_data_list[i][4][t:t + horizon] for i in batch_idx.tolist()
                    ]).to(device)
                    act_horizons_norm = (act_horizons - action_mean) / action_std

                    if needs_obs_dim:
                        observation = {
                            "agent_state": batch_cur_states.unsqueeze(1),
                            "static_affordance": batch_cur_affs.unsqueeze(1),
                            "pointcloud": pcds.unsqueeze(1),
                            "action_horizon": act_horizons_norm,
                        }
                    else:
                        observation = {
                            "agent_state": batch_cur_states,
                            "static_affordance": batch_cur_affs,
                            "pointcloud": pcds,
                            "action_horizon": act_horizons_norm,
                        }
                else:
                    # Multi-obs: build per-window observations with GT history
                    obs_states_list, obs_affs_list, obs_pcds_list, obs_acts_list = [], [], [], []
                    for i in batch_idx.tolist():
                        d = ep_data_list[i]
                        obs_start = max(0, t - n_obs + 1)
                        obs_s = d[1][obs_start:t + 1]
                        obs_a = d[2][obs_start:t + 1]
                        obs_p = d[3][obs_start:t + 1]
                        if obs_s.shape[0] < n_obs:
                            pad = n_obs - obs_s.shape[0]
                            obs_s = torch.cat([d[1][0:1].expand(pad, -1), obs_s])
                            obs_a = torch.cat([d[2][0:1].expand(pad, -1), obs_a])
                            obs_p = torch.cat([d[3][0:1].expand(pad, -1, -1), obs_p])
                        obs_states_list.append(obs_s)
                        obs_affs_list.append(obs_a)
                        obs_pcds_list.append(obs_p)
                        act_h = d[4][t:t + horizon]
                        obs_acts_list.append((act_h - action_mean) / action_std)

                    observation = {
                        "agent_state": torch.stack(obs_states_list).to(device),
                        "static_affordance": torch.stack(obs_affs_list).to(device),
                        "pointcloud": torch.stack(obs_pcds_list).to(device),
                        "action_horizon": torch.stack(obs_acts_list).to(device),
                    }

                gt_states_next = torch.stack([
                    ep_data_list[i][1][t + 1] for i in batch_idx.tolist()
                ]).to(device) if not predict_agent_state else None

                next_states, next_affs = _batched_step(
                    lwm, observation, batch_cur_states, batch_cur_affs,
                    agent_delta_mean, agent_delta_std, aff_delta_mean, aff_delta_std,
                    predict_agent_state, compose_affordance_fn,
                    gt_states_next=gt_states_next,
                )

                # Compute errors and store results
                for j, i in enumerate(batch_idx.tolist()):
                    gt_aff = ep_data_list[i][2][t + 1].to(device)
                    gt_state = ep_data_list[i][1][t + 1].to(device)

                    aff_err = torch.norm(next_affs[j] - gt_aff).item() * 100.0
                    gt_disp = torch.norm(gt_aff - aff_starts[i]).item() * 100.0
                    ep_aff_errors[i].append(aff_err)
                    ep_gt_displacements[i].append(gt_disp)

                    if predict_agent_state:
                        agent_err = torch.norm(next_states[j, :3] - gt_state[:3]).item() * 100.0
                        ep_agent_errors[i].append(agent_err)

                    ep_pred_aff[i].append(next_affs[j].detach().cpu().numpy())
                    ep_gt_aff[i].append(gt_aff.detach().cpu().numpy())
                    ep_pred_state[i].append(next_states[j].detach().cpu().numpy())
                    ep_gt_state[i].append(gt_state.detach().cpu().numpy())

                # Update tracked state
                cur_states[batch_idx] = next_states.detach()
                cur_affs[batch_idx] = next_affs.detach()

    # Store results in cache
    for i, (ep_idx, _, _, _, _, max_t) in enumerate(ep_data_list):
        cache[ep_idx] = {
            "pred_affordance": np.array(ep_pred_aff[i]),
            "gt_affordance": np.array(ep_gt_aff[i]),
            "pred_agent_state": np.array(ep_pred_state[i]),
            "gt_agent_state": np.array(ep_gt_state[i]),
            "per_step_agent_errors": ep_agent_errors[i] if predict_agent_state else None,
            "per_step_affordance_errors": ep_aff_errors[i],
            "per_step_gt_displacement_cm": ep_gt_displacements[i],
            "episode_idx": ep_idx,
        }

    return cache


def aggregate_avg_error_from_cache(cache, episode_indices, data_type="expert", predict_agent_state=True):
    """Aggregate cached per-episode rollout results into average error vs step."""
    per_step_agent_errors = {}
    per_step_affordance_errors = {}
    per_step_gt_displacement = {}
    num_evaluated = 0

    for ep_idx in episode_indices:
        if ep_idx not in cache:
            continue

        result = cache[ep_idx]
        num_evaluated += 1
        aff_errors = result["per_step_affordance_errors"]
        agent_errors = result["per_step_agent_errors"]
        gt_displacements = result["per_step_gt_displacement_cm"]

        for step_idx, aff_err in enumerate(aff_errors):
            if step_idx not in per_step_affordance_errors:
                per_step_affordance_errors[step_idx] = []
            per_step_affordance_errors[step_idx].append(aff_err)

            if step_idx not in per_step_gt_displacement:
                per_step_gt_displacement[step_idx] = []
            per_step_gt_displacement[step_idx].append(gt_displacements[step_idx])

            if predict_agent_state and agent_errors is not None:
                if step_idx not in per_step_agent_errors:
                    per_step_agent_errors[step_idx] = []
                per_step_agent_errors[step_idx].append(agent_errors[step_idx])

    if num_evaluated == 0:
        cprint(f"No {data_type} episodes could be evaluated", "yellow")
        return None

    step_indices = sorted(per_step_affordance_errors.keys())
    avg_affordance_errors = [np.mean(per_step_affordance_errors[s]) for s in step_indices]
    avg_gt_displacement_cm = [np.mean(per_step_gt_displacement[s]) for s in step_indices]

    disp_threshold_cm = 0.5
    avg_relative_errors = []
    for err, disp in zip(avg_affordance_errors, avg_gt_displacement_cm):
        if disp > disp_threshold_cm:
            avg_relative_errors.append(err / disp)
        else:
            avg_relative_errors.append(None)

    if predict_agent_state and len(per_step_agent_errors) > 0:
        avg_agent_errors = [np.mean(per_step_agent_errors[s]) for s in step_indices]
        avg_total_errors = [a + f for a, f in zip(avg_agent_errors, avg_affordance_errors)]
    else:
        avg_agent_errors = None
        avg_total_errors = avg_affordance_errors

    cprint(f"Evaluated {num_evaluated} {data_type} episodes, max step index: {max(step_indices)}", "green")

    return {
        "step_indices": step_indices,
        "avg_total_errors": avg_total_errors,
        "avg_agent_errors": avg_agent_errors,
        "avg_affordance_errors": avg_affordance_errors,
        "avg_gt_displacement_cm": avg_gt_displacement_cm,
        "avg_relative_errors": avg_relative_errors,
        "num_episodes": num_evaluated,
        "data_type": data_type,
    }


def evaluate_episodes_for_avg_error(
    lwm, replay_buffer, episode_indices, device, normalization_stats,
    max_steps=256, predict_agent_state=True, data_type="expert",
    compose_affordance_fn=None, preloaded_data=None,
):
    """Backward-compatible wrapper: compute + aggregate in one call."""
    cprint(f"Evaluating {len(episode_indices)} {data_type} episodes for average error vs step...", "yellow")
    cache = evaluate_episodes_cached(
        lwm, replay_buffer, episode_indices, device, normalization_stats,
        max_steps=max_steps, predict_agent_state=predict_agent_state,
        compose_affordance_fn=compose_affordance_fn,
        preloaded_data=preloaded_data,
    )
    return aggregate_avg_error_from_cache(cache, episode_indices, data_type=data_type,
                                          predict_agent_state=predict_agent_state)


# =============================================================================
# Batched Windowed N-Step Evaluation
# =============================================================================

def evaluate_windowed(
    lwm, replay_buffer, episode_indices, device, normalization_stats,
    window_steps=6, windows_per_episode=5, predict_agent_state=True,
    compose_affordance_fn=None, seed=42, preloaded_data=None,
    inference_batch_size=_DEFAULT_BATCH_SIZE, windows=None,
):
    """
    Windowed N-step eval with cross-episode batched inference.

    Collects all windows upfront (collect_windows, or the explicit ``windows`` list of
    (ep_idx, t_start)), then processes step-by-step in batches.
    Returns list of per-window result dicts (each with ep_idx for filtering) with
    per-step errors: aff_errors (handle position, cm), agent_errors (EE position, cm),
    and additionally finger_errors_mm (with agent-state prediction) and, for 9-D models,
    rot_errors_deg (geodesic angle between the predicted cumulative part rotation since
    the window start and the true one, Kabsch between the part clouds).
    """
    action_mean, action_std, agent_delta_mean, agent_delta_std, \
        aff_delta_mean, aff_delta_std = _make_norm_tensors(normalization_stats, device)

    horizon = lwm.n_future_steps
    n_obs = lwm.n_obs_steps
    needs_obs_dim = not hasattr(lwm, 'backbone')

    # Phase 1: Collect all windows
    # Each window: (ep_idx, t_start, agent_states, static_affordances, point_clouds, actions)
    # Episode data is kept on CPU; only moved to GPU during batched inference.
    if windows is None:
        windows = collect_windows(replay_buffer, episode_indices, window_steps, horizon,
                                  windows_per_episode, seed=seed)
    ep_cache = {}
    window_data = []
    for ep_idx, t_start in windows:
        if ep_idx not in ep_cache:
            if preloaded_data is not None and ep_idx in preloaded_data:
                ep_cache[ep_idx] = tuple(preloaded_data[ep_idx][:4])
            else:
                ep_cache[ep_idx] = tuple(load_episode_data(replay_buffer, ep_idx, device="cpu")[:4])
        window_data.append((ep_idx, int(t_start)) + ep_cache[ep_idx])
    windows = window_data

    W = len(windows)
    if W == 0:
        return []

    cprint(f"    {W} windows collected, batched inference...", "cyan")

    # Phase 2: Initialize tracked state (on GPU for inference)
    # Window tuple: (ep_idx, t_start, agent_states, static_affordances, point_clouds, actions)
    # Note: window episode data is on CPU; we move slices to GPU per batch.
    cur_states = torch.stack([w[2][w[1]].clone() for w in windows]).to(device)  # [W, state_dim]
    cur_affs = torch.stack([w[3][w[1]].clone() for w in windows]).to(device)    # [W, aff_dim]
    aff_starts = cur_affs.clone()

    # Results storage
    all_aff_errors = torch.zeros(W, window_steps, device=device)
    all_gt_displacements = torch.zeros(W, window_steps, device=device)
    all_agent_errors = torch.zeros(W, window_steps, device=device) if predict_agent_state else None
    all_finger_errors = torch.zeros(W, window_steps, device=device) if predict_agent_state else None
    track_rot = getattr(lwm, "affordance_output_dim", 3) == 9 and compose_affordance_fn is not None
    all_rot_errors = torch.zeros(W, window_steps, device=device) if track_rot else None
    cum_R = torch.eye(3, dtype=torch.float64, device=device).repeat(W, 1, 1) if track_rot else None

    # Phase 3: Step-by-step batched forward passes
    lwm.eval()
    use_amp = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for k in range(window_steps):
            for b_start in range(0, W, inference_batch_size):
                b_end = min(b_start + inference_batch_size, W)
                bs = slice(b_start, b_end)
                batch_windows = windows[b_start:b_end]

                batch_cur_states = cur_states[bs]
                batch_cur_affs = cur_affs[bs]

                if n_obs == 1:
                    pcds = torch.stack([w[4][w[1] + k] for w in batch_windows]).to(device)
                    act_h = torch.stack([w[5][w[1] + k : w[1] + k + horizon] for w in batch_windows]).to(device)
                    act_h_norm = (act_h - action_mean) / action_std

                    if needs_obs_dim:
                        observation = {
                            "agent_state": batch_cur_states.unsqueeze(1),
                            "static_affordance": batch_cur_affs.unsqueeze(1),
                            "pointcloud": pcds.unsqueeze(1),
                            "action_horizon": act_h_norm,
                        }
                    else:
                        observation = {
                            "agent_state": batch_cur_states,
                            "static_affordance": batch_cur_affs,
                            "pointcloud": pcds,
                            "action_horizon": act_h_norm,
                        }
                else:
                    obs_s, obs_a, obs_p, obs_act = [], [], [], []
                    for w in batch_windows:
                        t = w[1] + k
                        obs_start = max(0, t - n_obs + 1)
                        s = w[2][obs_start:t + 1]
                        a = w[3][obs_start:t + 1]
                        p = w[4][obs_start:t + 1]
                        if s.shape[0] < n_obs:
                            pad = n_obs - s.shape[0]
                            s = torch.cat([w[2][0:1].expand(pad, -1), s])
                            a = torch.cat([w[3][0:1].expand(pad, -1), a])
                            p = torch.cat([w[4][0:1].expand(pad, -1, -1), p])
                        obs_s.append(s)
                        obs_a.append(a)
                        obs_p.append(p)
                        act_h = w[5][t:t + horizon]
                        obs_act.append((act_h - action_mean) / action_std)

                    observation = {
                        "agent_state": torch.stack(obs_s).to(device),
                        "static_affordance": torch.stack(obs_a).to(device),
                        "pointcloud": torch.stack(obs_p).to(device),
                        "action_horizon": torch.stack(obs_act).to(device),
                    }

                gt_states_next = torch.stack([
                    w[2][w[1] + k + 1] for w in batch_windows
                ]).to(device) if not predict_agent_state else None

                next_states, next_affs, aff_deltas = _batched_step(
                    lwm, observation, batch_cur_states, batch_cur_affs,
                    agent_delta_mean, agent_delta_std, aff_delta_mean, aff_delta_std,
                    predict_agent_state, compose_affordance_fn,
                    gt_states_next=gt_states_next, return_aff_deltas=True,
                )

                # Compute errors
                gt_affs = torch.stack([w[3][w[1] + k + 1] for w in batch_windows]).to(device)
                all_aff_errors[bs, k] = torch.norm(next_affs - gt_affs, dim=-1) * 100.0
                all_gt_displacements[bs, k] = torch.norm(gt_affs - aff_starts[bs], dim=-1) * 100.0

                if predict_agent_state:
                    gt_states = torch.stack([w[2][w[1] + k + 1] for w in batch_windows]).to(device)
                    all_agent_errors[bs, k] = torch.norm(next_states[:, :3] - gt_states[:, :3], dim=-1) * 100.0
                    all_finger_errors[bs, k] = (next_states[:, 9] - gt_states[:, 9]).abs() * 1000.0

                if track_rot:
                    R_step = rotation_transfer_6D_to_matrix_torch(aff_deltas[:, :6].float()).double()
                    cum_R[bs] = R_step @ cum_R[bs]
                    pcd0 = torch.stack([w[4][w[1]] for w in batch_windows]).to(device)
                    pcdk = torch.stack([w[4][w[1] + k + 1] for w in batch_windows]).to(device)
                    all_rot_errors[bs, k] = rotation_angle_deg(cum_R[bs], kabsch_rotation(pcd0, pcdk)).to(all_rot_errors.dtype)

                cur_states[bs] = next_states.detach()
                cur_affs[bs] = next_affs.detach()

    # Phase 4: Convert to result dicts
    all_window_results = []
    for w_idx in range(W):
        result = {
            "aff_errors": all_aff_errors[w_idx].cpu().tolist(),
            "gt_displacements": all_gt_displacements[w_idx].cpu().tolist(),
            "ep_idx": windows[w_idx][0],
        }
        if predict_agent_state:
            result["agent_errors"] = all_agent_errors[w_idx].cpu().tolist()
            result["finger_errors_mm"] = all_finger_errors[w_idx].cpu().tolist()
        if track_rot:
            result["rot_errors_deg"] = all_rot_errors[w_idx].cpu().tolist()
        result["t_start"] = windows[w_idx][1]
        all_window_results.append(result)

    return all_window_results


def aggregate_windowed_results(all_window_results, episode_indices=None, window_steps=6):
    """
    Aggregate windowed eval results into per-step statistics.

    If episode_indices is provided, only include windows from those episodes.
    """
    if episode_indices is not None:
        ep_set = set(episode_indices)
        filtered = [w for w in all_window_results if w["ep_idx"] in ep_set]
    else:
        filtered = all_window_results

    if not filtered:
        return None

    has_agent = "agent_errors" in filtered[0]
    has_finger = "finger_errors_mm" in filtered[0]

    per_step_aff_errors = [[] for _ in range(window_steps)]
    per_step_gt_displacements = [[] for _ in range(window_steps)]
    per_step_agent_errors = [[] for _ in range(window_steps)]
    per_step_finger_errors = [[] for _ in range(window_steps)]

    for w in filtered:
        for k in range(window_steps):
            per_step_aff_errors[k].append(w["aff_errors"][k])
            per_step_gt_displacements[k].append(w["gt_displacements"][k])
            if has_agent:
                per_step_agent_errors[k].append(w["agent_errors"][k])
            if has_finger:
                per_step_finger_errors[k].append(w["finger_errors_mm"][k])

    return {
        "step_indices": list(range(1, window_steps + 1)),
        "avg_aff_drift_cm": [float(np.mean(e)) for e in per_step_aff_errors],
        "std_aff_drift_cm": [float(np.std(e)) for e in per_step_aff_errors],
        "avg_gt_displacement_cm": [float(np.mean(d)) for d in per_step_gt_displacements],
        "avg_agent_drift_cm": [float(np.mean(e)) for e in per_step_agent_errors] if has_agent else None,
        "std_agent_drift_cm": [float(np.std(e)) for e in per_step_agent_errors] if has_agent else None,
        "avg_finger_err_mm": [float(np.mean(e)) for e in per_step_finger_errors] if has_finger else None,
        "num_windows": len(filtered),
    }


# =============================================================================
# Phase-Based Evaluation (Reach / Grasp / Pull) — batched
# =============================================================================

def detect_episode_phases(
    agent_states, static_affordances,
    proximity_threshold_m, window_steps, pull_movement_threshold_m,
    horizon, rng,
):
    """
    Detect reach/grasp/pull phase windows in a single episode.

    Returns:
        dict mapping phase name -> window start timestep (int or None)
    """
    ep_len = agent_states.shape[0]
    ee_pos = agent_states[:, :3]
    distances = torch.norm(ee_pos - static_affordances, dim=-1)

    near_mask = distances < proximity_threshold_m
    near_indices = torch.where(near_mask)[0]

    result = {p: None for p in PHASE_NAMES}

    max_valid_start = ep_len - window_steps - horizon
    if max_valid_start <= 0:
        return result

    if len(near_indices) == 0:
        result["reach"] = int(rng.randint(0, max_valid_start))
        return result

    t_near = int(near_indices[0].item())

    max_reach_start = t_near - window_steps
    if max_reach_start > 0:
        result["reach"] = int(rng.randint(0, max_reach_start))

    if t_near <= max_valid_start:
        result["grasp"] = t_near

    grasp_end = t_near + window_steps
    if grasp_end <= max_valid_start:
        aff_at_near = static_affordances[t_near]
        pull_range = slice(grasp_end, max_valid_start + 1)
        aff_disps = torch.norm(
            static_affordances[pull_range] - aff_at_near, dim=-1
        )
        ee_dists = distances[pull_range]

        valid_pull = (
            (aff_disps > pull_movement_threshold_m)
            & (ee_dists < proximity_threshold_m * 3)
        )
        valid_idx = torch.where(valid_pull)[0]
        if len(valid_idx) > 0:
            chosen = int(valid_idx[rng.randint(0, len(valid_idx))].item())
            result["pull"] = grasp_end + chosen

    return result


def _episode_seed(episode_path, base_seed):
    """Deterministic per-episode seed (hashlib, not built-in hash) for reproducible phase sampling."""
    h = hashlib.sha256(str(episode_path).encode()).digest()
    path_hash = int.from_bytes(h[:4], 'little')
    return (base_seed + path_hash) % (2**32)


def evaluate_phase_based(
    lwm, replay_buffer, episode_indices, device, normalization_stats,
    proximity_threshold_cm=1.0, window_steps=18,
    pull_movement_threshold_cm=0.5,
    predict_agent_state=True, compose_affordance_fn=None, seed=42,
    phase_cache=None, preloaded_data=None,
    inference_batch_size=_DEFAULT_BATCH_SIZE,
):
    """
    Phase-based windowed evaluation with batched inference.

    Phase detection runs per-episode, then all detected windows are batched together.
    """
    action_mean, action_std, agent_delta_mean, agent_delta_std, \
        aff_delta_mean, aff_delta_std = _make_norm_tensors(normalization_stats, device)

    proximity_m = proximity_threshold_cm / 100.0
    pull_m = pull_movement_threshold_cm / 100.0
    horizon = lwm.n_future_steps
    n_obs = lwm.n_obs_steps
    needs_obs_dim = not hasattr(lwm, 'backbone')

    phase_results = {p: [] for p in PHASE_NAMES}
    phase_counts = {p: 0 for p in PHASE_NAMES}

    ep_paths = {ep_idx: replay_buffer.all_path_list[ep_idx] for ep_idx in episode_indices}
    cached_eps = phase_cache.get("episodes", {}) if phase_cache is not None else {}

    # Separate cached vs uncached
    cached_indices = []
    uncached_indices = []
    for ep_idx in episode_indices:
        if ep_paths[ep_idx] in cached_eps:
            cached_indices.append(ep_idx)
        else:
            uncached_indices.append(ep_idx)

    # Restore cached results
    for ep_idx in cached_indices:
        ep_data = cached_eps[ep_paths[ep_idx]]
        for phase_name in PHASE_NAMES:
            cached_win = ep_data.get(phase_name)
            if cached_win is None:
                continue
            phase_counts[phase_name] += 1
            win = dict(cached_win)
            win["ep_idx"] = ep_idx
            phase_results[phase_name].append(win)

    if cached_indices:
        cprint(f"    Phase cache: {len(cached_indices)} from cache, "
               f"{len(uncached_indices)} to evaluate", "green")

    if not uncached_indices:
        return phase_results, phase_counts

    # Phase 1: Detect phases and collect windows
    # windows: list of (ep_idx, phase_name, t_start, agent_states, static_affordances, point_clouds, actions)
    windows = []
    ep_phase_map = {}  # ep_idx -> {phase_name: window_index_in_list}

    for ep_idx in tqdm(uncached_indices, desc="Phase detection", disable=len(uncached_indices) < 50):
        ep_rng = np.random.RandomState(_episode_seed(ep_paths[ep_idx], seed))
        # Keep episode data on CPU; only move to GPU during batched inference
        if preloaded_data is not None and ep_idx in preloaded_data:
            agent_states, static_affordances, point_clouds, actions, _ = preloaded_data[ep_idx]
        else:
            agent_states, static_affordances, point_clouds, actions, _ = \
                load_episode_data(replay_buffer, ep_idx, device="cpu")

        phases = detect_episode_phases(
            agent_states, static_affordances,
            proximity_m, window_steps, pull_m, horizon, ep_rng,
        )

        ep_phase_map[ep_idx] = {}
        for phase_name in PHASE_NAMES:
            t_start = phases[phase_name]
            if t_start is None:
                ep_phase_map[ep_idx][phase_name] = None
                continue

            phase_counts[phase_name] += 1
            w_idx = len(windows)
            ep_phase_map[ep_idx][phase_name] = w_idx
            windows.append((ep_idx, phase_name, t_start, agent_states, static_affordances, point_clouds, actions))

    W = len(windows)
    if W == 0:
        # Still need to update cache for episodes with no phases
        if phase_cache is not None:
            for ep_idx in uncached_indices:
                ep_cache_entry = {}
                for phase_name in PHASE_NAMES:
                    ep_cache_entry[phase_name] = None
                phase_cache["episodes"][ep_paths[ep_idx]] = ep_cache_entry
        return phase_results, phase_counts

    cprint(f"    {W} phase windows collected, batched inference...", "cyan")

    # Phase 2: Initialize tracked state (on GPU; window data stays on CPU)
    cur_states = torch.stack([w[3][w[2]].clone() for w in windows]).to(device)
    cur_affs = torch.stack([w[4][w[2]].clone() for w in windows]).to(device)
    aff_starts = cur_affs.clone()

    all_aff_errors = torch.zeros(W, window_steps, device=device)
    all_gt_displacements = torch.zeros(W, window_steps, device=device)
    all_agent_errors = torch.zeros(W, window_steps, device=device) if predict_agent_state else None

    # Phase 3: Batched step-by-step
    lwm.eval()
    use_amp = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for k in range(window_steps):
            for b_start in range(0, W, inference_batch_size):
                b_end = min(b_start + inference_batch_size, W)
                bs = slice(b_start, b_end)
                batch_windows = windows[b_start:b_end]

                batch_cur_states = cur_states[bs]
                batch_cur_affs = cur_affs[bs]

                if n_obs == 1:
                    pcds = torch.stack([w[5][w[2] + k] for w in batch_windows]).to(device)
                    act_h = torch.stack([w[6][w[2] + k : w[2] + k + horizon] for w in batch_windows]).to(device)
                    act_h_norm = (act_h - action_mean) / action_std

                    if needs_obs_dim:
                        observation = {
                            "agent_state": batch_cur_states.unsqueeze(1),
                            "static_affordance": batch_cur_affs.unsqueeze(1),
                            "pointcloud": pcds.unsqueeze(1),
                            "action_horizon": act_h_norm,
                        }
                    else:
                        observation = {
                            "agent_state": batch_cur_states,
                            "static_affordance": batch_cur_affs,
                            "pointcloud": pcds,
                            "action_horizon": act_h_norm,
                        }
                else:
                    obs_s, obs_a, obs_p, obs_act = [], [], [], []
                    for w in batch_windows:
                        t = w[2] + k
                        obs_start = max(0, t - n_obs + 1)
                        s = w[3][obs_start:t + 1]
                        a = w[4][obs_start:t + 1]
                        p = w[5][obs_start:t + 1]
                        if s.shape[0] < n_obs:
                            pad = n_obs - s.shape[0]
                            s = torch.cat([w[3][0:1].expand(pad, -1), s])
                            a = torch.cat([w[4][0:1].expand(pad, -1), a])
                            p = torch.cat([w[5][0:1].expand(pad, -1, -1), p])
                        obs_s.append(s)
                        obs_a.append(a)
                        obs_p.append(p)
                        obs_act.append((w[6][t:t + horizon] - action_mean) / action_std)

                    observation = {
                        "agent_state": torch.stack(obs_s).to(device),
                        "static_affordance": torch.stack(obs_a).to(device),
                        "pointcloud": torch.stack(obs_p).to(device),
                        "action_horizon": torch.stack(obs_act).to(device),
                    }

                gt_states_next = torch.stack([
                    w[3][w[2] + k + 1] for w in batch_windows
                ]).to(device) if not predict_agent_state else None

                next_states, next_affs = _batched_step(
                    lwm, observation, batch_cur_states, batch_cur_affs,
                    agent_delta_mean, agent_delta_std, aff_delta_mean, aff_delta_std,
                    predict_agent_state, compose_affordance_fn,
                    gt_states_next=gt_states_next,
                )

                gt_affs = torch.stack([w[4][w[2] + k + 1] for w in batch_windows]).to(device)
                all_aff_errors[bs, k] = torch.norm(next_affs - gt_affs, dim=-1) * 100.0
                all_gt_displacements[bs, k] = torch.norm(gt_affs - aff_starts[bs], dim=-1) * 100.0

                if predict_agent_state:
                    gt_states = torch.stack([w[3][w[2] + k + 1] for w in batch_windows]).to(device)
                    all_agent_errors[bs, k] = torch.norm(next_states[:, :3] - gt_states[:, :3], dim=-1) * 100.0

                cur_states[bs] = next_states.detach()
                cur_affs[bs] = next_affs.detach()

    # Phase 4: Distribute results and update cache
    for ep_idx in uncached_indices:
        ep_cache_entry = {}
        for phase_name in PHASE_NAMES:
            w_idx = ep_phase_map[ep_idx][phase_name]
            if w_idx is None:
                ep_cache_entry[phase_name] = None
                continue

            win = {
                "aff_errors": all_aff_errors[w_idx].cpu().tolist(),
                "gt_displacements": all_gt_displacements[w_idx].cpu().tolist(),
                "ep_idx": ep_idx,
                "phase": phase_name,
                "t_start": windows[w_idx][2],
            }
            if predict_agent_state:
                win["agent_errors"] = all_agent_errors[w_idx].cpu().tolist()
            phase_results[phase_name].append(win)

            # Cache without ep_idx
            ep_cache_entry[phase_name] = {k: v for k, v in win.items() if k != "ep_idx"}

        if phase_cache is not None:
            phase_cache["episodes"][ep_paths[ep_idx]] = ep_cache_entry

    return phase_results, phase_counts


def aggregate_phase_results(phase_results, phase_name, episode_indices=None, window_steps=18):
    """
    Aggregate phase eval results into per-step statistics.
    Same output format as aggregate_windowed_results.
    """
    windows = phase_results.get(phase_name, [])
    if episode_indices is not None:
        ep_set = set(episode_indices)
        windows = [w for w in windows if w["ep_idx"] in ep_set]

    if not windows:
        return None

    has_agent = "agent_errors" in windows[0]

    per_step_aff = [[] for _ in range(window_steps)]
    per_step_gt_disp = [[] for _ in range(window_steps)]
    per_step_agent = [[] for _ in range(window_steps)]

    for w in windows:
        for k in range(window_steps):
            per_step_aff[k].append(w["aff_errors"][k])
            per_step_gt_disp[k].append(w["gt_displacements"][k])
            if has_agent:
                per_step_agent[k].append(w["agent_errors"][k])

    return {
        "step_indices": list(range(1, window_steps + 1)),
        "avg_aff_drift_cm": [float(np.mean(e)) for e in per_step_aff],
        "std_aff_drift_cm": [float(np.std(e)) for e in per_step_aff],
        "avg_gt_displacement_cm": [float(np.mean(d)) for d in per_step_gt_disp],
        "avg_agent_drift_cm": [float(np.mean(e)) for e in per_step_agent] if has_agent else None,
        "std_agent_drift_cm": [float(np.std(e)) for e in per_step_agent] if has_agent else None,
        "num_windows": len(windows),
    }
