"""
Utility functions for evaluation during LWM training.
Contains shared code for building observations, normalizing, and predicting states.
"""

import torch
import numpy as np

from lwm.world_model.targets import decode_affordance_delta, observation_centroid


def rotation_transfer_6D_to_matrix_torch(orient):
    """Convert 6D rotation representation to 3x3 rotation matrix (PyTorch version)."""
    # orient: [6] or [B, 6]
    if orient.dim() == 1:
        orient = orient.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Reshape to [B, 2, 3]
    orient = orient.view(-1, 2, 3)
    a1 = orient[:, 0]  # [B, 3]
    a2 = orient[:, 1]  # [B, 3]
    
    # Gram-Schmidt orthogonalization
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Stack to create rotation matrix [B, 3, 3]
    rot_mat = torch.stack([b1, b2, b3], dim=-1)
    
    if squeeze_output:
        rot_mat = rot_mat.squeeze(0)
    return rot_mat


def rotation_transfer_matrix_to_6D_torch(rotate_matrix):
    """Convert 3x3 rotation matrix to 6D representation (PyTorch version)."""
    # rotate_matrix: [3, 3] or [B, 3, 3]
    if rotate_matrix.dim() == 2:
        rotate_matrix = rotate_matrix.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Extract first two columns
    a1 = rotate_matrix[:, :, 0]  # [B, 3]
    a2 = rotate_matrix[:, :, 1]  # [B, 3]
    
    result = torch.cat([a1, a2], dim=-1)  # [B, 6]
    
    if squeeze_output:
        result = result.squeeze(0)
    return result


def compose_state_with_delta(cur_state, pred_delta):
    """
    Compose current state with predicted delta, handling rotation correctly.
    
    State format: [pos(3), ori_6d(6), finger(1)] = 10D
    
    For position and finger: simple addition
    For orientation: proper rotation composition via matrix multiplication
    
    Args:
        cur_state: [state_dim] current state
        pred_delta: [state_dim] predicted delta
        
    Returns:
        next_state: [state_dim] properly composed next state
    """
    next_state = cur_state.clone()
    
    # Position (indices 0:3) - simple addition
    next_state[:3] = cur_state[:3] + pred_delta[:3]
    
    # Finger (index 9) - simple addition
    next_state[9] = cur_state[9] + pred_delta[9]
    
    # Orientation (indices 3:9) - proper rotation composition
    # cur_ori_mat @ delta_ori_mat = next_ori_mat
    cur_ori_6d = cur_state[3:9]
    delta_ori_6d = pred_delta[3:9]
    
    cur_ori_mat = rotation_transfer_6D_to_matrix_torch(cur_ori_6d)
    delta_ori_mat = rotation_transfer_6D_to_matrix_torch(delta_ori_6d)
    
    # Compose rotations: R_next = R_cur @ R_delta
    next_ori_mat = cur_ori_mat @ delta_ori_mat
    next_ori_6d = rotation_transfer_matrix_to_6D_torch(next_ori_mat)
    
    next_state[3:9] = next_ori_6d
    
    return next_state


def compose_affordance_with_9dof_delta(cur_aff, pred_delta_9d):
    """
    Apply 9-DoF rigid body transform to affordance point.

    The 9-DoF delta encodes how the part point cloud moves between timesteps.
    Since the affordance point lies on the part, it moves by the same transform:
        next_aff = R @ cur_aff + t

    Args:
        cur_aff: [3] current affordance point position (torch tensor)
        pred_delta_9d: [9] = [rot_6d(6), translation(3)] (torch tensor)

    Returns:
        [3] next affordance point position
    """
    rot_6d = pred_delta_9d[:6]
    t = pred_delta_9d[6:]
    R = rotation_transfer_6D_to_matrix_torch(rot_6d)  # [3, 3]
    return R @ cur_aff + t


def compose_state_with_delta_batched(cur_states, pred_deltas):
    """Batched version of compose_state_with_delta. [B, 10] x [B, 10] -> [B, 10]."""
    next_states = cur_states.clone()
    next_states[:, :3] = cur_states[:, :3] + pred_deltas[:, :3]
    next_states[:, 9] = cur_states[:, 9] + pred_deltas[:, 9]

    cur_ori_mat = rotation_transfer_6D_to_matrix_torch(cur_states[:, 3:9])    # [B, 3, 3]
    delta_ori_mat = rotation_transfer_6D_to_matrix_torch(pred_deltas[:, 3:9])  # [B, 3, 3]
    next_ori_mat = torch.bmm(cur_ori_mat, delta_ori_mat)                       # [B, 3, 3]
    next_states[:, 3:9] = rotation_transfer_matrix_to_6D_torch(next_ori_mat)   # [B, 6]
    return next_states


def compose_affordance_with_9dof_delta_batched(cur_affs, pred_deltas_9d):
    """Batched version of compose_affordance_with_9dof_delta. [B, 3] x [B, 9] -> [B, 3]."""
    R = rotation_transfer_6D_to_matrix_torch(pred_deltas_9d[:, :6])  # [B, 3, 3]
    t = pred_deltas_9d[:, 6:]                                         # [B, 3]
    return torch.bmm(R, cur_affs.unsqueeze(-1)).squeeze(-1) + t       # [B, 3]


def get_normalization_stats(lwm_dataset, device, action_mean=None, action_std=None):
    """
    Get normalization statistics from dataset.
    
    Returns:
        Tuple of (action_mean, action_std, agent_state_delta_mean, agent_state_delta_std,
                  affordance_delta_mean, affordance_delta_std)
    """
    if action_mean is None:
        action_mean = torch.from_numpy(lwm_dataset.action_mean).float().to(device)
    if action_std is None:
        action_std = torch.from_numpy(lwm_dataset.action_std).float().to(device)
    
    agent_state_delta_mean = torch.from_numpy(lwm_dataset.agent_state_delta_mean).float().to(device)
    agent_state_delta_std = torch.from_numpy(lwm_dataset.agent_state_delta_std).float().to(device)
    affordance_delta_mean = torch.from_numpy(lwm_dataset.affordance_delta_mean).float().to(device)
    affordance_delta_std = torch.from_numpy(lwm_dataset.affordance_delta_std).float().to(device)
    
    return (action_mean, action_std, agent_state_delta_mean, agent_state_delta_std,
            affordance_delta_mean, affordance_delta_std)


def load_episode_data(replay_buffer, ep_idx, device):
    """
    Load full episode data from replay buffer.

    Returns:
        Tuple of (agent_states, static_affordances, point_clouds, actions, ep_len)
    """
    ep_len = int(replay_buffer.episode_lengths[ep_idx])
    zarr_path = replay_buffer.all_path_list[ep_idx]

    data = replay_buffer._get_data_single_file(zarr_path, 0, ep_len)
    agent_states = torch.from_numpy(data["state"]).float().to(device)
    static_affordances = torch.from_numpy(data["static_affordance"]).float().to(device)
    point_clouds = torch.from_numpy(data["part_point_cloud"]).float().to(device)
    actions = torch.from_numpy(data["action"]).float().to(device)

    return agent_states, static_affordances, point_clouds, actions, ep_len


def preload_episodes(replay_buffer, episode_indices, device):
    """
    Preload all episode data into memory at once.

    Returns:
        dict mapping ep_idx -> (agent_states, static_affordances, point_clouds, actions, ep_len)
    """
    from tqdm import tqdm
    from termcolor import cprint

    unique_indices = sorted(set(episode_indices))
    cprint(f"Preloading {len(unique_indices)} episodes to CPU...", "cyan")
    preloaded = {}
    for ep_idx in tqdm(unique_indices, desc="Preloading episodes"):
        preloaded[ep_idx] = load_episode_data(replay_buffer, ep_idx, device="cpu")
    cprint(f"Preloaded {len(preloaded)} episodes to CPU.", "green")
    return preloaded


def get_episode_data(replay_buffer, ep_idx, device, preloaded_data=None):
    """
    Get episode data from preloaded cache or load from disk.

    Returns:
        Tuple of (agent_states, static_affordances, point_clouds, actions, ep_len)
    """
    if preloaded_data is not None and ep_idx in preloaded_data:
        # Preloaded data lives on CPU; move to device on access
        agent_states, static_affordances, point_clouds, actions, ep_len = preloaded_data[ep_idx]
        return (
            agent_states.to(device),
            static_affordances.to(device),
            point_clouds.to(device),
            actions.to(device),
            ep_len,
        )
    return load_episode_data(replay_buffer, ep_idx, device)


def build_observation(
    t,
    cur_state,
    cur_aff,
    agent_states,
    static_affordances,
    point_clouds,
    action_horizon_normalized,
    n_obs_steps,
    horizon,
):
    """
    Build observation dictionary for model input.
    
    Handles both single and multiple observation steps with proper padding.
    
    Args:
        t: Current timestep
        cur_state: Current predicted agent state [agent_state_dim]
        cur_aff: Current predicted affordance [3]
        agent_states: Ground truth agent states [T, agent_state_dim] (for history)
        static_affordances: Ground truth affordances [T, 3] (for history)
        point_clouds: Point clouds [T, N, 3]
        action_horizon_normalized: Normalized action horizon [horizon, action_dim]
        n_obs_steps: Number of observation steps
        horizon: Horizon length
    
    Returns:
        Observation dict with keys: agent_state, static_affordance, pointcloud, action_horizon
    """
    if n_obs_steps == 1:
        # Single timestep
        obs_agent_state = cur_state.unsqueeze(0)  # [1, agent_state_dim]
        obs_pointcloud = point_clouds[t].unsqueeze(0)  # [1, N, 3]
        obs_affordance = cur_aff.unsqueeze(0)  # [1, 3]
    else:
        # Multiple timesteps: get history from t-n_obs_steps+1 to t
        obs_start = max(0, t - n_obs_steps + 1)
        obs_end = t + 1
        if obs_start == 0:
            # Not enough history, pad with first timestep
            obs_agent_states = agent_states[:obs_end]  # [obs_end, agent_state_dim]
            obs_pointclouds = point_clouds[:obs_end]  # [obs_end, N, 3]
            obs_affordances = static_affordances[:obs_end]  # [obs_end, 3]
            if obs_agent_states.shape[0] < n_obs_steps:
                padding_size = n_obs_steps - obs_agent_states.shape[0]
                first_state = agent_states[0:1]  # [1, agent_state_dim]
                first_pc = point_clouds[0:1]  # [1, N, 3]
                first_aff = static_affordances[0:1]  # [1, 3]
                obs_agent_states = torch.cat([first_state.repeat(padding_size, 1), obs_agent_states], dim=0)
                obs_pointclouds = torch.cat([first_pc.repeat(padding_size, 1, 1), obs_pointclouds], dim=0)
                obs_affordances = torch.cat([first_aff.repeat(padding_size, 1), obs_affordances], dim=0)
        else:
            obs_agent_states = agent_states[obs_start:obs_end]  # [n_obs_steps, agent_state_dim]
            obs_pointclouds = point_clouds[obs_start:obs_end]  # [n_obs_steps, N, 3]
            obs_affordances = static_affordances[obs_start:obs_end]  # [n_obs_steps, 3]
        obs_agent_state = obs_agent_states.unsqueeze(0)  # [1, n_obs_steps, agent_state_dim]
        obs_pointcloud = obs_pointclouds.unsqueeze(0)  # [1, n_obs_steps, N, 3]
        obs_affordance = obs_affordances.unsqueeze(0)  # [1, n_obs_steps, 3]
    
    observation = {
        "agent_state": obs_agent_state,
        "static_affordance": obs_affordance,
        "pointcloud": obs_pointcloud,
        "action_horizon": action_horizon_normalized.unsqueeze(0),  # [1, horizon, action_dim]
    }
    
    return observation


def predict_next_state_and_affordance(
    lwm,
    observation,
    cur_state,
    cur_aff,
    agent_state_delta_mean,
    agent_state_delta_std,
    affordance_delta_mean,
    affordance_delta_std,
    predict_agent_state,
    compose_state_with_delta_fn,
    gt_state_fallback=None,
    compose_affordance_fn=None,
):
    """
    Predict next state and affordance from model output.

    Args:
        lwm: The LWM model
        observation: Observation dict
        cur_state: Current state [agent_state_dim]
        cur_aff: Current affordance [3]
        agent_state_delta_mean: Agent state delta mean for denormalization
        agent_state_delta_std: Agent state delta std for denormalization
        affordance_delta_mean: Affordance delta mean for denormalization
        affordance_delta_std: Affordance delta std for denormalization
        predict_agent_state: Whether to predict agent state
        compose_state_with_delta_fn: Function to compose state with delta
        gt_state_fallback: Optional ground truth state to use if predict_agent_state=False
        compose_affordance_fn: Optional function to compose affordance with delta.
            If None, uses simple addition (cur_aff + delta).
            For v8 9-DoF, use compose_affordance_with_9dof_delta.

    Returns:
        Tuple of (next_state, next_aff, pred_aff_delta_normalized, pred_agent_delta_normalized)
    """
    output = lwm(observation)

    # Take first-step delta (normalized)
    pred_aff_delta_normalized = output["affordance_delta"][0, 0]  # [affordance_output_dim]

    # Denormalize deltas
    pred_aff_delta = pred_aff_delta_normalized * affordance_delta_std + affordance_delta_mean
    # Model frame -> world form (no-op for affordance_target_frame="world")
    frame = getattr(lwm, "affordance_target_frame", "world")
    if frame != "world" and pred_aff_delta.shape[-1] == 9:
        pred_aff_delta = decode_affordance_delta(
            pred_aff_delta.reshape(1, 1, 9), observation_centroid(observation)[:1], frame
        ).reshape(9)

    if predict_agent_state:
        pred_agent_delta_normalized = output["agent_state_delta"][0, 0]  # [agent_state_dim]
        pred_agent_delta = pred_agent_delta_normalized * agent_state_delta_std + agent_state_delta_mean
        # Compose state with delta
        if compose_state_with_delta_fn is not None:
            next_state = compose_state_with_delta_fn(cur_state, pred_agent_delta)
        else:
            # Fallback: simple addition (not correct for rotation but better than error)
            next_state = cur_state + pred_agent_delta
    else:
        pred_agent_delta_normalized = None
        if gt_state_fallback is not None:
            next_state = gt_state_fallback
        else:
            next_state = cur_state  # No change if no fallback

    # Compose affordance with delta
    if compose_affordance_fn is not None:
        next_aff = compose_affordance_fn(cur_aff, pred_aff_delta)
    else:
        next_aff = cur_aff + pred_aff_delta

    return next_state, next_aff, pred_aff_delta_normalized, pred_agent_delta_normalized


def get_near_affordance_step_indices(
    episode_indices,
    lwm_dataset,
    distance_threshold=0.2,
    rollout_length=18,
    mode="all",  # "all", "first_contact", or "initial_phase"
    initial_phase_window=16,  # Steps after first contact to include
    initial_phase_stride=4,  # Stride for sampling within initial phase
):
    """
    Pre-compute step indices where end-effector is within distance_threshold of affordance point.

    Args:
        episode_indices: List of episode indices to process
        lwm_dataset: LWMDataset instance
        distance_threshold: Distance threshold in meters (default 0.2m = 20cm)
        rollout_length: Length of rollout to perform (default 18 steps)
        mode: "all" = all near-affordance steps (original behavior)
              "first_contact" = only the first timestep within threshold
              "initial_phase" = first contact + a few steps after with stride
        initial_phase_window: How many steps after first contact to consider (default 16)
        initial_phase_stride: Stride for sampling within initial phase (default 4)

    Returns:
        dict[int, list[int]]: Mapping from episode_idx to list of start timesteps
    """
    if len(episode_indices) == 0:
        return {}

    rb = lwm_dataset.replay_buffer
    step_indices_dict = {}

    for ep_idx in episode_indices:
        ep_len = int(rb.episode_lengths[ep_idx])
        if ep_len < rollout_length + 1:
            continue  # Episode too short for rollout

        # Load ground truth data
        agent_states, static_affordances, _, _, _ = load_episode_data(rb, ep_idx, "cpu")

        # Extract end-effector positions (first 3 dims)
        ee_positions = agent_states[:, :3]  # [T, 3]

        # Compute distances between EE and affordance point
        distances = torch.norm(ee_positions - static_affordances, dim=-1)  # [T]

        # Find timesteps where distance < threshold and we can do full rollout
        timestep_indices = torch.arange(ep_len, dtype=torch.long)
        valid_mask = (distances < distance_threshold) & (timestep_indices <= ep_len - rollout_length)
        valid_steps = torch.where(valid_mask)[0].tolist()

        if len(valid_steps) == 0:
            continue

        if mode == "all":
            # Original behavior: all near-affordance steps
            step_indices_dict[ep_idx] = valid_steps
        elif mode == "first_contact":
            # Only the first timestep within threshold
            step_indices_dict[ep_idx] = [valid_steps[0]]
        elif mode == "initial_phase":
            # First contact + a few steps after with stride
            first_contact = valid_steps[0]
            selected_steps = [first_contact]

            # Add steps at stride intervals within the window
            for offset in range(initial_phase_stride, initial_phase_window + 1, initial_phase_stride):
                step = first_contact + offset
                if step in valid_steps:  # Make sure it's still within threshold and valid
                    selected_steps.append(step)

            step_indices_dict[ep_idx] = selected_steps

    return step_indices_dict
