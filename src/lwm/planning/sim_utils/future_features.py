"""Utilities for future timesteps and features."""
import numpy as np
import torch


def stack_for_obs_steps(data, n_obs_steps, axis=0):
    """Stack data for n_obs_steps."""
    return np.stack([data] * n_obs_steps, axis=axis)


def get_future_feature_config(cfg):
    """Extract future feature configuration from policy config. use_wm enables both future_static_affordance and future_agent_pos."""
    use_wm = cfg.policy.get("use_wm", False)
    n_future_steps = cfg.policy.get("n_future_steps", 0)
    
    # Support both lwm_encoder_cfg (new) and act3d_encoder_cfg (backward compatibility)
    encoder_cfg = getattr(cfg.policy, "lwm_encoder_cfg", None) or getattr(cfg.policy, "act3d_encoder_cfg", None)
    if encoder_cfg is not None:
        use_wm = use_wm or encoder_cfg.get("use_wm", False)
        if n_future_steps == 0:
            n_future_steps = encoder_cfg.get("n_future_steps", 0)
    
    return {
        "use_wm": use_wm,
        "n_future_steps": n_future_steps,
    }


def initialize_future_timesteps(obs, cfg, n_obs_steps, info=None):
    """Initialize future timesteps in observation. If use_wm=True, initializes both future_static_affordance and future_agent_pos."""
    future_config = get_future_feature_config(cfg)
    if not future_config["use_wm"]:
        return obs
    
    n_future_steps = future_config["n_future_steps"]
    
    current_affordance = obs.get("static_affordance", np.zeros((n_obs_steps, 3), dtype=np.float32))
    future_static_affordance = stack_for_obs_steps(current_affordance, n_future_steps, axis=1)
    current_agent_pos = obs.get("agent_pos", np.zeros((n_obs_steps, 10), dtype=np.float32))
    future_agent_pos = stack_for_obs_steps(current_agent_pos, n_future_steps, axis=1)
    
    obs["future_static_affordance"] = future_static_affordance
    obs["future_agent_pos"] = future_agent_pos.astype(np.float32)
    
    return obs


def pad_future_feature_if_needed(feature, expected_steps, feature_name):
    """Pad future feature by repeating last timestep if it falls short of expected steps.

    Args:
        feature: np.ndarray of shape (actual_steps, feature_dim)
        expected_steps: int, number of expected timesteps
        feature_name: str, name for warning message

    Returns:
        np.ndarray of shape (expected_steps, feature_dim), padded if necessary
    """
    actual_steps = feature.shape[0]
    if actual_steps < expected_steps:
        print(f"[WARNING] {feature_name} has {actual_steps} steps but expected {expected_steps}. "
              f"Padding by repeating last timestep. This is a hacky workaround.")
        last_step = feature[-1:]
        padding = np.tile(last_step, (expected_steps - actual_steps, 1))
        feature = np.concatenate([feature, padding], axis=0)
    return feature


def update_future_timesteps_in_obs(obs, future_data, cfg, n_obs_steps):
    """Update future timesteps in observation with observed future data from simulator."""
    future_config = get_future_feature_config(cfg)
    if not future_config["use_wm"]:
        return obs

    expected_steps = future_config["n_future_steps"]

    if "future_static_affordance" in future_data:
        future_affordance = future_data["future_static_affordance"]
        future_affordance = pad_future_feature_if_needed(
            future_affordance, expected_steps, "future_static_affordance"
        )
        obs["future_static_affordance"] = stack_for_obs_steps(future_affordance, n_obs_steps).astype(np.float32)

    if "future_agent_pos" in future_data:
        future_agent_pos = future_data["future_agent_pos"]
        future_agent_pos = pad_future_feature_if_needed(
            future_agent_pos, expected_steps, "future_agent_pos"
        )
        obs["future_agent_pos"] = stack_for_obs_steps(future_agent_pos, n_obs_steps).astype(np.float32)

    return obs


def extract_future_features_with_lwm(lwm_predictor, obs, predicted_actions, n_future_steps):
    """Extract future features using a trained LWM instead of the simulator.

    This is the LWM equivalent of simulation.extract_future_features_with_saved_state().
    It predicts future affordance positions from the current observation and predicted
    actions, without modifying the environment.

    Args:
        lwm_predictor: WorldModelPredictor instance (loaded on CPU).
        obs: Current observation dict with keys:
            - agent_pos: [n_obs_steps, 10]
            - static_affordance: [n_obs_steps, 3]
            - part_point_cloud or point_cloud: [n_obs_steps, N, 3]
        predicted_actions: Actions predicted by policy, shape [n_action_steps, 10].
        n_future_steps: Number of future steps expected by the policy.

    Returns:
        dict with keys:
            - future_static_affordance: np.ndarray [n_future_steps, 3]
            - future_agent_pos: np.ndarray [n_future_steps, 10]
    """
    # Extract current state from last observation step
    agent_pos = obs["agent_pos"]
    if agent_pos.ndim == 2:
        agent_pos = agent_pos[-1]  # [10]

    static_affordance = obs["static_affordance"]
    if static_affordance.ndim == 2:
        static_affordance = static_affordance[-1]  # [3]

    # Prefer part_point_cloud (LWM trained on 512-point part clouds)
    if "part_point_cloud" in obs:
        point_cloud = obs["part_point_cloud"]
    else:
        point_cloud = obs["point_cloud"]
    if point_cloud.ndim == 3:
        point_cloud = point_cloud[-1]  # [N, 3]

    # Pad or truncate actions to n_future_steps
    if predicted_actions.shape[0] < n_future_steps:
        last_action = predicted_actions[-1:]
        padding = np.tile(last_action, (n_future_steps - predicted_actions.shape[0], 1))
        actions_for_lwm = np.concatenate([predicted_actions, padding], axis=0)
    else:
        actions_for_lwm = predicted_actions[:n_future_steps]

    # Convert to tensors on LWM device
    device = lwm_predictor.device
    agent_state_t = torch.from_numpy(agent_pos).float().unsqueeze(0).to(device)
    affordance_t = torch.from_numpy(static_affordance).float().unsqueeze(0).to(device)
    point_cloud_t = torch.from_numpy(point_cloud).float().unsqueeze(0).to(device)
    actions_t = torch.from_numpy(actions_for_lwm).float().unsqueeze(0).to(device)

    # Run LWM autoregressive prediction
    with torch.no_grad():
        result = lwm_predictor.predict_trajectory(
            agent_state=agent_state_t,
            affordance=affordance_t,
            point_cloud=point_cloud_t,
            actions=actions_t,
        )

    # [1, n_future_steps, dim] -> [n_future_steps, dim]
    future_affordances = result["predicted_affordances"][0].cpu().numpy()
    future_agent_pos = result["predicted_states"][0].cpu().numpy()

    return {
        "future_static_affordance": future_affordances.astype(np.float32),
        "future_agent_pos": future_agent_pos.astype(np.float32),
    }

