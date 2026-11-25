"""Utilities for observation processing and validation."""
import numpy as np
from termcolor import cprint

# IMPORTANT: use a single, consistent implementation for handle / link utilities
# across evaluation code so that:
#   - handle selection (closest vs link_0) is done once per episode
#   - all consumers (metrics, static affordance, part point cloud) see the same
#     handle ordering and joint IDs.
#
# We therefore import get_handle_pos / get_link_handle / get_link_pc from
# manipulation.utils, which is also used by eval.utils.part_pointcloud.
from lwm.sim.utils import get_handle_pos, get_link_handle, get_link_pc

from lwm.planning.sim_utils.future_features import stack_for_obs_steps, get_future_feature_config
from lwm.planning.sim_utils.part_pointcloud import get_closest_handle_to_eef
from lwm.planning.sim_utils.articulation_points import (
    load_articulation_points_json,
    sample_articulation_point,
    get_handle_link_transform_from_joint,
    transform_handle_local_to_world,
)


def assert_observation_shapes(obs, n_obs_steps, context=""):
    """Assert that observation has expected shapes."""
    if "agent_pos" in obs:
        agent_pos = obs["agent_pos"]
        if agent_pos.ndim == 2:
            assert agent_pos.shape == (n_obs_steps, 10), f"{context}: agent_pos shape {agent_pos.shape} != ({n_obs_steps}, 10)"
        elif agent_pos.ndim == 1:
            assert agent_pos.shape == (10,), f"{context}: agent_pos shape {agent_pos.shape} != (10,)"
    
    if "gripper_pcd" in obs:
        gp = obs["gripper_pcd"]
        if gp.ndim == 3:
            assert gp.shape[1:] == (4, 3), f"{context}: gripper_pcd shape {gp.shape} should have (4, 3) at end"
        elif gp.ndim == 2:
            assert gp.shape == (4, 3), f"{context}: gripper_pcd shape {gp.shape} != (4, 3)"
    
    if "static_affordance" in obs:
        sa = obs["static_affordance"]
        if sa.ndim == 2:
            assert sa.shape == (n_obs_steps, 3), f"{context}: static_affordance shape {sa.shape} != ({n_obs_steps}, 3)"
        elif sa.ndim == 1:
            assert sa.shape == (3,), f"{context}: static_affordance shape {sa.shape} != (3,)"


def get_gripper_ee_pos(obs, context=""):
    """Extract end-effector position from observation's gripper_pcd."""
    if "gripper_pcd" not in obs:
        cprint(f"{context}: No gripper_pcd in obs, using zero position", "yellow")
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    gp = obs["gripper_pcd"]
    if gp.ndim == 3:
        gp = gp[-1]  # Take last observation step
    
    assert gp.ndim == 2 and gp.shape[0] >= 4, f"{context}: gripper_pcd shape {gp.shape} invalid"
    return gp[3, :3].astype(np.float32)


def get_ee_pos_for_verification(env):
    """Get current EE position for state verification."""
    pos, _ = env.env._env.robot.get_pos_orient(env.env._env.robot.right_end_effector)
    return np.array(pos, dtype=np.float32)


def normalize_handle_pos(handle_pos):
    """Normalize handle position to shape (3,)."""
    handle_pos = np.asarray(handle_pos)
    if handle_pos.shape == (3,):
        return handle_pos
    elif handle_pos.shape == (1, 3):
        return handle_pos[0]
    elif len(handle_pos.shape) == 2:
        return handle_pos[0] if handle_pos.shape[0] > 0 else handle_pos.flatten()[:3]
    else:
        return handle_pos.flatten()[:3].reshape(3,)


def normalize_static_affordance(static_aff):
    """Normalize static affordance to shape (3,)."""
    if isinstance(static_aff, np.ndarray) and static_aff.shape != (3,):
        return static_aff.flatten()[:3]
    return static_aff


def sample_centerline_handle_point(points, k=10):
    """Select a point in the middle of the handle, directly on the handle surface.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) containing the handle point cloud.
    k : int
        Number of neighbouring points around the centerline point to average over.

    Returns
    -------
    np.ndarray
        A single point of shape (3,), approximately in the middle of the
        handle in y, and averaged in x/z over its k nearest neighbours.
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 3:
        pts = pts.reshape(-1, 3)

    # 1) Find the \"middle\" in y (between min and max y)
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])
    target_y = 0.5 * (y_min + y_max)

    # 2) Find the handle point whose y is closest to this middle y
    y_dists = np.abs(pts[:, 1] - target_y)
    center_idx = int(np.argmin(y_dists))

    # 3) Take k nearest neighbours around this center point in Euclidean space
    diffs = pts - pts[center_idx]
    dists = np.linalg.norm(diffs, axis=1)
    neighbour_indices = np.argsort(dists)[: min(k, pts.shape[0])]
    neighbours = pts[neighbour_indices]

    # 4) Average x and z over the neighbours, keep y at the center point
    mean_x = float(np.mean(neighbours[:, 0]))
    mean_z = float(np.mean(neighbours[:, 2]))
    center_y = float(pts[center_idx, 1])

    return np.array([mean_x, center_y, mean_z], dtype=pts.dtype)


def sample_lowest_x_middle_yz_point(handle_points):
    """Select affordance point using lowest x coordinate with middle y and z.
    
    Parameters
    ----------
    handle_points : np.ndarray
        Array of shape (N, 3) containing the handle point cloud.
        
    Returns
    -------
    np.ndarray
        A single point of shape (3,), the point closest to (lowest_x, middle_y, middle_z)
    """
    pts = np.asarray(handle_points)
    if pts.ndim != 2 or pts.shape[1] < 3:
        pts = pts.reshape(-1, 3)
    
    # Calculate min/max for y and z
    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])
    middle_y = 0.5 * (y_min + y_max)
    
    z_min = np.min(pts[:, 2])
    z_max = np.max(pts[:, 2])
    middle_z = 0.5 * (z_min + z_max)
    
    # Find point with lowest x coordinate
    lowest_x_idx = int(np.argmin(pts[:, 0]))
    lowest_x = float(pts[lowest_x_idx, 0])
    
    # Target point: (lowest_x, middle_y, middle_z)
    target_point = np.array([lowest_x, middle_y, middle_z], dtype=pts.dtype)
    
    # Find closest point in handle point cloud to target
    distances = np.linalg.norm(pts - target_point, axis=1)
    closest_idx = int(np.argmin(distances))
    
    return pts[closest_idx]


def get_handle_pos_from_env(env, object_name, use_closest_handle=False, extracted_dataset_path=None, affordance_selection_method="lowest_x_middle_yz"):
    """Get handle position using link_0 selection logic or closest handle selection.

    This function is called at every timestep. We:
    1. Select which handle to use ONCE at the start (cache the index)
    2. Cache the handle points in object frame and mobility info
    3. Select affordance point ONCE at the start using the specified method
    4. Cache the index of the selected point
    5. Each timestep, just look up that index in the new handle point cloud (so the point tracks drawer movement)

    Parameters
    ----------
    env : MultiStepWrapper
        The wrapped environment.
    object_name : str
        Name of the articulated object.
    use_closest_handle : bool
        If False (default), select handle using link_0-based logic (get_link_handle).
        If True, select handle closest to end-effector (get_closest_handle_to_eef).
    extracted_dataset_path : str, optional
        Path to extracted dataset directory (where articulation_points.json is located).
        If None, will try to get from base_env._extracted_dataset_path.
    affordance_selection_method : str
        Method to use for selecting affordance point. Options:
        - "lowest_x_middle_yz" (default): Use heuristic: lowest x with middle y and z
        - "articulation_points": Sample from saved articulation points
        - "lowest_x_middle_yz": Use lowest x coordinate with middle y and z
    """
    base_env = env.env._env if hasattr(env, "env") and hasattr(env.env, "_env") else env
    
    # Get extracted_dataset_path from base_env if not provided
    if extracted_dataset_path is None and hasattr(base_env, "_extracted_dataset_path"):
        extracted_dataset_path = base_env._extracted_dataset_path

    # Cache attribute names for handle selection and object-frame data
    cached_idx_attr = (
        "_static_affordance_handle_idx_closest"
        if use_closest_handle
        else "_static_affordance_handle_idx_link0"
    )
    cached_obj_frame_attr = "_static_affordance_handle_pts_obj_frame"
    cached_mobility_attr = "_static_affordance_mobility_info"
    cached_articulation_point_local_attr = "_static_affordance_articulation_point_local"
    cached_handle_point_idx_attr = f"_static_affordance_handle_point_idx_{affordance_selection_method}"

    # If evaluation code has already chosen a global handle index (via
    # get_handle_joint_idx_from_env), respect that choice and propagate it
    # into this function's cached index attribute so that we never re-select
    # a different handle for static affordance.
    if hasattr(base_env, "_eval_min_link_idx") and not hasattr(base_env, cached_idx_attr):
        setattr(base_env, cached_idx_attr, base_env._eval_min_link_idx)

    # If we already know which handle index to use for this mode, just use it
    # with cached object-frame data (if available).
    if hasattr(base_env, cached_idx_attr):
        selected_handle_idx = getattr(base_env, cached_idx_attr)
        handle_pts_obj_frame = getattr(base_env, cached_obj_frame_attr, None)
        mobility_info = getattr(base_env, cached_mobility_attr, None)

        # If we don't yet have cached object-frame data, fetch it once.
        if handle_pts_obj_frame is None or mobility_info is None:
            _, _, handle_pts_obj_frame, mobility_info = get_handle_pos(
                base_env,
                object_name.lower(),
                return_median=False,
                return_info=True,
            )
            setattr(base_env, cached_obj_frame_attr, handle_pts_obj_frame)
            setattr(base_env, cached_mobility_attr, mobility_info)

        # Transform handle points into world frame using cached object-frame data.
        all_handle_pos, _ = get_handle_pos(
            base_env,
            object_name.lower(),
            return_median=False,
            handle_pts_obj_frame=handle_pts_obj_frame,
            mobility_info=mobility_info,
        )

        if len(all_handle_pos) == 0 or selected_handle_idx >= len(all_handle_pos):
            return None

        # Get the handle point cloud for the selected handle
        selected_handle_points = all_handle_pos[selected_handle_idx]
        if selected_handle_points.ndim == 1:
            selected_handle_points = selected_handle_points.reshape(1, -1)
        
        # Check if we have a cached handle point index
        # If so, just use that index to get the point from the handle point cloud
        handle_point_idx = getattr(base_env, cached_handle_point_idx_attr, None)
        if handle_point_idx is not None:
            if handle_point_idx < len(selected_handle_points):
                handle_point = selected_handle_points[handle_point_idx]
                base_env.static_affordance_handle_pos_closest = handle_point
                return handle_point
        
        # If no cached index, we need to select the point (first time only)
        if affordance_selection_method == "lowest_x_middle_yz":
            # Use the new method: lowest x with middle y and z
            handle_point = sample_lowest_x_middle_yz_point(selected_handle_points)
            # Find the index of this point (or closest match)
            distances = np.linalg.norm(selected_handle_points - handle_point, axis=1)
            handle_point_idx = int(np.argmin(distances))
            # Cache the index for future timesteps
            setattr(base_env, cached_handle_point_idx_attr, handle_point_idx)
            base_env.static_affordance_handle_pos_closest = handle_point
            cprint(
                f"[StaticAffordance] Selected point using lowest_x_middle_yz method: {handle_point.tolist()} (idx: {handle_point_idx})",
                "cyan",
            )
            return handle_point
        else:
            # Default: use articulation points
            selected_handle_joint_id = None
            if hasattr(base_env, "_eval_handle_joint"):
                selected_handle_joint_id = base_env._eval_handle_joint
            
            # Try to get object_id from experiment_folder
            object_id = None
            if hasattr(base_env, "_experiment_folder"):
                experiment_folder = base_env._experiment_folder
                # Extract object_id from experiment_folder (last part of path)
                object_id = experiment_folder.split("/")[-1]
            
            # Try to load and use saved articulation points
            if object_id is not None and extracted_dataset_path is not None and selected_handle_joint_id is not None:
                articulation_points = load_articulation_points_json(extracted_dataset_path)
                point_handle_local = sample_articulation_point(
                    articulation_points, object_id, selected_handle_joint_id
                )
                
                if point_handle_local is not None:
                    # Cache the sampled local point for future reference
                    setattr(base_env, cached_articulation_point_local_attr, point_handle_local)
                    # Transform to world coordinates
                    T_body_to_world = get_handle_link_transform_from_joint(
                        base_env, object_name.lower(), selected_handle_joint_id, mobility_info
                    )
                    handle_point_world = transform_handle_local_to_world(point_handle_local, T_body_to_world)
                    
                    # Find the closest point in the handle point cloud
                    distances = np.linalg.norm(selected_handle_points - handle_point_world, axis=1)
                    handle_point_idx = int(np.argmin(distances))
                    
                    # Cache the index for future timesteps
                    setattr(base_env, cached_handle_point_idx_attr, handle_point_idx)
                    
                    # Get the actual point from the handle point cloud
                    handle_point = selected_handle_points[handle_point_idx]
                    base_env.static_affordance_handle_pos_closest = handle_point
                    cprint(
                        f"[StaticAffordance] Sampled articulation point (local) for object {object_id}, joint {selected_handle_joint_id}: {point_handle_local.tolist()} | world: {handle_point_world.tolist()} | closest handle point idx: {handle_point_idx} | closest handle point: {handle_point.tolist()}",
                        "cyan",
                    )
                    return handle_point
            
            # Articulation point not found — fall back to lowest_x_middle_yz
            cprint(
                f"[StaticAffordance] WARNING: no articulation point for object_id={object_id}, "
                f"joint_id={selected_handle_joint_id} — falling back to lowest_x_middle_yz",
                "yellow",
            )
            handle_point = sample_lowest_x_middle_yz_point(selected_handle_points)
            distances = np.linalg.norm(selected_handle_points - handle_point, axis=1)
            handle_point_idx = int(np.argmin(distances))
            setattr(base_env, cached_handle_point_idx_attr, handle_point_idx)
            base_env.static_affordance_handle_pos_closest = handle_point
            return handle_point

    # ------------------------------------------------------------------
    # First time: select which handle to use and cache everything.
    # This is only reached if no global selection exists yet.
    # ------------------------------------------------------------------
    all_handle_pos, all_handle_joint_id, handle_pts_obj_frame, mobility_info = get_handle_pos(
        base_env, object_name.lower(), return_median=False, return_info=True
    )

    if len(all_handle_pos) == 0 or len(all_handle_joint_id) == 0:
        return None

    # Cache object-frame data for consistent tracking across timesteps
    setattr(base_env, cached_obj_frame_attr, handle_pts_obj_frame)
    setattr(base_env, cached_mobility_attr, mobility_info)

    if use_closest_handle:
        # Match demo-generation "closest handle" logic once at the start:
        # select the handle whose median is closest to the end-effector.
        robot_eef_pos, _ = base_env.robot.get_pos_orient(
            base_env.robot.right_end_effector
        )
        robot_eef_pos = np.array(robot_eef_pos)
        _, selected_handle_joint_id, _, selected_handle_idx = get_closest_handle_to_eef(
            all_handle_pos, all_handle_joint_id, robot_eef_pos
        )
        # Set on base_env so _get_info uses the same handle for joint angle reporting
        base_env._eval_handle_joint = selected_handle_joint_id
        base_env._eval_min_link_idx = selected_handle_idx
    else:
        # Default behaviour: use link_0-based selection (original implementation).
        link_pc = get_link_pc(base_env, object_name.lower(), "link_0")
        _, selected_handle_joint_id, _, selected_handle_idx = get_link_handle(
            all_handle_pos, all_handle_joint_id, link_pc
        )
        base_env._eval_handle_joint = selected_handle_joint_id
        base_env._eval_min_link_idx = selected_handle_idx

    # Cache the selected handle index for this mode
    setattr(base_env, cached_idx_attr, selected_handle_idx)

    # Get the handle point cloud for the selected handle
    selected_handle_points = all_handle_pos[selected_handle_idx]
    if selected_handle_points.ndim == 1:
        selected_handle_points = selected_handle_points.reshape(1, -1)
    
    # Select affordance point based on method
    if affordance_selection_method == "lowest_x_middle_yz":
        # Use the new method: lowest x with middle y and z
        handle_point = sample_lowest_x_middle_yz_point(selected_handle_points)
        # Find the index of this point (or closest match)
        distances = np.linalg.norm(selected_handle_points - handle_point, axis=1)
        handle_point_idx = int(np.argmin(distances))
        # Cache the index for future timesteps
        setattr(base_env, cached_handle_point_idx_attr, handle_point_idx)
        base_env.static_affordance_handle_pos_closest = handle_point
        cprint(
            f"[StaticAffordance] Selected point using lowest_x_middle_yz method: {handle_point.tolist()} (idx: {handle_point_idx})",
            "cyan",
        )
        return handle_point
    else:
        # Default: use articulation points
        # Try to get object_id from experiment_folder
        object_id = None
        if hasattr(base_env, "_experiment_folder"):
            experiment_folder = base_env._experiment_folder
            # Extract object_id from experiment_folder (last part of path)
            object_id = experiment_folder.split("/")[-1]
        
        # Try to load and use saved articulation points
        if object_id is not None and extracted_dataset_path is not None:
            articulation_points = load_articulation_points_json(extracted_dataset_path)
            point_handle_local = sample_articulation_point(
                articulation_points, object_id, selected_handle_joint_id
            )
            
            if point_handle_local is not None:
                # Cache the sampled local point for future reference
                setattr(base_env, cached_articulation_point_local_attr, point_handle_local)
                # Transform to world coordinates
                T_body_to_world = get_handle_link_transform_from_joint(
                    base_env, object_name.lower(), selected_handle_joint_id, mobility_info
                )
                handle_point_world = transform_handle_local_to_world(point_handle_local, T_body_to_world)
                
                # Find the closest point in the handle point cloud
                distances = np.linalg.norm(selected_handle_points - handle_point_world, axis=1)
                handle_point_idx = int(np.argmin(distances))
                
                # Cache the index for future timesteps
                setattr(base_env, cached_handle_point_idx_attr, handle_point_idx)
                
                # Get the actual point from the handle point cloud
                handle_point = selected_handle_points[handle_point_idx]
                base_env.static_affordance_handle_pos_closest = handle_point
                cprint(
                    f"[StaticAffordance] Sampled articulation point (local) for object {object_id}, joint {selected_handle_joint_id}: {point_handle_local.tolist()} | world: {handle_point_world.tolist()} | closest handle point idx: {handle_point_idx} | closest handle point: {handle_point.tolist()}",
                    "cyan",
                )
                return handle_point

        # Articulation point not found — fall back to lowest_x_middle_yz
        cprint(
            f"[StaticAffordance] WARNING: no articulation point for object_id={object_id}, "
            f"joint_id={selected_handle_joint_id} — falling back to lowest_x_middle_yz (first call)",
            "yellow",
        )
        handle_point = sample_lowest_x_middle_yz_point(selected_handle_points)
        distances = np.linalg.norm(selected_handle_points - handle_point, axis=1)
        handle_point_idx = int(np.argmin(distances))
        setattr(base_env, cached_handle_point_idx_attr, handle_point_idx)
        base_env.static_affordance_handle_pos_closest = handle_point
        return handle_point


def process_observation_for_model(obs, cfg, info=None, n_obs_steps=None, env=None, object_name=None, use_closest_handle=False, extracted_dataset_path=None, affordance_selection_method="lowest_x_middle_yz"):
    """Process observation to match model expectations.
    
    Parameters
    ----------
    obs : dict
        Observation dictionary.
    cfg : OmegaConf
        Configuration object.
    info : dict, optional
        Info dictionary from environment.
    n_obs_steps : int, optional
        Number of observation steps.
    env : MultiStepWrapper, optional
        Environment wrapper.
    object_name : str, optional
        Name of the articulated object.
    use_closest_handle : bool, optional
        If True, use closest handle selection for static affordance (matches closest_target_handle behavior).
    extracted_dataset_path : str, optional
        Path to extracted dataset directory (where articulation_points.json is located).
    """
    processed_obs = obs.copy()
    n_obs_steps = n_obs_steps or cfg.get("n_obs_steps", 2)
    
    # Use config value for use_part_point_cloud (should match training)
    use_part_point_cloud = cfg.task.dataset.get("use_part_point_cloud", False)
    
    if use_part_point_cloud:
        if "part_point_cloud" in processed_obs:
            part_pc = processed_obs["part_point_cloud"]
            if len(part_pc.shape) == 2:
                part_pc = stack_for_obs_steps(part_pc, n_obs_steps)
            processed_obs["point_cloud"] = part_pc
        else:
            cprint("Warning: use_part_point_cloud=True but part_point_cloud not found", "yellow")
    else:
        # When not using part point cloud, ensure we're using the full point cloud
        if "point_cloud" in processed_obs:
            # Ensure point_cloud has correct shape for n_obs_steps
            # MultiStepWrapper should already stack it, but check just in case
            pc = processed_obs["point_cloud"]
            if len(pc.shape) == 2:
                # Single timestep, needs stacking
                pc = stack_for_obs_steps(pc, n_obs_steps)
                processed_obs["point_cloud"] = pc
            elif len(pc.shape) == 3 and pc.shape[0] != n_obs_steps:
                # Wrong number of timesteps, this shouldn't happen but handle it
                cprint(f"Warning: point_cloud has shape {pc.shape} but expected ({n_obs_steps}, N, 3)", "yellow")

    # Check both policy.use_static_affordance and lwm_encoder_cfg.use_static_affordance (or act3d_encoder_cfg for backward compatibility)
    use_static_affordance = cfg.policy.get("use_static_affordance", False)
    encoder_cfg = getattr(cfg.policy, "lwm_encoder_cfg", None) or getattr(cfg.policy, "act3d_encoder_cfg", None)
    if encoder_cfg is not None:
        use_static_affordance = use_static_affordance or encoder_cfg.get("use_static_affordance", False)
    
    if use_static_affordance:
        handle_pos = None
        if env is not None and object_name is not None:
            handle_pos = get_handle_pos_from_env(env, object_name, use_closest_handle=use_closest_handle, extracted_dataset_path=extracted_dataset_path, affordance_selection_method=affordance_selection_method)
        if handle_pos is None:
            if info is not None and "handle_pos" in info:
                handle_pos = info["handle_pos"]
        
        if handle_pos is not None:
            handle_pos = normalize_handle_pos(handle_pos)
            processed_obs["static_affordance"] = stack_for_obs_steps(handle_pos, n_obs_steps).astype(np.float32)
        else:
            cprint("Warning: use_static_affordance=True but could not extract handle position", "yellow")
    
    if "affordance_point" in processed_obs and "static_affordance" in processed_obs:
        del processed_obs["affordance_point"]
    
    future_config = get_future_feature_config(cfg)
    if future_config["use_wm"]:
        if "future_static_affordance" in processed_obs:
            future_aff = processed_obs["future_static_affordance"]
            if len(future_aff.shape) == 2:
                processed_obs["future_static_affordance"] = stack_for_obs_steps(future_aff, n_obs_steps).astype(np.float32)
        if "future_agent_pos" in processed_obs:
            future_agent_pos = processed_obs["future_agent_pos"]
            if len(future_agent_pos.shape) == 2:
                processed_obs["future_agent_pos"] = stack_for_obs_steps(future_agent_pos, n_obs_steps).astype(np.float32)
    
    return processed_obs


def post_step_process_obs(obs, cfg, env, info, n_obs_steps, object_name, enable_future_features=False, use_closest_handle=False, extracted_dataset_path=None, affordance_selection_method="lowest_x_middle_yz", goal_static_affordance=None):
    """Process observation after env.reset() or env.step().

    Parameters
    ----------
    goal_static_affordance : np.ndarray, optional
        Goal affordance point of shape (3,). If provided, will be added to the observation
        dict as 'goal_static_affordance' with shape (n_obs_steps, 3) for policy consumption.
    """
    from lwm.planning.sim_utils.future_features import initialize_future_timesteps

    if "point_cloud" in obs:
        obs["full_point_cloud"] = obs["point_cloud"].copy()
    obs = process_observation_for_model(obs, cfg, info=info, n_obs_steps=n_obs_steps, env=env, object_name=object_name, use_closest_handle=use_closest_handle, extracted_dataset_path=extracted_dataset_path, affordance_selection_method=affordance_selection_method)
    if enable_future_features:
        obs = initialize_future_timesteps(obs, cfg, n_obs_steps, info=info)
        if "point_cloud" in obs:
            obs["full_point_cloud"] = obs["point_cloud"].copy()
        obs = process_observation_for_model(obs, cfg, info=info, n_obs_steps=n_obs_steps, env=env, object_name=object_name, use_closest_handle=use_closest_handle, extracted_dataset_path=extracted_dataset_path, affordance_selection_method=affordance_selection_method)

    # Add goal_static_affordance if provided (for goal-based evaluation)
    if goal_static_affordance is not None:
        goal_static_affordance = np.asarray(goal_static_affordance).flatten()[:3].astype(np.float32)
        # Stack to match n_obs_steps: shape (n_obs_steps, 3)
        goal_stacked = stack_for_obs_steps(goal_static_affordance, n_obs_steps)
        obs["goal_static_affordance"] = goal_stacked.astype(np.float32)

    return obs


def append_to_observation_trajectory(observation_trajectory, obs, use_part_point_cloud):
    """Append current observation to observation trajectory."""
    if observation_trajectory is None:
        return
    observation_trajectory["point_cloud"].append(obs["point_cloud"].copy())
    observation_trajectory["gripper_pcd"].append(obs["gripper_pcd"].copy())
    if "static_affordance" in obs:
        static_aff = normalize_static_affordance(obs["static_affordance"])
        observation_trajectory["static_affordance"].append(static_aff.copy())
    if "part_point_cloud" in obs:
        observation_trajectory["part_point_cloud"].append(obs["part_point_cloud"].copy())
    elif use_part_point_cloud and "point_cloud" in obs:
        observation_trajectory["part_point_cloud"].append(obs["point_cloud"].copy())

