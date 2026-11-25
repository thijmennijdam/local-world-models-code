"""Utilities for part point cloud extraction."""
import numpy as np
import scipy.spatial

from lwm.sim.utils import save_env, get_handle_pos, get_link_handle, get_link_pc
from lwm.planning.sim_utils.object_state import extract_object_state


def get_handle_joint_idx_from_env(env, object_name, use_closest_handle=False):
    """Get handle joint index from environment.

    Parameters
    ----------
    env : MultiStepWrapper
        The wrapped environment.
    object_name : str
        Name of the articulated object.
    use_closest_handle : bool
        If False (default), match the original behaviour: select the handle
        whose median is closest to the point cloud of ``link_0`` using
        ``get_link_handle``.
        If True, mirror the demo-generation behaviour and select the handle
        whose median is closest to the current end-effector position using
        ``get_closest_handle_to_eef``.
    
    Note: When use_closest_handle=True, this also sets _eval_handle_joint and
    _eval_min_link_idx on the base environment so that _get_info() uses the
    correct joint for tracking how much the drawer has opened.
    """
    object_name_for_handle = object_name.lower() if object_name else None
    if not object_name_for_handle:
        return None

    base_env = env.env._env

    # If evaluation code (or another helper) has already selected a handle
    # joint/index for this episode, simply reuse it so that:
    #   - part point cloud extraction
    #   - static affordance
    #   - metrics (opened_joint_angle, etc.)
    # all stay perfectly consistent.
    if hasattr(base_env, "_eval_handle_joint"):
        return base_env._eval_handle_joint

    # Otherwise, perform selection once and record it as the global choice.
    all_handle_pos, all_handle_joint_id = get_handle_pos(
        base_env, object_name_for_handle, return_median=False
    )
    if len(all_handle_pos) == 0 or len(all_handle_joint_id) == 0:
        return None

    if use_closest_handle:
        # Match training-time "closest handle" logic:
        # select the handle whose median is closest to the end-effector.
        robot_eef_pos, _ = base_env.robot.get_pos_orient(
            base_env.robot.right_end_effector
        )
        robot_eef_pos = np.array(robot_eef_pos)
        _, handle_joint_idx, _, selected_handle_idx = get_closest_handle_to_eef(
            all_handle_pos, all_handle_joint_id, robot_eef_pos
        )
        # Set on base_env so _get_info and observation code use the same handle.
        base_env._eval_handle_joint = handle_joint_idx
        base_env._eval_min_link_idx = selected_handle_idx
        return handle_joint_idx

    # Default behaviour: use link_0-based selection (original implementation).
    link_pc = get_link_pc(base_env, object_name_for_handle, "link_0")
    _, handle_joint_idx, _, selected_handle_idx = get_link_handle(
        all_handle_pos, all_handle_joint_id, link_pc
    )
    base_env._eval_handle_joint = handle_joint_idx
    base_env._eval_min_link_idx = selected_handle_idx
    return handle_joint_idx


def get_closest_handle_to_eef(all_handle_pos, handle_joint_id, eef_pos):
    """
    Select the handle closest to the end effector position.
    Simplified version - just returns the handle's own point cloud.

    Copied from the training/demo code so evaluation can compute the target
    in exactly the same way, without depending on the original repo.

    Args
    ----
    all_handle_pos : list[np.ndarray]
        List of handle point clouds.
    handle_joint_id : list[int]
        List of handle joint IDs.
    eef_pos : np.ndarray
        End effector position (3D array).

    Returns
    -------
    handle_pc : np.ndarray
        Selected handle point cloud.
    handle_joint_id : int
        Selected handle joint index.
    handle_median : np.ndarray
        Median point of the selected handle.
    selected_handle_idx : int
        Index of the selected handle in the input lists.
    """
    handle_median_points = np.array(
        [np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]
    ).reshape(-1, 3)

    # Calculate distance from each handle median to the end effector
    eef_pos_reshaped = eef_pos.reshape(1, -1)
    distances_to_eef = scipy.spatial.distance.cdist(
        handle_median_points, eef_pos_reshaped
    ).flatten()

    # Find the handle with minimum distance to end effector
    min_distance_handle_idx = np.argmin(distances_to_eef)

    handle_joint_id = handle_joint_id[min_distance_handle_idx]
    handle_pc = all_handle_pos[min_distance_handle_idx]
    handle_median = handle_median_points[min_distance_handle_idx]

    return handle_pc, handle_joint_id, handle_median, min_distance_handle_idx


def extract_part_point_cloud_from_state(env, object_geometry, handle_joint_idx, object_name, part_pointcloud_num, state_dict=None):
    """Extract part point cloud from state dict or current environment state.
    
    Parameters
    ----------
    env : MultiStepWrapper
        The wrapped environment
    object_geometry : ArticulatedObjectGeometry
        Object geometry for point cloud extraction
    handle_joint_idx : int
        Joint index for the handle
    object_name : str
        Name of the object
    part_pointcloud_num : int
        Number of points in the point cloud
    state_dict : dict, optional
        State dictionary to extract from. If None, uses current env state.
    
    Returns
    -------
    part_point_cloud : np.ndarray
        Shape (part_pointcloud_num, 3) in world frame (consistent with regular point_cloud)
    """
    if object_geometry is None or handle_joint_idx is None:
        return None
    object_name_for_state = object_name.lower() if object_name else None
    if not object_name_for_state:
        return None
    
    if state_dict is not None:
        object_state = extract_object_state(state_dict, object_name_for_state)
    else:
        current_state = save_env(env.env._env)
        object_state = extract_object_state(current_state, object_name_for_state)
    
    object_geometry.set_state(object_state["base_pos"], object_state["base_quat"], object_state["joint_dict"])
    part_point_cloud_world, _ = object_geometry.get_part_point_cloud(handle_joint_idx, total_points=part_pointcloud_num)
    
    # Return in world frame (consistent with regular point_cloud from RobogenPointCloudWrapper)
    return part_point_cloud_world.astype(np.float32)


def extract_part_point_cloud_for_env_state(env, object_geometry, handle_joint_idx, object_name, part_pointcloud_num):
    """Extract part point cloud for current environment state.
    
    Returns point cloud in world frame (consistent with regular point_cloud).
    """
    return extract_part_point_cloud_from_state(env, object_geometry, handle_joint_idx, object_name, part_pointcloud_num, state_dict=None)

