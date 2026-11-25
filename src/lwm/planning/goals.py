"""Goal affordance computation utilities.

Provides functions to compute goal affordance positions from goal percentages,
following the same methodology used in rollout generation.
"""

import numpy as np
import pybullet as p
from typing import Optional, Tuple
from termcolor import cprint


def _get_base_env(env):
    """Unwrap environment to get the base SimpleEnv instance."""
    if hasattr(env, 'env'):
        if hasattr(env.env, '_env'):
            return env.env._env
        return env.env
    return env


def get_object_body_id(base_env):
    """Get PyBullet body ID for the articulated object.

    Finds the object by excluding 'robot' and 'plane' from urdf_ids.
    This is robust regardless of the object type name.
    """
    for name, body_id in base_env.urdf_ids.items():
        if name not in ("robot", "plane"):
            return body_id
    raise ValueError(
        f"Could not find object body ID in env.urdf_ids: {list(base_env.urdf_ids.keys())}"
    )


def _get_env_object_name(base_env):
    """Get the object type name used in the environment (e.g., 'storagefurniture').

    All objects in this dataset are StorageFurniture (doors/drawers of cabinets),
    matching the hardcoded name in construct_env().
    """
    for name in base_env.urdf_ids:
        if name not in ("robot", "plane"):
            return name
    return "storagefurniture"


def compute_goal_affordance(
    env,
    object_id: str,
    goal_percentage: float,
    affordance_selection_method: str = "articulation_points",
    extracted_dataset_path: Optional[str] = None,
    use_closest_handle: bool = False,
    goal_joint_angle_override: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Compute goal affordance position from goal percentage.

    Follows the same approach as _recompute_goal_affordance() in rollout_worker.py:
    1. Establish handle point index at current angle
    2. Compute goal joint angle (from override, or goal_percentage clamped to max_goal)
    3. Set PyBullet joint to goal angle
    4. Extract handle position at goal angle (reuses point index)
    5. Restore joint to original position
    6. Return goal affordance [3] and goal joint angle

    Args:
        env: Environment instance (wrapped with RobogenPointCloudWrapper + MultiStepWrapper)
        object_id: Numeric object ID (e.g., "45526")
        goal_percentage: Target joint angle value directly (e.g., 0.4 = 0.4 rad).
            Clamped to [0, max_goal]. For doors max_goal=0.4, for drawers max_goal=0.8.
            Ignored if goal_joint_angle_override is set.
        affordance_selection_method: Method for selecting affordance point
        extracted_dataset_path: Optional path to extracted dataset
        use_closest_handle: Whether to use closest handle selection
        goal_joint_angle_override: If set, use this as goal joint angle (e.g. from
            joint_angle_constants.OBJECT_JOINT_ANGLE_AT_GOAL for canonical 040/080).

    Returns:
        Tuple of:
            goal_affordance: [3] goal affordance position in world frame
            goal_joint_angle: float, the clamped target joint angle (rad/m)
    """
    from lwm.planning.sim_utils import observations as _obs_mod
    get_handle_pos_from_env = _obs_mod.get_handle_pos_from_env
    from lwm.planning.sim_utils import file_utils as _file_mod
    get_object_category_by_id = _file_mod.get_object_category_by_id
    GOAL_BASED_OBJECT_CATEGORIES = _file_mod.GOAL_BASED_OBJECT_CATEGORIES

    base_env = _get_base_env(env)

    # Get object category and compute goal joint angle
    category = get_object_category_by_id(object_id)
    if category is None:
        raise ValueError(f"Unknown object ID: {object_id}")

    max_goal = GOAL_BASED_OBJECT_CATEGORIES[category]["max_goal"]
    if goal_joint_angle_override is not None:
        goal_joint_angle = float(goal_joint_angle_override)
    else:
        goal_joint_angle = min(goal_percentage, max_goal)

    cprint(
        f"Computing goal affordance: object_id={object_id}, category={category}, "
        f"goal_angle={goal_joint_angle:.3f}"
        + (f" (override)" if goal_joint_angle_override is not None else f" (from goal_pct={goal_percentage:.2f}, max_goal={max_goal})"),
        "cyan"
    )

    # Get PyBullet object body ID (robust: finds by excluding robot/plane)
    obj_id_pb = get_object_body_id(base_env)

    # Get env object type name for get_handle_pos_from_env calls
    env_obj_name = _get_env_object_name(base_env)

    # Step 1: Establish handle point index at CURRENT angle
    get_handle_pos_from_env(
        env,
        env_obj_name,
        use_closest_handle=use_closest_handle,
        extracted_dataset_path=extracted_dataset_path,
        affordance_selection_method=affordance_selection_method,
    )

    # Get handle joint (use eval handle if set via closest-handle selection)
    handle_jid = getattr(base_env, "_eval_handle_joint", base_env.handle_joint)

    # Step 2: Save current joint state
    cur_state = p.getJointState(obj_id_pb, handle_jid, physicsClientId=base_env.id)
    cur_pos, cur_vel = cur_state[0], cur_state[1]

    # Step 3: Set joint to goal angle
    p.resetJointState(
        obj_id_pb, handle_jid, goal_joint_angle,
        physicsClientId=base_env.id
    )

    # Step 4: Extract handle position at goal angle
    # Reuses the same cached point index and obj-frame data
    goal_point = get_handle_pos_from_env(
        env,
        env_obj_name,
        use_closest_handle=use_closest_handle,
        extracted_dataset_path=extracted_dataset_path,
        affordance_selection_method=affordance_selection_method,
    )

    if goal_point is not None:
        goal_affordance = np.asarray(goal_point).flatten()[:3].astype(np.float32)
    else:
        # Fallback to centerline
        goal_affordance = np.asarray(base_env.handle_pos).flatten()[:3].astype(np.float32)
        cprint(
            "WARNING: get_handle_pos_from_env returned None at goal angle, "
            "falling back to centerline",
            "yellow"
        )

    # Step 5: Restore original joint state
    p.resetJointState(
        obj_id_pb, handle_jid, cur_pos,
        targetVelocity=cur_vel, physicsClientId=base_env.id
    )

    # Step 6: Refresh current affordance
    # (Step 4 overwrote cached affordance with goal-angle position)
    get_handle_pos_from_env(
        env,
        env_obj_name,
        use_closest_handle=use_closest_handle,
        extracted_dataset_path=extracted_dataset_path,
        affordance_selection_method=affordance_selection_method,
    )
    # Restore _get_info() internal state
    base_env._get_info()

    cprint(
        f"Goal affordance computed: {goal_affordance.tolist()}, "
        f"goal_joint_angle={goal_joint_angle:.3f}",
        "green"
    )

    return goal_affordance, goal_joint_angle


# Cache the imported function
_cached_get_handle_pos_from_env = None


def get_current_affordance(
    env,
    affordance_selection_method: str = "articulation_points",
) -> np.ndarray:
    """Get current static affordance position, tracking the same point over time.

    Calls get_handle_pos_from_env() which uses a cached point index to track the
    same physical point on the handle across timesteps. This matches what the
    rollout code does via post_step_process_obs() at every step, and ensures the
    affordance tracks handle movement as the joint angle changes.

    Must be called AFTER compute_goal_affordance() which establishes the cached
    point index on the base environment.

    Args:
        env: Wrapped environment (MultiStepWrapper → RobogenPointCloudWrapper → base)
        affordance_selection_method: Method for selecting affordance point

    Returns:
        affordance: [3] position in world frame
    """
    global _cached_get_handle_pos_from_env
    if _cached_get_handle_pos_from_env is None:
        from lwm.planning.sim_utils import observations as _obs_mod
        _cached_get_handle_pos_from_env = _obs_mod.get_handle_pos_from_env

    base_env = _get_base_env(env)
    env_obj_name = _get_env_object_name(base_env)

    handle_pos = _cached_get_handle_pos_from_env(
        env,
        env_obj_name,
        affordance_selection_method=affordance_selection_method,
    )

    if handle_pos is not None:
        return np.asarray(handle_pos).flatten()[:3].astype(np.float32)

    # Fallback to centerline from _get_info()
    info = base_env._get_info()
    cprint("WARNING: get_handle_pos_from_env returned None, using centerline fallback", "yellow")
    return np.asarray(info["handle_pos"]).flatten()[:3].astype(np.float32)


def get_object_category(object_id: str) -> str:
    """Get category (door/drawer) for an object ID.

    Args:
        object_id: Object ID string (e.g., "45526")

    Returns:
        Category string ("door" or "drawer")
    """
    from lwm.planning.sim_utils import file_utils as _file_mod
    category = _file_mod.get_object_category_by_id(object_id)
    if category is None:
        raise ValueError(f"Unknown object ID: {object_id}")
    return category


def get_max_goal_for_object(object_id: str) -> float:
    """Get maximum goal percentage for an object.

    Args:
        object_id: Object ID string (e.g., "45526")

    Returns:
        Maximum goal percentage (e.g., 0.4 for doors, 0.8 for drawers)
    """
    from lwm.planning.sim_utils import file_utils as _file_mod
    category = get_object_category(object_id)
    return _file_mod.GOAL_BASED_OBJECT_CATEGORIES[category]["max_goal"]


def validate_goal_percentage(object_id: str, goal_percentage: float) -> bool:
    """Check if goal percentage is valid.

    In the MPC convention, goal_percentage is a fraction (0.0 to 1.0)
    of the object's max_goal joint angle.

    Args:
        object_id: Object ID string
        goal_percentage: Desired goal fraction (0.0 to 1.0)

    Returns:
        True if valid, False otherwise
    """
    return 0.0 <= goal_percentage <= 1.0
