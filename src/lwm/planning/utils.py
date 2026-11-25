"""Utility functions for MPC planning.

Provides helper functions for environment setup, trajectory saving,
visualization, and result logging.
"""

import os
import numpy as np
import pybullet as p
import pickle
import json
import pathlib
from typing import Dict, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from termcolor import cprint


def create_env_from_config(
    task_config_file: str,
    solution_path: str = "lwm.sim",
    task_name: str = "sim",
    object_name: str = "storagefurniture",
    num_points: int = 1280,
    n_obs_steps: int = 1,
    init_state_file: Optional[str] = None,
):
    """Create properly wrapped PyBullet environment for MPC planning.

    Builds the same wrapper chain used in training/evaluation:
    build_up_env() → RobogenPointCloudWrapper → MultiStepWrapper

    This ensures observations contain point clouds, agent state, and affordance
    data in the same format expected by the World Model.

    Args:
        task_config_file: Path to task config YAML from generated data
        solution_path: Python package path for the environment (default: "lwm.sim")
        task_name: Module name within solution_path (default: "sim")
        object_name: Object type name for the environment (default: "storagefurniture")
        num_points: Number of points in point cloud (default: 1280)
        n_obs_steps: Number of observation steps to stack (default: 1 for MPC)
        init_state_file: Optional path to initial state file (.pkl)

    Returns:
        Wrapped environment (MultiStepWrapper → RobogenPointCloudWrapper → base env)
        producing dict observations with keys: agent_pos, point_cloud,
        affordance_point, static_affordance, gripper_pcd, etc.
    """
    if not pathlib.Path(task_config_file).exists():
        raise FileNotFoundError(f"Task config not found: {task_config_file}")

    from lwm.sim.utils import build_up_env
    from lwm.sim.robogen_wrapper import RobogenPointCloudWrapper
    from lwm.planning.multistep_wrapper import MultiStepWrapper

    # 1. Create base environment
    env, _ = build_up_env(
        task_config_file, solution_path, task_name,
        init_state_file, render=False, horizon=600,
    )
    env.reset()

    # 2. Wrap with point cloud generation
    pointcloud_env = RobogenPointCloudWrapper(
        env, object_name, num_points=num_points,
        observation_mode="act3d_displacement_gripper_to_object_static_affordance",
        horizon=600,
    )

    # 3. Wrap with multi-step (n_action_steps=1 for single-step MPC execution)
    wrapped_env = MultiStepWrapper(
        pointcloud_env,
        n_obs_steps=n_obs_steps,
        n_action_steps=1,
        max_episode_steps=600,
        reward_agg_method="sum",
    )

    cprint(
        f"Environment created: {task_name} with {object_name}, "
        f"n_obs_steps={n_obs_steps}, num_points={num_points}",
        "green",
    )

    return wrapped_env


def _get_base_env(env):
    """Unwrap MultiStepWrapper → RobogenPointCloudWrapper → base SimpleEnv."""
    if hasattr(env, 'env'):
        if hasattr(env.env, '_env'):
            return env.env._env
        return env.env
    return env


def _get_object_body_id(base_env):
    """Get PyBullet body ID for the articulated object (excludes robot/plane)."""
    for name, body_id in base_env.urdf_ids.items():
        if name not in ("robot", "plane", "init_table"):
            return body_id
    raise ValueError(f"No object found in urdf_ids: {list(base_env.urdf_ids.keys())}")


def create_part_pcd_generator(
    env,
    object_id: str,
    num_points: int = 512,
) -> "PartPointCloudGenerator":
    """Create a part point cloud generator for the given environment.

    Uses ArticulatedObjectGeometry to generate part point clouds from mesh
    data, matching the extraction pipeline used for WM training.

    Args:
        env: Wrapped environment (MultiStepWrapper → RobogenPointCloudWrapper → base)
        object_id: Numeric object ID (e.g., "45526")
        num_points: Number of points to sample (default: 512)

    Returns:
        PartPointCloudGenerator instance
    """
    import time as _time

    t0 = _time.time()
    from lwm.sim.object_geometry import ArticulatedObjectGeometry
    from lwm.planning.sim_utils.part_pointcloud import get_handle_joint_idx_from_env
    cprint(f"  [2b] imports: {_time.time() - t0:.2f}s", "white")

    t1 = _time.time()
    base_env = _get_base_env(env)
    from lwm.sim.utils import data_dir as partnet_dir  # PARTNET_DATASET_DIR
    urdf_path = os.path.join(partnet_dir, str(object_id), "mobility.urdf")

    # Select handle closest to EE (matches eval/demo-generation logic)
    handle_jid = get_handle_joint_idx_from_env(
        env, "storagefurniture", use_closest_handle=True
    )
    cprint(f"  [2b] handle joint selection: {_time.time() - t1:.2f}s", "white")

    obj_body_id = _get_object_body_id(base_env)

    # Object scale must match training data (extraction uses scale from state).
    # Use env's simulator_sizes if set (e.g. after loading init_state_file), else 1.0.
    object_names = [n for n in base_env.urdf_ids if n not in ("robot", "plane", "init_table")]
    object_scale = 1.0
    if object_names:
        object_scale = float(getattr(base_env, "simulator_sizes", {}).get(object_names[0], 1.0))

    t2 = _time.time()
    geometry = ArticulatedObjectGeometry(urdf_path, object_scale=object_scale)
    cprint(f"  [2b] ArticulatedObjectGeometry init: {_time.time() - t2:.2f}s", "white")

    t3 = _time.time()
    generator = PartPointCloudGenerator(
        geometry=geometry,
        base_env=base_env,
        obj_body_id=obj_body_id,
        handle_joint_idx=handle_jid,
        num_points=num_points,
    )

    cprint(
        f"Part PCD generator created: object={object_id}, "
        f"handle_joint={handle_jid}, num_points={num_points}",
        "green",
    )
    return generator


class PartPointCloudGenerator:
    """Generates part point clouds and tracks affordance from the live simulation state.

    Syncs joint states from the main simulation into a separate
    ArticulatedObjectGeometry instance and returns the part point cloud
    in robot base frame (matching the extraction pipeline).

    Also tracks the affordance point from the same handle mesh, ensuring
    perfect geometric consistency between the part PCD and affordance.
    """

    def __init__(
        self,
        geometry,
        base_env,
        obj_body_id: int,
        handle_joint_idx: int,
        num_points: int = 512,
    ):
        self.geometry = geometry
        self.base_env = base_env
        self.obj_body_id = obj_body_id
        self.handle_joint_idx = handle_joint_idx
        self.num_points = num_points
        self._affordance_index = None  # cached index into handle_points

    def init_affordance_tracking(self, reference_affordance_world: np.ndarray):
        """Initialize affordance tracking by finding the closest handle point.

        Call once after creation, using the affordance from the old pipeline
        (get_current_affordance) as the reference. This finds the nearest point
        in the geometry's handle mesh and caches its index.

        Args:
            reference_affordance_world: [3] affordance position in world frame
        """
        # Sync current state and get handle points
        self._sync_state()
        _, handle_pts_world = self.geometry.get_part_point_cloud(
            self.handle_joint_idx, total_points=self.num_points
        )
        dists = np.linalg.norm(handle_pts_world - reference_affordance_world[None, :], axis=1)
        self._affordance_index = int(np.argmin(dists))
        min_dist = dists[self._affordance_index]
        from termcolor import cprint
        cprint(
            f"Affordance tracking initialized: handle_point_idx={self._affordance_index}, "
            f"dist_to_reference={min_dist:.4f}m",
            "green",
        )

    def _sync_state(self):
        """Read object state from main simulation and sync to geometry."""
        base_pos, base_quat = p.getBasePositionAndOrientation(
            self.obj_body_id, physicsClientId=self.base_env.id
        )
        base_pos = np.array(base_pos, dtype=np.float32)
        base_quat = np.array(base_quat, dtype=np.float32)

        joint_dict = {}
        num_joints = p.getNumJoints(self.obj_body_id, physicsClientId=self.base_env.id)
        for j in range(num_joints):
            info = p.getJointInfo(self.obj_body_id, j, physicsClientId=self.base_env.id)
            jname = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            state = p.getJointState(self.obj_body_id, j, physicsClientId=self.base_env.id)
            joint_dict[jname] = state[0]

        self.geometry.set_state(base_pos, base_quat, joint_dict)
        return joint_dict

    def get_part_pcd(self):
        """Get current part point cloud and affordance point, both in world frame.

        The caller is responsible for transforming to robot base frame using
        a consistent robot pose for both outputs (avoiding frame mismatch).

        Returns:
            Tuple of:
                part_pcd_world: [num_points, 3] numpy array in world frame
                affordance_world: [3] numpy array in world frame (or None if
                    init_affordance_tracking was not called)
        """
        self._sync_state()

        # Get part point cloud and handle points in world frame
        part_pcd_world, handle_pts_world = self.geometry.get_part_point_cloud(
            self.handle_joint_idx, total_points=self.num_points
        )

        # Get tracked affordance point
        affordance_world = None
        if self._affordance_index is not None:
            affordance_world = handle_pts_world[self._affordance_index].astype(np.float32)

        return part_pcd_world.astype(np.float32), affordance_world

    def get_goal_affordance(self, goal_joint_angle: float) -> np.ndarray:
        """Get affordance position at a specific joint angle.

        Temporarily sets the geometry to the goal joint state, reads the
        affordance at the cached index, then restores the original state.

        Args:
            goal_joint_angle: Target joint angle (rad for revolute, m for prismatic)

        Returns:
            goal_affordance: [3] position in world frame
        """
        if self._affordance_index is None:
            raise RuntimeError("Call init_affordance_tracking() first")

        # Save current state
        joint_dict = self._sync_state()

        # Find the handle joint name
        handle_info = p.getJointInfo(
            self.geometry.body, self.handle_joint_idx,
            physicsClientId=self.geometry.client_id
        )
        handle_jname = handle_info[1].decode("utf-8") if isinstance(handle_info[1], (bytes, bytearray)) else str(handle_info[1])

        # Set to goal angle
        goal_dict = dict(joint_dict)
        goal_dict[handle_jname] = goal_joint_angle
        # Read base pos/quat from geometry (already synced)
        base_pos, base_quat = p.getBasePositionAndOrientation(
            self.geometry.body, physicsClientId=self.geometry.client_id
        )
        self.geometry.set_state(
            np.array(base_pos, dtype=np.float32),
            np.array(base_quat, dtype=np.float32),
            goal_dict,
        )

        # Get handle points at goal
        _, handle_pts_world = self.geometry.get_part_point_cloud(
            self.handle_joint_idx, total_points=self.num_points
        )
        goal_affordance = handle_pts_world[self._affordance_index].astype(np.float32)

        # Restore original state
        self.geometry.set_state(
            np.array(base_pos, dtype=np.float32),
            np.array(base_quat, dtype=np.float32),
            joint_dict,
        )

        return goal_affordance

    def close(self):
        """Clean up the geometry's PyBullet instance."""
        self.geometry.close()


def save_trajectory(trajectory: Dict, save_path: str, metadata: Optional[Dict] = None):
    """Save planned trajectory to file.

    Args:
        trajectory: Dict with keys:
            - states: [T, state_dim] observed states
            - actions: [T, action_dim] executed actions
            - affordances: [T, 3] observed affordances
            - costs: [T] costs at each step
            - goal_distances: [T] distance to goal at each step
        save_path: Path to save trajectory (directory or .pkl file)
        metadata: Optional metadata to save alongside trajectory
    """
    save_path = pathlib.Path(save_path)

    # If directory, create timestamped filename
    if save_path.is_dir():
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = save_path / f"trajectory_{timestamp}.pkl"

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data
    data = {
        "trajectory": trajectory,
        "metadata": metadata or {},
    }

    # Save as pickle
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    cprint(f"Trajectory saved to {save_path}", "green")


def save_metrics(metrics: Dict, save_path: str):
    """Save planning metrics to JSON file.

    Args:
        metrics: Dict with planning metrics
        save_path: Path to save metrics JSON
    """
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    cprint(f"Metrics saved to {save_path}", "green")


def visualize_planned_trajectory(
    trajectory: Dict,
    save_path: str,
    goal_affordance: Optional[np.ndarray] = None,
):
    """Create visualization of planned trajectory.

    Generates plots showing:
    - EE and affordance positions over time
    - Distance to goal over time
    - Cost over time

    Args:
        trajectory: Trajectory dict from run_mpc_planning
        save_path: Path to save visualization
        goal_affordance: Optional goal position for reference
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        cprint("matplotlib not available, skipping visualization", "yellow")
        return

    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract data
    states = np.array(trajectory.get("states", []))
    affordances = np.array(trajectory.get("affordances", []))
    costs = np.array(trajectory.get("costs", []))
    observed_costs = np.array(trajectory.get("observed_costs", []))
    goal_distances = np.array(trajectory.get("goal_distances", []))

    if len(states) == 0:
        cprint("No trajectory data to visualize", "yellow")
        return

    # Check if WM diagnostic data is available
    pred_errors_ee = np.array(trajectory.get("prediction_errors_ee", []))
    pred_errors_aff = np.array(trajectory.get("prediction_errors_aff", []))
    wm_predicted_cost = np.array(trajectory.get("wm_predicted_cost", []))
    has_wm_diag = len(pred_errors_ee) > 0

    # Check if contact prediction data is available
    contact_logits = np.array(trajectory.get("contact_logits", []))
    has_contact = len(contact_logits) > 0 and not np.all(np.isnan(contact_logits))

    num_plots = 3
    if has_wm_diag:
        num_plots += 1
    if has_contact:
        num_plots += 1

    # Create figure with subplots
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))

    # Plot 1: Positions (EE pos → affordance → goal; cost is EE→aff then aff→goal)
    ax = axes[0]
    ee_positions = states[:, :3]  # First 3 dims are EE position
    timesteps = np.arange(len(states))

    ax.plot(timesteps, ee_positions[:, 0], label='EE X', color='C0', linestyle='-', linewidth=1.5)
    ax.plot(timesteps, ee_positions[:, 1], label='EE Y', color='C1', linestyle='-', linewidth=1.5)
    ax.plot(timesteps, ee_positions[:, 2], label='EE Z', color='C2', linestyle='-', linewidth=1.5)

    ax.plot(timesteps, affordances[:, 0], label='Aff X', color='C0', linestyle='--', linewidth=1.2)
    ax.plot(timesteps, affordances[:, 1], label='Aff Y', color='C1', linestyle='--', linewidth=1.2)
    ax.plot(timesteps, affordances[:, 2], label='Aff Z', color='C2', linestyle='--', linewidth=1.2)

    if goal_affordance is not None:
        ax.axhline(goal_affordance[0], color='C0', linestyle=':', alpha=0.7, linewidth=1.2, label='Goal X')
        ax.axhline(goal_affordance[1], color='C1', linestyle=':', alpha=0.7, linewidth=1.2, label='Goal Y')
        ax.axhline(goal_affordance[2], color='C2', linestyle=':', alpha=0.7, linewidth=1.2, label='Goal Z')

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position (m)')
    ax.set_title('EE pos (solid) → Affordance (dashed) → Goal (dotted)')
    ax.legend(ncol=3, fontsize='small')
    ax.grid(True, alpha=0.3)

    # Plot 2: Both cost distances (EE→affordance, affordance→goal)
    ax = axes[1]
    ee_to_aff = np.linalg.norm(ee_positions - affordances, axis=1)
    if len(goal_distances) > 0:
        ax.plot(timesteps, ee_to_aff, color='C0', linewidth=2, label='EE → Affordance')
        ax.plot(timesteps, goal_distances, color='C1', linewidth=2, label='Affordance → Goal')

        # Show pull_threshold line (read from trajectory metadata if available)
        pull_thresh = trajectory.get("pull_threshold", None)
        if pull_thresh is not None:
            ax.axhline(pull_thresh, color='red', linestyle='--', alpha=0.7,
                       linewidth=1, label=f'Pull threshold ({pull_thresh}m)')

        # Mark phase change step
        phase_step = trajectory.get("phase_change_step", None)
        if phase_step is not None:
            ax.axvline(phase_step, color='cyan', linestyle='--', alpha=0.8,
                       linewidth=1.5, label='Planner reset')

        ax.set_xlabel('Timestep')
        ax.set_ylabel('Distance (m)')
        ax.set_title('Cost distances: EE→Aff and Aff→Goal')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add final distances text
        text = f'Final EE→Aff: {ee_to_aff[-1]:.4f}m  |  Final Aff→Goal: {goal_distances[-1]:.4f}m'
        ax.text(
            0.95, 0.95, text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize='small',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Plot 3: Cost comparison — actual observed vs WM predicted 1-step cost
    ax = axes[2]
    if len(observed_costs) > 0:
        ax.plot(timesteps, observed_costs, color='green', linewidth=2, label='Actual step cost')
    if len(wm_predicted_cost) > 0:
        ax.plot(timesteps, wm_predicted_cost, color='red', linewidth=2, linestyle='--',
                label='WM predicted 1-step cost')
    if len(costs) > 0:
        ax.plot(timesteps, costs, color='orange', linewidth=1.5, linestyle=':',
                alpha=0.6, label='Best horizon cost (plan)')
    if len(observed_costs) > 0 or len(costs) > 0:
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Cost')
        ax.set_title('Actual vs WM-predicted cost (1-step)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: WM prediction errors + clip effect (if available)
    if has_wm_diag:
        ax = axes[3]
        ax.plot(timesteps, pred_errors_ee, color='limegreen', linewidth=2,
                label=f'EE pos error (mean={pred_errors_ee.mean():.4f}m)')
        ax.plot(timesteps, pred_errors_aff, color='red', linewidth=2,
                label=f'Aff pos error (mean={pred_errors_aff.mean():.4f}m)')

        # Clip delta: how much clipping shifted aff→goal (positive = WM was optimistic)
        clip_deltas = np.array(trajectory.get("clip_deltas", []))
        if len(clip_deltas) > 0 and np.any(np.abs(clip_deltas) > 1e-6):
            ax2 = ax.twinx()
            ax2.fill_between(timesteps, 0, clip_deltas, alpha=0.2, color='blue',
                             label=f'Clip Δ aff→goal (mean={clip_deltas.mean():+.4f}m)')
            ax2.axhline(0, color='blue', linewidth=0.5, alpha=0.3)
            ax2.set_ylabel('Clip effect (m)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.legend(loc='lower right', fontsize='small')

        ax.set_xlabel('Timestep')
        ax.set_ylabel('Prediction error (m)')
        ax.set_title('WM 1-step prediction error + affordance clip effect')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Summary stats
        clip_text = ''
        if len(clip_deltas) > 0:
            clip_text = f'\nClip Δ — mean: {clip_deltas.mean():+.4f}m, max: {clip_deltas.max():+.4f}m'
        text = (
            f'EE — mean: {pred_errors_ee.mean():.4f}m, max: {pred_errors_ee.max():.4f}m\n'
            f'Aff — mean: {pred_errors_aff.mean():.4f}m, max: {pred_errors_aff.max():.4f}m'
            + clip_text
        )
        ax.text(
            0.95, 0.95, text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize='small',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7)
        )

    # Plot 5: Contact prediction (if available)
    if has_contact:
        ax = axes[num_plots - 1]
        contact_preds = np.array(trajectory.get("contact_predictions", []))

        # Plot logit as a line
        ax.plot(timesteps, contact_logits, color='blue', linewidth=1.5, label='Contact logit')
        ax.axhline(0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Threshold (logit=0)')

        # Shade contact/no-contact regions
        if len(contact_preds) > 0 and not np.all(np.isnan(contact_preds)):
            contact_mask = contact_preds.astype(bool)
            ymin, ymax = ax.get_ylim()
            ax.fill_between(timesteps, ymin, ymax,
                            where=contact_mask, alpha=0.15, color='green', label='Contact predicted')
            ax.fill_between(timesteps, ymin, ymax,
                            where=~contact_mask, alpha=0.10, color='red', label='No contact')
            ax.set_ylim(ymin, ymax)  # restore limits after fill

        # Mark phase change step
        phase_step = trajectory.get("phase_change_step", None)
        if phase_step is not None:
            ax.axvline(phase_step, color='cyan', linestyle='--', alpha=0.8,
                       linewidth=1.5, label='Planner reset')

        ax.set_xlabel('Timestep')
        ax.set_ylabel('Contact logit')
        ax.set_title('Learned contact prediction (logit > 0 = contact)')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)

        # Summary stats
        n_contact = int(np.nansum(contact_preds)) if len(contact_preds) > 0 else 0
        n_total = len(contact_preds) if len(contact_preds) > 0 else 0
        text = f'Contact steps: {n_contact}/{n_total} ({100*n_contact/max(n_total,1):.0f}%)'
        ax.text(
            0.95, 0.95, text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize='small',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7)
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    cprint(f"Visualization saved to {save_path}", "green")


def compute_trajectory_metrics(
    trajectory: Dict,
    goal_affordance: np.ndarray,
    initial_joint_angle: float = 0.0,
    goal_joint_angle: float = 0.4,
    joint_upper_limit: Optional[float] = None,
) -> Dict:
    """Compute summary metrics for a trajectory.

    Args:
        trajectory: Trajectory dict
        goal_affordance: Goal affordance position [3]
        initial_joint_angle: Joint angle at start (rad/m)
        goal_joint_angle: Target joint angle (rad/m)
        joint_upper_limit: Actual joint upper limit from PyBullet. If set, we cap the
            effective goal so "fully open" for this object counts as 1.0 (avoids
            penalizing objects with smaller physical range than category max).

    Returns:
        Dict with metrics including angle-based normalized_performance
        (matching eval_robogen_affordance convention).
    """
    states = np.array(trajectory["states"])
    affordances = np.array(trajectory["affordances"])
    costs = np.array(trajectory.get("costs", []))
    joint_angles = np.array(trajectory.get("joint_angles", []))

    # Extract EE positions
    ee_positions = states[:, :3]

    # Compute distance-based metrics (kept for plots/diagnostics)
    ee_to_aff_distances = np.linalg.norm(ee_positions - affordances, axis=1)
    aff_to_goal_distances = np.linalg.norm(affordances - goal_affordance, axis=1)

    # Effective goal: never ask for more than this object's joint can do
    effective_goal = goal_joint_angle
    if joint_upper_limit is not None and np.isfinite(joint_upper_limit) and joint_upper_limit > 0:
        effective_goal = min(goal_joint_angle, float(joint_upper_limit))

    # Angle-based normalized performance: (final - initial) / (effective_goal - initial)
    final_joint_angle = float(joint_angles[-1]) if len(joint_angles) > 0 else initial_joint_angle
    if effective_goal > initial_joint_angle:
        normalized_performance = min(1.0,
            (final_joint_angle - initial_joint_angle) / (effective_goal - initial_joint_angle))
    else:
        normalized_performance = 0.0

    metrics = {
        "final_distance_to_goal": float(aff_to_goal_distances[-1]),
        "mean_ee_to_aff_distance": float(ee_to_aff_distances.mean()),
        "total_cost": float(costs.sum()) if len(costs) > 0 else 0.0,
        "trajectory_length": len(states),
        # Angle-based metrics (primary for SR computation)
        "initial_joint_angle": initial_joint_angle,
        "final_joint_angle": final_joint_angle,
        "goal_joint_angle": goal_joint_angle,
        "normalized_performance": normalized_performance,
    }

    return metrics


def print_planning_summary(
    metrics: Dict,
    object_id: str,
    goal_percentage: float,
):
    """Print summary of planning results.

    Args:
        metrics: Trajectory metrics
        object_id: Object ID
        goal_percentage: Goal percentage used
    """
    cprint("\n" + "="*60, "cyan")
    cprint("PLANNING SUMMARY", "cyan")
    cprint("="*60, "cyan")
    cprint(f"Object: {object_id}", "white")
    cprint(f"Goal Percentage: {goal_percentage:.2f}", "white")
    cprint(f"Trajectory Length: {metrics['trajectory_length']} steps", "white")
    cprint(f"Final Distance to Goal: {metrics['final_distance_to_goal']:.4f} m", "white")
    cprint(f"Mean EE-Affordance Distance: {metrics['mean_ee_to_aff_distance']:.4f} m", "white")
    cprint(f"Total Cost: {metrics['total_cost']:.2f}", "white")

    # Angle-based metrics
    init_angle = metrics.get("initial_joint_angle", 0.0)
    final_angle = metrics.get("final_joint_angle", 0.0)
    goal_angle = metrics.get("goal_joint_angle", 0.0)
    norm_perf = metrics.get("normalized_performance", 0.0)
    cprint(f"Joint Angle: {init_angle:.4f} -> {final_angle:.4f} (goal: {goal_angle:.4f})", "white")
    cprint(f"Normalized Performance: {norm_perf:.1%}", "white")

    if norm_perf >= 0.8:
        cprint("SUCCESS (SR@0.8): Goal reached!", "green")
    elif norm_perf >= 0.4:
        cprint("PARTIAL (SR@0.4): Partial progress", "yellow")
    else:
        cprint("INCOMPLETE: Insufficient progress", "yellow")

    cprint("="*60 + "\n", "cyan")
