"""Run MPC planning with a trained Local World Model in the PyBullet simulator.

This script orchestrates the full MPC planning loop:
1. Load WM checkpoint
2. Setup environment (with proper wrapper chain)
3. Compute goal affordance
4. Run receding horizon MPC
5. Save results and visualizations

NOTE: task.object_id, task.task_config_file, and task.init_state_file must all refer
to the SAME object. The object_id is used for goal computation and part PCD, while
task_config_file/init_state_file determine the actual object loaded in the simulator.

PartNet-Mobility assets are found through the PARTNET_DATASET_DIR environment variable.

Example:

    export PARTNET_DATASET_DIR=/path/to/partnet-mobility
    python -m lwm.planning.run_mpc \
        checkpoint_path=/path/to/LWM-200M_d1024_L16/best.ckpt \
        task.object_id=45526 \
        task.goal_percentage=0.4 \
        task.task_config_file=/path/to/<session>/task_config.yaml \
        task.init_state_file=/path/to/<session>/states/state_0.pkl \
        mpc.planner_type=icem
"""

import os
import sys
import copy
import json
import shutil
import pickle
import pathlib
import numpy as np
import torch
from termcolor import cprint
import hydra
from omegaconf import DictConfig, OmegaConf

from lwm.planning.planners import create_planner
from lwm.planning.costs import create_cost_function
from lwm.planning.wm_predictor import WorldModelPredictor
from lwm.planning.goals import compute_goal_affordance, get_current_affordance, get_object_body_id, _get_base_env as _get_base_env_goal
from lwm.planning.joint_angles import OBJECT_JOINT_ANGLE_AT_GOAL
from lwm.planning.utils import (
    create_env_from_config,
    create_part_pcd_generator,
    save_trajectory,
    save_metrics,
    visualize_planned_trajectory,
    compute_trajectory_metrics,
    print_planning_summary,
    _get_base_env,
)

# Import save_env for saving PyBullet states (extraction-compatible format)
from lwm.sim.utils import save_env
from lwm.paths import CONFIG_DIR


def _annotate_frame(
    frame, step, total_steps, ee_to_aff, aff_to_goal,
    actual_cost=None, predicted_cost=None, ee_error=None, aff_error=None,
    phase_change=False,
    predict_agent_state=True,
    contact_logit=None, contact_pred=None,
):
    """Add text overlay to a rendered frame showing actual vs WM-predicted costs.

    When predict_agent_state=False, EE 'prediction' is current_ee+action (no WM);
    EE err then means execution/dynamics error, not WM prediction error.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to get a readable font
    font = None
    for path in [
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
    ]:
        try:
            font = ImageFont.truetype(path, 16)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default()

    h, w = frame.shape[:2]

    def _draw_outlined(x, y, text, color=(255, 255, 255)):
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=color, font=font)

    # Step counter — top right
    step_text = f"Step {step}/{total_steps}"
    bbox = draw.textbbox((0, 0), step_text, font=font)
    tw = bbox[2] - bbox[0]
    _draw_outlined(w - tw - 10, 10, step_text)

    # Contact prediction — top right, below step counter
    if contact_pred is not None:
        contact_text = f"Contact: {'YES' if contact_pred else 'NO'}"
        contact_color = (100, 255, 100) if contact_pred else (255, 100, 100)
        bbox_c = draw.textbbox((0, 0), contact_text, font=font)
        tw_c = bbox_c[2] - bbox_c[0]
        _draw_outlined(w - tw_c - 10, 30, contact_text, color=contact_color)
        if contact_logit is not None:
            logit_text = f"logit: {contact_logit:.2f}"
            bbox_l = draw.textbbox((0, 0), logit_text, font=font)
            tw_l = bbox_l[2] - bbox_l[0]
            _draw_outlined(w - tw_l - 10, 50, logit_text, color=(200, 200, 200))

    # Left column: actual observations
    y = 10
    _draw_outlined(10, y, "--- Actual ---", color=(100, 255, 100))
    y += 20
    _draw_outlined(10, y, f"EE->Aff: {ee_to_aff:.3f}m")
    y += 20
    _draw_outlined(10, y, f"Aff->Goal: {aff_to_goal:.3f}m")
    y += 20
    if actual_cost is not None:
        _draw_outlined(10, y, f"Cost: {actual_cost:.3f}")
        y += 20

    # WM prediction comparison (if available)
    if predicted_cost is not None:
        y += 5
        _draw_outlined(10, y, "--- WM Predicted ---", color=(255, 200, 100))
        y += 20
        _draw_outlined(10, y, f"Cost: {predicted_cost:.3f}")
        y += 20
        cost_err = abs(predicted_cost - (actual_cost or 0))
        _draw_outlined(10, y, f"Cost err: {cost_err:.3f}")
        y += 20

    if ee_error is not None or aff_error is not None:
        _draw_outlined(10, y, "--- WM Errors ---", color=(255, 130, 130))
        y += 20
        if ee_error is not None:
            ee_label = "EE err (exec):" if not predict_agent_state else "EE err:"
            _draw_outlined(10, y, f"{ee_label} {ee_error:.4f}m")
            y += 20
        if aff_error is not None:
            _draw_outlined(10, y, f"Aff err: {aff_error:.4f}m")
            y += 20

    if phase_change:
        y += 10
        _draw_outlined(10, y, ">> PLANNER RESET <<", color=(0, 255, 255))

    return np.array(img)


def _world_to_base(points, robot_base_pos, robot_base_R):
    """Transform points from world frame to robot base frame.

    Args:
        points: [3] or [N, 3] numpy array in world frame
        robot_base_pos: [3] robot base position in world frame
        robot_base_R: [3, 3] rotation matrix (world → base)

    Returns:
        Transformed points in robot base frame (same shape as input)
    """
    return (points - robot_base_pos) @ robot_base_R.T


def _base_to_world(points, robot_base_pos, robot_base_R):
    """Transform points from robot base frame to world frame (inverse of _world_to_base).

    Args:
        points: [3] or [N, 3] numpy array in robot base frame
        robot_base_pos: [3] robot base position in world frame
        robot_base_R: [3, 3] rotation matrix (world → base)

    Returns:
        Transformed points in world frame (same shape as input)
    """
    return points @ robot_base_R + robot_base_pos


def _render_wm_input_frames(wm_input_data, goal_affordance_base, output_path):
    """Render 3-view 3D scatter plots of WM inputs and save as GIF.

    All data must already be in the same frame (robot base frame).
    Three views: top-down, front-left, front-right.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    # Compute consistent axis limits from all data
    all_pts = []
    for d in wm_input_data:
        all_pts.append(d["part_pcd"])
        all_pts.append(d["ee_pos"].reshape(1, 3))
        all_pts.append(d["affordance"].reshape(1, 3))
    all_pts.append(goal_affordance_base.reshape(1, 3))
    all_pts = np.concatenate(all_pts, axis=0)

    margin = 0.05
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    # Three camera views: (elevation, azimuth, title)
    views = [
        (90, -90, "Top-down"),
        (20, -60, "Front-left"),
        (20, 30, "Front-right"),
    ]

    total = len(wm_input_data) - 1  # first entry is "Init"
    wm_frames = []

    for i, d in enumerate(wm_input_data):
        fig = plt.figure(figsize=(18, 5), dpi=80)
        label = "Init" if i == 0 else f"Step {i}/{total}"

        for vi, (elev, azim, vtitle) in enumerate(views):
            ax = fig.add_subplot(1, 3, vi + 1, projection="3d")

            pcd = d["part_pcd"]
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2],
                       s=2, c="steelblue", alpha=0.4, label="Part PCD")
            ax.scatter(*d["affordance"], s=120, c="red", marker="o",
                       label="Affordance", zorder=5)
            ax.scatter(*d["ee_pos"], s=120, c="limegreen", marker="^",
                       label="EE Pos", zorder=5)
            ax.scatter(*goal_affordance_base, s=120, c="orange", marker="*",
                       label="Goal Aff", zorder=5)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{vtitle} — {label}", fontsize=10)

            if vi == 0:
                ax.legend(loc="upper left", fontsize=7)

        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        wm_frames.append(img)
        plt.close(fig)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), wm_frames, fps=5, loop=0)
    return len(wm_frames)


def _render_wm_diagnostic_frames(wm_diagnostic_data, goal_affordance_base, output_path):
    """Render side-by-side 3D scatter: WM-predicted vs actual observations.

    Each frame shows two columns (Predicted | Actual) with two camera views each,
    overlaid on the same part PCD. Errors are shown as dashed lines connecting
    predicted → actual positions.

    All data must be in robot base frame.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    if not wm_diagnostic_data:
        return 0

    # Compute consistent axis limits from all data
    all_pts = []
    for d in wm_diagnostic_data:
        all_pts.append(d["part_pcd"])
        if "pred_pcd" in d:
            all_pts.append(d["pred_pcd"])
        all_pts.append(d["actual_ee_base"].reshape(1, 3))
        all_pts.append(d["actual_aff_base"].reshape(1, 3))
        all_pts.append(d["pred_ee_base"].reshape(1, 3))
        all_pts.append(d["pred_aff_base"].reshape(1, 3))
    all_pts.append(goal_affordance_base.reshape(1, 3))
    all_pts = np.concatenate(all_pts, axis=0)

    margin = 0.05
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    views = [
        (90, -90, "Top-down"),
        (20, -60, "Front-left"),
    ]

    total = len(wm_diagnostic_data)
    diag_frames = []

    for i, d in enumerate(wm_diagnostic_data):
        fig = plt.figure(figsize=(16, 10), dpi=80)
        label = f"Step {i+1}/{total}"

        ee_err = np.linalg.norm(d["pred_ee_base"] - d["actual_ee_base"])
        aff_err = np.linalg.norm(d["pred_aff_base"] - d["actual_aff_base"])
        fig.suptitle(
            f"{label}  |  EE err: {ee_err:.4f}m  |  Aff err: {aff_err:.4f}m",
            fontsize=12, fontweight="bold",
        )

        for vi, (elev, azim, vtitle) in enumerate(views):
            # Left: WM Predicted (uses WM's predicted PCD after rigid body transform)
            ax = fig.add_subplot(2, 2, vi * 2 + 1, projection="3d")
            pred_pcd = d.get("pred_pcd", d["part_pcd"])  # fallback to GT if not available
            ax.scatter(pred_pcd[:, 0], pred_pcd[:, 1], pred_pcd[:, 2],
                       s=2, c="steelblue", alpha=0.3)
            ax.scatter(*d["pred_ee_base"], s=150, c="limegreen", marker="^",
                       label="Pred EE", zorder=5, edgecolors="black", linewidths=0.5)
            ax.scatter(*d["pred_aff_base"], s=150, c="red", marker="o",
                       label="Pred Aff", zorder=5, edgecolors="black", linewidths=0.5)
            ax.scatter(*goal_affordance_base, s=100, c="orange", marker="*",
                       label="Goal", zorder=5)
            # Show actual as faint reference
            ax.scatter(*d["actual_ee_base"], s=80, c="limegreen", marker="^",
                       alpha=0.3, zorder=4)
            ax.scatter(*d["actual_aff_base"], s=80, c="red", marker="o",
                       alpha=0.3, zorder=4)
            # Error lines
            ax.plot(*zip(d["pred_ee_base"], d["actual_ee_base"]),
                    c="limegreen", ls="--", lw=1.5, alpha=0.7)
            ax.plot(*zip(d["pred_aff_base"], d["actual_aff_base"]),
                    c="red", ls="--", lw=1.5, alpha=0.7)

            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"WM Predicted — {vtitle}", fontsize=9)
            if vi == 0:
                ax.legend(loc="upper left", fontsize=6)

            # Right: Actual (uses GT part PCD from simulation)
            ax = fig.add_subplot(2, 2, vi * 2 + 2, projection="3d")
            actual_pcd = d["part_pcd"]
            ax.scatter(actual_pcd[:, 0], actual_pcd[:, 1], actual_pcd[:, 2],
                       s=2, c="steelblue", alpha=0.3)
            ax.scatter(*d["actual_ee_base"], s=150, c="limegreen", marker="^",
                       label="Actual EE", zorder=5, edgecolors="black", linewidths=0.5)
            ax.scatter(*d["actual_aff_base"], s=150, c="red", marker="o",
                       label="Actual Aff", zorder=5, edgecolors="black", linewidths=0.5)
            ax.scatter(*goal_affordance_base, s=100, c="orange", marker="*",
                       label="Goal", zorder=5)
            # Show predicted as faint reference
            ax.scatter(*d["pred_ee_base"], s=80, c="limegreen", marker="^",
                       alpha=0.3, zorder=4)
            ax.scatter(*d["pred_aff_base"], s=80, c="red", marker="o",
                       alpha=0.3, zorder=4)
            # Error lines
            ax.plot(*zip(d["pred_ee_base"], d["actual_ee_base"]),
                    c="limegreen", ls="--", lw=1.5, alpha=0.7)
            ax.plot(*zip(d["pred_aff_base"], d["actual_aff_base"]),
                    c="red", ls="--", lw=1.5, alpha=0.7)

            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"Actual — {vtitle}", fontsize=9)
            if vi == 0:
                ax.legend(loc="upper left", fontsize=6)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        diag_frames.append(img)
        plt.close(fig)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), diag_frames, fps=3, loop=0)
    return len(diag_frames)


def _extract_obs_arrays(obs_dict, n_obs_steps=1):
    """Extract numpy arrays from wrapped env observation dict.

    MultiStepWrapper stacks observations with shape (n_obs_steps, ...).
    For MPC we use n_obs_steps=1, so we take the last (most recent) timestep.

    Args:
        obs_dict: Observation dict from env.reset() or env.step()
        n_obs_steps: Number of observation steps (for indexing)

    Returns:
        agent_pos: [10] numpy array
        point_cloud: [N, 3] numpy array
        affordance: [3] numpy array
    """
    # Take most recent timestep (last index along obs_steps dim)
    idx = -1

    # agent_pos: (n_obs_steps, 10) → (10,)
    agent_pos = obs_dict["agent_pos"][idx]

    # point_cloud: (n_obs_steps, N, 3) → (N, 3)
    point_cloud = obs_dict["point_cloud"][idx]

    # Affordance: try static_affordance first (added by post_step_process_obs),
    # then fall back to affordance_point (from RobogenPointCloudWrapper)
    if "static_affordance" in obs_dict:
        affordance = obs_dict["static_affordance"][idx]
    elif "affordance_point" in obs_dict:
        affordance = obs_dict["affordance_point"][idx]
    else:
        raise KeyError(
            f"No affordance key found in obs. Available keys: {list(obs_dict.keys())}"
        )

    return agent_pos, point_cloud, affordance


def _obs_to_tensors(agent_pos, point_cloud, affordance, device):
    """Convert numpy observation arrays to batched tensors for the WM.

    Args:
        agent_pos: [10] numpy array
        point_cloud: [N, 3] numpy array
        affordance: [3] numpy array
        device: torch device

    Returns:
        Dict with tensors of shape [1, ...] (batch_size=1)
    """
    return {
        "agent_state": torch.from_numpy(agent_pos).float().unsqueeze(0).to(device),
        "affordance": torch.from_numpy(affordance).float().unsqueeze(0).to(device),
        "point_cloud": torch.from_numpy(point_cloud).float().unsqueeze(0).to(device),
    }


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIR / "mpc"),
    config_name="planner"
)
def main(cfg: DictConfig):
    """Main MPC planning loop.

    Args:
        cfg: Hydra configuration
    """
    cprint("\n" + "="*80, "cyan")
    cprint(" "*25 + "MPC PLANNING WITH WORLD MODEL", "cyan")
    cprint("="*80 + "\n", "cyan")

    # Resolve output directory from Hydra (works regardless of chdir setting)
    from hydra.core.hydra_config import HydraConfig
    output_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cprint(f"Output directory: {output_dir}", "cyan")

    # Print configuration
    cprint("Configuration:", "yellow")
    print(OmegaConf.to_yaml(cfg))

    device = cfg.execution.device

    # Set random seeds for reproducibility (if specified)
    seed = cfg.execution.get("seed", None)
    if seed is not None:
        seed = int(seed)
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cprint(f"Random seed set to {seed}", "yellow")

    # 1. Load World Model checkpoint
    import time as _time
    cprint("\n[1/7] Loading World Model...", "cyan")
    _t1 = _time.time()
    aff_gate_scale = cfg.mpc.get("affordance_proximity_gate_scale", None)
    if aff_gate_scale is not None:
        aff_gate_scale = float(aff_gate_scale)
    finger_thresh = cfg.mpc.get("finger_close_threshold", None)
    if finger_thresh is not None:
        finger_thresh = float(finger_thresh)
    use_gt_agent_state = cfg.mpc.get("use_gt_agent_state", False)
    wm_predictor = WorldModelPredictor(
        cfg.checkpoint_path, device,
        affordance_proximity_gate_scale=aff_gate_scale,
        finger_close_threshold=finger_thresh,
        use_gt_agent_state=use_gt_agent_state,
        cumulative_agent_state_delta=cfg.mpc.get("cumulative_agent_state_delta", True),
    )
    cprint(f"  [1/7] done in {_time.time() - _t1:.2f}s", "white")

    # 2. Setup environment (with proper wrapper chain)
    cprint("\n[2/7] Creating environment...", "cyan")
    _t2 = _time.time()
    num_points = cfg.task.get("num_points", 512)
    env = create_env_from_config(
        task_config_file=cfg.task.task_config_file,
        solution_path=cfg.task.get("solution_path", "lwm.sim"),
        task_name=cfg.task.get("task_name", "sim"),
        num_points=num_points,
        n_obs_steps=1,  # MPC uses single-step observations
        init_state_file=cfg.task.get("init_state_file", None),
    )
    cprint(f"  [2/7] create_env_from_config: {_time.time() - _t2:.2f}s", "white")

    # Reset environment and get initial observation
    _t2b = _time.time()
    obs = env.reset()
    cprint(f"  [2/7] env.reset(): {_time.time() - _t2b:.2f}s", "white")

    # Setup camera for rendering GIF
    base_env = _get_base_env(env)
    base_env.setup_camera_rpy(
        camera_target=None,  # auto-center on object AABB
        distance=1.2,
        rpy=[0, -30, -45],
        fov=60,
        camera_width=640,
        camera_height=480,
    )
    def _capture_frame():
        frame = base_env.render()
        return np.asarray(frame, dtype=np.uint8)
    frames = [_capture_frame()]  # capture initial frame

    # 3. Compute goal affordance (must run before part PCD generator so _eval_handle_joint is set;
    #    part PCD and affordance must use the same handle as in WM training data.)
    # Use canonical goal (040 doors / 080 drawers) from joint_angle_constants when available,
    # so MPC stops when we reach the same goal we use for success rate.
    canonical_goal = OBJECT_JOINT_ANGLE_AT_GOAL.get(str(cfg.task.object_id))
    cprint("\n[3/7] Computing goal affordance...", "cyan")
    _t3 = _time.time()
    goal_affordance, goal_joint_angle = compute_goal_affordance(
        env,
        cfg.task.object_id,
        cfg.task.goal_percentage,
        use_closest_handle=True,
        goal_joint_angle_override=canonical_goal,
    )
    cprint(f"  [3/7] done in {_time.time() - _t3:.2f}s", "white")

    # Read initial joint angle and joint upper limit from PyBullet (for angle-based SR metrics)
    import pybullet as p
    obj_body_id = get_object_body_id(base_env)
    handle_jid = getattr(base_env, "_eval_handle_joint", base_env.handle_joint)
    initial_joint_angle = float(p.getJointState(obj_body_id, handle_jid, physicsClientId=base_env.id)[0])
    joint_info = p.getJointInfo(obj_body_id, handle_jid, physicsClientId=base_env.id)
    # getJointInfo: index 9 = jointUpperLimit (8 = lower). Use it so "fully open" = 1.0 for objects with small range.
    joint_upper_limit = float(joint_info[9]) if joint_info[9] is not None else None
    cprint(f"  Initial joint angle: {initial_joint_angle:.4f}, goal: {goal_joint_angle:.4f}, joint_upper_limit: {joint_upper_limit}", "cyan")

    # Apply affordance offset (hacky displacement for collision avoidance etc.)
    aff_offset = np.array(cfg.cost.get("affordance_offset", [0.0, 0.0, 0.0]), dtype=np.float32)
    if np.any(aff_offset != 0):
        cprint(f"Applying affordance offset: {aff_offset.tolist()} (world frame)", "yellow")
        goal_affordance = goal_affordance + aff_offset

    # 2b. Setup part point cloud generator (after goal affordance so it uses _eval_handle_joint)
    cprint("\n[2b/7] Setting up part point cloud generator...", "cyan")
    _t_pcd_start = _time.time()
    part_pcd_gen = create_part_pcd_generator(
        env, str(cfg.task.object_id), num_points=num_points,
    )

    # Initialize affordance tracking from the same geometry pipeline as the part PCD.
    # Use the old-pipeline affordance as a reference to find the matching handle point.
    _ref_aff = get_current_affordance(env)
    part_pcd_gen.init_affordance_tracking(_ref_aff)

    # Recompute goal affordance from the geometry pipeline so goal and current
    # affordance use the same mesh + transform (no systematic offset).
    goal_affordance = part_pcd_gen.get_goal_affordance(goal_joint_angle)
    if np.any(aff_offset != 0):
        goal_affordance = goal_affordance + aff_offset
    cprint(f"  Goal affordance (geometry): {goal_affordance.tolist()}", "green")
    cprint(f"  [2b] total create_part_pcd_generator: {_time.time() - _t_pcd_start:.2f}s", "white")

    # 4. Create planner (MPPI or CEM)
    planner_type = cfg.mpc.get("planner_type", "mppi")
    cprint(f"\n[4/7] Initializing {planner_type.upper()} planner...", "cyan")

    # Build the "no-op" action: zero position/finger deltas, identity rotation
    # in 6D representation.  Action layout: [pos_delta(3), orient_6d(6), finger(1)]
    # Identity rotation in 6D = first two columns of I₃ flattened = [1,0,0, 0,1,0]
    neutral_action = np.zeros(cfg.task.action_dim)
    neutral_action[3:9] = [1, 0, 0, 0, 1, 0]  # 6D identity rotation

    # Use data-driven action bounds from checkpoint if available, else fall back to config
    if wm_predictor.action_min is not None and cfg.task.get("action_bounds_source", "checkpoint") == "checkpoint":
        action_bounds_low = wm_predictor.action_min.cpu().numpy()
        action_bounds_high = wm_predictor.action_max.cpu().numpy()
        cprint("Using action bounds from WM checkpoint", "green")
    else:
        action_bounds_low = np.array(cfg.task.action_bounds.low)
        action_bounds_high = np.array(cfg.task.action_bounds.high)
        cprint("Using action bounds from config (no bounds in checkpoint)", "yellow")

    wm_batch_size = cfg.mpc.get("wm_batch_size", None)
    if wm_batch_size is not None:
        wm_batch_size = int(wm_batch_size)
        cprint(f"WM batch size: {wm_batch_size} (chunking {cfg.mpc.num_samples} samples)", "cyan")

    planner_kwargs = dict(
        horizon=cfg.mpc.horizon,
        num_samples=cfg.mpc.num_samples,
        action_dim=cfg.task.action_dim,
        noise_sigma=np.array(cfg.mpc.noise_sigma),
        action_bounds=(action_bounds_low, action_bounds_high),
        neutral_action=neutral_action,
        wm_batch_size=wm_batch_size,
        device=device,
    )
    if planner_type == "mppi":
        planner_kwargs["temperature"] = cfg.mpc.temperature
    elif planner_type == "cem":
        planner_kwargs["elite_fraction"] = cfg.mpc.get("elite_fraction", 0.1)
        planner_kwargs["alpha"] = cfg.mpc.get("alpha", 0.25)
    elif planner_type == "icem":
        planner_kwargs["elite_fraction"] = cfg.mpc.get("elite_fraction", 0.1)
        planner_kwargs["alpha"] = cfg.mpc.get("alpha", 0.1)
        planner_kwargs["elites_keep_fraction"] = cfg.mpc.get("elites_keep_fraction", 0.5)
        planner_kwargs["noise_beta"] = cfg.mpc.get("noise_beta", 3.0)
        planner_kwargs["history_length"] = cfg.mpc.get("history_length", 3)
        planner_kwargs["history_elite_frac"] = cfg.mpc.get("history_elite_frac", 0.3)
        planner_kwargs["warmup_iters"] = cfg.mpc.get("warmup_iters", 5)
        planner_kwargs["online_iters"] = cfg.mpc.get("online_iters", 3)
        planner_kwargs["execute_best"] = cfg.mpc.get("execute_best", True)
        planner_kwargs["noise_correlate_time"] = cfg.mpc.get("noise_correlate_time", True)

    planner = create_planner(planner_type, **planner_kwargs)

    # 5. Create cost function
    cost_type = cfg.cost.get("cost_type", "dual")
    cprint(f"\n[5/7] Creating cost function (type={cost_type})...", "cyan")
    cost_fn = create_cost_function(
        cost_type=cost_type,
        ee_aff_weight=cfg.cost.ee_aff_weight,
        aff_goal_weight=cfg.cost.aff_goal_weight,
        terminal_weight=cfg.cost.get("terminal_weight", 5.0),
        clip_affordance=cfg.cost.get("clip_affordance", True),
        grip_cost_weight=cfg.cost.get("grip_cost_weight", 0.0),
        proximity_scale=cfg.cost.get("proximity_scale", 0.04),
        progress_weight=cfg.cost.get("progress_weight", 15.0),
        pull_threshold=cfg.cost.get("pull_threshold", 0.03),
    )

    # 6. Run receding horizon MPC
    cprint("\n[6/7] Running receding horizon MPC...", "cyan")
    cprint(f"Planning for {cfg.execution.num_replanning_steps} steps", "white")

    # Storage for trajectory
    trajectory_data = {
        "states": [],
        "actions": [],
        "affordances": [],
        "costs": [],           # best predicted trajectory cost (per MPC step)
        "observed_costs": [],  # w1*EE→aff + w2*aff→goal at executed step (matches distance plot)
        "goal_distances": [],
        "joint_angles": [],    # actual joint angle from PyBullet per step
        # WM diagnostic fields (predicted vs actual, all world frame)
        "wm_predicted_ee": [],       # WM predicted EE pos [3] (world frame)
        "wm_predicted_aff": [],      # WM predicted affordance [3] (world frame)
        "wm_predicted_cost": [],     # WM predicted 1-step cost (world frame)
        "prediction_errors_ee": [],  # ||pred_ee - actual_ee||
        "prediction_errors_aff": [], # ||pred_aff - actual_aff||
        "ee_to_aff_distances": [],   # ||EE - affordance|| per step
        "clip_deltas": [],           # mean aff→goal shift from clipping (positive = WM optimistic)
        # Contact prediction (from learned contact head, if available)
        "contact_logits": [],        # raw logit per step (>0 = contact predicted)
        "contact_predictions": [],   # binary 0/1 per step
    }

    # Robot base transform: world → robot base frame.
    # WM training data has affordance & part PCD in robot base frame,
    # but agent_state in world frame. We must match this convention.
    from scipy.spatial.transform import Rotation as R_transform
    robot_base_pos, robot_base_quat = base_env.robot.get_base_pos_orient()
    robot_base_pos = np.array(robot_base_pos, dtype=np.float32)
    robot_base_R = R_transform.from_quat(robot_base_quat).as_matrix().astype(np.float32)
    goal_affordance_base = _world_to_base(goal_affordance, robot_base_pos, robot_base_R)

    # Extract initial observation arrays and use part PCD + affordance from geometry pipeline.
    # Both returned in world frame; transform to base frame with same robot pose for consistency.
    agent_pos, _scene_pcd, _aff_from_obs = _extract_obs_arrays(obs)
    _t_first_pcd = _time.time()
    part_pcd_world, affordance_world = part_pcd_gen.get_part_pcd()
    affordance_world = affordance_world + aff_offset
    cprint(f"  [2b] first get_part_pcd() call (mesh sampling + FPS): {_time.time() - _t_first_pcd:.2f}s", "white")
    part_pcd = _world_to_base(part_pcd_world, robot_base_pos, robot_base_R)
    affordance_base = _world_to_base(affordance_world, robot_base_pos, robot_base_R)
    # WM expects: agent_state (world), affordance (base), point_cloud (base)
    current_obs = _obs_to_tensors(agent_pos, part_pcd, affordance_base, device)

    # Save initial PyBullet state for extraction compatibility
    save_states = cfg.logging.get("save_states", True)
    state_buffer = []
    if save_states:
        states_dir = output_dir / "states"
        states_dir.mkdir(parents=True, exist_ok=True)
        state_buffer.append(save_env(base_env))
        cprint(f"  State saving enabled → {states_dir}", "cyan")

    # Annotate initial frame with distances (world-frame distances for display)
    total_steps = cfg.execution.num_replanning_steps
    init_ee_to_aff = np.linalg.norm(agent_pos[:3] - affordance_world)
    init_aff_to_goal = np.linalg.norm(affordance_world - goal_affordance)
    frames[0] = _annotate_frame(
        frames[0], 0, total_steps, init_ee_to_aff, init_aff_to_goal,
    )

    # Collect WM input data for visualization GIF (all in robot base frame)
    wm_input_data = [{
        "part_pcd": part_pcd.copy(),
        "affordance": affordance_base.copy(),
        "ee_pos": _world_to_base(agent_pos[:3], robot_base_pos, robot_base_R),
    }]

    # WM diagnostic data: predicted vs actual per step (all in robot base frame)
    wm_diagnostic_data = []

    # Track whether we've reached the handle (to clear planner history once)
    pull_threshold = float(cfg.cost.get("pull_threshold", 0.01))
    reached_handle = False

    # Gripper schedule: after hovering near handle for N steps, ramp finger toward close
    grip_hover_threshold = float(cfg.cost.get("grip_hover_threshold", 0.03))  # meters
    grip_warmup_steps = int(cfg.cost.get("grip_warmup_steps", 5))             # steps before full close
    grip_close_value = float(cfg.cost.get("grip_close_value", -0.04))         # finger action for closed
    grip_open_value = float(cfg.cost.get("grip_open_value", 0.04))             # finger action for fully open (episode start)
    grip_enabled = bool(cfg.cost.get("grip_enabled", True))
    hover_count = 0  # consecutive steps EE is within grip_hover_threshold of affordance

    # Receding horizon loop
    for step in range(cfg.execution.num_replanning_steps):
        cprint(f"\n--- Step {step+1}/{cfg.execution.num_replanning_steps} ---", "yellow")

        # Optimize action sequence using WM
        # goal_affordance_base is in robot base frame (matches WM training)
        optimal_actions, info = planner.optimize(
            current_obs=current_obs,
            goal_affordance=goal_affordance_base,
            cost_fn=cost_fn,
            world_model=wm_predictor,
            num_iterations=cfg.mpc.num_iterations,
            verbose=True,
        )

        # --- Gripper: first step = fully open; then schedule bias toward close when hovering ---
        action = optimal_actions[0].copy()
        if step == 0:
            action[-1] = grip_open_value
            cprint(f"  Gripper at episode start: finger delta = {grip_open_value:.4f} (fully open)", "cyan")
        if grip_enabled and step > 0:
            # Check current ee-to-aff distance (from previous step's observation)
            cur_ee_raw = current_obs["agent_state"]
            if hasattr(cur_ee_raw, 'cpu'):
                cur_ee_raw = cur_ee_raw.cpu().numpy()
            cur_ee = np.asarray(cur_ee_raw).flatten()[:3]
            _, cur_aff_world = part_pcd_gen.get_part_pcd()
            cur_aff_world = cur_aff_world + aff_offset
            cur_ee_to_aff = float(np.linalg.norm(cur_ee - cur_aff_world))

            if cur_ee_to_aff < grip_hover_threshold:
                hover_count += 1
            else:
                hover_count = 0

            if hover_count > 0 and grip_warmup_steps > 0:
                grip_bias = min(1.0, hover_count / grip_warmup_steps)
                planned_finger = action[-1]
                action[-1] = (1.0 - grip_bias) * planned_finger + grip_bias * grip_close_value
                if grip_bias > 0.01:
                    cprint(
                        f"  Grip schedule: hover={hover_count}, bias={grip_bias:.2f}, "
                        f"finger {planned_finger:.4f} -> {action[-1]:.4f}",
                        "cyan",
                    )

        # --- WM 1-step prediction (before executing action) ---
        # Run the WM with just the chosen first action to see what it predicts
        action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)
        wm_1step = wm_predictor.predict_single_step(
            current_obs["agent_state"],
            current_obs["affordance"],
            current_obs["point_cloud"],
            action_tensor,
        )
        pred_state = wm_1step["next_state"][0].cpu().numpy()    # [10] world frame
        pred_aff_base = wm_1step["next_affordance"][0].cpu().numpy()  # [3] base frame
        pred_pcd_base = wm_1step["predicted_point_cloud"][0].cpu().numpy()  # [N, 3] base frame
        pred_ee_world = pred_state[:3]
        pred_aff_world = _base_to_world(pred_aff_base, robot_base_pos, robot_base_R)

        # Contact prediction from learned contact head (if available)
        step_contact_logit = None
        step_contact_pred = None
        if "contact_logit" in wm_1step:
            step_contact_logit = float(wm_1step["contact_logit"][0, 0].cpu())
            step_contact_pred = int(step_contact_logit > 0.0)

        # Execute first action in simulator
        # MultiStepWrapper expects (n_action_steps, action_dim) = (1, 10)
        action_for_env = action[np.newaxis, :]  # (1, action_dim)
        obs, reward, done, info_env = env.step(action_for_env)

        # Save PyBullet state after this step (for extraction pipeline)
        if save_states:
            state_buffer.append(save_env(base_env))

        # Extract new observation arrays (use part PCD, not scene PCD).
        # Both returned in world frame; transform with same robot pose for consistency.
        new_agent_pos, _scene_pcd, _aff_from_obs = _extract_obs_arrays(obs)
        new_part_pcd_world, new_affordance_world = part_pcd_gen.get_part_pcd()
        new_affordance_world = new_affordance_world + aff_offset
        new_part_pcd = _world_to_base(new_part_pcd_world, robot_base_pos, robot_base_R)
        new_affordance_base = _world_to_base(new_affordance_world, robot_base_pos, robot_base_R)

        # --- Compare WM prediction with reality (all world frame) ---
        actual_ee_world = new_agent_pos[:3]
        ee_pred_error = float(np.linalg.norm(pred_ee_world - actual_ee_world))
        aff_pred_error = float(np.linalg.norm(pred_aff_world - new_affordance_world))

        # Predicted 1-step cost (world frame, same formula as observed cost)
        pred_ee_to_aff = np.linalg.norm(pred_ee_world - pred_aff_world)
        pred_aff_to_goal = np.linalg.norm(pred_aff_world - goal_affordance)
        predicted_cost = (
            float(cfg.cost.ee_aff_weight) * pred_ee_to_aff
            + float(cfg.cost.aff_goal_weight) * pred_aff_to_goal
        )

        # Compute actual distances and observed step cost (world frame)
        ee_to_aff = np.linalg.norm(actual_ee_world - new_affordance_world)
        aff_to_goal_dist = np.linalg.norm(new_affordance_world - goal_affordance)
        observed_cost = (
            float(cfg.cost.ee_aff_weight) * ee_to_aff
            + float(cfg.cost.aff_goal_weight) * aff_to_goal_dist
        )

        contact_str = ""
        if step_contact_logit is not None:
            contact_str = f"  |  Contact: {'YES' if step_contact_pred else 'NO'} (logit={step_contact_logit:.2f})"
        cprint(
            f"  WM prediction error — EE: {ee_pred_error:.4f}m, Aff: {aff_pred_error:.4f}m  |  "
            f"Cost predicted: {predicted_cost:.4f}, actual: {observed_cost:.4f}{contact_str}",
            "magenta",
        )
        # Progress cost diagnostics
        if hasattr(cost_fn, 'last_gate_mean') and cost_fn.last_gate_mean > 0:
            cprint(
                f"  Progress cost: gate={cost_fn.last_gate_mean:.3f}, "
                f"progress={cost_fn.last_progress_mean:.4f}m, "
                f"bonus={cost_fn.last_bonus_mean:.4f}",
                "blue",
            )

        # Capture and annotate frame for GIF (now with dual costs)
        frame = _capture_frame()
        # Check if this step triggers the phase change (for annotation)
        just_reached = (not reached_handle and ee_to_aff < pull_threshold)
        frame = _annotate_frame(
            frame, step + 1, total_steps, ee_to_aff, aff_to_goal_dist,
            actual_cost=observed_cost, predicted_cost=predicted_cost,
            ee_error=ee_pred_error, aff_error=aff_pred_error,
            phase_change=just_reached,
            predict_agent_state=wm_predictor.predict_agent_state,
            contact_logit=step_contact_logit, contact_pred=step_contact_pred,
        )
        frames.append(frame)

        # Collect WM input data for visualization (all in robot base frame)
        wm_input_data.append({
            "part_pcd": new_part_pcd.copy(),
            "affordance": new_affordance_base.copy(),
            "ee_pos": _world_to_base(actual_ee_world, robot_base_pos, robot_base_R),
        })

        # Collect WM diagnostic data for predicted-vs-actual scatter GIF
        wm_diagnostic_data.append({
            "part_pcd": new_part_pcd.copy(),
            "pred_pcd": pred_pcd_base.copy(),
            "pred_ee_base": _world_to_base(pred_ee_world, robot_base_pos, robot_base_R),
            "pred_aff_base": pred_aff_base.copy(),
            "actual_ee_base": _world_to_base(actual_ee_world, robot_base_pos, robot_base_R),
            "actual_aff_base": new_affordance_base.copy(),
        })

        # Log trajectory (world frame for metrics/plots)
        trajectory_data["states"].append(new_agent_pos.copy())
        trajectory_data["actions"].append(action.copy())
        trajectory_data["affordances"].append(new_affordance_world.copy())
        trajectory_data["costs"].append(info["final_best_cost"])
        trajectory_data["observed_costs"].append(observed_cost)
        trajectory_data["goal_distances"].append(aff_to_goal_dist)
        trajectory_data["wm_predicted_ee"].append(pred_ee_world.copy())
        trajectory_data["wm_predicted_aff"].append(pred_aff_world.copy())
        trajectory_data["wm_predicted_cost"].append(predicted_cost)
        trajectory_data["prediction_errors_ee"].append(ee_pred_error)
        trajectory_data["prediction_errors_aff"].append(aff_pred_error)
        trajectory_data["ee_to_aff_distances"].append(ee_to_aff)
        trajectory_data["clip_deltas"].append(0.0)
        trajectory_data["contact_logits"].append(step_contact_logit if step_contact_logit is not None else float('nan'))
        trajectory_data["contact_predictions"].append(step_contact_pred if step_contact_pred is not None else float('nan'))

        # Read current joint angle from PyBullet
        current_joint_angle = float(p.getJointState(obj_body_id, handle_jid, physicsClientId=base_env.id)[0])
        trajectory_data["joint_angles"].append(current_joint_angle)

        finger_delta = float(action[-1])
        # Read gripper openness from PyBullet (0=closed, 0.04=open → normalize to 0-1)
        gripper_joint = float(p.getJointState(
            base_env.robot.body, base_env.robot.right_gripper_indices[0],
            physicsClientId=base_env.id
        )[0])
        gripper_openness = max(0.0, min(1.0, gripper_joint / 0.04))
        finger_delta_pct = finger_delta / 0.04 * 100  # % of full range
        cprint(
            f"EE→Aff: {ee_to_aff:.4f}m  |  Aff→Goal: {aff_to_goal_dist:.4f}m  |  "
            f"Gripper: {gripper_openness:.0%} open (delta: {finger_delta_pct:+.1f}%)  |  "
            f"Angle: {current_joint_angle:.4f}",
            "white",
        )


        # Check goal condition
        if aff_to_goal_dist < cfg.execution.goal_tolerance:
            cprint(f"\nGoal reached at step {step+1}!", "green")
            break

        if done:
            cprint(f"\nEnvironment done at step {step+1}", "yellow")
            break

        # Update WM input tensors for next planning iteration
        # agent_state (world), affordance (base), point_cloud (base)
        current_obs = _obs_to_tensors(new_agent_pos, new_part_pcd, new_affordance_base, device)

        # When EE first reaches the handle, do a FULL planner reset:
        # - Clears elite history (stale "approach" trajectories)
        # - Resets mean_actions to neutral (stops forward momentum)
        # - Triggers warmup iterations on next call
        # - Skips shift (which would re-bias toward approach)
        if not reached_handle and ee_to_aff < pull_threshold:
            reached_handle = True
            planner.reset()
            cprint(
                f"  >> PHASE CHANGE: EE reached handle "
                f"(ee_to_aff={ee_to_aff:.4f}m < {pull_threshold}m), "
                f"full planner reset",
                "green",
            )
        else:
            # Normal warm-start: shift mean actions forward by one step
            planner.shift_and_init_last_action()


    # 7. Save results
    cprint("\n[7/7] Saving results...", "cyan")

    # Add metadata for visualization and angle-based metrics
    trajectory_data["initial_joint_angle"] = initial_joint_angle
    trajectory_data["goal_joint_angle"] = goal_joint_angle
    trajectory_data["pull_threshold"] = pull_threshold
    if reached_handle:
        # Find the step where phase change occurred
        ee_aff_dists = trajectory_data["ee_to_aff_distances"]
        for i, d in enumerate(ee_aff_dists):
            if d < pull_threshold:
                trajectory_data["phase_change_step"] = i
                break

    # Convert lists to arrays (skip non-list metadata)
    for key in trajectory_data:
        if isinstance(trajectory_data[key], list):
            trajectory_data[key] = np.array(trajectory_data[key])

    # Sanity check: if we ran multiple steps, state/affordance (and thus cost) should change
    n_steps = len(trajectory_data["states"])
    if n_steps > 1:
        states = trajectory_data["states"]
        affs = trajectory_data["affordances"]
        obs_costs = trajectory_data["observed_costs"]
        state_unchanged = np.abs(states - states[0]).max() < 1e-6
        aff_unchanged = np.abs(affs - affs[0]).max() < 1e-6
        cost_unchanged = np.abs(obs_costs - obs_costs[0]).max() < 1e-6
        if (state_unchanged and aff_unchanged) or cost_unchanged:
            cprint(
                "WARNING: Multiple steps ran but states/affordances or observed cost are effectively unchanged. "
                "Check that actions are applied correctly (e.g. env.step, action scale).",
                "yellow",
            )

    # Compute metrics (includes angle-based normalized_performance; joint_upper_limit caps goal so full open = 1.0)
    metrics = compute_trajectory_metrics(trajectory_data, goal_affordance,
                                         initial_joint_angle=initial_joint_angle,
                                         goal_joint_angle=goal_joint_angle,
                                         joint_upper_limit=joint_upper_limit)

    # Add planning configuration to metrics
    metrics["config"] = {
        "object_id": cfg.task.object_id,
        "goal_percentage": float(cfg.task.goal_percentage),
        "planner_type": planner_type,
        "horizon": int(cfg.mpc.horizon),
        "num_samples": int(cfg.mpc.num_samples),
        "seed": seed,
    }

    # Save trajectory
    if cfg.logging.save_trajectories:
        save_trajectory(
            trajectory_data,
            output_dir / "trajectory.pkl",
            metadata=metrics["config"],
        )

    # Save metrics
    save_metrics(
        metrics,
        output_dir / "metrics.json",
    )

    # Save PyBullet states and extraction-compatible metadata files
    if save_states and state_buffer:
        # Write state_*.pkl files
        for idx, state in enumerate(state_buffer):
            state_path = states_dir / f"state_{idx}.pkl"
            with open(state_path, "wb") as f:
                pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)
        cprint(f"Saved {len(state_buffer)} PyBullet states to {states_dir}", "green")
        state_buffer.clear()

        # task_config.yaml (copy from the config used to create this episode)
        task_config_src = str(cfg.task.task_config_file)
        task_config_dst = output_dir / "task_config.yaml"
        if os.path.exists(task_config_src) and not task_config_dst.exists():
            shutil.copy(task_config_src, task_config_dst)

        # stage_lengths.json (all zeros, same convention as rollouts)
        stage_lengths_path = output_dir / "stage_lengths.json"
        with open(stage_lengths_path, "w") as f:
            json.dump({"reach_handle": 0, "reach_to_contact": 0,
                        "close_gripper": 0, "open_door": 0}, f, indent=4)

        # handle_metadata.json (required by extraction in rollout mode)
        handle_meta = {
            "handle_joint_idx": int(handle_jid),
            "object_id": str(cfg.task.object_id),
            "object_scale": float(getattr(base_env, "simulator_sizes", {}).get(
                next((k for k in getattr(base_env, "urdf_ids", {}) if str(cfg.task.object_id) in k), ""), 1.0)),
            "partnet_object_path": os.path.join(
                os.environ.get("PARTNET_DATASET_DIR", ""), str(cfg.task.object_id)),
        }
        with open(output_dir / "handle_metadata.json", "w") as f:
            json.dump(handle_meta, f, indent=2)

        # opened_angle.txt (final angle, lower limit, upper limit)
        with open(output_dir / "opened_angle.txt", "w") as f:
            final_ja = float(trajectory_data["joint_angles"][-1]) if len(trajectory_data["joint_angles"]) > 0 else initial_joint_angle
            f.write(f"{final_ja}\n")
            f.write(f"{float(joint_info[8])}\n")   # jointLowerLimit
            f.write(f"{float(joint_info[9])}\n")   # jointUpperLimit

        cprint("Saved extraction-compatible metadata (task_config, handle_metadata, stage_lengths, opened_angle)", "green")

    # Determine final joint angle for GIF filenames
    joint_angles_arr = trajectory_data.get("joint_angles")
    if isinstance(joint_angles_arr, np.ndarray) and len(joint_angles_arr) > 0:
        final_joint_angle = float(joint_angles_arr[-1])
    else:
        final_joint_angle = initial_joint_angle
    angle_suffix = f"_final{final_joint_angle:.3f}"

    # Save rendered GIF
    if frames:
        import imageio
        gif_path = output_dir / f"mpc_episode{angle_suffix}.gif"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(gif_path), frames, fps=10, loop=0)
        cprint(f"GIF saved to {gif_path} ({len(frames)} frames)", "green")

    # Save WM input visualization GIF (3D scatter: part PCD + affordance + EE)
    if wm_input_data:
        wm_gif_path = output_dir / f"mpc_wm_inputs{angle_suffix}.gif"
        n_wm = _render_wm_input_frames(wm_input_data, goal_affordance_base, wm_gif_path)
        cprint(f"WM input GIF saved to {wm_gif_path} ({n_wm} frames)", "green")

    # Save WM diagnostic GIF (predicted vs actual, side-by-side 3D scatter)
    if wm_diagnostic_data:
        diag_gif_path = output_dir / f"mpc_wm_diagnostic{angle_suffix}.gif"
        n_diag = _render_wm_diagnostic_frames(
            wm_diagnostic_data, goal_affordance_base, diag_gif_path,
        )
        cprint(f"WM diagnostic GIF saved to {diag_gif_path} ({n_diag} frames)", "green")

    # Visualize
    if cfg.logging.visualize:
        visualize_planned_trajectory(
            trajectory_data,
            output_dir / "trajectory_plot.png",
            goal_affordance=goal_affordance,
        )

    # Print summary
    print_planning_summary(
        metrics,
        cfg.task.object_id,
        float(cfg.task.goal_percentage),
    )

    # Cleanup
    part_pcd_gen.close()

    cprint("\nDone!", "green")


if __name__ == "__main__":
    main()
