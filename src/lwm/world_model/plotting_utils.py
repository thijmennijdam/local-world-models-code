"""
Utility functions for creating 3D plots for long horizon validation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lwm.common.wandb_compat import wandb
import io
from PIL import Image


def plot_affordance_trajectory_3d(ax, pred_aff, gt_aff, point_clouds=None, elev=20, azim=45):
    """
    Plot affordance point trajectories in 3D.
    
    Args:
        ax: Matplotlib 3D axis
        pred_aff: Predicted affordance trajectory [T, 3]
        gt_aff: Ground truth affordance trajectory [T, 3]
        point_clouds: Optional list of point clouds for context
        elev: Elevation angle for view
        azim: Azimuth angle for view
    """
    ax.set_facecolor("white")
    
    # Plot ground truth affordance trajectory as thin lines with decreasing opacity
    T = len(gt_aff)
    alpha_start = 0.9
    alpha_end = 0.2
    for t in range(T - 1):
        # Alpha decreases linearly from start to end
        alpha = alpha_start - (alpha_start - alpha_end) * (t / (T - 1))
        ax.plot(gt_aff[t:t+2, 0], gt_aff[t:t+2, 1], gt_aff[t:t+2, 2],
               c="green", linewidth=1, alpha=alpha, zorder=3)
    # Add label only once
    ax.plot([], [], c="green", linewidth=1, label="GT Affordance", zorder=3)
    
    # Plot predicted affordance trajectory as thin lines with decreasing opacity
    T_pred = len(pred_aff)
    for t in range(T_pred - 1):
        # Alpha decreases linearly from start to end
        alpha = alpha_start - (alpha_start - alpha_end) * (t / (T_pred - 1))
        ax.plot(pred_aff[t:t+2, 0], pred_aff[t:t+2, 1], pred_aff[t:t+2, 2],
               c="red", linewidth=1, linestyle="--", alpha=alpha, zorder=3)
    # Add label only once
    ax.plot([], [], c="red", linewidth=1, linestyle="--", label="Pred Affordance", zorder=3)
    
    # Set bounds
    all_points = np.vstack([gt_aff, pred_aff])
    
    if all_points.shape[0] > 0:
        padding = 0.05
        x_min, x_max = all_points[:, 0].min() - padding, all_points[:, 0].max() + padding
        y_min, y_max = all_points[:, 1].min() - padding, all_points[:, 1].max() + padding
        z_min, z_max = all_points[:, 2].min() - padding, all_points[:, 2].max() + padding
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=elev, azim=azim)


def plot_agent_state_trajectory_3d(ax, pred_agent, gt_agent, elev=20, azim=45):
    """
    Plot agent state trajectories in 3D (first 3 dimensions).
    
    Args:
        ax: Matplotlib 3D axis
        pred_agent: Predicted agent state trajectory [T, agent_state_dim]
        gt_agent: Ground truth agent state trajectory [T, agent_state_dim]
        elev: Elevation angle for view
        azim: Azimuth angle for view
    """
    ax.set_facecolor("white")
    
    agent_dim = min(3, gt_agent.shape[1])
    
    # Plot ground truth agent state trajectory as thin lines with decreasing opacity
    z_gt = gt_agent[:, 2] if agent_dim > 2 else np.zeros(len(gt_agent))
    T = len(gt_agent)
    alpha_start = 0.9
    alpha_end = 0.2
    for t in range(T - 1):
        # Alpha decreases linearly from start to end
        alpha = alpha_start - (alpha_start - alpha_end) * (t / (T - 1))
        ax.plot(gt_agent[t:t+2, 0], gt_agent[t:t+2, 1], z_gt[t:t+2],
               c="blue", linewidth=1, alpha=alpha, zorder=3)
    # Add label only once
    ax.plot([], [], c="blue", linewidth=1, label="GT Agent State", zorder=3)
    
    # Plot predicted agent state trajectory as thin lines with decreasing opacity
    if pred_agent is not None and len(pred_agent) > 0:
        z_pred = pred_agent[:, 2] if agent_dim > 2 else np.zeros(len(pred_agent))
        T_pred = len(pred_agent)
        for t in range(T_pred - 1):
            # Alpha decreases linearly from start to end
            alpha = alpha_start - (alpha_start - alpha_end) * (t / (T_pred - 1))
            ax.plot(pred_agent[t:t+2, 0], pred_agent[t:t+2, 1], z_pred[t:t+2],
                   c="orange", linewidth=1, linestyle="--", alpha=alpha, zorder=3)
        # Add label only once
        ax.plot([], [], c="orange", linewidth=1, linestyle="--", label="Pred Agent State", zorder=3)
    
    # Set bounds
    if agent_dim >= 3:
        if pred_agent is not None and len(pred_agent) > 0:
            all_points = np.vstack([gt_agent[:, :3], pred_agent[:, :3]])
        else:
            all_points = gt_agent[:, :3]
    else:
        # Pad with zeros for z dimension
        gt_padded = np.zeros((len(gt_agent), 3))
        gt_padded[:, :agent_dim] = gt_agent[:, :agent_dim]
        if pred_agent is not None and len(pred_agent) > 0:
            pred_padded = np.zeros((len(pred_agent), 3))
            pred_padded[:, :agent_dim] = pred_agent[:, :agent_dim]
            all_points = np.vstack([gt_padded, pred_padded])
        else:
            all_points = gt_padded
    
    if all_points.shape[0] > 0:
        padding = 0.05
        x_min, x_max = all_points[:, 0].min() - padding, all_points[:, 0].max() + padding
        y_min, y_max = all_points[:, 1].min() - padding, all_points[:, 1].max() + padding
        if agent_dim > 2:
            z_min, z_max = all_points[:, 2].min() - padding, all_points[:, 2].max() + padding
        else:
            z_min, z_max = -0.1, 0.1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z" if agent_dim > 2 else "Z (const)")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=elev, azim=azim)


def create_long_horizon_plots(episode_trajectories, num_plots=5):
    """
    Create 3D plots for long horizon validation trajectories.
    
    Args:
        episode_trajectories: Dict mapping episode index to trajectory dict with:
            - pred_affordance: [T, 3]
            - gt_affordance: [T, 3]
            - pred_agent_state: [T, agent_state_dim]
            - gt_agent_state: [T, agent_state_dim]
            - point_clouds: List of point clouds
        num_plots: Number of episodes to plot
    
    Returns:
        Dict mapping wandb keys to wandb.Image objects
    """
    plot_metrics = {}
    num_plots = min(num_plots, len(episode_trajectories))
    
    views = [(20, 45), (20, 225), (20, 135)]
    
    for plot_idx in range(num_plots):
        if plot_idx not in episode_trajectories:
            continue
            
        traj = episode_trajectories[plot_idx]
        pred_aff = traj["pred_affordance"]
        gt_aff = traj["gt_affordance"]
        pred_agent = traj.get("pred_agent_state", None)
        gt_agent = traj["gt_agent_state"]
        point_clouds = traj["point_clouds"]
        
        # Create figure with 6 subplots (3 views for affordance, 3 for agent state)
        fig = plt.figure(figsize=(16, 12), dpi=200)
        fig.patch.set_facecolor("white")
        
        # Plot affordance trajectories (top row)
        for view_idx, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 3, view_idx + 1, projection="3d")
            plot_affordance_trajectory_3d(ax, pred_aff, gt_aff, point_clouds, elev, azim)
            ax.set_title(f"Affordance View {view_idx + 1}")
        
        # Plot agent state trajectories (bottom row)
        for view_idx, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 3, view_idx + 4, projection="3d")
            plot_agent_state_trajectory_3d(ax, pred_agent, gt_agent, elev, azim)
            ax.set_title(f"Agent State View {view_idx + 1}")
        
        plt.tight_layout()
        
        # Convert to image and log to wandb
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        pil_image = Image.open(buf)
        plot_metrics[f"val_long_horizon_plots/episode_{plot_idx}_affordance_agent_state"] = wandb.Image(pil_image)
        plt.close(fig)
    
    return plot_metrics


def plot_error_vs_step(step_indices, step_total_errors, step_agent_errors=None, step_affordance_errors=None, title=None):
    """
    Create a matplotlib plot for error vs step and convert to wandb.Image.
    
    Args:
        step_indices: List of step indices
        step_total_errors: List of total errors at each step
        step_agent_errors: Optional list of agent state errors at each step
        step_affordance_errors: Optional list of affordance errors at each step
        title: Optional title for the plot (default: "Error vs Step")
    
    Returns:
        wandb.Image object
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("white")
    
    # Use 1-based step labels
    x_labels = [s + 1 for s in step_indices]

    # Plot total error
    ax.plot(x_labels, step_total_errors, 'b-', linewidth=2, marker='o', markersize=4, label='Total Error')

    # Optionally plot agent and affordance errors separately
    if step_agent_errors is not None:
        ax.plot(x_labels, step_agent_errors, 'g--', linewidth=1.5, alpha=0.7, label='Agent State Error')
    if step_affordance_errors is not None:
        ax.plot(x_labels, step_affordance_errors, 'r--', linewidth=1.5, alpha=0.7, label='Affordance Error')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Cumulative Error (MSE)', fontsize=12)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x_labels)
    if title is None:
        title = 'Cumulative Error vs Step'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Convert to image and return as wandb.Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close(fig)
    
    return wandb.Image(pil_image)
