"""
Training script for Local World Model (LWM).
"""

import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau
from termcolor import cprint
from lwm.common.wandb_compat import wandb
import pathlib
import datetime
import time
import random
import numpy as np
import json
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
import shutil
from lwm.world_model.plotting_utils import create_long_horizon_plots, plot_error_vs_step
from lwm.world_model.training_utils import setup_checkpoint_dir, save_config_to_checkpoint, initialize_wandb, save_checkpoint
from lwm.world_model.validation_utils import (
    validate_separate, evaluate_near_affordance_rollout, log_evaluation_results,
    select_val_windows, compute_val_rollout_metrics,
)
from lwm.world_model.evaluation_utils import (
    rotation_transfer_6D_to_matrix_torch,
    rotation_transfer_matrix_to_6D_torch,
    compose_state_with_delta,
    get_normalization_stats,
    load_episode_data,
    build_observation,
    predict_next_state_and_affordance,
    get_near_affordance_step_indices,
)

from lwm.common.pytorch_util import dict_apply
from lwm.paths import CONFIG_DIR
from lwm.world_model.dataset import LWMDataset
from lwm.world_model.targets import check_frames


def set_seed(seed: int = 42):
    """Set all random seeds for maximum determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set CuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def compute_component_grad_norms(component_losses, params):
    """Compute gradient L2 norms for individual loss components w.r.t. model parameters.

    Each component loss is differentiated independently via torch.autograd.grad
    (retain_graph=True) so the main backward pass can still run afterwards.

    Note: with DDP these are local (per-rank) gradient norms, not all-reduced.

    Args:
        component_losses: dict of {name: scalar_loss_tensor}
        params: list of model parameters (requires_grad=True)

    Returns:
        dict of {name: float gradient_norm}
    """
    grad_norms = {}
    for name, loss in component_losses.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        total_norm_sq = 0.0
        for g in grads:
            if g is not None:
                total_norm_sq += g.data.norm(2).item() ** 2
        grad_norms[name] = total_norm_sq ** 0.5
    return grad_norms


# How often (in training steps) to compute per-component gradient norms.
# Each probe runs ~5 extra backward-like passes, so 100 keeps overhead at ~5%.
GRAD_NORM_LOG_INTERVAL = 100


def is_distributed():
    return "LOCAL_RANK" in os.environ

def get_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process():
    return get_rank() == 0

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    torch.cuda.set_device(get_rank())


class DistributedWeightedSampler(torch.utils.data.Sampler):
    """Weighted sampling (e.g. 50/50 expert/interaction) with each rank seeing different indices for DDP."""

    def __init__(self, weights, num_replicas, rank, seed=0):
        self.weights = weights if isinstance(weights, torch.Tensor) else torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(self.weights) // num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.rank + self.epoch * 10000)
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def wandb_active():
    """True when a W&B run is live (WANDB_MODE=online/offline and wandb installed)."""
    return getattr(wandb, "run", None) is not None


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr_scheduler(optimizer, scheduler_type: str = "none", num_epochs: int = 1, 
                     num_batches_per_epoch: int = 1, **kwargs):
    """Get learning rate scheduler based on type."""
    if scheduler_type.lower() == "none" or scheduler_type.lower() == "":
        return None
    elif scheduler_type.lower() == "cosine":
        T_max = num_epochs * num_batches_per_epoch
        # CosineAnnealingLR doesn't accept step_size or gamma, filter them out
        cosine_kwargs = {k: v for k, v in kwargs.items() if k not in ["step_size", "gamma"]}
        return CosineAnnealingLR(optimizer, T_max=T_max, **cosine_kwargs)
    elif scheduler_type.lower() == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type.lower() == "exponential":
        gamma = kwargs.get("gamma", 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type.lower() == "plateau":
        # ReduceLROnPlateau has different default parameters
        plateau_kwargs = {
            "mode": kwargs.get("mode", "min"),
            "factor": kwargs.get("factor", 0.5),
            "patience": kwargs.get("patience", 5),
            "verbose": kwargs.get("verbose", True),
        }
        return ReduceLROnPlateau(optimizer, **plateau_kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Choose from: none, cosine, step, exponential, plateau")


def evaluate_long_horizon(
    lwm,
    lwm_dataset,
    device,
    fixed_val_episodes,
    num_episodes=8,
    max_steps_per_episode=256,
    data_type="combined",
    checkpoint_dir=None,
    epoch=None,
    action_mean=None,
    action_std=None,
    predict_agent_state: bool = True,
    global_step: int = None,
):
    """
    Long-horizon autoregressive validation on a fixed subset of validation episodes.

    - Uses pre-selected validation episodes (fixed_val_episodes)
    - Autoregressively rolls out agent_state and static_affordance
      using expert actions and pointclouds from the dataset.
    - Logs aggregate per-step MSE and per-step error curve.
    
    Args:
        data_type: "expert", "rollout", or "combined" - used for metric naming
    """
    lwm.eval()
    criterion = nn.MSELoss(reduction="mean")

    rb = lwm_dataset.replay_buffer
    n_obs_steps = lwm_dataset.n_obs_steps  # Get n_obs_steps from dataset
    horizon = lwm.n_future_steps
    
    # Get normalization stats from dataset
    (action_mean, action_std, agent_state_delta_mean, agent_state_delta_std,
     affordance_delta_mean, affordance_delta_std) = get_normalization_stats(
        lwm_dataset, device, action_mean, action_std
    )

    if len(fixed_val_episodes) == 0:
        return None

    # Use fixed episodes (already subset if needed)
    selected_episodes = fixed_val_episodes[: min(num_episodes, len(fixed_val_episodes))]

    total_agent_mse = 0.0
    total_affordance_mse = 0.0
    total_steps = 0

    # Track per-step errors (step index -> list of errors at that step)
    per_step_agent_errors = {}
    per_step_affordance_errors = {}
    
    # Store trajectories for plotting (episode_idx -> trajectories dict)
    episode_trajectories = {}

    with torch.no_grad():
        for ep_idx_idx, ep_idx in enumerate(selected_episodes):
            # Episode length and path
            ep_len = int(rb.episode_lengths[ep_idx])
            if ep_len <= horizon + 1:
                continue  # too short for our horizon

            # Load full episode from disk
            agent_states, static_affordances, point_clouds, actions, _ = load_episode_data(
                rb, ep_idx, device
            )

            # Initial state from ground truth
            cur_state = agent_states[0]          # [agent_state_dim]
            cur_aff = static_affordances[0]      # [3]

            # We can only form full action horizons up to T - horizon - 1
            max_t = min(ep_len - horizon - 1, max_steps_per_episode)
            if max_t <= 0:
                continue

            # Initialize trajectory storage for this episode
            pred_affordance_traj = []
            gt_affordance_traj = []
            pred_agent_state_traj = []
            gt_agent_state_traj = []
            point_cloud_samples = []  # Store a few point clouds for context

            for t in range(max_t):
                # Build observation for this step
                action_horizon = actions[t : t + horizon]  # [horizon, action_dim]
                action_horizon_normalized = (action_horizon - action_mean) / action_std
                
                observation = build_observation(
                    t, cur_state, cur_aff, agent_states, static_affordances,
                    point_clouds, action_horizon_normalized, n_obs_steps, horizon
                )
                
                # Predict next state and affordance
                next_state, next_aff, _, _ = predict_next_state_and_affordance(
                    lwm, observation, cur_state, cur_aff,
                    agent_state_delta_mean, agent_state_delta_std,
                    affordance_delta_mean, affordance_delta_std,
                    predict_agent_state, compose_state_with_delta,
                    gt_state_fallback=agent_states[t + 1] if not predict_agent_state else None
                )

                gt_state = agent_states[t + 1]
                gt_aff = static_affordances[t + 1]

                agent_mse = criterion(next_state, gt_state)
                affordance_mse = criterion(next_aff, gt_aff)

                if predict_agent_state:
                    total_agent_mse += agent_mse.item()
                total_affordance_mse += affordance_mse.item()
                total_steps += 1

                # Store trajectories (convert to numpy for plotting)
                pred_affordance_traj.append(next_aff.detach().cpu().numpy())
                gt_affordance_traj.append(gt_aff.detach().cpu().numpy())
                pred_agent_state_traj.append(next_state.detach().cpu().numpy())
                gt_agent_state_traj.append(gt_state.detach().cpu().numpy())
                
                # Store point cloud samples (every 10 steps to avoid too much data)
                if t % 10 == 0:
                    point_cloud_samples.append(point_clouds[t].detach().cpu().numpy())

                # Track per-step errors (step index in rollout, not episode timestep)
                step_idx = t
                # Always ensure affordance error list exists
                if step_idx not in per_step_affordance_errors:
                    per_step_affordance_errors[step_idx] = []
                # Optionally track agent errors
                if predict_agent_state:
                    if step_idx not in per_step_agent_errors:
                        per_step_agent_errors[step_idx] = []
                    per_step_agent_errors[step_idx].append(agent_mse.item())
                per_step_affordance_errors[step_idx].append(affordance_mse.item())

                # Autoregressive: feed prediction back in
                cur_state = next_state.detach()
                cur_aff = next_aff.detach()
            
            # Store trajectories for this episode
            if len(pred_affordance_traj) > 0:
                episode_trajectories[ep_idx_idx] = {
                    "pred_affordance": np.array(pred_affordance_traj),
                    "gt_affordance": np.array(gt_affordance_traj),
                    "pred_agent_state": np.array(pred_agent_state_traj),
                    "gt_agent_state": np.array(gt_agent_state_traj),
                    "point_clouds": point_cloud_samples,
                }

    if total_steps == 0:
        return None

    if predict_agent_state:
        avg_agent_mse = total_agent_mse / total_steps
    avg_affordance_mse = total_affordance_mse / total_steps
    if predict_agent_state:
        avg_total_mse = avg_agent_mse + avg_affordance_mse
    else:
        avg_total_mse = avg_affordance_mse

    # Compute per-step averages and create matplotlib plot
    step_indices = sorted(per_step_affordance_errors.keys())
    step_agent_errors = [np.mean(per_step_agent_errors[s]) for s in step_indices] if predict_agent_state else None
    step_affordance_errors = [np.mean(per_step_affordance_errors[s]) for s in step_indices]
    if predict_agent_state:
        step_total_errors = [a + f for a, f in zip(step_agent_errors, step_affordance_errors)]
    else:
        step_total_errors = step_affordance_errors

    # Create matplotlib plot and convert to wandb.Image (like affordance plots)
    error_plot = plot_error_vs_step(
        step_indices=step_indices,
        step_total_errors=step_total_errors,
        step_agent_errors=step_agent_errors,
        step_affordance_errors=step_affordance_errors,
        title=f'Long-horizon validation ({data_type}): Error vs Rollout Step',
    )

    # Save raw error data for later reuse
    error_data = {
        "step_indices": step_indices,
        "step_total_errors": step_total_errors,
        "step_agent_errors": step_agent_errors,
        "step_affordance_errors": step_affordance_errors,
        "avg_total_mse": float(avg_total_mse),
        "avg_affordance_mse": float(avg_affordance_mse),
        "total_steps": int(total_steps),
        "num_episodes": len(selected_episodes),
        "data_type": data_type,
    }
    if predict_agent_state:
        error_data["avg_agent_mse"] = float(avg_agent_mse)
    
    # Save to checkpoint directory if available
    if checkpoint_dir is not None:
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create validation data subdirectory
        val_data_dir = checkpoint_dir / "validation_data"
        val_data_dir.mkdir(exist_ok=True)
        
        # Save with epoch and data_type in filename
        epoch_str = f"epoch_{epoch:04d}_" if epoch is not None else ""
        error_data_file = val_data_dir / f"{epoch_str}error_vs_step_{data_type}.json"
        
        with open(error_data_file, 'w') as f:
            json.dump(error_data, f, indent=2)
        
        cprint(f"Saved raw error data to {error_data_file}", "cyan")
    
    # Determine metric prefix based on data_type
    if data_type.startswith("train_"):
        # Strip "train_" prefix from data_type for metric naming
        data_type_suffix = data_type[len("train_"):]
        metric_prefix = f"train_long_horizon_{data_type_suffix}"
    else:
        metric_prefix = f"val_long_horizon_{data_type}"
    
    metrics = {
        f"{metric_prefix}/total_loss": avg_total_mse,
        f"{metric_prefix}/affordance_loss": avg_affordance_mse,
        f"{metric_prefix}/error_vs_step": error_plot,
    }
    if predict_agent_state:
        metrics[f"{metric_prefix}/agent_state_loss"] = avg_agent_mse

    # Create 3D plots for affordance points and agent states using utility function
    plot_metrics = create_long_horizon_plots(episode_trajectories, num_plots=5)
    
    # Update plot metric keys with data type
    if plot_metrics:
        updated_plot_metrics = {}
        for key, value in plot_metrics.items():
            # Replace val_long_horizon_plots with appropriate prefix
            if data_type.startswith("train_"):
                # Strip "train_" prefix from data_type for metric naming
                data_type_suffix = data_type[len("train_"):]
                new_key = key.replace("val_long_horizon_plots", f"train_long_horizon_plots_{data_type_suffix}")
            else:
                new_key = key.replace("val_long_horizon_plots", f"val_long_horizon_plots_{data_type}")
            updated_plot_metrics[new_key] = value
        plot_metrics = updated_plot_metrics

    # Log to wandb under separate folder
    if global_step is not None:
        wandb.log(metrics, step=global_step)
        if plot_metrics:
            wandb.log(plot_metrics, step=global_step)
    else:
        wandb.log(metrics)
        if plot_metrics:
            wandb.log(plot_metrics)

    return metrics


def compute_horizon_weights(n_future_steps: int, decay_type: str = "linear") -> torch.Tensor:
    """
    Compute weights for each horizon step. Earlier steps get higher weight.
    
    Args:
        n_future_steps: Number of future steps (horizon)
        decay_type: Type of decay - "linear", "exponential", or "none"
    
    Returns:
        Tensor of shape [n_future_steps] with weights that sum to n_future_steps (so mean is 1.0)
    """
    if decay_type == "none":
        return torch.ones(n_future_steps)
    elif decay_type == "linear":
        # Linear decay: step 1 gets highest weight, step n gets lowest
        # weights = [n, n-1, n-2, ..., 1]
        weights = torch.arange(n_future_steps, 0, -1, dtype=torch.float32)
    elif decay_type == "exponential":
        # Exponential decay: weights = [1, 0.8, 0.64, 0.512, ...]
        decay_factor = 0.8
        weights = torch.tensor([decay_factor ** i for i in range(n_future_steps)], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}. Choose from: none, linear, exponential")
    
    # Normalize so weights sum to n_future_steps (mean weight = 1.0)
    weights = weights * (n_future_steps / weights.sum())
    return weights


def train_lwm(lwm: nn.Module, dataloader: DataLoader, val_dataloader: DataLoader = None, num_epochs: int = 1, device: str = "cuda",
              learning_rate: float = 0.001, batch_size: int = 32,
              pointcloud_encoder_size: int = 256,
              checkpoint_dir: str = None,
              lr_scheduler_type: str = "none", lr_scheduler_kwargs: dict = None,
              lr_warmup_steps: int = 0,
              max_grad_norm: float = None, exp_name: str = "lwm_training", output_dir: str = None,
              wandb_cfg = None, full_cfg = None, lwm_dataset = None, long_horizon_num_episodes: int = 50,
              # New parameters for improved training
              weight_decay: float = 0.01,
              affordance_loss_weight: float = 1.0,
              horizon_weight_decay: str = "none",
              # Whether to predict agent state (if False, only affordance is trained/logged)
              predict_agent_state: bool = False,
              train_sampler: DistributedSampler = None,
              use_amp: bool = False,
              # Contact prediction
              predict_contact: str = "none",
              contact_loss_weight: float = 1.0):
    distributed = is_distributed()
    rank0 = is_main_process()

    lwm = lwm.to(device)
    if distributed:
        lwm = DDP(lwm, device_ids=[get_rank()], find_unused_parameters=True)
    
    # Attach weight_decay to model instance so it can be picked up by wandb config
    lwm.weight_decay = weight_decay

    # Count and log model parameters (use unwrapped model for DDP)
    model_unwrapped = lwm.module if distributed else lwm
    num_params = count_parameters(model_unwrapped)
    num_params_m = num_params / 1e6
    if rank0:
        cprint(f"Model has {num_params:,} trainable parameters ({num_params_m:.2f}M)", "cyan")

    # Use AdamW with weight decay for regularization
    optimizer = torch.optim.AdamW(lwm.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if rank0:
        cprint(f"Using AdamW optimizer with weight_decay={weight_decay}", "cyan")

    # Mixed precision training (AMP)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if rank0 and use_amp:
        cprint("Using automatic mixed precision (AMP)", "cyan")

    # Per-element MSE for horizon weighting
    criterion = nn.MSELoss(reduction='none')

    # Per-component weights for agent state loss: [agent_state_dim]
    # Indices: 0:3 = translation, 3:9 = 6D rotation, 9 = gripper
    agent_state_dim = model_unwrapped.agent_state_dim
    agent_component_weights = torch.ones(agent_state_dim, device=device)

    # Compute horizon step weights
    horizon_weights = compute_horizon_weights(model_unwrapped.n_future_steps, horizon_weight_decay).to(device)
    if rank0 and horizon_weight_decay != "none":
        cprint(f"Using horizon weight decay: {horizon_weight_decay}", "cyan")
        cprint(f"  Horizon weights: {horizon_weights.cpu().numpy()}", "cyan")

    # Contact prediction loss (BCEWithLogitsLoss)
    contact_criterion = None
    if predict_contact != "none":
        contact_criterion = nn.BCEWithLogitsLoss(reduction='none')
        if rank0:
            cprint(f"Using contact prediction mode: {predict_contact}, loss_weight={contact_loss_weight}", "cyan")

    # Log affordance loss weight
    if rank0 and affordance_loss_weight != 1.0:
        cprint(f"Using affordance_loss_weight={affordance_loss_weight}", "cyan")
    
    # Setup learning rate scheduler
    num_batches_per_epoch = len(dataloader)
    if lr_scheduler_kwargs is None:
        lr_scheduler_kwargs = {}
    lr_scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type=lr_scheduler_type,
        num_epochs=num_epochs,
        num_batches_per_epoch=num_batches_per_epoch,
        **lr_scheduler_kwargs
    )
    if rank0 and lr_scheduler is not None:
        cprint(f"Using LR scheduler: {lr_scheduler_type}", "cyan")

    # Setup learning rate warmup
    if rank0 and lr_warmup_steps > 0:
        cprint(f"Using linear warmup for {lr_warmup_steps} steps", "cyan")

    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = setup_checkpoint_dir(
            output_dir=output_dir,
            checkpoint_base=None,
            exp_name=exp_name
        )
    else:
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config to checkpoint directory (rank 0 only for DDP)
    # full_cfg is passed by reference from main(), so open_dict() overrides are already applied
    if rank0:
        save_config_to_checkpoint(checkpoint_dir, full_cfg)

    # 9-DoF PCD transform target (always on by default)
    use_pcd_transform_target = full_cfg.training.get("use_pcd_transform_target", True) if full_cfg is not None else True

    # Initialize wandb (rank 0 only for DDP)
    if rank0:
        initialize_wandb(
            exp_name=exp_name,
            lwm=model_unwrapped,
            wandb_cfg=wandb_cfg,
            pointcloud_encoder_size=pointcloud_encoder_size,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr_scheduler_type=lr_scheduler_type,
            lr_warmup_steps=lr_warmup_steps,
            max_grad_norm=max_grad_norm,
            checkpoint_dir=checkpoint_dir,
            num_params=num_params,
            predict_agent_state=predict_agent_state,
            weight_decay=weight_decay,
            affordance_loss_weight=affordance_loss_weight,
            data_splits=lwm_dataset.data_splits if lwm_dataset is not None else None,
            model_version="v8",
        )

    # Checkpoint save interval from config (periodic epoch_*.ckpt)
    save_every = 100
    if full_cfg is not None and hasattr(full_cfg, "checkpoint") and hasattr(full_cfg.checkpoint, "save_every"):
        save_every = int(full_cfg.checkpoint.save_every)
    if rank0:
        cprint(f"Saving checkpoint every {save_every} epochs", "cyan")

    # Track best validation loss for saving best model
    best_val_loss = float('inf')
    # Track most recent combined validation loss for comparison with training loss
    last_val_loss = None
    
    # Extract normalization stats from dataset to save in checkpoints
    normalization_stats = None
    if lwm_dataset is not None:
        normalization_stats = {
            "agent_state_delta_mean": lwm_dataset.agent_state_delta_mean,
            "agent_state_delta_std": lwm_dataset.agent_state_delta_std,
            "affordance_delta_mean": lwm_dataset.affordance_delta_mean,
            "affordance_delta_std": lwm_dataset.affordance_delta_std,
            "action_mean": lwm_dataset.action_mean,
            "action_std": lwm_dataset.action_std,
            "action_min": lwm_dataset.action_min,
            "action_max": lwm_dataset.action_max,
            "skip_rotation_normalization": lwm_dataset.skip_rotation_normalization,
            "affordance_target_frame": getattr(lwm_dataset, "affordance_target_frame", "world"),
        }

    # Set normalized identity for gumbel_gate (must happen after normalization stats are available)
    if predict_contact == "gumbel_gate" and lwm_dataset is not None:
        # Access the underlying model on DDP-wrapped modules
        model_unwrapped = lwm.module if hasattr(lwm, 'module') else lwm
        model_unwrapped.set_normalized_identity(
            mean=torch.from_numpy(lwm_dataset.affordance_delta_mean).float(),
            std=torch.from_numpy(lwm_dataset.affordance_delta_std).float(),
        )
        if rank0:
            cprint(f"Set normalized identity for gumbel_gate: {model_unwrapped.identity_normalized.tolist()}", "cyan")

    # Windowed rollout metrics on held-out val episodes (paper metric), logged to W&B at
    # every validation epoch. Computed when W&B is active, or always with
    # LWM_ROLLOUT_METRICS=1 (then also printed). Fixed windows: comparable across epochs/runs.
    rollout_metrics_on = rank0 and lwm_dataset is not None and (
        wandb_active() or os.environ.get("LWM_ROLLOUT_METRICS") == "1")
    val_rollout_windows = {}
    if rollout_metrics_on:
        n_win = int(os.environ.get("LWM_ROLLOUT_WINDOWS", 2048))
        val_rollout_windows = select_val_windows(lwm_dataset, n_windows=n_win, window_steps=6, seed=0)
        cprint(f"Rollout metrics on held-out val windows: "
               f"{ {k: len(v) for k, v in val_rollout_windows.items()} }", "cyan")
    if rank0 and wandb_active() and full_cfg is not None:
        wandb.config.update({"cfg": OmegaConf.to_container(full_cfg, resolve=True)}, allow_val_change=True)

    # Fix validation episodes for long-horizon evaluation (deterministic, fixed at start)
    # Helper to select fixed episodes for a given mask
    def _select_fixed_episodes(mask, suffix):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []
        rng = np.random.RandomState(hash(exp_name + suffix) % (2**31))
        rng.shuffle(indices)
        selected = indices[:min(long_horizon_num_episodes, len(indices))].tolist()
        if rank0:
            cprint(f"Fixed {len(selected)} {suffix.strip('_')} episodes for long-horizon evaluation", "cyan")
        return selected

    fixed_expert_val_episodes = []
    fixed_new_config_val_episodes = []
    fixed_replay_val_episodes = []
    fixed_expert_train_episodes = []
    fixed_new_config_train_episodes = []
    fixed_replay_train_episodes = []

    if lwm_dataset is not None:
        # Validation episodes per split
        fixed_expert_val_episodes = _select_fixed_episodes(
            lwm_dataset.val_mask & lwm_dataset.expert_episode_mask, "_val_expert")
        fixed_new_config_val_episodes = _select_fixed_episodes(
            lwm_dataset.val_mask & lwm_dataset.new_config_episode_mask, "_val_new_config")
        fixed_replay_val_episodes = _select_fixed_episodes(
            lwm_dataset.val_mask & lwm_dataset.replay_episode_mask, "_val_replay")

        # Training episodes per split (for training-set long-horizon eval)
        fixed_expert_train_episodes = _select_fixed_episodes(
            lwm_dataset.train_mask & lwm_dataset.expert_episode_mask, "_train_expert")
        fixed_new_config_train_episodes = _select_fixed_episodes(
            lwm_dataset.train_mask & lwm_dataset.new_config_episode_mask, "_train_new_config")
        fixed_replay_train_episodes = _select_fixed_episodes(
            lwm_dataset.train_mask & lwm_dataset.replay_episode_mask, "_train_replay")

    # Helper function to sample random subset of near-affordance steps
    def sample_near_affordance_steps(step_indices_dict, max_steps=100):
        """Sample up to max_steps random (episode_idx, step_idx) pairs from the dictionary."""
        all_pairs = []
        for ep_idx, step_list in step_indices_dict.items():
            for step_idx in step_list:
                all_pairs.append((ep_idx, step_idx))

        if len(all_pairs) <= max_steps:
            return step_indices_dict

        sampled_pairs = random.sample(all_pairs, max_steps)
        sampled_dict = {}
        for ep_idx, step_idx in sampled_pairs:
            if ep_idx not in sampled_dict:
                sampled_dict[ep_idx] = []
            sampled_dict[ep_idx].append(step_idx)

        return sampled_dict

    # Pre-compute near-affordance step indices for efficient evaluation
    def _compute_near_aff_steps(fixed_episodes, split_name):
        if len(fixed_episodes) == 0:
            return {}
        steps = get_near_affordance_step_indices(
            fixed_episodes, lwm_dataset, distance_threshold=0.2, rollout_length=18
        )
        total_before = sum(len(s) for s in steps.values())
        steps = sample_near_affordance_steps(steps, max_steps=100)
        total_after = sum(len(s) for s in steps.values())
        if rank0:
            cprint(f"Pre-computed {total_after} near-affordance steps (sampled from {total_before}) across {len(steps)} {split_name} episodes", "cyan")
        return steps

    fixed_expert_val_near_aff_steps = {}
    fixed_new_config_val_near_aff_steps = {}
    fixed_replay_val_near_aff_steps = {}
    fixed_expert_train_near_aff_steps = {}
    fixed_new_config_train_near_aff_steps = {}
    fixed_replay_train_near_aff_steps = {}

    if lwm_dataset is not None:
        fixed_expert_val_near_aff_steps = _compute_near_aff_steps(fixed_expert_val_episodes, "expert val")
        fixed_new_config_val_near_aff_steps = _compute_near_aff_steps(fixed_new_config_val_episodes, "new_config val")
        fixed_replay_val_near_aff_steps = _compute_near_aff_steps(fixed_replay_val_episodes, "replay val")
        fixed_expert_train_near_aff_steps = _compute_near_aff_steps(fixed_expert_train_episodes, "expert train")
        fixed_new_config_train_near_aff_steps = _compute_near_aff_steps(fixed_new_config_train_episodes, "new_config train")
        fixed_replay_train_near_aff_steps = _compute_near_aff_steps(fixed_replay_train_episodes, "replay train")
    
    global_step = 0
    world_size = torch.distributed.get_world_size() if distributed else 1
    on_cuda = torch.device(device).type == "cuda"
    epoch_wall_times = []
    for epoch in range(num_epochs):
        epoch_t0 = time.time()
        epoch_samples = 0
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_agent_loss = 0.0
        epoch_affordance_loss = 0.0
        epoch_rotation_loss = 0.0
        epoch_translation_loss = 0.0
        epoch_contact_loss = 0.0
        num_batches = 0

        if rank0:
            batch_pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{num_epochs}", unit="step")
        else:
            batch_pbar = enumerate(dataloader)
        for batch_idx, batch in batch_pbar:
            observation = batch["observation"]
            target = batch["target"]  # dict with "agent_state_delta" and "affordance_delta"
            
            # Move to device
            observation = dict_apply(observation, lambda x: x.to(device))
            target = dict_apply(target, lambda x: x.to(device))
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = lwm(observation)  # dict with "agent_state_delta" and "affordance_delta"

                # Compute losses for outputs with horizon weighting
                # criterion returns per-element loss: [B, n_future_steps, dim]
                if predict_agent_state:
                    agent_loss_per_elem = criterion(output["agent_state_delta"], target["agent_state_delta"])
                    # Apply per-component weights (e.g. down-weight gripper)
                    agent_loss_per_elem = agent_loss_per_elem * agent_component_weights  # [B, H, D] * [D]

                aff_target = target["affordance_delta"]
                # For gumbel_gate: gate target to normalized identity for no-contact steps,
                # matching the model's gated output (loss = 0 when gate agrees with label)
                if predict_contact == "gumbel_gate" and "contact" in target:
                    contact_label = target["contact"]  # [B, H, 1]
                    model_unwrapped = lwm.module if hasattr(lwm, 'module') else lwm
                    norm_id = model_unwrapped.identity_normalized  # [affordance_output_dim]
                    aff_target = contact_label * aff_target + (1 - contact_label) * norm_id
                affordance_loss_per_elem = criterion(output["affordance_delta"], aff_target)
                affordance_loss_per_step = affordance_loss_per_elem.mean(dim=(0, 2))  # [n_future_steps]

                # Split rotation (6D) and translation (3D) losses for logging
                rotation_loss_val = None
                translation_loss_val = None
                if use_pcd_transform_target:
                    with torch.no_grad():
                        rotation_loss_val = affordance_loss_per_elem[:, :, :6].mean().item()
                        translation_loss_val = affordance_loss_per_elem[:, :, 6:].mean().item()

                # Average over batch and feature dimensions, keep horizon dimension: [n_future_steps]
                if predict_agent_state:
                    agent_loss_per_step = agent_loss_per_elem.mean(dim=(0, 2))  # [n_future_steps]

                # Apply horizon weights and average
                if predict_agent_state:
                    agent_loss = (agent_loss_per_step * horizon_weights).mean()
                affordance_loss = (affordance_loss_per_step * horizon_weights).mean()

                # Apply affordance loss weight (and optional agent loss)
                if predict_agent_state:
                    total_loss = agent_loss + affordance_loss_weight * affordance_loss
                else:
                    total_loss = affordance_loss

                # Contact prediction loss (BCE)
                contact_loss_val = None
                if contact_criterion is not None and "contact_logit" in output:
                    contact_loss_per_elem = contact_criterion(
                        output["contact_logit"], target["contact"]
                    )  # [B, n_future_steps, 1]
                    contact_loss_per_step = contact_loss_per_elem.mean(dim=(0, 2))  # [n_future_steps]
                    contact_loss = (contact_loss_per_step * horizon_weights).mean()
                    total_loss = total_loss + contact_loss_weight * contact_loss
                    contact_loss_val = contact_loss.item()

            # Per-component gradient norm probing (every N steps, rank 0 only)
            comp_grad_norms = None
            if rank0 and global_step % GRAD_NORM_LOG_INTERVAL == 0:
                comp_losses = {}
                if predict_agent_state:
                    comp_losses["agent_translation"] = (
                        agent_loss_per_elem[:, :, :3].mean(dim=(0, 2)) * horizon_weights
                    ).mean()
                    comp_losses["agent_rotation"] = (
                        agent_loss_per_elem[:, :, 3:9].mean(dim=(0, 2)) * horizon_weights
                    ).mean()
                    comp_losses["agent_gripper"] = (
                        agent_loss_per_elem[:, :, 9:].mean(dim=(0, 2)) * horizon_weights
                    ).mean()
                if use_pcd_transform_target:
                    comp_losses["aff_rotation"] = affordance_loss_weight * (
                        affordance_loss_per_elem[:, :, :6].mean(dim=(0, 2)) * horizon_weights
                    ).mean()
                    comp_losses["aff_translation"] = affordance_loss_weight * (
                        affordance_loss_per_elem[:, :, 6:].mean(dim=(0, 2)) * horizon_weights
                    ).mean()
                params_for_grad = [p for p in lwm.parameters() if p.requires_grad]
                comp_grad_norms = compute_component_grad_norms(comp_losses, params_for_grad)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()

            # Unscale before grad clipping so clip threshold is in true gradient scale
            scaler.unscale_(optimizer)

            # Compute gradient norm (always logged) and optionally clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                lwm.parameters(), max_grad_norm if max_grad_norm is not None else float('inf')
            )

            scaler.step(optimizer)
            scaler.update()

            # Simple linear warmup: scale learning rate from 0 to target over warmup_steps
            if lr_warmup_steps > 0 and global_step < lr_warmup_steps:
                warmup_factor = global_step / lr_warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * warmup_factor
            # After warmup, update learning rate scheduler (step per batch for cosine, etc.)
            elif lr_scheduler is not None and lr_scheduler_type.lower() in ["cosine", "exponential"]:
                lr_scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_loss += total_loss.item()
            if predict_agent_state:
                epoch_agent_loss += agent_loss.item()
            epoch_affordance_loss += affordance_loss.item()
            if use_pcd_transform_target and rotation_loss_val is not None:
                epoch_rotation_loss += rotation_loss_val
                epoch_translation_loss += translation_loss_val
            if contact_loss_val is not None:
                epoch_contact_loss += contact_loss_val
            num_batches += 1
            global_step += 1
            epoch_samples += target["agent_state_delta"].shape[0]

            if rank0:
                # Log to wandb
                log_dict = {
                    "train/total_loss": total_loss.item(),
                    "train/affordance_loss": affordance_loss.item(),
                    "train/epoch": epoch,
                    "train/learning_rate": current_lr,
                }
                if predict_agent_state:
                    log_dict["train/agent_state_loss"] = agent_loss.item()
                if use_pcd_transform_target and rotation_loss_val is not None:
                    log_dict["train/rotation_loss"] = rotation_loss_val
                    log_dict["train/translation_loss"] = translation_loss_val
                if contact_loss_val is not None:
                    log_dict["train/contact_loss"] = contact_loss_val
                log_dict["train/grad_norm"] = grad_norm.item()
                if comp_grad_norms is not None:
                    for comp_name, comp_norm in comp_grad_norms.items():
                        log_dict[f"grad_norm/{comp_name}"] = comp_norm
                wandb.log(log_dict, step=global_step)

                # Update batch progress bar
                postfix = {"loss": f"{total_loss.item():.4f}", "aff": f"{affordance_loss.item():.4f}", "lr": f"{current_lr:.1e}"}
                if predict_agent_state:
                    postfix["agent"] = f"{agent_loss.item():.4f}"
                if contact_loss_val is not None:
                    postfix["contact"] = f"{contact_loss_val:.4f}"
                batch_pbar.set_postfix(postfix)
        
        train_time = time.time() - epoch_t0
        avg_loss = epoch_loss / num_batches
        avg_agent_loss = epoch_agent_loss / num_batches if predict_agent_state else None
        avg_affordance_loss = epoch_affordance_loss / num_batches
        
        # Update LR scheduler for step-based schedulers
        current_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler is not None and lr_scheduler_type.lower() in ["step"]:
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        if rank0:
            log_epoch = {
                "train/epoch_total_loss": avg_loss,
                "train/epoch_affordance_loss": avg_affordance_loss,
                "train/epoch": epoch,
            }
            if predict_agent_state:
                log_epoch["train/epoch_agent_loss"] = avg_agent_loss
            if use_pcd_transform_target and num_batches > 0:
                log_epoch["train/epoch_rotation_loss"] = epoch_rotation_loss / num_batches
                log_epoch["train/epoch_translation_loss"] = epoch_translation_loss / num_batches
            if predict_contact != "none" and num_batches > 0:
                log_epoch["train/epoch_contact_loss"] = epoch_contact_loss / num_batches
            wandb.log(log_epoch, step=global_step)

            if predict_agent_state:
                cprint(
                    f"Epoch {epoch} completed. Avg Total Loss: {avg_loss:.6f}, "
                    f"Avg Agent Loss: {avg_agent_loss:.6f}, "
                    f"Avg Affordance Loss: {avg_affordance_loss:.6f}, LR: {current_lr:.6e}",
                    "green",
                )
            else:
                cprint(
                    f"Epoch {epoch} completed. Avg Total Loss: {avg_loss:.6f}, "
                    f"Avg Affordance Loss: {avg_affordance_loss:.6f}, LR: {current_lr:.6e}",
                    "green",
                )
        
        # Validation (every 10 epochs) - rank 0 only for DDP
        if rank0 and val_dataloader is not None and (epoch + 1) % 10 == 0:
            cprint(f"Running validation at epoch {epoch} (global_step={global_step})...", "yellow")
            lwm.eval()

            # Use separate validation utility (pass unwrapped model for DDP)
            val_model = model_unwrapped if distributed else lwm
            combined_metrics, expert_metrics, new_config_metrics, replay_metrics, is_best, unseen_expert_metrics, unseen_new_config_metrics = validate_separate(
                val_model,
                lwm_dataset,
                device,
                criterion,
                epoch,
                optimizer.param_groups[0]['lr'],
                best_val_loss,
                batch_size,
                predict_agent_state=predict_agent_state,
                global_step=global_step,
            )

            # Update LR scheduler for plateau-based scheduler (use combined loss)
            if lr_scheduler is not None and lr_scheduler_type.lower() == "plateau":
                if combined_metrics:
                    last_val_loss = combined_metrics["val_combined/total_loss"]
                    lr_scheduler.step(combined_metrics["val_combined/total_loss"])
                    current_lr = optimizer.param_groups[0]['lr']
                else:
                    current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = optimizer.param_groups[0]['lr']
                if combined_metrics:
                    last_val_loss = combined_metrics["val_combined/total_loss"]

            # Update best validation loss
            if is_best and combined_metrics:
                best_val_loss = combined_metrics["val_combined/total_loss"]

                # Save best model checkpoint (use unwrapped model for DDP)
                best_checkpoint_path = save_checkpoint(
                    checkpoint_dir, model_unwrapped, optimizer, epoch + 1, exp_name,
                    current_lr, batch_size, best_val_loss, lr_scheduler, suffix="best",
                    normalization_stats=normalization_stats
                )
                cprint(f"Best model checkpoint saved to {best_checkpoint_path}", "green")
                wandb.summary["checkpoint_path_best"] = str(best_checkpoint_path)
                wandb.log({"checkpoint_path_best": str(best_checkpoint_path)}, step=global_step)

            # Windowed rollout metrics (handle / EE / part rotation / finger) on held-out windows
            if val_rollout_windows:
                t_roll = time.time()
                roll = compute_val_rollout_metrics(val_model, lwm_dataset, val_rollout_windows,
                                                   normalization_stats, device, window_steps=6)
                roll["rollout/eval_time_s"] = time.time() - t_roll
                wandb.log(roll, step=global_step)
                cprint("Rollout metrics: " + ", ".join(
                    f"{k}={v:.3f}" for k, v in roll.items() if "step6" in k or "eval_time" in k), "cyan")

            lwm.train()
        
        # Long-horizon and near-affordance validations disabled for now (too slow, ~5+ min every 10 epochs)
        # TODO: Re-enable when training is stable and you want detailed rollout metrics
        # if lwm_dataset is not None and epoch > 0 and epoch % 10 == 0:
        #     ... (long-horizon + near-affordance evaluations)
        
        # Save checkpoint every save_every epochs (rank 0 only)
        if rank0 and (epoch + 1) % save_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            checkpoint_path = save_checkpoint(
                checkpoint_dir, model_unwrapped, optimizer, epoch + 1, exp_name,
                current_lr, batch_size, best_val_loss, lr_scheduler,
                suffix=f"epoch_{epoch + 1}", normalization_stats=normalization_stats
            )
            cprint(f"Checkpoint saved to {checkpoint_path}", "green")
            wandb.summary["checkpoint_path_latest"] = str(checkpoint_path)
            wandb.log({"checkpoint_path": str(checkpoint_path), "checkpoint_epoch": epoch + 1}, step=global_step)

        # Throughput / time / memory / ETA
        if rank0:
            epoch_wall = time.time() - epoch_t0
            epoch_wall_times.append(epoch_wall)
            eta_s = float(np.mean(epoch_wall_times[-10:])) * (num_epochs - epoch - 1)
            perf = {
                "perf/samples_per_s": epoch_samples * world_size / max(train_time, 1e-9),
                "perf/epoch_train_time_s": train_time,
                "perf/epoch_wall_time_s": epoch_wall,
                "perf/eta_h": eta_s / 3600.0,
            }
            if on_cuda:
                perf["perf/max_gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            wandb.log(perf, step=global_step)
            cprint(f"Epoch {epoch}: {perf['perf/samples_per_s']:.0f} samples/s, "
                   f"epoch {epoch_wall:.1f}s, ETA {perf['perf/eta_h']:.2f} h", "cyan")

    # Ensure all ranks have left the training loop before rank 0 does wandb.finish()
    # (avoids NCCL timeout / BrokenPipeError when rank 0 closes wandb while rank 1 is in a collective)
    if distributed:
        barrier()

    # Save final checkpoint (rank 0 only)
    if rank0:
        current_lr = optimizer.param_groups[0]['lr']
        final_checkpoint_path = save_checkpoint(
            checkpoint_dir, model_unwrapped, optimizer, num_epochs, exp_name,
            current_lr, batch_size, best_val_loss, lr_scheduler, suffix="last",
            normalization_stats=normalization_stats
        )
        cprint(f"Final checkpoint saved to {final_checkpoint_path}", "green")
        cprint(f"Best validation loss (min over epochs): {best_val_loss:.6f}", "green")
        if last_val_loss is not None:
            cprint(f"Last validation loss (most recent val_combined/total_loss): {last_val_loss:.6f}", "green")
        wandb.summary["best_val_loss"] = float(best_val_loss)
        if last_val_loss is not None:
            wandb.summary["final_val_loss"] = float(last_val_loss)
        wandb.summary["checkpoint_path_final"] = str(final_checkpoint_path)
        wandb.log({"checkpoint_path_final": str(final_checkpoint_path)}, step=global_step)

        wandb.finish()

    if distributed:
        destroy_process_group()


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIR / "lwm"),
    config_name="train",
)
def main(cfg):
    # DDP setup (if launched with torchrun)
    distributed = is_distributed()
    if distributed:
        ddp_setup()
    rank0 = is_main_process()

    # Set maximum determinism for reproducible results
    # Allow environment variable override for seed
    seed = int(os.environ.get("RANDOM_SEED", cfg.dataset.seed))
    set_seed(seed)
    if rank0:
        cprint(f"Set random seed to {seed} for maximum determinism", "cyan")
        if distributed:
            cprint(f"DDP enabled: {torch.distributed.get_world_size()} GPUs, rank {get_rank()}", "cyan")
    
    # Get output directory from Hydra
    output_dir = HydraConfig.get().runtime.output_dir
    if rank0:
        cprint(f"Output directory: {output_dir}", "cyan")
    
    # Read dataset paths from config/environment (local directories; download the
    # HuggingFace datasets first, see README)
    from lwm.world_model.dataset import resolve_dataset_path

    expert_dataset_path = os.environ.get("EXPERT_DATASET_PATH") or cfg.get("expert_dataset_path")
    if expert_dataset_path:
        expert_dataset_path = resolve_dataset_path(expert_dataset_path)
    else:
        raise ValueError("expert_dataset_path must be set (local path)")

    rollout_dataset_path = os.environ.get("ROLLOUT_DATASET_PATH") or cfg.get("rollout_dataset_path")
    if rollout_dataset_path and rollout_dataset_path not in ("null", ""):
        rollout_dataset_path = resolve_dataset_path(rollout_dataset_path)
    else:
        rollout_dataset_path = None
    
    # Read data splits configuration
    data_splits_str = os.environ.get("DATA_SPLITS")
    if data_splits_str:
        data_splits = json.loads(data_splits_str)
    else:
        raise ValueError("DATA_SPLITS environment variable is required. Set it via the launch script.")

    # Episode filtering configuration
    include_object_ids_str = os.environ.get("INCLUDE_OBJECT_IDS")
    include_object_ids = json.loads(include_object_ids_str) if include_object_ids_str else None
    include_object_ids_new_config_str = os.environ.get("INCLUDE_OBJECT_IDS_NEW_CONFIG")
    include_object_ids_new_config = json.loads(include_object_ids_new_config_str) if include_object_ids_new_config_str else None
    allowed_goal_door_str = os.environ.get("ALLOWED_GOAL_PERCENTAGES_DOOR")
    allowed_goal_percentages_door = json.loads(allowed_goal_door_str) if allowed_goal_door_str else None
    allowed_goal_drawer_str = os.environ.get("ALLOWED_GOAL_PERCENTAGES_DRAWER")
    allowed_goal_percentages_drawer = json.loads(allowed_goal_drawer_str) if allowed_goal_drawer_str else None

    # Log resolved dataset paths (no local copying)
    if rank0:
        cprint(f"Expert dataset path: {expert_dataset_path}", "green")
        if rollout_dataset_path:
            cprint(f"Rollout dataset path: {rollout_dataset_path}", "green")
        cprint(f"Data splits: {json.dumps(data_splits, indent=2)}", "green")
        if include_object_ids:
            cprint(f"Include object IDs (expert/replay): {include_object_ids}", "green")
        if include_object_ids_new_config:
            cprint(f"Include object IDs (new_config): {include_object_ids_new_config}", "green")
        if allowed_goal_percentages_door:
            cprint(f"Allowed goal percentages (door): {allowed_goal_percentages_door}", "green")
        if allowed_goal_percentages_drawer:
            cprint(f"Allowed goal percentages (drawer): {allowed_goal_percentages_drawer}", "green")
    
    # Read training parameters from config, with environment variable overrides (backward compatibility)
    horizon = int(os.environ.get("HORIZON", cfg.training.horizon))
    n_future_steps = horizon  # Always equal to horizon for WM
    n_obs_steps = int(os.environ.get("N_OBS_STEPS", cfg.training.get("n_obs_steps", 1)))
    batch_size = int(os.environ.get("BATCH_SIZE", cfg.training.batch_size))
    num_epochs = int(os.environ.get("NUM_EPOCHS", cfg.training.num_epochs))
    learning_rate = float(os.environ.get("LEARNING_RATE", cfg.training.learning_rate))
    lr_scheduler_type = os.environ.get("LR_SCHEDULER_TYPE", cfg.training.lr_scheduler_type).lower()
    lr_scheduler_kwargs = cfg.training.get("lr_scheduler_kwargs", {})
    # Override lr_scheduler_kwargs from environment if set
    if os.environ.get("LR_SCHEDULER_STEP_SIZE"):
        lr_scheduler_kwargs["step_size"] = int(os.environ.get("LR_SCHEDULER_STEP_SIZE"))
    if os.environ.get("LR_SCHEDULER_GAMMA"):
        lr_scheduler_kwargs["gamma"] = float(os.environ.get("LR_SCHEDULER_GAMMA"))
    lr_warmup_steps = int(os.environ.get("LR_WARMUP_STEPS", cfg.training.get("lr_warmup_steps", 0)))
    max_grad_norm = os.environ.get("MAX_GRAD_NORM")
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
    else:
        max_grad_norm = cfg.training.get("max_grad_norm", None)
    
    # New training parameters for regularization and loss weighting
    weight_decay = float(os.environ.get("WEIGHT_DECAY", cfg.training.get("weight_decay", 0.01)))
    affordance_loss_weight = float(os.environ.get("AFFORDANCE_LOSS_WEIGHT", cfg.training.get("affordance_loss_weight", 1.0)))
    horizon_weight_decay = os.environ.get("HORIZON_WEIGHT_DECAY", cfg.training.get("horizon_weight_decay", "none"))
    use_amp = str(os.environ.get("USE_AMP", cfg.training.get("use_amp", False))).lower() in ("true", "1", "yes")


    # Contact prediction parameters
    predict_contact = os.environ.get("PREDICT_CONTACT", cfg.training.get("predict_contact", "none"))
    contact_loss_weight = float(os.environ.get("CONTACT_LOSS_WEIGHT", cfg.training.get("contact_loss_weight", 1.0)))
    contact_threshold = float(os.environ.get("CONTACT_THRESHOLD", cfg.training.get("contact_threshold", 1e-4)))
    gumbel_tau = float(os.environ.get("GUMBEL_TAU", cfg.model.get("gumbel_tau", 0.5)))
    
    # Support both "name" and "exp_name" in config, prefer "name"
    # Also support EXP_NAME environment variable
    if "name" in cfg:
        default_name = cfg.name
    elif "exp_name" in cfg:
        default_name = cfg.exp_name
    else:
        default_name = "lwm_training"
    exp_name = os.environ.get("EXP_NAME", default_name)
    
    # Long horizon validation number of episodes
    long_horizon_num_episodes = int(os.environ.get("LONG_HORIZON_NUM_EPISODES", cfg.get("long_horizon_num_episodes", 50)))
    
    # Model parameters from config, with environment variable overrides
    model_cfg = cfg.model

    # All supported versions (v3/v7/v8) use LWMv3 with d_model for all token dimensions
    d_model = int(os.environ.get("D_MODEL", model_cfg.get("d_model", 128)))
    n_layers = int(os.environ.get("N_LAYERS", model_cfg.get("n_layers", 3)))
    n_heads = int(os.environ.get("N_HEADS", model_cfg.get("n_heads", 4)))
    # Point-cloud tokens: 1 = one global token (paper models), K > 1 = K local patch tokens
    pcd_tokens = int(os.environ.get("PCD_TOKENS", model_cfg.get("pcd_tokens", 1)))
    pcd_group_size = int(os.environ.get("PCD_GROUP_SIZE", model_cfg.get("pcd_group_size", 32)))
    pointcloud_encoder_size = int(os.environ.get("POINTCLOUD_ENCODER_SIZE", model_cfg.get("pointcloud_encoder_size", 256)))
    affordance_output_dim = int(os.environ.get("AFFORDANCE_OUTPUT_DIM", model_cfg.get("affordance_output_dim", 9)))

    dataset_k_nearest = 0
    # Enable 9-DoF PCD transform target (SVD-based) — always on, all versions predict rigid body transforms
    use_pcd_transform_target = cfg.training.get("use_pcd_transform_target", True)
    # Skip normalization for 6D rotation components (best practice: rotation representations are inherently bounded)
    skip_rotation_normalization = str(os.environ.get("SKIP_ROTATION_NORMALIZATION", cfg.training.get("skip_rotation_normalization", False))).lower() in ("true", "1", "yes")

    # Experiment switches (env var > config; defaults reproduce the paper models)
    balanced_sampling = str(os.environ.get("BALANCED_SAMPLING", cfg.dataset.get("balanced_sampling", False))).lower() in ("true", "1", "yes")
    affordance_target_frame = str(os.environ.get("AFFORDANCE_TARGET_FRAME", cfg.dataset.get("affordance_target_frame", "world")))
    input_frame = str(os.environ.get("INPUT_FRAME", cfg.dataset.get("input_frame", "world")))
    check_frames(affordance_target_frame, input_frame)
    augment_yaw = float(os.environ.get("AUGMENT_YAW", cfg.dataset.get("augment_yaw", 0) or 0))
    augment_translation = float(os.environ.get("AUGMENT_TRANSLATION", cfg.dataset.get("augment_translation", 0) or 0))
    if rank0:
        cprint(f"Frames: affordance_target_frame={affordance_target_frame}, input_frame={input_frame}; "
               f"augment_yaw={augment_yaw} deg, augment_translation={augment_translation} m; "
               f"balanced_sampling={balanced_sampling}", "cyan")

    if rank0:
        cprint(f"Loading LWM dataset with horizon={horizon}, n_obs_steps={n_obs_steps}, skip_rotation_normalization={skip_rotation_normalization}...", "yellow")

    lwm_dataset = LWMDataset(
        expert_dataset_path=expert_dataset_path,
        rollout_dataset_path=rollout_dataset_path,
        horizon=horizon,
        pad_before=cfg.dataset.pad_before,
        pad_after=cfg.dataset.pad_after,
        seed=cfg.dataset.seed,
        val_ratio=cfg.dataset.val_ratio,
        train_ratio=cfg.dataset.train_ratio,
        is_pickle=cfg.dataset.is_pickle,
        n_obs_steps=n_obs_steps,
        data_splits=data_splits,
        use_part_pointcloud=True,
        k_nearest=dataset_k_nearest,
        use_pcd_transform_target=use_pcd_transform_target,
        skip_rotation_normalization=skip_rotation_normalization,
        include_object_ids=include_object_ids,
        include_object_ids_new_config=include_object_ids_new_config,
        allowed_goal_percentages_door=allowed_goal_percentages_door,
        allowed_goal_percentages_drawer=allowed_goal_percentages_drawer,
        load_unseen_val=True,
        affordance_target_frame=affordance_target_frame,
        augment_yaw=augment_yaw,
        augment_translation=augment_translation,
    )
    # Set contact threshold on dataset
    if predict_contact != "none":
        lwm_dataset.contact_threshold = contact_threshold
        if rank0:
            cprint(f"Contact threshold: {contact_threshold}", "cyan")

    if rank0:
        cprint(f"LWM dataset size: {len(lwm_dataset)}", "green")

    # Create validation dataset
    val_dataset = lwm_dataset.get_validation_dataset()
    if rank0:
        cprint(f"Validation dataset size: {len(val_dataset)}", "green")

    # Save episode splits immediately so eval can run in parallel with training
    if rank0:
        splits_metadata = lwm_dataset.get_episode_splits_metadata()
        splits_path = pathlib.Path(output_dir) / "episode_splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits_metadata, f, indent=2)
        cprint(f"Saved episode splits to {splits_path}", "green")

    # Create generator for deterministic DataLoader shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Worker init function for deterministic DataLoader workers
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
    
    # Create samplers for DDP (DistributedSampler handles data sharding across GPUs)
    if distributed and balanced_sampling:
        weights = lwm_dataset.get_balanced_sample_weights()
        world_size = torch.distributed.get_world_size()
        train_sampler = DistributedWeightedSampler(weights, num_replicas=world_size, rank=get_rank(), seed=seed)
        if rank0:
            cprint("Using balanced sampling (equal expert / interaction probability) with DDP", "green")
    elif distributed:
        train_sampler = DistributedSampler(lwm_dataset, shuffle=True)
    elif balanced_sampling:
        weights = lwm_dataset.get_balanced_sample_weights()
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        if rank0:
            cprint("Using balanced sampling (equal expert / interaction probability)", "green")
    else:
        train_sampler = None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    # Create dataloaders from config (use batch_size from training config, which may be overridden by env vars)
    dataloader = DataLoader(
        lwm_dataset,
        batch_size=batch_size,  # Use the batch_size that may have been overridden by env vars
        shuffle=(train_sampler is None and cfg.dataloader.shuffle),
        sampler=train_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.get("persistent_workers", False),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Use the batch_size that may have been overridden by env vars
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.val_dataloader.num_workers,
        pin_memory=cfg.val_dataloader.pin_memory,
        persistent_workers=cfg.val_dataloader.get("persistent_workers", False),
        worker_init_fn=worker_init_fn,
    )
    
    # Create model - detect dimensions from dataset, then instantiate from config
    if rank0:
        cprint("Creating LWM model...", "yellow")
    sample = lwm_dataset[0]
    agent_state_shape = sample["observation"]["agent_state"].shape
    if len(agent_state_shape) == 1:
        # Single timestep: [agent_state_dim]
        agent_state_dim = agent_state_shape[0]
    else:
        # Multiple timesteps: [n_obs_steps, agent_state_dim]
        agent_state_dim = agent_state_shape[-1]
    action_dim = sample["observation"]["action_horizon"].shape[-1]
    if rank0:
        cprint(f"Detected agent_state_dim: {agent_state_dim}, action_dim: {action_dim}, n_obs_steps: {n_obs_steps}", "cyan")
        if pcd_tokens > 1:
            cprint(f"Point cloud: {pcd_tokens} patch tokens of {pcd_group_size} points", "cyan")
    
    # Update model config with detected dimensions and env var overrides
    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
    model_cfg_dict["agent_state_dim"] = agent_state_dim
    model_cfg_dict["action_dim"] = action_dim
    model_cfg_dict["n_future_steps"] = horizon

    # All supported versions (v3/v7/v8) use LWMv3 architecture
    model_cfg_dict["d_model"] = d_model
    model_cfg_dict["n_layers"] = n_layers
    model_cfg_dict["n_heads"] = n_heads
    model_cfg_dict["pcd_tokens"] = pcd_tokens
    model_cfg_dict["pcd_group_size"] = pcd_group_size
    model_cfg_dict["n_obs_steps"] = n_obs_steps
    model_cfg_dict["affordance_target_frame"] = affordance_target_frame
    model_cfg_dict["input_frame"] = input_frame
    if os.environ.get("POINTCLOUD_ENCODER_SIZE"):
        model_cfg_dict["pointcloud_encoder_size"] = pointcloud_encoder_size

    # Add contact prediction params
    if predict_contact != "none":
        model_cfg_dict["predict_contact"] = predict_contact
        model_cfg_dict["gumbel_tau"] = gumbel_tau
        if rank0:
            cprint(f"Contact prediction: mode={predict_contact}, gumbel_tau={gumbel_tau}", "cyan")

    # Override predict_agent_state from environment variable if set
    predict_agent_state_env = os.environ.get("PREDICT_AGENT_STATE")
    if predict_agent_state_env is not None:
        predict_agent_state_value = predict_agent_state_env.lower() == "true"
        model_cfg_dict["predict_agent_state"] = predict_agent_state_value
        if rank0:
            cprint(f"Overriding predict_agent_state from environment: {predict_agent_state_value}", "cyan")

    # Override use_pointnet2 from environment variable if set
    use_pointnet2_env = os.environ.get("USE_POINTNET2")
    if use_pointnet2_env is not None:
        use_pointnet2_value = use_pointnet2_env.lower() in ("true", "1", "yes")
        model_cfg_dict["use_pointnet2"] = use_pointnet2_value
        if rank0:
            cprint(f"Overriding use_pointnet2 from environment: {use_pointnet2_value}", "cyan")

    # Update full config with all overrides so they get saved to checkpoint
    with open_dict(cfg):
        cfg.model.agent_state_dim = agent_state_dim
        cfg.model.action_dim = action_dim
        cfg.model.n_future_steps = horizon
        cfg.model.d_model = d_model
        cfg.model.n_layers = n_layers
        cfg.model.n_heads = n_heads
        cfg.model.pcd_tokens = pcd_tokens
        cfg.model.pcd_group_size = pcd_group_size
        cfg.training.seed = seed
        cfg.model.n_obs_steps = n_obs_steps
        if os.environ.get("POINTCLOUD_ENCODER_SIZE"):
            cfg.model.pointcloud_encoder_size = pointcloud_encoder_size

        if predict_agent_state_env is not None:
            cfg.model.predict_agent_state = predict_agent_state_value
        if use_pointnet2_env is not None:
            cfg.model.use_pointnet2 = use_pointnet2_value
        # Save contact prediction params so load_checkpoint can reconstruct the model
        if predict_contact != "none":
            cfg.model.predict_contact = predict_contact
            cfg.model.gumbel_tau = gumbel_tau
        # Save rotation normalization flag so checkpoint config is complete
        cfg.training.skip_rotation_normalization = skip_rotation_normalization
        # Experiment switches: dataset.* records the training setup; model.* is what
        # load_checkpoint passes to LWMv3 (and what every consumer reads back).
        cfg.dataset.balanced_sampling = balanced_sampling
        cfg.dataset.affordance_target_frame = affordance_target_frame
        cfg.dataset.input_frame = input_frame
        cfg.dataset.augment_yaw = augment_yaw
        cfg.dataset.augment_translation = augment_translation
        cfg.model.affordance_target_frame = affordance_target_frame
        cfg.model.input_frame = input_frame

    # Instantiate model using Hydra
    lwm = hydra.utils.instantiate(model_cfg_dict)
    predict_agent_state = getattr(lwm, "predict_agent_state", True)
    
    # Train
    if rank0:
        cprint("Starting training...", "yellow")
    device = torch.device(get_rank()) if distributed else ("cuda" if torch.cuda.is_available() else "cpu")
    if rank0:
        cprint(f"Using device: {device}", "cyan")
    
    train_lwm(
        lwm, 
        dataloader, 
        val_dataloader=val_dataloader, 
        num_epochs=num_epochs, 
        device=device, 
        learning_rate=learning_rate, 
        batch_size=batch_size, 
        pointcloud_encoder_size=pointcloud_encoder_size,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=dict(lr_scheduler_kwargs),
        lr_warmup_steps=lr_warmup_steps,
        max_grad_norm=max_grad_norm,
        exp_name=exp_name,
        output_dir=output_dir,
        wandb_cfg=OmegaConf.to_container(cfg.wandb, resolve=True) if hasattr(cfg, "wandb") else None,
        full_cfg=cfg,
        lwm_dataset=lwm_dataset,
        long_horizon_num_episodes=long_horizon_num_episodes,
        # New parameters
        weight_decay=weight_decay,
        affordance_loss_weight=affordance_loss_weight,
        horizon_weight_decay=horizon_weight_decay,
        predict_agent_state=predict_agent_state,
        train_sampler=train_sampler,
        use_amp=use_amp,
        # Contact prediction
        predict_contact=predict_contact,
        contact_loss_weight=contact_loss_weight,
    )
    
    # Save episode splits after training (rank 0 only)
    if rank0:
        splits_metadata = lwm_dataset.get_episode_splits_metadata()
        splits_path = pathlib.Path(output_dir) / "episode_splits.json"
        with open(splits_path, 'w') as f:
            json.dump(splits_metadata, f, indent=2)
        cprint(f"Saved episode splits to {splits_path}", "green")
        cprint(f"  Train: {len(splits_metadata['train_episodes'])} episodes from {len(splits_metadata['train_objects'])} objects", "green")
        cprint(f"  Val: {len(splits_metadata['val_episodes'])} episodes from {len(splits_metadata['val_objects'])} objects", "green")

        cprint("Training completed!", "green")


if __name__ == "__main__":
    main()
