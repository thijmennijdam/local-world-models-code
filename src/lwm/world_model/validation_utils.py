"""
Utility functions for validation during LWM training.
Provides separate validation for expert and rollout data.
"""

import torch
from torch import nn
from termcolor import cprint
from tqdm import tqdm
from lwm.common.wandb_compat import wandb
import numpy as np
from lwm.common.pytorch_util import dict_apply
from lwm.world_model.plotting_utils import plot_error_vs_step
from lwm.world_model.evaluation_utils import (
    get_normalization_stats,
    load_episode_data,
    build_observation,
    predict_next_state_and_affordance,
    compose_state_with_delta,
)


def validate_model(lwm, dataloader, device, criterion, data_type="combined", predict_agent_state: bool = True):
    """
    Validate model on a dataloader.
    
    Args:
        lwm: The LWM model
        dataloader: DataLoader for validation
        device: Device to run on
        criterion: Loss criterion
        data_type: Type of data ("expert", "rollout", or "combined")
    
    Returns:
        Dict with validation metrics
    """
    lwm.eval()
    val_loss = 0.0
    val_agent_loss = 0.0
    val_affordance_loss = 0.0
    val_rotation_loss = 0.0
    val_translation_loss = 0.0
    val_contact_loss = 0.0
    val_contact_correct = 0
    val_contact_total = 0
    val_num_batches = 0
    is_9dof = hasattr(lwm, 'affordance_output_dim') and lwm.affordance_output_dim == 9
    has_contact = hasattr(lwm, 'predict_contact') and lwm.predict_contact != "none"
    contact_criterion = nn.BCEWithLogitsLoss(reduction='mean') if has_contact else None
    
    # Track per-step errors within the horizon
    n_future_steps = lwm.n_future_steps
    per_step_agent_errors = {step: [] for step in range(n_future_steps)}
    per_step_affordance_errors = {step: [] for step in range(n_future_steps)}
    
    # Use reduction='none' for per-step error tracking
    criterion_none = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Val ({data_type})", unit="step"):
            observation = batch["observation"]
            target = batch["target"]
            
            observation = dict_apply(observation, lambda x: x.to(device))
            target = dict_apply(target, lambda x: x.to(device))
            
            output = lwm(observation)

            # Compute per-step errors within the horizon
            if predict_agent_state:
                # criterion_none gives [B, n_future_steps, dim]
                agent_loss_per_elem = criterion_none(output["agent_state_delta"], target["agent_state_delta"])
                # Average over batch and feature dimensions, keep horizon dimension: [n_future_steps]
                agent_loss_per_step = agent_loss_per_elem.mean(dim=(0, 2))  # [n_future_steps]
                agent_loss = agent_loss_per_step.mean()
                
                # Track per-step errors
                for step in range(n_future_steps):
                    per_step_agent_errors[step].append(agent_loss_per_step[step].item())
            else:
                agent_loss = None
            
            # v6 outputs knn_point_delta instead of affordance_delta
            is_v6 = hasattr(lwm, 'k_nearest') and lwm.k_nearest > 0
            if is_v6:
                knn_loss_per_elem = criterion_none(output["knn_point_delta"], target["knn_point_delta"])
                affordance_loss_per_step = knn_loss_per_elem.mean(dim=(0, 2, 3))  # [n_future_steps]
            else:
                aff_target = target["affordance_delta"]
                # For gumbel_gate: gate target to normalized identity for no-contact steps
                if has_contact and lwm.predict_contact == "gumbel_gate" and "contact" in target:
                    contact_label = target["contact"]  # [B, H, 1]
                    norm_id = lwm.identity_normalized  # [affordance_output_dim]
                    aff_target = contact_label * aff_target + (1 - contact_label) * norm_id
                affordance_loss_per_elem = criterion_none(output["affordance_delta"], aff_target)
                affordance_loss_per_step = affordance_loss_per_elem.mean(dim=(0, 2))  # [n_future_steps]
            affordance_loss = affordance_loss_per_step.mean()
            
            # Track per-step errors
            for step in range(n_future_steps):
                per_step_affordance_errors[step].append(affordance_loss_per_step[step].item())
            
            # Also compute average loss using the original criterion for consistency
            total_loss = affordance_loss + (agent_loss if predict_agent_state else 0.0)
            
            val_loss += total_loss.item()
            if predict_agent_state:
                val_agent_loss += agent_loss.item()
            val_affordance_loss += affordance_loss.item()
            if is_9dof and not is_v6:
                val_rotation_loss += affordance_loss_per_elem[:, :, :6].mean().item()
                val_translation_loss += affordance_loss_per_elem[:, :, 6:].mean().item()
            if has_contact and "contact_logit" in output:
                c_loss = contact_criterion(output["contact_logit"], target["contact"])
                val_contact_loss += c_loss.item()
                # Contact accuracy
                preds = (output["contact_logit"] > 0.0).float()
                val_contact_correct += (preds == target["contact"]).sum().item()
                val_contact_total += target["contact"].numel()
            val_num_batches += 1
    
    if val_num_batches == 0:
        return None
    
    avg_val_loss = val_loss / val_num_batches
    avg_val_agent_loss = val_agent_loss / val_num_batches if predict_agent_state else None
    avg_val_affordance_loss = val_affordance_loss / val_num_batches
    
    # Compute per-step averages
    step_indices = list(range(n_future_steps))
    step_agent_errors = [np.mean(per_step_agent_errors[s]) for s in step_indices] if predict_agent_state else None
    step_affordance_errors = [np.mean(per_step_affordance_errors[s]) for s in step_indices]
    step_total_errors = [a + f for a, f in zip(step_agent_errors, step_affordance_errors)] if predict_agent_state else step_affordance_errors

    # Compute cumulative errors (sum of errors up to each step)
    cum_affordance_errors = list(np.cumsum(step_affordance_errors))
    cum_agent_errors = list(np.cumsum(step_agent_errors)) if step_agent_errors is not None else None
    cum_total_errors = [a + f for a, f in zip(cum_agent_errors, cum_affordance_errors)] if cum_agent_errors is not None else cum_affordance_errors

    # Create cumulative error vs step plot
    error_plot = plot_error_vs_step(
        step_indices=step_indices,
        step_total_errors=cum_total_errors,
        step_agent_errors=cum_agent_errors,
        step_affordance_errors=cum_affordance_errors,
        title=f'Validation ({data_type}): Cumulative Error vs Step',
    )
    
    metrics = {
        f"val_{data_type}/total_loss": avg_val_loss,
        f"val_{data_type}/affordance_loss": avg_val_affordance_loss,
        f"val_{data_type}/num_batches": val_num_batches,
        f"val_{data_type}/error_vs_step": error_plot,
    }
    if predict_agent_state:
        metrics[f"val_{data_type}/agent_state_loss"] = avg_val_agent_loss
    if is_9dof and val_num_batches > 0:
        metrics[f"val_{data_type}/rotation_loss"] = val_rotation_loss / val_num_batches
        metrics[f"val_{data_type}/translation_loss"] = val_translation_loss / val_num_batches
    if has_contact and val_num_batches > 0:
        metrics[f"val_{data_type}/contact_loss"] = val_contact_loss / val_num_batches
        if val_contact_total > 0:
            metrics[f"val_{data_type}/contact_accuracy"] = val_contact_correct / val_contact_total

    return metrics


def _validate_split(lwm, lwm_dataset, device, criterion, batch_size, split_name, split_getter, predict_agent_state, epoch):
    """Validate a single split. Returns metrics dict or None if no data."""
    from torch.utils.data import DataLoader

    val_dataset = split_getter()
    if len(val_dataset) == 0:
        return None

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    metrics = validate_model(lwm, val_dataloader, device, criterion, split_name, predict_agent_state=predict_agent_state)

    if metrics:
        total_key = f"val_{split_name}/total_loss"
        aff_key = f"val_{split_name}/affordance_loss"
        agent_key = f"val_{split_name}/agent_state_loss"
        if predict_agent_state and agent_key in metrics:
            cprint(
                f"Epoch {epoch} {split_name} validation. "
                f"Total: {metrics[total_key]:.6f}, Agent: {metrics[agent_key]:.6f}, Affordance: {metrics[aff_key]:.6f}",
                "cyan"
            )
        else:
            cprint(
                f"Epoch {epoch} {split_name} validation. "
                f"Total: {metrics[total_key]:.6f}, Affordance: {metrics[aff_key]:.6f}",
                "cyan"
            )

    return metrics


def _compute_combined_from_splits(split_metrics_list, predict_agent_state):
    """Compute weighted average of split metrics (weighted by num_batches)."""
    total_weight = 0
    weighted_total_loss = 0.0
    weighted_affordance_loss = 0.0
    weighted_agent_loss = 0.0

    for metrics, split_name in split_metrics_list:
        if metrics is None:
            continue
        n = metrics[f"val_{split_name}/num_batches"]
        total_weight += n
        weighted_total_loss += metrics[f"val_{split_name}/total_loss"] * n
        weighted_affordance_loss += metrics[f"val_{split_name}/affordance_loss"] * n
        if predict_agent_state and f"val_{split_name}/agent_state_loss" in metrics:
            weighted_agent_loss += metrics[f"val_{split_name}/agent_state_loss"] * n

    if total_weight == 0:
        return None

    combined = {
        "val_combined/total_loss": weighted_total_loss / total_weight,
        "val_combined/affordance_loss": weighted_affordance_loss / total_weight,
        "val_combined/num_batches": total_weight,
    }
    if predict_agent_state:
        combined["val_combined/agent_state_loss"] = weighted_agent_loss / total_weight

    return combined


def validate_separate(
    lwm,
    lwm_dataset,
    device,
    criterion,
    epoch,
    current_lr,
    best_val_loss,
    batch_size,
    predict_agent_state: bool = True,
    global_step: int = None,
):
    """
    Validate model separately on expert, new_config, replay, and combined data.

    Returns:
        Tuple of (combined_metrics, expert_metrics, new_config_metrics, replay_metrics, is_best, unseen_expert_metrics, unseen_new_config_metrics)
    """
    from torch.utils.data import DataLoader

    # Check which splits have data
    has_expert = lwm_dataset.expert_episode_mask.sum() > 0
    has_new_config = lwm_dataset.new_config_episode_mask.sum() > 0
    has_replay = lwm_dataset.replay_episode_mask.sum() > 0
    has_unseen_expert = (
        hasattr(lwm_dataset, 'unseen_expert_episode_mask') and
        lwm_dataset.unseen_expert_episode_mask.sum() > 0
    )
    has_unseen_new_config = (
        hasattr(lwm_dataset, 'unseen_new_config_episode_mask') and
        lwm_dataset.unseen_new_config_episode_mask.sum() > 0
    )

    # Validate each split
    expert_metrics = None
    if has_expert:
        expert_metrics = _validate_split(
            lwm, lwm_dataset, device, criterion, batch_size,
            "expert", lwm_dataset.get_expert_validation_dataset,
            predict_agent_state, epoch
        )

    new_config_metrics = None
    if has_new_config:
        new_config_metrics = _validate_split(
            lwm, lwm_dataset, device, criterion, batch_size,
            "new_config", lwm_dataset.get_new_config_validation_dataset,
            predict_agent_state, epoch
        )

    replay_metrics = None
    if has_replay:
        replay_metrics = _validate_split(
            lwm, lwm_dataset, device, criterion, batch_size,
            "replay", lwm_dataset.get_replay_validation_dataset,
            predict_agent_state, epoch
        )

    unseen_expert_metrics = None
    if has_unseen_expert:
        unseen_expert_metrics = _validate_split(
            lwm, lwm_dataset, device, criterion, batch_size,
            "unseen_expert", lwm_dataset.get_unseen_expert_validation_dataset,
            predict_agent_state, epoch
        )

    unseen_new_config_metrics = None
    if has_unseen_new_config:
        unseen_new_config_metrics = _validate_split(
            lwm, lwm_dataset, device, criterion, batch_size,
            "unseen_new_config", lwm_dataset.get_unseen_new_config_validation_dataset,
            predict_agent_state, epoch
        )

    # Compute combined metrics as weighted average of individual splits (no extra forward pass)
    # Note: unseen is NOT included in combined (it's out-of-distribution)
    split_metrics_list = [
        (expert_metrics, "expert"),
        (new_config_metrics, "new_config"),
        (replay_metrics, "replay"),
    ]
    combined_metrics = _compute_combined_from_splits(split_metrics_list, predict_agent_state)

    # Aggregate all metrics
    all_metrics = {}
    is_best = False

    for m in [expert_metrics, new_config_metrics, replay_metrics, unseen_expert_metrics, unseen_new_config_metrics, combined_metrics]:
        if m:
            all_metrics.update(m)

    if combined_metrics:
        # Add simple val/ metrics that mirror train/ for easy comparison in wandb
        all_metrics["val/total_loss"] = combined_metrics["val_combined/total_loss"]
        all_metrics["val/affordance_loss"] = combined_metrics["val_combined/affordance_loss"]
        if predict_agent_state and "val_combined/agent_state_loss" in combined_metrics:
            all_metrics["val/agent_state_loss"] = combined_metrics["val_combined/agent_state_loss"]

        cprint(f"val/total_loss={all_metrics['val/total_loss']:.6f}, val/affordance_loss={all_metrics['val/affordance_loss']:.6f}", "green")

        is_best = combined_metrics["val_combined/total_loss"] < best_val_loss
        if is_best:
            cprint(f"New best validation loss: {combined_metrics['val_combined/total_loss']:.6f} at epoch {epoch}", "green")

    cprint(f"Logging {len(all_metrics)} validation metrics to wandb at step={global_step}", "cyan")

    if global_step is not None:
        wandb.log(all_metrics, step=global_step)
    else:
        wandb.log(all_metrics)

    return combined_metrics, expert_metrics, new_config_metrics, replay_metrics, is_best, unseen_expert_metrics, unseen_new_config_metrics


def run_evaluation_with_logging(
    eval_func,
    fixed_episodes_or_steps,
    data_type,
    epoch,
    eval_name,
    predict_agent_state,
    **eval_kwargs
):
    """
    Run evaluation function and log results.
    
    Args:
        eval_func: Evaluation function to call
        fixed_episodes_or_steps: Fixed episodes list or step indices dict
        data_type: Data type string (e.g., "expert", "train_expert")
        epoch: Current epoch
        eval_name: Name for logging (e.g., "expert long-horizon validation")
        predict_agent_state: Whether agent state is predicted
        **eval_kwargs: Additional kwargs to pass to eval_func
    
    Returns:
        Metrics dict or None
    """
    if len(fixed_episodes_or_steps) == 0:
        return None
    
    cprint(f"Running {eval_name} at epoch {epoch}...", "yellow")
    
    # Determine which parameter name to use
    if isinstance(fixed_episodes_or_steps, dict):
        eval_kwargs["fixed_step_indices"] = fixed_episodes_or_steps
    else:
        eval_kwargs["fixed_val_episodes"] = fixed_episodes_or_steps
    
    eval_kwargs["data_type"] = data_type
    eval_kwargs["epoch"] = epoch
    
    metrics = eval_func(**eval_kwargs)
    
    # Determine prefix for logging
    if data_type.startswith("train_"):
        data_type_suffix = data_type[len("train_"):]
        if isinstance(fixed_episodes_or_steps, dict):
            prefix = f"train_near_affordance_{data_type_suffix}"
            eval_type = "near_affordance"
        else:
            prefix = f"train_long_horizon_{data_type_suffix}"
            eval_type = "long_horizon"
    else:
        if isinstance(fixed_episodes_or_steps, dict):
            prefix = f"val_near_affordance_{data_type}"
            eval_type = "near_affordance"
        else:
            prefix = f"val_long_horizon_{data_type}"
            eval_type = "long_horizon"
    
    log_evaluation_results(metrics, prefix, epoch, predict_agent_state, eval_type)
    
    return metrics


def log_evaluation_results(metrics, prefix, epoch, predict_agent_state, eval_type="long_horizon"):
    """
    Log evaluation results to console.
    
    Args:
        metrics: Dict with metrics (can be None)
        prefix: Metric prefix (e.g., "val_long_horizon_expert")
        epoch: Current epoch
        predict_agent_state: Whether agent state is predicted
        eval_type: Type of evaluation ("long_horizon" or "near_affordance")
    """
    if metrics is None:
        return
    
    if eval_type == "near_affordance":
        total_key = f"{prefix}/total_loss"
        affordance_key = f"{prefix}/affordance_loss"
        agent_key = f"{prefix}/agent_state_loss"
        steps_key = f"{prefix}/total_evaluated_steps"
        
        if predict_agent_state:
            cprint(
                f"{prefix.split('/')[0].replace('_', ' ').title()} at epoch {epoch}: "
                f"total={metrics[total_key]:.6f}, "
                f"agent={metrics[agent_key]:.6f}, "
                f"affordance={metrics[affordance_key]:.6f}, "
                f"steps={metrics[steps_key]}",
                "magenta",
            )
        else:
            cprint(
                f"{prefix.split('/')[0].replace('_', ' ').title()} at epoch {epoch}: "
                f"total={metrics[total_key]:.6f}, "
                f"affordance={metrics[affordance_key]:.6f}, "
                f"steps={metrics[steps_key]}",
                "magenta",
            )
    else:  # long_horizon
        total_key = f"{prefix}/total_loss"
        affordance_key = f"{prefix}/affordance_loss"
        agent_key = f"{prefix}/agent_state_loss"
        
        if predict_agent_state:
            cprint(
                f"{prefix.split('/')[0].replace('_', ' ').title()} at epoch {epoch}: "
                f"total={metrics[total_key]:.6f}, "
                f"agent={metrics[agent_key]:.6f}, "
                f"affordance={metrics[affordance_key]:.6f}",
                "magenta",
            )
        else:
            cprint(
                f"{prefix.split('/')[0].replace('_', ' ').title()} at epoch {epoch}: "
                f"total={metrics[total_key]:.6f}, "
                f"affordance={metrics[affordance_key]:.6f}",
                "magenta",
            )


def evaluate_near_affordance_rollout(
    lwm,
    lwm_dataset,
    device,
    fixed_step_indices,
    rollout_length=18,
    data_type="expert",
    checkpoint_dir=None,
    epoch=None,
    action_mean=None,
    action_std=None,
    predict_agent_state: bool = True,
    global_step: int = None,
):
    """
    Evaluate model on 18-step rollouts starting from pre-filtered timesteps where
    end-effector is within 20cm of affordance point.
    
    Args:
        lwm: The LWM model
        lwm_dataset: LWMDataset instance
        device: Device to run on
        fixed_step_indices: dict[int, list[int]] mapping episode_idx -> list of start timesteps
        rollout_length: Length of rollout (default 18 steps, 3x horizon)
        data_type: "expert", "rollout", "train_expert", "train_rollout" - used for metric naming
        checkpoint_dir: Optional checkpoint directory for saving data
        epoch: Current epoch number
        action_mean: Optional action mean for normalization
        action_std: Optional action std for normalization
        predict_agent_state: Whether to predict agent state
    
    Returns:
        Dict with evaluation metrics or None if no valid steps
    """
    if len(fixed_step_indices) == 0:
        return None
    
    lwm.eval()
    criterion = nn.MSELoss(reduction="mean")
    
    rb = lwm_dataset.replay_buffer
    horizon = lwm.n_future_steps
    
    # Get normalization stats from dataset
    (action_mean, action_std, agent_state_delta_mean, agent_state_delta_std,
     affordance_delta_mean, affordance_delta_std) = get_normalization_stats(
        lwm_dataset, device, action_mean, action_std
    )
    
    total_agent_mse = 0.0
    total_affordance_mse = 0.0
    total_evaluated_steps = 0
    
    with torch.no_grad():
        for ep_idx, start_timesteps in fixed_step_indices.items():
            if len(start_timesteps) == 0:
                continue
            
            ep_len = int(rb.episode_lengths[ep_idx])
            if ep_len < rollout_length + 1:
                continue
            
            # Load full episode from disk
            agent_states, static_affordances, point_clouds, actions, _ = load_episode_data(
                rb, ep_idx, device
            )
            
            # Perform rollout from each pre-selected start timestep
            for start_t in start_timesteps:
                if start_t + rollout_length >= ep_len:
                    continue  # Can't do full rollout from this timestep
                
                # Initialize state from ground truth at start timestep
                cur_state = agent_states[start_t].clone()  # [agent_state_dim]
                cur_aff = static_affordances[start_t].clone()  # [3]
                
                # Perform 18-step autoregressive rollout
                for step_offset in range(rollout_length):
                    t = start_t + step_offset
                    if t + 1 >= ep_len:
                        break
                    
                    # Build observation for this step
                    action_horizon = actions[t : t + horizon]  # [horizon, action_dim]
                    action_horizon_normalized = (action_horizon - action_mean) / action_std
                    
                    observation = build_observation(
                        t, cur_state, cur_aff, agent_states, static_affordances,
                        point_clouds, action_horizon_normalized, lwm.n_obs_steps, horizon
                    )
                    
                    # Predict next state and affordance
                    next_state, next_aff, _, _ = predict_next_state_and_affordance(
                        lwm, observation, cur_state, cur_aff,
                        agent_state_delta_mean, agent_state_delta_std,
                        affordance_delta_mean, affordance_delta_std,
                        predict_agent_state, compose_state_with_delta,
                        gt_state_fallback=agent_states[t + 1] if not predict_agent_state else None
                    )
                    
                    # Ground truth
                    gt_state = agent_states[t + 1]
                    gt_aff = static_affordances[t + 1]
                    
                    # Compute losses
                    agent_mse = criterion(next_state, gt_state)
                    affordance_mse = criterion(next_aff, gt_aff)
                    
                    if predict_agent_state:
                        total_agent_mse += agent_mse.item()
                    total_affordance_mse += affordance_mse.item()
                    total_evaluated_steps += 1
                    
                    # Autoregressive: feed prediction back in
                    cur_state = next_state.detach()
                    cur_aff = next_aff.detach()
    
    if total_evaluated_steps == 0:
        return None
    
    # Compute average losses
    if predict_agent_state:
        avg_agent_mse = total_agent_mse / total_evaluated_steps
    avg_affordance_mse = total_affordance_mse / total_evaluated_steps
    if predict_agent_state:
        avg_total_mse = avg_agent_mse + avg_affordance_mse
    else:
        avg_total_mse = avg_affordance_mse
    
    # Determine metric prefix based on data_type
    if data_type.startswith("train_"):
        data_type_suffix = data_type[len("train_"):]
        metric_prefix = f"train_near_affordance_{data_type_suffix}"
    else:
        metric_prefix = f"val_near_affordance_{data_type}"
    
    metrics = {
        f"{metric_prefix}/total_loss": avg_total_mse,
        f"{metric_prefix}/affordance_loss": avg_affordance_mse,
        f"{metric_prefix}/total_evaluated_steps": total_evaluated_steps,
    }
    if predict_agent_state:
        metrics[f"{metric_prefix}/agent_state_loss"] = avg_agent_mse
    
    # Log to wandb
    if global_step is not None:
        wandb.log(metrics, step=global_step)
    else:
        wandb.log(metrics)

    return metrics


# =============================================================================
# Windowed rollout metrics on held-out validation episodes (the paper metric)
# =============================================================================

def heldout_episode_groups(lwm_dataset):
    """Held-out validation episodes by group: {seen,unseen}_{expert,interaction}.

    "seen" = the val split of the training objects (episode_splits val_episodes),
    "unseen" = episodes of objects not trained on. Empty groups are omitted.
    """
    n = len(lwm_dataset.val_mask)
    zeros = np.zeros(n, dtype=bool)
    unseen = getattr(lwm_dataset, "unseen_episode_mask", zeros)
    seen_val = lwm_dataset.val_mask & ~unseen
    interaction = lwm_dataset.new_config_episode_mask | lwm_dataset.replay_episode_mask
    groups = {
        "seen_expert": seen_val & lwm_dataset.expert_episode_mask,
        "seen_interaction": seen_val & interaction,
        "unseen_expert": getattr(lwm_dataset, "unseen_expert_episode_mask", zeros),
        "unseen_interaction": getattr(lwm_dataset, "unseen_new_config_episode_mask", zeros),
    }
    return {k: np.where(m)[0].tolist() for k, m in groups.items() if m.any()}


def select_val_windows(lwm_dataset, n_windows=2048, window_steps=6, horizon=None, seed=0):
    """Fixed random subset of windows (ep_idx, t_start) per held-out group.

    n_windows is split evenly over the groups; the same seed and data give the same
    windows in every epoch and run.
    """
    horizon = lwm_dataset.horizon if horizon is None else horizon
    rb = lwm_dataset.replay_buffer
    groups = heldout_episode_groups(lwm_dataset)
    if not groups:
        return {}
    per_group = max(1, n_windows // len(groups))
    rng = np.random.RandomState(seed)
    out = {}
    for name, eps in groups.items():
        cands = [(ep, t) for ep in eps
                 for t in range(max(0, int(rb.episode_lengths[ep]) - window_steps - horizon))]
        if not cands:
            continue
        idx = rng.choice(len(cands), size=min(per_group, len(cands)), replace=False)
        out[name] = [cands[i] for i in sorted(idx)]
    return out


def summarize_window_results(results, prefix):
    """Mean step-1 / step-6 metrics of evaluate_windowed results."""
    last = len(results[0]["aff_errors"]) - 1
    m = {
        f"{prefix}/n_windows": len(results),
        f"{prefix}/aff_err_step1_cm": float(np.mean([r["aff_errors"][0] for r in results])),
        f"{prefix}/aff_err_step{last + 1}_cm": float(np.mean([r["aff_errors"][last] for r in results])),
    }
    if "agent_errors" in results[0]:
        m[f"{prefix}/ee_err_step1_cm"] = float(np.mean([r["agent_errors"][0] for r in results]))
        m[f"{prefix}/ee_err_step{last + 1}_cm"] = float(np.mean([r["agent_errors"][last] for r in results]))
        m[f"{prefix}/finger_err_step{last + 1}_mm"] = float(np.mean([r["finger_errors_mm"][last] for r in results]))
    if "rot_errors_deg" in results[0]:
        m[f"{prefix}/part_rot_err_step{last + 1}_deg"] = float(np.mean([r["rot_errors_deg"][last] for r in results]))
    return m


def compute_val_rollout_metrics(lwm, lwm_dataset, windows_by_group, normalization_stats, device,
                                window_steps=6, inference_batch_size=256):
    """Windowed rollout metrics (analysis.core.evaluate_windowed) per held-out group.

    Returns {"rollout_<group>/<metric>": value}: handle-point and EE position error at
    step 1 and step ``window_steps`` (cm), part rotation error (deg) and finger error (mm)
    at the last step.
    """
    from lwm.world_model.analysis import core
    from lwm.world_model.evaluation_utils import compose_affordance_with_9dof_delta

    compose = compose_affordance_with_9dof_delta if getattr(lwm, "affordance_output_dim", 3) == 9 else None
    was_training = lwm.training
    metrics = {}
    for name, windows in windows_by_group.items():
        res = core.evaluate_windowed(
            lwm, lwm_dataset.replay_buffer, None, device, normalization_stats,
            window_steps=window_steps, predict_agent_state=lwm.predict_agent_state,
            compose_affordance_fn=compose, windows=windows, inference_batch_size=inference_batch_size,
        )
        if res:
            metrics.update(summarize_window_results(res, f"rollout_{name}"))
    if was_training:
        lwm.train()
    return metrics
