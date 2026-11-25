"""
Utility functions for training setup and checkpoint management.
"""

import os
import pathlib
import datetime
from termcolor import cprint
from lwm.common.wandb_compat import wandb
from omegaconf import OmegaConf


def setup_checkpoint_dir(output_dir=None, checkpoint_base=None, exp_name="lwm_training"):
    """
    Setup checkpoint directory structure.
    
    Args:
        output_dir: Output directory from Hydra (if available)
        checkpoint_base: Base directory for checkpoints (from env or default)
        exp_name: Experiment name
    
    Returns:
        Path to checkpoint directory
    """
    if checkpoint_base is None:
        if output_dir is not None:
            checkpoint_dir = pathlib.Path(output_dir) / "checkpoints"
        else:
            checkpoint_base = os.environ.get("CHECKPOINT_BASE_DIR")
            if checkpoint_base is None:
                checkpoint_base = pathlib.Path("checkpoints")
            else:
                checkpoint_base = pathlib.Path(checkpoint_base)
            now = datetime.datetime.now()
            datetime_str = now.strftime("%Y.%m.%d_%H.%M.%S")
            experiment_dir = checkpoint_base / exp_name / exp_name / datetime_str
            experiment_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = experiment_dir / "checkpoints"
    else:
        checkpoint_dir = pathlib.Path(checkpoint_base)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    
    cprint("==================================", "green")
    cprint("================ CHECKPOINT DIR ================", "green")
    cprint(f"{checkpoint_dir}", "green")
    cprint("==================================", "green")
    
    return checkpoint_dir


def save_config_to_checkpoint(checkpoint_dir, full_cfg):
    """
    Save the full config to .hydra/config.yaml for later loading.
    
    Args:
        checkpoint_dir: Checkpoint directory path
        full_cfg: Full Hydra config
    """
    if full_cfg is not None:
        hydra_dir = checkpoint_dir.parent / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
        config_path = hydra_dir / "config.yaml"
        OmegaConf.save(full_cfg, config_path)
        cprint(f"Saved config to {config_path}", "cyan")


def initialize_wandb(exp_name, lwm, wandb_cfg=None,
                    pointcloud_encoder_size=256,
                    learning_rate=0.001, batch_size=32, num_epochs=1,
                    lr_scheduler_type="none", lr_warmup_steps=0, max_grad_norm=None,
                    checkpoint_dir=None, num_params=None,
                    predict_agent_state=True,
                    weight_decay=None, affordance_loss_weight=1.0,
                    data_splits=None, model_version="v8"):
    """
    Initialize wandb for logging.

    Args:
        exp_name: Experiment name
        lwm: LWM model instance
        wandb_cfg: Wandb config dict
        Other args: Model and training parameters

    Returns:
        None (wandb is initialized globally)
    """
    wandb_cfg_dict = wandb_cfg if wandb_cfg is not None else {}
    wandb_project = wandb_cfg_dict.get("project", "actual_wm_training")
    wandb_entity = wandb_cfg_dict.get("entity", None)
    wandb_name = exp_name

    # Logging is disabled unless WANDB_MODE is set (e.g. "online" or "offline");
    # credentials come from the usual wandb login / WANDB_API_KEY mechanism.
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name,
        mode=os.environ.get("WANDB_MODE", "disabled"),
        config={
            "model_version": model_version,
            "agent_state_dim": lwm.agent_state_dim,
            "pointcloud_encoder_size": pointcloud_encoder_size,
            "d_model": lwm.d_model,
            "n_future_steps": lwm.n_future_steps,
            "n_obs_steps": lwm.n_obs_steps,
            "affordance_output_dim": lwm.affordance_output_dim,
            "predict_agent_state": predict_agent_state,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "exp_name": exp_name,
            "num_parameters": num_params,
            "lr_scheduler_type": lr_scheduler_type,
            "lr_warmup_steps": lr_warmup_steps,
            "max_grad_norm": max_grad_norm,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
            "weight_decay": weight_decay,
            "affordance_loss_weight": affordance_loss_weight,
            "data_splits": data_splits,
        }
    )


def save_checkpoint(checkpoint_dir, lwm, optimizer, epoch, exp_name, learning_rate,
                   batch_size, best_val_loss, lr_scheduler=None, suffix="", normalization_stats=None):
    """
    Save a checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        lwm: LWM model
        optimizer: Optimizer
        epoch: Current epoch
        exp_name: Experiment name
        learning_rate: Current learning rate
        batch_size: Batch size
        best_val_loss: Best validation loss
        lr_scheduler: Optional LR scheduler
        suffix: Suffix for checkpoint filename (e.g., "best", "last", "epoch_10")
        normalization_stats: Optional dict with normalization stats (agent_state_delta_mean, agent_state_delta_std,
                           affordance_delta_mean, affordance_delta_std, action_mean, action_std)
    
    Returns:
        Path to saved checkpoint
    """
    if suffix:
        checkpoint_path = checkpoint_dir / f"{suffix}.ckpt"
    else:
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.ckpt"
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": lwm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "exp_name": exp_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_val_loss": best_val_loss,
    }
    if lr_scheduler is not None:
        checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if normalization_stats is not None:
        checkpoint["normalization_stats"] = normalization_stats
    
    import torch
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
