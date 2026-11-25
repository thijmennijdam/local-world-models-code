"""Checkpoint and config loading utilities for LWM evaluation."""

import json
import pathlib
import pickle
import types

import torch
import hydra
from omegaconf import OmegaConf
from termcolor import cprint


# Checkpoints trained before the package was renamed store the model class under
# its old module path (in .hydra/config.yaml). Map those to the current class.
DEFAULT_MODEL_TARGET = "lwm.world_model.model.LWMv3"
LEGACY_TARGETS = {
    "local_world_model.lwm_v3.LWMv3": DEFAULT_MODEL_TARGET,
    "lwm_v3.LWMv3": DEFAULT_MODEL_TARGET,
}

# Old module paths that may appear in pickled checkpoint objects.
_LEGACY_MODULE_PREFIXES = (
    ("local_world_model.lwm_v3", "lwm.world_model.model"),
    ("local_world_model.lwm_dataset", "lwm.world_model.dataset"),
    ("local_world_model.", "lwm.world_model."),
    ("lwm_v3", "lwm.world_model.model"),
)


def _remap_module(module: str) -> str:
    for old, new in _LEGACY_MODULE_PREFIXES:
        if old.endswith("."):
            if module.startswith(old):
                return new + module[len(old):]
        elif module == old or module.startswith(old + "."):
            return new + module[len(old):]
    return module


class _LegacyUnpickler(pickle.Unpickler):
    """Unpickler that resolves classes pickled under the old module paths."""

    def find_class(self, module, name):
        return super().find_class(_remap_module(module), name)


# torch.load(pickle_module=...) needs a module-like object with Unpickler and load.
_legacy_pickle = types.SimpleNamespace(
    Unpickler=_LegacyUnpickler,
    load=lambda f, **kw: _LegacyUnpickler(f, **kw).load(),
    __name__="lwm_legacy_pickle",
)


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """
    Load LWM model from checkpoint.

    Returns:
        Tuple of (model, checkpoint_dict, normalization_stats, config)
    """
    cprint(f"Loading checkpoint from {checkpoint_path}", "yellow")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False, pickle_module=_legacy_pickle
    )

    # Load config from .hydra/config.yaml next to checkpoint
    # Check both: same dir as checkpoint (flat layout) and parent dir (nested checkpoints/ layout)
    checkpoint_dir = pathlib.Path(checkpoint_path).parent
    config_path = None
    for candidate in [checkpoint_dir / ".hydra" / "config.yaml", checkpoint_dir.parent / ".hydra" / "config.yaml"]:
        if candidate.exists():
            config_path = candidate
            break

    if config_path is None:
        raise FileNotFoundError(f"Config not found. Searched {checkpoint_dir}/.hydra/ and {checkpoint_dir.parent}/.hydra/")

    cprint(f"Loading config from {config_path}", "cyan")
    full_config = OmegaConf.load(config_path)
    model_config = full_config.model

    # Get normalization stats from checkpoint
    normalization_stats = checkpoint.get("normalization_stats", None)
    if normalization_stats is None:
        cprint("Warning: No normalization stats found in checkpoint", "yellow")

    model_config_dict = OmegaConf.to_container(model_config, resolve=True)

    # Ensure _target_ is set so hydra.utils.instantiate knows which class to use
    if "_target_" not in model_config_dict:
        model_config_dict["_target_"] = DEFAULT_MODEL_TARGET
    model_config_dict["_target_"] = LEGACY_TARGETS.get(
        model_config_dict["_target_"], model_config_dict["_target_"]
    )

    cprint(f"Model config: _target_={model_config_dict['_target_']}, "
           f"n_future_steps={model_config_dict.get('n_future_steps')}, "
           f"agent_state_dim={model_config_dict.get('agent_state_dim')}, "
           f"predict_agent_state={model_config_dict.get('predict_agent_state')}, "
           f"affordance_output_dim={model_config_dict.get('affordance_output_dim', 3)}", "cyan")

    # Infer actual model dimensions from state_dict, since the saved Hydra config
    # may not reflect env var overrides (d_model, n_layers) applied during training.
    state_dict = checkpoint["model_state_dict"]

    if "state_encoder.weight" in state_dict:
        actual_d_model = state_dict["state_encoder.weight"].shape[0]
        if actual_d_model != model_config_dict.get("d_model", -1):
            cprint(f"Overriding d_model: config={model_config_dict.get('d_model')} -> state_dict={actual_d_model}", "yellow")
            model_config_dict["d_model"] = actual_d_model

    layer_indices = [int(k.split(".")[2]) for k in state_dict if k.startswith("transformer.layers.")]
    if layer_indices:
        actual_n_layers = max(layer_indices) + 1
        if actual_n_layers != model_config_dict.get("n_layers", -1):
            cprint(f"Overriding n_layers: config={model_config_dict.get('n_layers')} -> state_dict={actual_n_layers}", "yellow")
            model_config_dict["n_layers"] = actual_n_layers

    # Infer n_obs_steps from state_dict: obs_step_embed exists only when n_obs_steps > 1
    if "obs_step_embed.weight" in state_dict:
        actual_n_obs_steps = state_dict["obs_step_embed.weight"].shape[0]
        if actual_n_obs_steps != model_config_dict.get("n_obs_steps", 1):
            cprint(f"Overriding n_obs_steps: config={model_config_dict.get('n_obs_steps', 1)} -> state_dict={actual_n_obs_steps}", "yellow")
            model_config_dict["n_obs_steps"] = actual_n_obs_steps

    # Infer pointcloud_encoder_size from pcd_proj weight shape
    if "pcd_proj.weight" in state_dict:
        actual_pcd_size = state_dict["pcd_proj.weight"].shape[1]
        if actual_pcd_size != model_config_dict.get("pointcloud_encoder_size", -1):
            cprint(f"Overriding pointcloud_encoder_size: config={model_config_dict.get('pointcloud_encoder_size')} -> state_dict={actual_pcd_size}", "yellow")
            model_config_dict["pointcloud_encoder_size"] = actual_pcd_size

    # Infer use_pointnet2 from state_dict: sa1 keys exist only for PointNet++
    has_pointnet2_keys = any(k.startswith("pcd_encoder.sa1.") for k in state_dict)
    if has_pointnet2_keys != model_config_dict.get("use_pointnet2", False):
        cprint(f"Overriding use_pointnet2: config={model_config_dict.get('use_pointnet2', False)} -> state_dict={has_pointnet2_keys}", "yellow")
        model_config_dict["use_pointnet2"] = has_pointnet2_keys

    # Infer predict_contact from state_dict: contact_head exists only when predict_contact != "none"
    has_contact_head = any(k.startswith("contact_head.") for k in state_dict)
    has_identity_normalized = "identity_normalized" in state_dict
    if has_contact_head:
        # Determine mode: gumbel_gate has identity_normalized buffer, bce does not
        inferred_mode = "gumbel_gate" if has_identity_normalized else "bce"
        if model_config_dict.get("predict_contact", "none") != inferred_mode:
            cprint(f"Overriding predict_contact: config={model_config_dict.get('predict_contact', 'none')} -> state_dict={inferred_mode}", "yellow")
            model_config_dict["predict_contact"] = inferred_mode

    # Infer pcd_tokens from state_dict: pcd_center_embed exists only for K > 1 patch tokens;
    # K follows from the stored attention mask (K + 2 obs tokens + n_future_steps actions).
    # pcd_group_size has no weights and is read from the config (default 32).
    if "pcd_center_embed.0.weight" in state_dict:
        actual_pcd_tokens = (state_dict["attn_mask"].shape[0]
                             - model_config_dict["n_future_steps"] - 2)
        if actual_pcd_tokens != model_config_dict.get("pcd_tokens", 1):
            cprint(f"Overriding pcd_tokens: config={model_config_dict.get('pcd_tokens', 1)} -> state_dict={actual_pcd_tokens}", "yellow")
            model_config_dict["pcd_tokens"] = actual_pcd_tokens
    elif model_config_dict.get("pcd_tokens", 1) != 1:
        cprint(f"Overriding pcd_tokens: config={model_config_dict['pcd_tokens']} -> state_dict=1", "yellow")
        model_config_dict["pcd_tokens"] = 1

    model = hydra.utils.instantiate(model_config_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    cprint(f"Model loaded successfully — {num_params:,} params ({num_params/1e6:.2f}M)", "green")
    return model, checkpoint, normalization_stats, full_config


def load_episode_splits(checkpoint_path: str):
    """
    Load episode splits from training.

    Returns:
        dict with train_episodes, val_episodes, etc. or None if not found
    """
    checkpoint_dir = pathlib.Path(checkpoint_path).parent

    possible_paths = [
        checkpoint_dir.parent / "episode_splits.json",
        checkpoint_dir / "episode_splits.json",
    ]

    for splits_path in possible_paths:
        if splits_path.exists():
            with open(splits_path, 'r') as f:
                return json.load(f)

    return None
