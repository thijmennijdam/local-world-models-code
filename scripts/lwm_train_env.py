#!/usr/bin/env python
"""Print shell `export` lines for one LWM training run in configs/lwm/runs.yaml.

Used by scripts/train_lwm.sh. The mapping from run config to environment variables
is the one used to train the paper's models (lwm.world_model.train reads them).

Usage: python scripts/lwm_train_env.py configs/lwm/runs.yaml <run name>
"""

import json
import os
import shlex
import sys

import yaml


def run_env(cfg, defaults):
    """Return the environment for one run (defaults + per-run overrides)."""
    # Apply defaults, then override with config-specific values
    # Support both "name" and "exp_name" for backward compatibility, but prefer "name"
    name = cfg.get("name", cfg.get("exp_name", defaults.get("name", defaults.get("exp_name"))))
    exp_name = name
    expert_dataset_path = cfg.get("expert_dataset_path", defaults.get("expert_dataset_path"))
    rollout_dataset_path = cfg.get("rollout_dataset_path", defaults.get("rollout_dataset_path"))
    horizon = cfg.get("horizon", defaults.get("horizon"))
    # For WM, n_future_steps always equals horizon
    n_future_steps = horizon
    n_obs_steps = cfg.get("n_obs_steps", defaults.get("n_obs_steps", 1))
    batch_size = cfg.get("batch_size", defaults.get("batch_size"))
    num_epochs = cfg.get("num_epochs", defaults.get("num_epochs"))
    learning_rate = cfg.get("learning_rate", defaults.get("learning_rate"))
    lr_warmup_steps = cfg.get("lr_warmup_steps", defaults.get("lr_warmup_steps"))
    weight_decay = cfg.get("weight_decay", defaults.get("weight_decay"))
    affordance_loss_weight = cfg.get("affordance_loss_weight", defaults.get("affordance_loss_weight"))
    affordance_encoder_size = cfg.get("affordance_encoder_size", defaults.get("affordance_encoder_size"))
    pointcloud_encoder_size = cfg.get("pointcloud_encoder_size", defaults.get("pointcloud_encoder_size"))
    action_encoder_size = cfg.get("action_encoder_size", defaults.get("action_encoder_size"))
    agent_state_encoder_size = cfg.get("agent_state_encoder_size", defaults.get("agent_state_encoder_size"))
    wm_hidden_dims = cfg.get("wm_hidden_dims", defaults.get("wm_hidden_dims"))
    affordance_output_dim = cfg.get("affordance_output_dim", defaults.get("affordance_output_dim"))
    use_cls_token = cfg.get("use_cls_token", defaults.get("use_cls_token"))
    # v3-specific: d_model, n_layers, n_heads
    d_model = cfg.get("d_model", defaults.get("d_model"))
    n_layers = cfg.get("n_layers", defaults.get("n_layers"))
    n_heads = cfg.get("n_heads", defaults.get("n_heads"))
    # v6-specific: k_nearest (number of nearest points to predict)
    k_nearest = cfg.get("k_nearest", defaults.get("k_nearest"))
    force_recopy_dataset = cfg.get("force_recopy_dataset", defaults.get("force_recopy_dataset", False))
    use_pointnet2 = cfg.get("use_pointnet2", defaults.get("use_pointnet2"))
    # Episode filtering (object IDs and goal percentages)
    include_object_ids = cfg.get("include_object_ids_expert_replay", defaults.get("include_object_ids_expert_replay",
                          cfg.get("include_object_ids", defaults.get("include_object_ids"))))
    include_object_ids_new_config = cfg.get("include_object_ids_new_config", defaults.get("include_object_ids_new_config"))
    allowed_goal_percentages_door = cfg.get("allowed_goal_percentages_door_expert_replay", defaults.get("allowed_goal_percentages_door_expert_replay",
                                     cfg.get("allowed_goal_percentages_door", defaults.get("allowed_goal_percentages_door"))))
    allowed_goal_percentages_drawer = cfg.get("allowed_goal_percentages_drawer_expert_replay", defaults.get("allowed_goal_percentages_drawer_expert_replay",
                                       cfg.get("allowed_goal_percentages_drawer", defaults.get("allowed_goal_percentages_drawer"))))
    # Contact prediction
    predict_contact = cfg.get("predict_contact", defaults.get("predict_contact"))
    contact_loss_weight = cfg.get("contact_loss_weight", defaults.get("contact_loss_weight"))
    contact_threshold = cfg.get("contact_threshold", defaults.get("contact_threshold"))
    gumbel_tau = cfg.get("gumbel_tau", defaults.get("gumbel_tau"))
    # Rotation normalization
    skip_rotation_normalization = cfg.get("skip_rotation_normalization", defaults.get("skip_rotation_normalization"))
    # Per-component agent state loss weights
    gpus = int(cfg.get("gpus", defaults.get("gpus", 1)))
    # Experiment switches (see configs/lwm/train.yaml, dataset section)
    balanced_sampling = cfg.get("balanced_sampling", defaults.get("balanced_sampling"))
    affordance_target_frame = cfg.get("affordance_target_frame", defaults.get("affordance_target_frame"))
    input_frame = cfg.get("input_frame", defaults.get("input_frame"))
    augment_yaw = cfg.get("augment_yaw", defaults.get("augment_yaw"))
    augment_translation = cfg.get("augment_translation", defaults.get("augment_translation"))
    pcd_tokens = cfg.get("pcd_tokens", defaults.get("pcd_tokens"))
    pcd_group_size = cfg.get("pcd_group_size", defaults.get("pcd_group_size"))
    # Training seed (model init, sampling); unset = dataset.seed (42). The train/val
    # split always uses dataset.seed, so runs with different seeds share the split.
    seed = cfg.get("seed", defaults.get("seed"))

    gpus_per_job = gpus
    # Support both "agent_state" and "predict_agent_state" for backward compatibility
    predict_agent_state = cfg.get("predict_agent_state", cfg.get("agent_state", defaults.get("predict_agent_state", defaults.get("agent_state"))))
    # Data splits configuration (deep merge: per-split override from cfg on top of defaults)
    default_data_splits = defaults.get("data_splits", {})
    cfg_data_splits = cfg.get("data_splits", {})
    data_splits = {}
    for split_name in ["expert", "new_config", "replay"]:
        default_split = default_data_splits.get(split_name, {})
        cfg_split = cfg_data_splits.get(split_name, {})
        merged = dict(default_split)
        merged.update(cfg_split)
        data_splits[split_name] = merged


    env = os.environ.copy()

    # Set environment variables from config
    if name is not None:
        env["EXP_NAME"] = str(name)
    elif exp_name is not None:
        env["EXP_NAME"] = str(exp_name)
    if expert_dataset_path:
        env["EXPERT_DATASET_PATH"] = str(expert_dataset_path)
    if rollout_dataset_path:
        env["ROLLOUT_DATASET_PATH"] = str(rollout_dataset_path)
    if horizon is not None:
        env["HORIZON"] = str(horizon)
        # n_future_steps is derived from horizon in train_lwm.py, but we can also set it here for clarity
        env["N_FUTURE_STEPS"] = str(n_future_steps)
    if n_obs_steps is not None:
        env["N_OBS_STEPS"] = str(n_obs_steps)
    if batch_size is not None:
        env["BATCH_SIZE"] = str(batch_size)
    if num_epochs is not None:
        env["NUM_EPOCHS"] = str(num_epochs)
    if learning_rate is not None:
        env["LEARNING_RATE"] = str(learning_rate)
    if lr_warmup_steps is not None:
        env["LR_WARMUP_STEPS"] = str(lr_warmup_steps)
    if weight_decay is not None:
        env["WEIGHT_DECAY"] = str(weight_decay)
    if affordance_loss_weight is not None:
        env["AFFORDANCE_LOSS_WEIGHT"] = str(affordance_loss_weight)
    if affordance_encoder_size is not None:
        env["AFFORDANCE_ENCODER_SIZE"] = str(affordance_encoder_size)
    if pointcloud_encoder_size is not None:
        env["POINTCLOUD_ENCODER_SIZE"] = str(pointcloud_encoder_size)
    if action_encoder_size is not None:
        env["ACTION_ENCODER_SIZE"] = str(action_encoder_size)
    if agent_state_encoder_size is not None:
        env["AGENT_STATE_ENCODER_SIZE"] = str(agent_state_encoder_size)
    if wm_hidden_dims is not None:
        # Convert list to string format that can be parsed by Python
        if isinstance(wm_hidden_dims, list):
            env["WM_HIDDEN_DIMS"] = json.dumps(wm_hidden_dims)
        else:
            env["WM_HIDDEN_DIMS"] = str(wm_hidden_dims)
    if affordance_output_dim is not None:
        env["AFFORDANCE_OUTPUT_DIM"] = str(affordance_output_dim)
    if use_cls_token is not None:
        env["USE_CLS_TOKEN"] = "true" if use_cls_token else "false"
    if d_model is not None:
        env["D_MODEL"] = str(int(d_model))
    if n_layers is not None:
        env["N_LAYERS"] = str(int(n_layers))
    if n_heads is not None:
        env["N_HEADS"] = str(int(n_heads))
    if k_nearest is not None:
        env["K_NEAREST"] = str(int(k_nearest))
    if predict_agent_state is not None:
        env["PREDICT_AGENT_STATE"] = "true" if predict_agent_state else "false"
    if force_recopy_dataset is not None:
        env["FORCE_RECOPY_DATASET"] = "true" if force_recopy_dataset else "false"
    if use_pointnet2 is not None:
        env["USE_POINTNET2"] = "true" if use_pointnet2 else "false"
    if predict_contact is not None:
        env["PREDICT_CONTACT"] = str(predict_contact)
    if contact_loss_weight is not None:
        env["CONTACT_LOSS_WEIGHT"] = str(contact_loss_weight)
    if contact_threshold is not None:
        env["CONTACT_THRESHOLD"] = str(contact_threshold)
    if gumbel_tau is not None:
        env["GUMBEL_TAU"] = str(gumbel_tau)
    if skip_rotation_normalization is not None:
        env["SKIP_ROTATION_NORMALIZATION"] = "true" if skip_rotation_normalization else "false"
    if balanced_sampling is not None:
        env["BALANCED_SAMPLING"] = "true" if balanced_sampling else "false"
    # Always exported (with their defaults), so a value left in the shell by another
    # run cannot leak into this one.
    env["AFFORDANCE_TARGET_FRAME"] = str(affordance_target_frame or "world")
    env["INPUT_FRAME"] = str(input_frame or "world")
    env["AUGMENT_YAW"] = str(augment_yaw or 0)
    env["AUGMENT_TRANSLATION"] = str(augment_translation or 0)
    env["PCD_TOKENS"] = str(int(pcd_tokens or 1))
    env["PCD_GROUP_SIZE"] = str(int(pcd_group_size or 32))
    if seed is not None:
        env["RANDOM_SEED"] = str(int(seed))
    if data_splits:
        env["DATA_SPLITS"] = json.dumps(data_splits)
    if include_object_ids is not None:
        env["INCLUDE_OBJECT_IDS"] = json.dumps(include_object_ids)
    if include_object_ids_new_config is not None:
        env["INCLUDE_OBJECT_IDS_NEW_CONFIG"] = json.dumps(include_object_ids_new_config)
    if allowed_goal_percentages_door is not None:
        env["ALLOWED_GOAL_PERCENTAGES_DOOR"] = json.dumps(allowed_goal_percentages_door)
    if allowed_goal_percentages_drawer is not None:
        env["ALLOWED_GOAL_PERCENTAGES_DRAWER"] = json.dumps(allowed_goal_percentages_drawer)
    env["NUM_GPUS"] = str(gpus_per_job)
    return env


def main():
    config_path, run_name = sys.argv[1], sys.argv[2]
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(os.path.expandvars(f.read()))
    defaults = config_data.get("defaults", {})
    configs = config_data.get("configs", [])
    matches = [c for c in configs if c.get("name") == run_name]
    if not matches:
        names = [c.get("name") for c in configs]
        sys.exit(f"Unknown run '{run_name}'. Available: {names}")
    env = run_env(matches[0], defaults)
    for key, value in sorted(env.items()):
        if os.environ.get(key) != value:
            print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
