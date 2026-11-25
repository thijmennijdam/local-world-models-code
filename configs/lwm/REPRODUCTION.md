# Reproducing the paper's world models

`configs/lwm/runs.yaml` entries `LWM-200M_d1024_L16` and `LWM-100M_d1024_L8` reproduce the
released checkpoints. Evidence sources:

- **ckpt**: the released `best.ckpt` (`normalization_stats`, `optimizer_state_dict`,
  `lr_scheduler_state_dict`, `epoch`, `batch_size`, `model_state_dict`).
- **cfg**: the released `.hydra/config.yaml`. It is the runtime config written by the
  trainer, but env-var overrides that the trainer did not write back are missing from it.
  `.hydra/overrides.yaml` is `[]` for both runs, so no Hydra CLI overrides were used.
- **log / history**: the training logs (`train_lwm.log`; the 100M run started 2026-02-14 09:27,
  the 200M run 2026-02-20 10:27) and the history of the training code and its launcher run tables.

| Setting | 200M | 100M | Evidence |
|---|---|---|---|
| d_model / n_layers | 1024 / 16 | 1024 / 8 | ckpt state_dict shapes |
| n_heads | 4 | 4 | cfg (`model.n_heads: 4`). The 200M launcher table set `n_heads: 8`, but neither the launcher nor the trainer handled `n_heads` before March 2026, so the yaml default 4 was used. Head count is not visible in the weights. |
| learning_rate | 6e-5 | 1e-4 | ckpt `lr_scheduler_state_dict.base_lrs` |
| lr_warmup_steps | 500 | 100 | ckpt scheduler: `T_max - last_epoch` = 112000-111500 = 500 and 66500-66400 = 100 (the cosine scheduler is not stepped during warmup) |
| weight_decay | 0.01 | **1e-3** | ckpt `optimizer_state_dict.param_groups[0].weight_decay` |
| num_epochs | 500 | 500 | ckpt `epoch`, scheduler `T_max` = 500 x batches/epoch (224 and 133) |
| batch_size (per GPU) / GPUs | 512 / 2 | 512 / 2 | ckpt `batch_size`, `train_lwm.log` (2 DDP ranks). 224 batches/epoch x 512 x 2 GPUs ~ 229k training pairs, consistent with 0.5e5 + 2.2e5 steps minus the val split and the horizon; same check for 100M (133 batches, 1.1e5) |
| skip_rotation_normalization | false | false | ckpt `normalization_stats`: the rotation dims are normalised. |
| balanced_sampling | false | false | cfg `dataset.balanced_sampling: false`. The trainer read it only from the Hydra config (no env var, no launcher key, no CLI overrides), so the saved value is the one used. (The 26M ablation of 2026-02-14 00:02 has `true` in its cfg.) |
| predict_agent_state | true | true | cfg, `agent_state_head` in state_dict |
| horizon / n_obs_steps | 6 / 1 | 6 / 1 | cfg `n_future_steps: 6`; no `obs_step_embed` in state_dict |
| data budgets (expert / interaction) | 0.5e5 / 2.2e5 | 0.5e5 / 1.1e5 | launcher run tables (`..._huge`, `..._large`) and the scheduler batch counts above |
| object / goal filters | 8 seen objects, goal % lists | same | launcher defaults and run entries; `episode_splits.json` train/val objects = the 8 ids |
| use_amp, max_grad_norm, cosine schedule, affordance_loss_weight, horizon_weight_decay | true, 1.0, cosine, 1.0, none | same | cfg |
| frames, augmentation | off (world / 0) | off | not used |

## Differences from the released training runs

- **Validation episodes.** The per-split validation (`val_expert`, `val_new_config`,
  `val_replay`, whose weighted mean selects `best.ckpt`) iterates the held-out val split of
  `episode_splits.json` (and the unseen objects). `best.ckpt` of a re-run is therefore chosen
  on held-out loss and can be a different epoch from the released checkpoints, which are the
  final epoch of their training runs.
- **Checkpoints.** The end-of-training checkpoint is `last.ckpt`.
- **Rollout metrics.** At every validation epoch, W&B also logs windowed rollout errors on
  held-out windows (`rollout_*`) and throughput (`perf/*`). These do not affect training.
