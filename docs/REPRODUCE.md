# Reproducing the paper

This guide walks through every step behind the paper's results: getting the data, training the
world models, the scaling figures and the planning table. For a first look, the
[README](../README.md) is enough.

## 1. Installation

Python 3.9 or newer. An editable install is needed because the Hydra configs live in `configs/`.

```bash
git clone https://github.com/thijmennijdam/local-world-models-code.git
cd local-world-models-code
pip install -e .            # or: pip install -e ".[wandb]" for Weights & Biases logging
```

PyTorch is installed from PyPI; pick a CUDA build from pytorch.org first if you need one.

## 2. Data and checkpoints

Everything is on the Hugging Face Hub as dataset repositories (each holds one zip):

| Repository | Content | Used by | Environment variable |
|---|---|---|---|
| `thijmennijdam/lwm-checkpoints` | LWM-100M and LWM-200M (`<run>/best.ckpt`, `.hydra/config.yaml`, `episode_splits.json`) | evaluation, MPC | `LWM_CKPT_DIR` |
| `thijmennijdam/lwm-expert-data` | expert demonstrations (one pickle per time step) | training, evaluation | `LWM_EXPERT_DATA_DIR` |
| `thijmennijdam/lwm-interaction-data` | interaction episodes of the diffusion-policy checkpoints | training, evaluation | `LWM_INTERACTION_DATA_DIR` |
| `thijmennijdam/lwm-expert-data-subset`, `…-interaction-data-subset` | small subsets for the tests | tests | same |
| `thijmennijdam/lwm-data-generation-assets` | PartNet-Mobility objects used in simulation | MPC | `PARTNET_DATASET_DIR` |

```bash
pip install -U huggingface_hub
for d in lwm-checkpoints lwm-expert-data lwm-interaction-data lwm-data-generation-assets; do
  hf download thijmennijdam/$d --repo-type dataset --local-dir data/$d
  (cd data/$d && for z in *.zip; do unzip -qn "$z" && rm "$z"; done)
done
export LWM_CKPT_DIR=$PWD/data/lwm-checkpoints
export LWM_EXPERT_DATA_DIR=$PWD/data/lwm-expert-data
export LWM_INTERACTION_DATA_DIR=$PWD/data/lwm-interaction-data
export PARTNET_DATASET_DIR=$PWD/data/lwm-data-generation-assets/partnet-mobility
```

`unzip -n` keeps the `.hydra/config.yaml` files of `lwm-checkpoints` that are already there: they
record the training settings that were used, the copies in the zip are the older, incomplete ones.
Planning needs only `lwm-checkpoints` and `lwm-data-generation-assets`; the start states of Table 1
are in this repository (`configs/mpc/table1/`).

Check the directory names after unzipping (the zips may carry one top-level directory) and point
the variables at the directories that hold `LWM-200M_d1024_L16/`, the `epoch-*/` interaction
episodes, and `partnet-mobility/`.

The scaling figures additionally use the ablation checkpoints named in `configs/lwm/eval/*.yaml`, which
are not part of the release.

Outputs go to `$LWM_OUTPUT_DIR` (default `./outputs`).

## 3. Training

```bash
bash scripts/train_lwm.sh LWM-200M_d1024_L16     # or LWM-100M_d1024_L8
```

`configs/lwm/runs.yaml` holds the paper models' recipes (the evidence for every setting is in
`configs/lwm/REPRODUCTION.md`) and two 26M examples; `configs/lwm/train.yaml` is the base config.
The paper models trained on 2 GPUs with a per-GPU batch of 512; `NUM_GPUS=1 BATCH_SIZE_OVERRIDE=1024`
keeps the recipe on one GPU. Checkpoints go to `$CHECKPOINT_BASE_DIR` (default `./checkpoints`), the
final one is `last.ckpt`. Set `WANDB_MODE=online` (and `WANDB_ENTITY`, `WANDB_PROJECT`) to log to
Weights & Biases; logging is off otherwise.

Training reads much faster from *packed* datasets (a few memory-mapped `.npy` files plus `index.json`
instead of one pickle per time step):

```bash
python scripts/pack_dataset.py --src data/lwm-expert-data --out data/packed/expert
python scripts/pack_dataset.py --src data/lwm-interaction-data --out data/packed/interaction
export LWM_EXPERT_DATA_DIR=$PWD/data/packed/expert LWM_INTERACTION_DATA_DIR=$PWD/data/packed/interaction
```

Episode order, splits, step budgets and filters are identical for both layouts, and the evaluation
accepts packed directories too.

## 4. Scaling figures (Figs. 4 and 5)

The windowed evaluation rolls the model out for 6 steps from random starting points of held-out
episodes (episodes that none of the compared models trained on) and reports the handle-point and
end-effector errors at every step; the figures plot the error at step 6.

```bash
bash scripts/eval_scaling.sh model_size_ablation_large     # model scale (Fig. 4)
bash scripts/eval_scaling.sh pointnet_data_ablations       # data scale (Fig. 5)
bash scripts/eval_scaling.sh no_agent_state                # affordance-only models (Fig. 5, stars)
python scripts/eval_table.py outputs/lwm_eval model_size_ablation_large   # table of the errors
python scripts/plot_scaling_figs.py configs/figures/model_scale.yaml      # figures/lwm_arch_combined.pdf
python scripts/plot_scaling_figs.py configs/figures/data_scale.yaml       # figures/lwm_data_ablation.pdf
```

The paper averages three window seeds: run each config with `LWM_EVAL_SEED=42`, `1` and `2`
(outputs land in `outputs/lwm_eval/<config>_seed<seed>`, which the figure specs list); with fewer
seeds, edit `eval_dirs` in `configs/figures/*.yaml`.

## 5. Zero-shot planning

One episode:

```bash
python scripts/table1_start_states.py      # writes the start states to data/table1-start-states
bash scripts/run_mpc_episode.sh $LWM_CKPT_DIR/LWM-200M_d1024_L16/best.ckpt 45526 \
    $PWD/data/table1-start-states/45526/new_config_45526_2026-02-13-14-06-49_0150 0.4 \
    bias=contact_kinematics execution.seed=14
```

`configs/mpc/paper.yaml` holds the planner settings of the paper; `bias=` selects the inductive
biases (`none`, `contact`, `kinematics`, `contact_kinematics`). Directly:
`python -m lwm.planning.run_mpc --config-name paper checkpoint_path=... task.object_id=...`.
Every run writes `metrics.json` (joint angles, task progress) and, with `logging.save_trajectories=true`,
the trajectory with per-step prediction errors.

### The planning table (Table 1)

The per-episode results of the paper are in `results/table1_metrics.tar.gz`, so the table can be
recomputed without a GPU:

```bash
mkdir -p outputs/paper && tar xzf results/table1_metrics.tar.gz -C outputs/paper
python scripts/table1_from_raw.py outputs/paper/table1 --first-n 50
```

`configs/mpc/table1/` holds the episodes of the table (`episodes.jsonl.gz`: start state, goal and
planner seed) and the planner settings of each variant (`planner.yaml`), taken from the archived MPC
runs (`thijmennijdam/lwm-mpc-data`, not needed here). `scripts/mpc_table1.py` runs the first `--n`
episodes of every object. The paper reports 50 episodes per object; `--n 15` gives a quick subset,
and a later run with a larger `--n` extends it, skipping the episodes that exist. One archived
episode of object 35059 has no stored start state, so a full row has 799 episodes.

The three LWM-200M rows use the same episodes and seeds. The LWM-100M no-bias row differs from the
other two LWM-100M rows in 3 episodes and in some seeds, and the LWM-100M and LWM-200M rows share
about 760 episodes, with mostly different seeds. `paired_cmp.py` compares on the common episodes.

```bash
for v in LWM-200M LWM-200M_contact LWM-200M_contact_kinematics; do
  python scripts/mpc_table1.py --variant $v --checkpoint $LWM_CKPT_DIR/LWM-200M_d1024_L16/best.ckpt --n 50 --workers 6
done
for v in LWM-100M LWM-100M_contact LWM-100M_contact_kinematics; do   # the LWM-100M checkpoint stores no action bounds
  python scripts/mpc_table1.py --variant $v --checkpoint $LWM_CKPT_DIR/LWM-100M_d1024_L8/best.ckpt --n 50 --workers 7 --bounds config
done
python scripts/table1_from_raw.py outputs/table1 --first-n 50           # SR@0.4 / SR@0.8, seen / unseen
python scripts/table1_latex.py outputs/table1/table1.json                # LaTeX rows
python scripts/paired_cmp.py --ref outputs/table1 --cmp outputs/table1_seed2 --variants LWM-200M_contact_kinematics
python scripts/plot_ood_figs.py outputs/table1 LWM-200M figures/ood      # appendix failure-mode figures
```

Success is task progress, the fraction of the target joint displacement reached at the end of the
episode (`lwm.planning.success`); SR@0.4 and SR@0.8 count episodes above those fractions, and the
table reports the mean and standard deviation over the eight seen and eight unseen object
instances. A full row (800 episodes) takes about 10 GPU-hours for LWM-200M (6 parallel episodes on
a 24 GB GPU) and about 6 for LWM-100M. GPU rollouts are not bit-reproducible across machines: the
same checkpoint, code and seeds give per-cell differences of a few points, so compare rows with
`paired_cmp.py` on the same episodes rather than by single cells. The planner settings are those of
`configs/mpc/planner.yaml` plus the per-variant values of `configs/mpc/table1/planner.yaml`; every switch is documented there.

## 6. Tests

```bash
LWM_CKPT_DIR=data/lwm-checkpoints python tests/test_imports.py
```

The other `tests/test_*.py` run on CPU with the data subsets (`LWM_EXPERT_DATA_DIR`,
`LWM_INTERACTION_DATA_DIR` pointing at `lwm-expert-data-subset` / `lwm-interaction-data-subset`)
and `LWM_CKPT_DIR`; `tests/test_defaults_unchanged.py` checks that the experiment switches at their
defaults leave the paper models' code path unchanged.
