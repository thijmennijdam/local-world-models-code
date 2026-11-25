#!/usr/bin/env bash
# Train one Local World Model on a single node.
#
# Usage: bash scripts/train_lwm.sh <run name> [extra hydra overrides]
#   run names: see configs/lwm/runs.yaml (paper: LWM-200M_d1024_L16, LWM-100M_d1024_L8;
#   example: LWM-26M_E0_base)
#
# Required env:
#   LWM_EXPERT_DATA_DIR       extracted expert episodes (HF dataset lwm-expert-data)
#   LWM_INTERACTION_DATA_DIR  extracted interaction episodes (HF dataset lwm-interaction-data)
# Optional env:
#   CHECKPOINT_BASE_DIR       output root (default ./checkpoints)
#   NUM_GPUS                  overrides the run's GPU count (the paper models used 2)
#   BATCH_SIZE_OVERRIDE       per-GPU batch size (keep NUM_GPUS x batch constant to keep the recipe)
#   WANDB_MODE=online         enable Weights & Biases logging (WANDB_ENTITY, WANDB_PROJECT)
set -euo pipefail

RUN_NAME=${1:?usage: train_lwm.sh <run name> [hydra overrides]}
shift
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}

: "${LWM_EXPERT_DATA_DIR:?set LWM_EXPERT_DATA_DIR}"
: "${LWM_INTERACTION_DATA_DIR:?set LWM_INTERACTION_DATA_DIR}"

USER_NUM_GPUS=${NUM_GPUS:-}
eval "$("$PYTHON" "$REPO_ROOT/scripts/lwm_train_env.py" "$REPO_ROOT/configs/lwm/runs.yaml" "$RUN_NAME")"
NUM_GPUS=${USER_NUM_GPUS:-$NUM_GPUS}
export NUM_GPUS
# BATCH_SIZE is per GPU. To train on fewer GPUs with the same effective batch, set
# BATCH_SIZE_OVERRIDE (e.g. NUM_GPUS=1 BATCH_SIZE_OVERRIDE=1024 for a 2 x 512 run).
BATCH_SIZE=${BATCH_SIZE_OVERRIDE:-$BATCH_SIZE}
export BATCH_SIZE

if [ "$NUM_GPUS" -gt 1 ]; then
    MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
    exec "$PYTHON" -m torch.distributed.run \
        --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
        --master_addr=localhost --nnodes=1 --node_rank=0 \
        -m lwm.world_model.train "$@"
else
    exec "$PYTHON" -m lwm.world_model.train "$@"
fi
