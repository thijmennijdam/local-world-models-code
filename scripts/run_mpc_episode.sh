#!/usr/bin/env bash
# Run one zero-shot MPC episode with a trained LWM in the PyBullet simulator.
#
# Usage: bash scripts/run_mpc_episode.sh <checkpoint> <object id> <session dir> [goal] [extra hydra overrides]
#   checkpoint   path to best.ckpt (with .hydra/config.yaml next to it)
#   object id    PartNet-Mobility id, e.g. 45526; must match the session
#   session dir  directory with task_config.yaml and states/state_0.pkl
#                (scripts/table1_start_states.py writes those of Table 1)
#   goal         goal fraction of the maximum opening (default 0.4)
#
# Required env:
#   PARTNET_DATASET_DIR  PartNet-Mobility assets (HF dataset lwm-data-generation-assets)
# Optional env:
#   MPC_CONFIG           config name in configs/mpc/ (default "paper" = the paper's planner
#                        settings; "planner" = the base defaults)
#   LWM_OUTPUT_DIR       output root (default ./outputs); runs go to $LWM_OUTPUT_DIR/mpc_results
#
# Extra arguments are Hydra overrides, e.g. bias=contact_kinematics execution.seed=0
# (bias: none | contact | kinematics | contact_kinematics; only with MPC_CONFIG=paper).
set -euo pipefail

CKPT=${1:?usage: run_mpc_episode.sh <checkpoint> <object id> <session dir> [goal] [overrides]}
OBJECT_ID=${2:?object id}
SESSION=${3:?session dir}
GOAL=${4:-0.4}
shift 3; [ $# -gt 0 ] && shift
PYTHON=${PYTHON:-python}

: "${PARTNET_DATASET_DIR:?set PARTNET_DATASET_DIR}"

exec "$PYTHON" -m lwm.planning.run_mpc --config-name "${MPC_CONFIG:-paper}" \
    "checkpoint_path=\"${CKPT}\"" \
    task.object_id="${OBJECT_ID}" \
    task.goal_percentage="${GOAL}" \
    "task.task_config_file=\"${SESSION}/task_config.yaml\"" \
    "task.init_state_file=\"${SESSION}/states/state_0.pkl\"" \
    "$@"
