#!/usr/bin/env bash
# Windowed / rollout evaluation of several LWM checkpoints (scaling figures).
#
# Usage: bash scripts/eval_scaling.sh [config name] [extra hydra overrides]
#   config names (configs/lwm/eval/): model_size_ablation_large (model scale),
#   pointnet_data_ablations (data scale), no_agent_state (affordance-only models)
#
# Required env:
#   LWM_CKPT_DIR              directory with one sub-directory per checkpoint (<run>/best.ckpt)
#   LWM_EXPERT_DATA_DIR       extracted expert episodes
#   LWM_INTERACTION_DATA_DIR  extracted interaction episodes
# Optional env:
#   LWM_OUTPUT_DIR            output root (default ./outputs); results go to $LWM_OUTPUT_DIR/lwm_eval
set -euo pipefail

CONFIG=${1:-model_size_ablation_large}
shift || true
PYTHON=${PYTHON:-python}

: "${LWM_CKPT_DIR:?set LWM_CKPT_DIR}"
: "${LWM_EXPERT_DATA_DIR:?set LWM_EXPERT_DATA_DIR}"
: "${LWM_INTERACTION_DATA_DIR:?set LWM_INTERACTION_DATA_DIR}"

exec "$PYTHON" -m lwm.world_model.analysis.eval_lwm_plots --config-name "eval/${CONFIG}" "$@"
