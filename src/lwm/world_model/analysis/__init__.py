"""
Evaluation package for Local World Model (LWM).

Provides modular components for:
  - Checkpoint loading
  - Episode discovery and classification
  - Core evaluation (autoregressive rollouts, windowed N-step eval)
  - Plotting (trajectory, drift, generalization)
  - Summary and reporting
"""

from lwm.world_model.analysis.constants import (
    SPLIT_COLORS, SPLIT_LABELS,
    SEEN_OBJECT_IDS, OBJECT_CATEGORIES,
    GEN_COLORS, GEN_LABELS, GEN_CATEGORIES,
)
from lwm.world_model.analysis.checkpoint import load_checkpoint, load_episode_splits
from lwm.world_model.analysis.episode_discovery import (
    collect_episode_paths, classify_episode, extract_object_id,
    classify_generalization, find_test_episodes, load_eval_replay_buffer,
    compute_test_intersection_exclusions,
)

from lwm.world_model.analysis.core import (
    evaluate_episode,
    evaluate_episodes_cached, aggregate_avg_error_from_cache,
    evaluate_episodes_for_avg_error,
    evaluate_windowed, aggregate_windowed_results,
)
from lwm.world_model.analysis.plots import (
    create_combined_plot,
    create_combined_avg_error_plot,
    create_generalization_avg_error_plot,
    create_windowed_eval_plot,
    create_one_step_summary_plot,
)
from lwm.world_model.analysis.summary import save_evaluation_summary, print_windowed_summary
from lwm.world_model.analysis.comparison_utils import (
    create_all_cross_model_comparisons,
    load_cached_model_results,
)
