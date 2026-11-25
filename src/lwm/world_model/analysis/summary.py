"""Evaluation summary: printing helpers and JSON output."""

import json

import numpy as np
from termcolor import cprint

from lwm.world_model.analysis.constants import GEN_LABELS


def print_windowed_summary(windowed_by_category, window_steps=6):
    """Print windowed evaluation summary table."""
    for cat, result in windowed_by_category.items():
        if result is None:
            continue
        label = GEN_LABELS.get(cat, cat)
        n_win = result["num_windows"]
        cprint(f"  {label} ({n_win} windows):", "cyan")
        for k in range(window_steps):
            step = k + 1
            drift = result["avg_aff_drift_cm"][k]
            std = result["std_aff_drift_cm"][k]
            gt_disp = result["avg_gt_displacement_cm"][k]
            line = f"    Step {step}: {drift:.3f} +/- {std:.3f} cm  (GT delta: {gt_disp:.3f} cm)"
            if gt_disp > 0.1:
                line += f"  [relative: {drift / gt_disp:.2f}]"
            cprint(line, "white")


def save_evaluation_summary(
    output_dir,
    model_name=None,
    checkpoint_path=None,
    num_params=None,
    split_avg_results=None,
    gen_avg_results=None,
    windowed_by_source=None,
    phase_by_source=None,
    test_set_hash=None,
):
    """Save JSON summary with all numerical results."""
    summary = {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "num_params": int(num_params) if num_params is not None else None,
        "test_set_hash": test_set_hash,
        "full_episode_rollouts": {},
        "generalization": {},
        "windowed_eval": {},
    }

    if split_avg_results:
        for split_name, result in split_avg_results.items():
            if result is not None:
                summary["full_episode_rollouts"][split_name] = {
                    "num_episodes": result["num_episodes"],
                    "max_step": max(result["step_indices"]) if result["step_indices"] else 0,
                    "step_indices": result["step_indices"],
                    "avg_affordance_errors": result["avg_affordance_errors"],
                    "avg_agent_errors": result["avg_agent_errors"],
                }

    if gen_avg_results:
        for source, source_results in gen_avg_results.items():
            summary["generalization"][source] = {"full_episode_rollouts": {}}
            for cat, result in source_results.items():
                if result is not None:
                    summary["generalization"][source]["full_episode_rollouts"][GEN_LABELS[cat]] = {
                        "num_episodes": result["num_episodes"],
                        "max_step": max(result["step_indices"]) if result["step_indices"] else 0,
                        "step_indices": result["step_indices"],
                        "avg_affordance_errors": result["avg_affordance_errors"],
                        "avg_agent_errors": result["avg_agent_errors"],
                    }

    if windowed_by_source:
        for source, windowed_by_cat in windowed_by_source.items():
            summary["windowed_eval"][source] = {}
            for cat, result in windowed_by_cat.items():
                if result is not None:
                    summary["windowed_eval"][source][GEN_LABELS[cat]] = result

    if phase_by_source:
        for source, phase_by_cat in phase_by_source.items():
            if source not in summary["generalization"]:
                summary["generalization"][source] = {}
            summary["generalization"][source]["phase_eval"] = {}
            for cat, cat_phases in phase_by_cat.items():
                summary["generalization"][source]["phase_eval"][GEN_LABELS[cat]] = cat_phases

    # Use hash-specific filename so different eval configs don't overwrite each other
    if test_set_hash:
        filename = f"evaluation_summary_{test_set_hash[:8]}.json"
    else:
        filename = "evaluation_summary.json"
    summary_path = output_dir / filename
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    cprint(f"Saved evaluation summary: {summary_path}", "green")
