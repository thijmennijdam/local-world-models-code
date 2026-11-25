"""
Functions for creating comparison plots across multiple LWM models.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from termcolor import cprint


def create_comparison_error_plot(
    models_results,
    output_path,
    data_type="expert",
    plot_type="avg_error",
    title_suffix="",
):
    """
    Create comparison plot showing error vs step for multiple models.

    Args:
        models_results: List of dicts, each containing:
            - model_name: str
            - step_indices: list
            - avg_total_errors: list (or cum_total_errors)
            - avg_agent_errors: list (optional)
            - avg_affordance_errors: list
        output_path: Path to save the plot
        data_type: "expert" or "rollout"
        plot_type: "avg_error" or "near_affordance"
        title_suffix: Additional text for title
    """
    fig, (ax_aff, ax_agent) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    fig.patch.set_facecolor("white")

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    # Plot affordance errors
    for idx, model_result in enumerate(models_results):
        model_name = model_result["model_name"]
        step_indices = model_result["step_indices"]
        affordance_errors = model_result["avg_affordance_errors"]

        # Compute cumulative errors
        cum_affordance_errors = np.cumsum(affordance_errors)

        color = colors[idx % len(colors)]
        ax_aff.plot(step_indices, cum_affordance_errors,
                   color=color, linewidth=2, marker='o', markersize=3,
                   label=model_name, alpha=0.8)

    ax_aff.set_xlabel('Rollout Step', fontsize=12)
    ax_aff.set_ylabel('Cumulative MSE', fontsize=12)
    ax_aff.set_title(f'Affordance Cumulative Error ({data_type.capitalize()}){title_suffix}',
                     fontsize=13, fontweight='bold')
    ax_aff.grid(True, alpha=0.3)
    ax_aff.legend(loc='best', fontsize=10)

    # Plot agent state errors (if available)
    has_agent_errors = any(model_result.get("avg_agent_errors") is not None
                           for model_result in models_results)

    if has_agent_errors:
        for idx, model_result in enumerate(models_results):
            model_name = model_result["model_name"]
            step_indices = model_result["step_indices"]
            agent_errors = model_result.get("avg_agent_errors")

            if agent_errors is not None:
                # Compute cumulative errors
                cum_agent_errors = np.cumsum(agent_errors)

                color = colors[idx % len(colors)]
                ax_agent.plot(step_indices, cum_agent_errors,
                            color=color, linewidth=2, marker='o', markersize=3,
                            label=model_name, alpha=0.8)

        ax_agent.set_xlabel('Rollout Step', fontsize=12)
        ax_agent.set_ylabel('Cumulative MSE', fontsize=12)
        ax_agent.set_title(f'Agent State Cumulative Error ({data_type.capitalize()}){title_suffix}',
                          fontsize=13, fontweight='bold')
        ax_agent.grid(True, alpha=0.3)
        ax_agent.legend(loc='best', fontsize=10)
    else:
        ax_agent.text(0.5, 0.5, 'Agent state prediction\nnot enabled',
                     ha='center', va='center', fontsize=14, transform=ax_agent.transAxes)
        ax_agent.set_title(f'Agent State Cumulative Error ({data_type.capitalize()}){title_suffix}',
                          fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    cprint(f"Saved comparison plot: {output_path}", "green")


def create_all_comparison_plots(
    all_model_results,
    output_dir,
):
    """
    Create all comparison plots for multiple models.

    Args:
        all_model_results: List of dicts, each containing:
            - model_name: str
            - expert_avg_results: dict (from evaluate_episodes_for_avg_error)
            - rollout_avg_results: dict
            - expert_near_aff_results: dict (from evaluate_near_affordance_rollouts)
            - rollout_near_aff_results: dict
        output_dir: Path to comparison output directory
    """
    # Average error vs step - Expert
    expert_models = []
    for model_result in all_model_results:
        if model_result.get("expert_avg_results") is not None:
            expert_models.append({
                "model_name": model_result["model_name"],
                "step_indices": model_result["expert_avg_results"]["step_indices"],
                "avg_total_errors": model_result["expert_avg_results"]["avg_total_errors"],
                "avg_agent_errors": model_result["expert_avg_results"]["avg_agent_errors"],
                "avg_affordance_errors": model_result["expert_avg_results"]["avg_affordance_errors"],
            })

    if len(expert_models) > 0:
        create_comparison_error_plot(
            expert_models,
            output_path=output_dir / "comparison_avg_error_expert.png",
            data_type="expert",
            plot_type="avg_error",
        )

    # Average error vs step - Rollout
    rollout_models = []
    for model_result in all_model_results:
        if model_result.get("rollout_avg_results") is not None:
            rollout_models.append({
                "model_name": model_result["model_name"],
                "step_indices": model_result["rollout_avg_results"]["step_indices"],
                "avg_total_errors": model_result["rollout_avg_results"]["avg_total_errors"],
                "avg_agent_errors": model_result["rollout_avg_results"]["avg_agent_errors"],
                "avg_affordance_errors": model_result["rollout_avg_results"]["avg_affordance_errors"],
            })

    if len(rollout_models) > 0:
        create_comparison_error_plot(
            rollout_models,
            output_path=output_dir / "comparison_avg_error_rollout.png",
            data_type="rollout",
            plot_type="avg_error",
        )

    # Near-affordance - Expert
    expert_near_models = []
    for model_result in all_model_results:
        if model_result.get("expert_near_aff_results") is not None:
            expert_near_models.append({
                "model_name": model_result["model_name"],
                "step_indices": model_result["expert_near_aff_results"]["step_indices"],
                "avg_total_errors": model_result["expert_near_aff_results"]["avg_total_errors"],
                "avg_agent_errors": model_result["expert_near_aff_results"]["avg_agent_errors"],
                "avg_affordance_errors": model_result["expert_near_aff_results"]["avg_affordance_errors"],
            })

    if len(expert_near_models) > 0:
        distance_threshold = all_model_results[0]["expert_near_aff_results"]["distance_threshold"]
        rollout_length = all_model_results[0]["expert_near_aff_results"]["rollout_length"]
        title_suffix = f"\nEE within {distance_threshold*100:.0f}cm, {rollout_length}-step rollouts"

        create_comparison_error_plot(
            expert_near_models,
            output_path=output_dir / "comparison_near_affordance_expert.png",
            data_type="expert",
            plot_type="near_affordance",
            title_suffix=title_suffix,
        )

    # Near-affordance - Rollout
    rollout_near_models = []
    for model_result in all_model_results:
        if model_result.get("rollout_near_aff_results") is not None:
            rollout_near_models.append({
                "model_name": model_result["model_name"],
                "step_indices": model_result["rollout_near_aff_results"]["step_indices"],
                "avg_total_errors": model_result["rollout_near_aff_results"]["avg_total_errors"],
                "avg_agent_errors": model_result["rollout_near_aff_results"]["avg_agent_errors"],
                "avg_affordance_errors": model_result["rollout_near_aff_results"]["avg_affordance_errors"],
            })

    if len(rollout_near_models) > 0:
        distance_threshold = all_model_results[0]["rollout_near_aff_results"]["distance_threshold"]
        rollout_length = all_model_results[0]["rollout_near_aff_results"]["rollout_length"]
        title_suffix = f"\nEE within {distance_threshold*100:.0f}cm, {rollout_length}-step rollouts"

        create_comparison_error_plot(
            rollout_near_models,
            output_path=output_dir / "comparison_near_affordance_rollout.png",
            data_type="rollout",
            plot_type="near_affordance",
            title_suffix=title_suffix,
        )

    cprint(f"\nAll comparison plots saved to: {output_dir}", "green")
