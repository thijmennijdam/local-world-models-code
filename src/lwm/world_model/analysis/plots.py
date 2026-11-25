"""Plotting functions for LWM evaluation."""

import pathlib
import csv
import io

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from termcolor import cprint

from lwm.world_model.analysis.constants import SPLIT_COLORS, SPLIT_LABELS, GEN_COLORS, GEN_LABELS, GEN_CATEGORIES, COMBINED_STYLES

# Standard figure size for single-panel plots (used across all eval plots for consistency)
SINGLE_FIG_SIZE = (5.5, 3.5)
FIG_HEIGHT = SINGLE_FIG_SIZE[1]

# Seaborn theme (paper-quality)
sns.set_theme(
    style="whitegrid",
    rc={
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "grid.alpha": 0.2,
    },
)


def _save_fig(fig, output_path):
    """Save figure as both PDF and PNG, then close."""
    output_path = pathlib.Path(output_path)
    fig.savefig(output_path)
    # Also save a PNG copy next to the PDF
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    cprint(f"Saved: {output_path} + .png", "green")

def create_combined_plot(eval_result, data_type, episode_idx, predict_agent_state=True):
    """
    Create a combined figure with affordance and agent state trajectories + error plots.

    Returns:
        PIL Image
    """
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("white")

    views = [(20, 45), (20, 135)]

    # Row 1: Affordance trajectories (2 views)
    pred_aff = eval_result["pred_affordance"]
    gt_aff = eval_result["gt_affordance"]

    for view_idx, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 3, view_idx + 1, projection="3d")
        ax.set_facecolor("white")

        T = len(gt_aff)
        alpha_start = 0.9
        alpha_end = 0.2

        for t in range(T - 1):
            alpha = alpha_start - (alpha_start - alpha_end) * (t / max(T - 1, 1))
            ax.plot(gt_aff[t:t+2, 0], gt_aff[t:t+2, 1], gt_aff[t:t+2, 2],
                   c="#228833", linewidth=2.0, alpha=alpha)
        ax.plot([], [], c="#228833", linewidth=2.0, label="GT")

        for t in range(len(pred_aff) - 1):
            alpha = alpha_start - (alpha_start - alpha_end) * (t / max(len(pred_aff) - 1, 1))
            ax.plot(pred_aff[t:t+2, 0], pred_aff[t:t+2, 1], pred_aff[t:t+2, 2],
                   c="#EE6677", linewidth=2.0, linestyle="--", alpha=alpha)
        ax.plot([], [], c="#EE6677", linewidth=2.0, linestyle="--", label="Pred")

        all_pts = np.vstack([gt_aff, pred_aff])
        pad = 0.05
        ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
        ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper left", frameon=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Affordance — View {view_idx + 1}")

    # Row 1: Affordance position error plot (rightmost)
    ax_aff_error = fig.add_subplot(2, 3, 3)
    step_indices = list(range(len(eval_result["per_step_affordance_errors"])))
    aff_errors_cm = eval_result["per_step_affordance_errors"]  # already L2 cm
    df_aff = pd.DataFrame({"step": step_indices, "error_cm": aff_errors_cm})
    sns.lineplot(data=df_aff, x="step", y="error_cm", color="#FF9800", linewidth=2,
                 marker="o", markersize=3, ax=ax_aff_error)
    ax_aff_error.set_xlabel('Prediction Step')
    ax_aff_error.set_ylabel('Position Error (cm)')
    ax_aff_error.set_title('Open-Loop Affordance Prediction Error')
    ax_aff_error.set_ylim(bottom=0)

    # Row 2: Agent state trajectories (2 views) - first 3 dims (position)
    pred_agent = eval_result["pred_agent_state"][:, :3]
    gt_agent = eval_result["gt_agent_state"][:, :3]

    for view_idx, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 3, view_idx + 4, projection="3d")
        ax.set_facecolor("white")

        T = len(gt_agent)
        alpha_start = 0.9
        alpha_end = 0.2

        for t in range(T - 1):
            alpha = alpha_start - (alpha_start - alpha_end) * (t / max(T - 1, 1))
            ax.plot(gt_agent[t:t+2, 0], gt_agent[t:t+2, 1], gt_agent[t:t+2, 2],
                   c="#2196F3", linewidth=2.0, alpha=alpha)
        ax.plot([], [], c="#2196F3", linewidth=2.0, label="GT")

        if predict_agent_state:
            for t in range(len(pred_agent) - 1):
                alpha = alpha_start - (alpha_start - alpha_end) * (t / max(len(pred_agent) - 1, 1))
                ax.plot(pred_agent[t:t+2, 0], pred_agent[t:t+2, 1], pred_agent[t:t+2, 2],
                       c="#FF9800", linewidth=2.0, linestyle="--", alpha=alpha)
            ax.plot([], [], c="#FF9800", linewidth=2.0, linestyle="--", label="Pred")

        all_pts = np.vstack([gt_agent, pred_agent]) if predict_agent_state else gt_agent
        pad = 0.05
        ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
        ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper left", frameon=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Agent Position — View {view_idx + 1}")

    # Row 2: Agent EE position error plot (rightmost)
    ax_agent_error = fig.add_subplot(2, 3, 6)
    if predict_agent_state and eval_result["per_step_agent_errors"]:
        agent_errors_cm = eval_result["per_step_agent_errors"]  # already L2 cm
        df_agent = pd.DataFrame({"step": step_indices, "error_cm": agent_errors_cm})
        sns.lineplot(data=df_agent, x="step", y="error_cm", color="#2196F3", linewidth=2,
                     marker="o", markersize=3, ax=ax_agent_error)
        ax_agent_error.set_title('Open-Loop Agent State Prediction Error')
    else:
        ax_agent_error.text(0.5, 0.5, 'Agent state prediction\nnot enabled',
                            ha='center', va='center', transform=ax_agent_error.transAxes)
        ax_agent_error.set_title('Open-Loop Agent State Prediction Error')
    ax_agent_error.set_xlabel('Prediction Step')
    ax_agent_error.set_ylabel('Position Error (cm)')
    ax_agent_error.set_ylim(bottom=0)

    fig.suptitle(f"LWM Evaluation — {data_type.capitalize()} Episode {episode_idx}")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    plt.close(fig)
    buf.close()

    return pil_image


def create_combined_avg_error_plot(split_results, output_path, title_suffix="Full Horizon"):
    """
    Create a two-panel position error plot:
      Left: absolute drift in cm
      Right: relative drift (error / GT displacement from start)

    Args:
        split_results: dict mapping split_name -> result dict from evaluate_episodes_for_avg_error
        output_path: Path to save the plot
        title_suffix: e.g. "Full Horizon" or "Short Horizon (6 steps)"
    """
    # Check if any split has relative errors
    has_relative = any(
        split_results.get(s) is not None and "avg_relative_errors" in split_results[s]
        for s in ["expert", "new_config"]
    )

    if has_relative:
        fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(10, 3.5))
    else:
        fig, ax_abs = plt.subplots(figsize=(5.5, 3.5))

    # --- Left: absolute error in cm ---
    rows = []
    for split_name in ["expert", "new_config"]:
        if split_name not in split_results or split_results[split_name] is None:
            continue
        result = split_results[split_name]
        for i, s in enumerate(result["step_indices"]):
            rows.append({"step": s + 1, "error_cm": result["avg_affordance_errors"][i],
                        "split": SPLIT_LABELS[split_name]})
    if rows:
        df = pd.DataFrame(rows)
        palette = {SPLIT_LABELS[k]: v for k, v in SPLIT_COLORS.items() if k in split_results}
        sns.lineplot(data=df, x="step", y="error_cm", hue="split", palette=palette,
                     linewidth=1.5, ax=ax_abs)
    ax_abs.set_xlabel('Prediction Step')
    ax_abs.set_ylabel('Affordance Error (cm)')
    ax_abs.set_ylim(bottom=0)
    ax_abs.set_title('Open-Loop Affordance Prediction Error')
    ax_abs.legend(frameon=False)

    # Set integer x-ticks for short horizons
    all_plotted_steps = set()
    for split_name in ["expert", "new_config"]:
        if split_name in split_results and split_results[split_name] is not None:
            all_plotted_steps.update(s + 1 for s in split_results[split_name]["step_indices"])
    if all_plotted_steps and len(all_plotted_steps) <= 30:
        ax_abs.set_xticks(sorted(all_plotted_steps))

    # --- Right: relative error (error / GT displacement) ---
    if has_relative:
        rows_rel = []
        for split_name in ["expert", "new_config"]:
            if split_name not in split_results or split_results[split_name] is None:
                continue
            result = split_results[split_name]
            for i, s in enumerate(result["step_indices"]):
                if result["avg_relative_errors"][i] is not None:
                    rows_rel.append({"step": s + 1,
                                     "relative_pct": result["avg_relative_errors"][i] * 100.0,
                                     "split": SPLIT_LABELS[split_name]})
        if rows_rel:
            df_rel = pd.DataFrame(rows_rel)
            palette_rel = {SPLIT_LABELS[k]: v for k, v in SPLIT_COLORS.items() if k in split_results}
            sns.lineplot(data=df_rel, x="step", y="relative_pct", hue="split",
                         palette=palette_rel, linewidth=1.5, ax=ax_rel)
        ax_rel.set_xlabel('Prediction Step')
        ax_rel.set_ylabel('Error / GT Displacement (%)')
        ax_rel.set_ylim(bottom=0)
        ax_rel.set_title(f'Relative Affordance Prediction Error')
        ax_rel.legend(frameon=False)

        if all_plotted_steps and len(all_plotted_steps) <= 30:
            ax_rel.set_xticks(sorted(all_plotted_steps))

    _save_fig(fig, output_path)

    # --- Save CSV with the plotted data ---
    csv_path = pathlib.Path(output_path).with_suffix('.csv')
    all_series = {}  # name -> {step: (error_cm, gt_disp_cm, relative_pct)}
    for split_name in ["expert", "new_config"]:
        if split_name not in split_results or split_results[split_name] is None:
            continue
        result = split_results[split_name]
        errors = result["avg_affordance_errors"]
        gt_disps = result.get("avg_gt_displacement_cm", [None] * len(errors))
        rel_errs = result.get("avg_relative_errors", [None] * len(errors))
        label = SPLIT_LABELS[split_name]
        all_series[f"{label}_error_cm"] = {s + 1: errors[i] for i, s in enumerate(result["step_indices"])}
        all_series[f"{label}_gt_disp_cm"] = {
            s + 1: gt_disps[i] for i, s in enumerate(result["step_indices"]) if gt_disps[i] is not None
        }
        all_series[f"{label}_relative_pct"] = {
            s + 1: rel_errs[i] * 100.0 for i, s in enumerate(result["step_indices"]) if rel_errs[i] is not None
        }
    all_steps_sorted = sorted(set(s for series in all_series.values() for s in series))
    col_names = list(all_series.keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + col_names)
        for s in all_steps_sorted:
            row = [s] + [all_series[c].get(s, "") for c in col_names]
            writer.writerow(row)
    cprint(f"Saved: {csv_path}", "green")


def create_windowed_eval_plot(windowed_by_category, output_path, window_steps=6):
    """
    Create windowed N-step evaluation plot: affordance drift vs rollout step.

    Args:
        windowed_by_category: dict mapping category label -> aggregated windowed result
        output_path: Path to save the plot
        window_steps: Number of rollout steps
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    rows = []
    for cat in GEN_CATEGORIES:
        if cat not in windowed_by_category or windowed_by_category[cat] is None:
            continue
        result = windowed_by_category[cat]
        for i, s in enumerate(result["step_indices"]):
            rows.append({"step": s, "drift_cm": result["avg_aff_drift_cm"][i],
                         "category": GEN_LABELS[cat]})
    if rows:
        df = pd.DataFrame(rows)
        palette = {GEN_LABELS[c]: GEN_COLORS[c] for c in GEN_CATEGORIES
                   if c in windowed_by_category and windowed_by_category[c] is not None}
        sns.lineplot(data=df, x="step", y="drift_cm", hue="category", palette=palette,
                     linewidth=1.5, marker="o", markersize=4, ax=ax)

    # Annotate avg GT displacement at final step
    for cat in GEN_CATEGORIES:
        if cat not in windowed_by_category or windowed_by_category[cat] is None:
            continue
        result = windowed_by_category[cat]
        gt_disp_final = result["avg_gt_displacement_cm"][-1]
        ax.axhline(y=gt_disp_final, color=GEN_COLORS[cat], linestyle=':', alpha=0.4, linewidth=0.8)
        ax.text(0.5, gt_disp_final, f'GT \u0394={gt_disp_final:.2f}cm ({GEN_LABELS[cat]})',
                fontsize=7, color=GEN_COLORS[cat], alpha=0.7, va='bottom')

    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Affordance Error (cm)')
    ax.set_ylim(bottom=0)
    ax.set_title(f'Windowed {window_steps}-Step Affordance Prediction Error')
    ax.set_xticks(list(range(1, window_steps + 1)))
    ax.legend(frameon=False)

    _save_fig(fig, output_path)

    # CSV
    csv_path = pathlib.Path(output_path).with_suffix('.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["step"]
        cats_present = [c for c in GEN_CATEGORIES if c in windowed_by_category and windowed_by_category[c] is not None]
        for cat in cats_present:
            label = GEN_LABELS[cat]
            header.extend([f"{label}_drift_cm", f"{label}_std_cm", f"{label}_gt_disp_cm"])
        writer.writerow(header)
        for k in range(window_steps):
            row = [k + 1]
            for cat in cats_present:
                r = windowed_by_category[cat]
                row.extend([f"{r['avg_aff_drift_cm'][k]:.4f}",
                            f"{r['std_aff_drift_cm'][k]:.4f}",
                            f"{r['avg_gt_displacement_cm'][k]:.4f}"])
            writer.writerow(row)
    cprint(f"Saved: {csv_path}", "green")


def create_one_step_summary_plot(windowed_by_category, output_path):
    """
    Bar chart of 1-step affordance drift (step 1 from windowed eval), seen vs unseen.
    """
    cats_present = [c for c in GEN_CATEGORIES
                    if c in windowed_by_category and windowed_by_category[c] is not None]
    if not cats_present:
        return

    df = pd.DataFrame([
        {"category": GEN_LABELS[c], "error_cm": windowed_by_category[c]["avg_aff_drift_cm"][0],
         "n_windows": windowed_by_category[c]["num_windows"]}
        for c in cats_present
    ])
    palette = {GEN_LABELS[c]: GEN_COLORS[c] for c in cats_present}

    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
    sns.barplot(data=df, x="category", y="error_cm", palette=palette, edgecolor="black",
                linewidth=0.5, width=0.5, alpha=0.85, ax=ax)

    for bar, (_, row) in zip(ax.patches, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{row["error_cm"]:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(bar.get_x() + bar.get_width() / 2, -0.08, f'n={int(row["n_windows"])}',
                ha='center', va='top', fontsize=7, color='gray',
                transform=ax.get_xaxis_transform())

    ax.set_ylabel('1-Step Error (cm)')
    ax.set_title('Single-Step Prediction Error')
    ax.set_ylim(bottom=0)

    _save_fig(fig, output_path)


# =============================================================================
# Generalization Category Plots
# =============================================================================

def create_generalization_avg_error_plot(gen_results, output_path, title_suffix="Full Horizon"):
    """
    Create position drift plot broken down by generalization category.

    Args:
        gen_results: dict mapping gen_category -> result dict from evaluate_episodes_for_avg_error
        output_path: Path to save the plot
        title_suffix: e.g. "Full Horizon" or "Short Horizon (6 steps)"
    """
    has_relative = any(
        gen_results.get(c) is not None and "avg_relative_errors" in gen_results[c]
        for c in GEN_CATEGORIES
    )

    if has_relative:
        fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(10, 3.5))
    else:
        fig, ax_abs = plt.subplots(figsize=(5.5, 3.5))

    rows_abs = []
    for cat in GEN_CATEGORIES:
        if cat not in gen_results or gen_results[cat] is None:
            continue
        result = gen_results[cat]
        for i, s in enumerate(result["step_indices"]):
            rows_abs.append({"step": s + 1, "error_cm": result["avg_affordance_errors"][i],
                            "category": GEN_LABELS[cat]})
    if rows_abs:
        df = pd.DataFrame(rows_abs)
        palette = {GEN_LABELS[c]: GEN_COLORS[c] for c in GEN_CATEGORIES
                   if c in gen_results and gen_results[c] is not None}
        sns.lineplot(data=df, x="step", y="error_cm", hue="category", palette=palette,
                     linewidth=1.5, ax=ax_abs)
    ax_abs.set_xlabel('Prediction Step')
    ax_abs.set_ylabel('Affordance Error (cm)')
    ax_abs.set_ylim(bottom=0)
    ax_abs.set_title('Open-Loop Affordance Prediction Error')
    ax_abs.legend(frameon=False)

    all_plotted_steps = set()
    for cat in GEN_CATEGORIES:
        if cat in gen_results and gen_results[cat] is not None:
            all_plotted_steps.update(s + 1 for s in gen_results[cat]["step_indices"])
    if all_plotted_steps and len(all_plotted_steps) <= 30:
        ax_abs.set_xticks(sorted(all_plotted_steps))

    if has_relative:
        rows_rel = []
        for cat in GEN_CATEGORIES:
            if cat not in gen_results or gen_results[cat] is None:
                continue
            result = gen_results[cat]
            for i, s in enumerate(result["step_indices"]):
                if result["avg_relative_errors"][i] is not None:
                    rows_rel.append({"step": s + 1,
                                     "relative_pct": result["avg_relative_errors"][i] * 100.0,
                                     "category": GEN_LABELS[cat]})
        if rows_rel:
            df_rel = pd.DataFrame(rows_rel)
            palette_rel = {GEN_LABELS[c]: GEN_COLORS[c] for c in GEN_CATEGORIES
                          if c in gen_results and gen_results[c] is not None}
            sns.lineplot(data=df_rel, x="step", y="relative_pct", hue="category",
                        palette=palette_rel, linewidth=1.5, ax=ax_rel)
        ax_rel.set_xlabel('Prediction Step')
        ax_rel.set_ylabel('Error / GT Displacement (%)')
        ax_rel.set_ylim(bottom=0)
        ax_rel.set_title(f'Relative Affordance Prediction Error')
        ax_rel.legend(frameon=False)
        if all_plotted_steps and len(all_plotted_steps) <= 30:
            ax_rel.set_xticks(sorted(all_plotted_steps))

    _save_fig(fig, output_path)

    # CSV
    csv_path = pathlib.Path(output_path).with_suffix('.csv')
    all_series = {}
    for cat in GEN_CATEGORIES:
        if cat not in gen_results or gen_results[cat] is None:
            continue
        result = gen_results[cat]
        errors = result["avg_affordance_errors"]
        gt_disps = result.get("avg_gt_displacement_cm", [None] * len(errors))
        rel_errs = result.get("avg_relative_errors", [None] * len(errors))
        label = GEN_LABELS[cat]
        all_series[f"{label}_error_cm"] = {s + 1: errors[i] for i, s in enumerate(result["step_indices"])}
        all_series[f"{label}_gt_disp_cm"] = {
            s + 1: gt_disps[i] for i, s in enumerate(result["step_indices"]) if gt_disps[i] is not None
        }
        all_series[f"{label}_relative_pct"] = {
            s + 1: rel_errs[i] * 100.0 for i, s in enumerate(result["step_indices"]) if rel_errs[i] is not None
        }
    all_steps_sorted = sorted(set(s for series in all_series.values() for s in series))
    col_names = list(all_series.keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + col_names)
        for s in all_steps_sorted:
            row = [s] + [all_series[c].get(s, "") for c in col_names]
            writer.writerow(row)
    cprint(f"Saved: {csv_path}", "green")


# =============================================================================
# Combined 4-Way Plots (Expert + Interaction × Seen + Unseen)
# =============================================================================

def create_combined_4way_drift_plot(all_gen_avg_results, output_path, title_suffix="Full Horizon"):
    """
    Combined drift plot with 4 lines: expert/seen, expert/unseen,
    interaction/seen, interaction/unseen.

    Args:
        all_gen_avg_results: {source: {gen_cat: result_dict}}
        output_path: Path to save
        title_suffix: title annotation
    """
    has_relative = False
    for source in ["expert", "new_config"]:
        for cat in GEN_CATEGORIES:
            r = all_gen_avg_results.get(source, {}).get(cat)
            if r is not None and any(v is not None for v in r.get("avg_relative_errors", [])):
                has_relative = True
                break

    if has_relative:
        fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(10, 3.5))
    else:
        fig, ax_abs = plt.subplots(figsize=(5.5, 3.5))

    rows_abs = []
    for source in ["expert", "new_config"]:
        gen_results = all_gen_avg_results.get(source, {})
        for cat in GEN_CATEGORIES:
            result = gen_results.get(cat)
            if result is None:
                continue
            style = COMBINED_STYLES.get((source, cat))
            if style is None:
                continue
            linestyle_label = "Seen" if cat == "seen_obj" else "Unseen"
            for i, s in enumerate(result["step_indices"]):
                rows_abs.append({"step": s + 1, "error_cm": result["avg_affordance_errors"][i],
                                "series": style["label"], "linestyle": linestyle_label})
    all_plotted_steps = {r["step"] for r in rows_abs} if rows_abs else set()

    if rows_abs:
        df = pd.DataFrame(rows_abs)
        palette = {st["label"]: st["color"] for (src, c), st in COMBINED_STYLES.items()
                   if all_gen_avg_results.get(src, {}).get(c) is not None}
        sns.lineplot(data=df, x="step", y="error_cm", hue="series", style="linestyle",
                     palette=palette, dashes={"Seen": "", "Unseen": (2, 2)}, linewidth=1.5, ax=ax_abs)
    ax_abs.set_xlabel('Prediction Step')
    ax_abs.set_ylabel('Affordance Error (cm)')
    ax_abs.set_ylim(bottom=0)
    ax_abs.set_title('Open-Loop Affordance Prediction Error')
    ax_abs.legend(frameon=False)
    if all_plotted_steps and len(all_plotted_steps) <= 30:
        ax_abs.set_xticks(sorted(all_plotted_steps))

    if has_relative:
        rows_rel = []
        for source in ["expert", "new_config"]:
            gen_results = all_gen_avg_results.get(source, {})
            for cat in GEN_CATEGORIES:
                result = gen_results.get(cat)
                if result is None:
                    continue
                style = COMBINED_STYLES.get((source, cat))
                if style is None:
                    continue
                linestyle_label = "Seen" if cat == "seen_obj" else "Unseen"
                for i, s in enumerate(result["step_indices"]):
                    if result["avg_relative_errors"][i] is not None:
                        rows_rel.append({"step": s + 1,
                                         "relative_pct": result["avg_relative_errors"][i] * 100.0,
                                         "series": style["label"], "linestyle": linestyle_label})
        if rows_rel:
            df_rel = pd.DataFrame(rows_rel)
            palette_rel = {st["label"]: st["color"] for (src, c), st in COMBINED_STYLES.items()
                          if all_gen_avg_results.get(src, {}).get(c) is not None}
            sns.lineplot(data=df_rel, x="step", y="relative_pct", hue="series", style="linestyle",
                         palette=palette_rel, dashes={"Seen": "", "Unseen": (2, 2)},
                         linewidth=1.5, ax=ax_rel)
        ax_rel.set_xlabel('Prediction Step')
        ax_rel.set_ylabel('Error / GT Displacement (%)')
        ax_rel.set_ylim(bottom=0)
        ax_rel.set_title('Relative Affordance Prediction Error')
        ax_rel.legend(frameon=False)
        if all_plotted_steps and len(all_plotted_steps) <= 30:
            ax_rel.set_xticks(sorted(all_plotted_steps))

    _save_fig(fig, output_path)

    # CSV
    csv_path = pathlib.Path(output_path).with_suffix('.csv')
    combined_series = {}
    for source in ["expert", "new_config"]:
        gen_results = all_gen_avg_results.get(source, {})
        for cat in GEN_CATEGORIES:
            result = gen_results.get(cat)
            if result is None:
                continue
            style = COMBINED_STYLES[(source, cat)]
            label = style["label"].replace(" / ", "_").replace(" ", "_")
            errors = result["avg_affordance_errors"]
            gt_disps = result.get("avg_gt_displacement_cm", [None] * len(errors))
            rel_errs = result.get("avg_relative_errors", [None] * len(errors))
            combined_series[f"{label}_error_cm"] = {
                s + 1: errors[i] for i, s in enumerate(result["step_indices"])
            }
            combined_series[f"{label}_gt_disp_cm"] = {
                s + 1: gt_disps[i] for i, s in enumerate(result["step_indices"]) if gt_disps[i] is not None
            }
            combined_series[f"{label}_relative_pct"] = {
                s + 1: rel_errs[i] * 100.0 for i, s in enumerate(result["step_indices"]) if rel_errs[i] is not None
            }
    if combined_series:
        combined_steps = sorted(set(s for series in combined_series.values() for s in series))
        col_names = list(combined_series.keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + col_names)
            for s in combined_steps:
                row = [s] + [combined_series[c].get(s, "") for c in col_names]
                writer.writerow(row)
        cprint(f"Saved: {csv_path}", "green")


def create_combined_4way_windowed_plot(all_windowed_by_source, output_path, window_steps=6):
    """
    Combined windowed N-step plot with 4 lines: expert/seen, expert/unseen,
    interaction/seen, interaction/unseen.

    Args:
        all_windowed_by_source: {source: {gen_cat: aggregated_result}}
        output_path: Path to save
        window_steps: Number of rollout steps
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    rows = []
    for source in ["expert", "new_config"]:
        windowed_by_cat = all_windowed_by_source.get(source, {})
        for cat in GEN_CATEGORIES:
            result = windowed_by_cat.get(cat)
            if result is None:
                continue
            style = COMBINED_STYLES.get((source, cat))
            if style is None:
                continue
            linestyle_label = "Seen" if cat == "seen_obj" else "Unseen"
            for i, s in enumerate(result["step_indices"]):
                rows.append({"step": s, "drift_cm": result["avg_aff_drift_cm"][i],
                            "series": style["label"], "linestyle": linestyle_label})
    if rows:
        df = pd.DataFrame(rows)
        palette = {st["label"]: st["color"] for (src, c), st in COMBINED_STYLES.items()
                   if all_windowed_by_source.get(src, {}).get(c) is not None}
        sns.lineplot(data=df, x="step", y="drift_cm", hue="series", style="linestyle",
                     palette=palette, dashes={"Seen": "", "Unseen": (2, 2)},
                     linewidth=1.5, marker="o", markersize=4, ax=ax)
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Affordance Error (cm)')
    ax.set_ylim(bottom=0)
    ax.set_title(f'Windowed {window_steps}-Step Affordance Prediction Error')
    ax.set_xticks(list(range(1, window_steps + 1)))
    ax.legend(frameon=False)

    _save_fig(fig, output_path)

    # CSV
    csv_path = pathlib.Path(output_path).with_suffix('.csv')
    lines_data = []
    header = ["step"]
    for source in ["expert", "new_config"]:
        windowed_by_cat = all_windowed_by_source.get(source, {})
        for cat in GEN_CATEGORIES:
            result = windowed_by_cat.get(cat)
            if result is None:
                continue
            style = COMBINED_STYLES[(source, cat)]
            label = style["label"].replace(" / ", "_").replace(" ", "_")
            header.extend([f"{label}_drift_cm", f"{label}_std_cm", f"{label}_gt_disp_cm"])
            lines_data.append(result)
    if lines_data:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for k in range(window_steps):
                row = [k + 1]
                for result in lines_data:
                    row.extend([
                        f"{result['avg_aff_drift_cm'][k]:.4f}",
                        f"{result['std_aff_drift_cm'][k]:.4f}",
                        f"{result['avg_gt_displacement_cm'][k]:.4f}",
                    ])
                writer.writerow(row)
        cprint(f"Saved: {csv_path}", "green")

