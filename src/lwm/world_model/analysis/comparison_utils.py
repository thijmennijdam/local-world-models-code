"""Cross-model comparison plots and cached-result loading.

Given results from multiple models, create plots grouped by generalization
category (seen / unseen) with each model as a separate line.
"""

import csv
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from termcolor import cprint

from lwm.world_model.analysis.constants import GEN_CATEGORIES, GEN_LABELS, GEN_COLORS, PHASE_NAMES, PHASE_LABELS, SPLIT_LABELS
from lwm.world_model.analysis.plots import SINGLE_FIG_SIZE, FIG_HEIGHT

sns.set_theme(
    style="whitegrid",
    rc={
        "grid.alpha": 0.14,
        "grid.color": "#CFCFCF",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    },
)

# Reverse map: "Seen Objects" -> "seen_obj" (also accepts old singular form from cached JSON)
_LABEL_TO_CAT = {v: k for k, v in GEN_LABELS.items()}
_LABEL_TO_CAT["Seen Object"] = "seen_obj"
_LABEL_TO_CAT["Unseen Object"] = "unseen_obj"

# Colorblind-safe fallback palette (Okabe-Ito inspired)
# Order: d512/110k, d1024-L8/110k, d1024-L16/110k, d1024-L16/220k, d2048/110k, d2048/220k
MODEL_COLORS = [
    "#0072B2",  # blue
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
]
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# ── Architecture-based styling (colorblind-safe) ───────────────────────────
ARCH_COLORS = {
    "d512-L8":   "#0072B2",  # blue
    "d1024-L8":  "#009E73",  # bluish green
    "d1024-L16": "#D55E00",  # vermillion
    "d2048-L8":  "#CC79A7",  # reddish purple
}
DATA_MARKERS = {"110k": "o", "220k": "^"}
DATA_MARKER_SIZE = {"110k": 4, "220k": 5}
# 110k = dark (solid), 220k = same color but lighter
DATA_ALPHA = {"110k": 1.0, "220k": 0.55}


def _lighten_color(color, amount=0.45):
    """Blend *color* toward white by *amount* (0 = unchanged, 1 = white). Returns hex."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def _parse_model_name(name):
    """Parse 'd1024-L16 / 220k' → ('d1024-L16', '220k')."""
    parts = name.split("/", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return name, None


def _get_model_style(name, idx):
    """Return (color, marker, markersize) for a model based on its name."""
    arch, data_tag = _parse_model_name(name)
    color = ARCH_COLORS.get(arch, MODEL_COLORS[idx % len(MODEL_COLORS)])
    marker = DATA_MARKERS.get(data_tag, MODEL_MARKERS[idx % len(MODEL_MARKERS)])
    ms = DATA_MARKER_SIZE.get(data_tag, 4)
    return color, marker, ms


def _get_alpha(name):
    """Return bar-chart alpha for the data size encoded in *name*."""
    _, data_tag = _parse_model_name(name)
    return DATA_ALPHA.get(data_tag, 1.0)


def _save_fig(fig, output_path):
    output_path = pathlib.Path(output_path)
    fig.savefig(output_path)
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    cprint(f"Saved: {output_path} + .png", "green")


def _save_csv(csv_path, header, rows):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    cprint(f"Saved: {csv_path}", "green")


# ── Windowed N-step comparison ──────────────────────────────────────────────

def create_comparison_windowed_plot(
    all_model_results, category, source, output_path, window_steps=6,
):
    """
    Per-category windowed drift comparison across models.

    Args:
        all_model_results: list of dicts from evaluate_single_checkpoint
        category: "seen_obj" or "unseen_obj"
        source: "expert" or "new_config"
        output_path: where to save the PDF
        window_steps: N in N-step eval
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    csv_header = ["step"]
    csv_data = {k + 1: [k + 1] for k in range(window_steps)}
    rows = []
    palette = {}

    for idx, model in enumerate(all_model_results):
        windowed = model.get("windowed_by_source", {}).get(source, {}).get(category)
        if windowed is None:
            continue
        name = model["model_name"]
        color, marker, ms = _get_model_style(name, idx)
        palette[name] = color
        for k, s in enumerate(windowed["step_indices"]):
            rows.append({"step": s, "drift_cm": windowed["avg_aff_drift_cm"][k], "model": name})
        csv_header.append(f"{name}_drift_cm")
        for k, s in enumerate(windowed["step_indices"]):
            csv_data.setdefault(s, [s])
            csv_data[s].append(f"{windowed['avg_aff_drift_cm'][k]:.4f}")

    if not rows:
        plt.close(fig)
        return

    df = pd.DataFrame(rows)
    sns.lineplot(data=df, x="step", y="drift_cm", hue="model", palette=palette,
                 linewidth=1.5, marker="o", markersize=4, ax=ax)
    for line, name in zip(ax.get_lines(), list(palette.keys())):
        _, data_tag = _parse_model_name(name)
        marker = DATA_MARKERS.get(data_tag, "o")
        ms = DATA_MARKER_SIZE.get(data_tag, 4)
        line.set_marker(marker)
        line.set_markersize(ms)
        if data_tag == "220k":
            line.set_linestyle('none')

    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Affordance Error (cm)')
    ax.set_ylim(bottom=0)
    source_label = SPLIT_LABELS.get(source, source)
    ax.set_title(f'{GEN_LABELS[category]} — Windowed {window_steps}-Step Affordance Prediction Error ({source_label})')
    ax.set_xticks(list(range(1, window_steps + 1)))
    _build_arch_data_legend(ax, list(palette.keys()))

    _save_fig(fig, output_path)
    _save_csv(
        pathlib.Path(output_path).with_suffix('.csv'),
        csv_header,
        [csv_data[k + 1] for k in range(window_steps)],
    )


# ── Full-horizon drift comparison ──────────────────────────────────────────

def create_comparison_drift_plot(
    all_model_results, category, source, output_path,
):
    """
    Per-category full-horizon drift comparison across models.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    csv_header = ["step"]
    csv_rows_map = {}
    rows = []
    palette = {}

    for idx, model in enumerate(all_model_results):
        gen = model.get("gen_avg_results", {}).get(source, {}).get(category)
        if gen is None:
            continue
        name = model["model_name"]
        color, marker, ms = _get_model_style(name, idx)
        palette[name] = color
        for i, s in enumerate(gen["step_indices"]):
            rows.append({"step": s + 1, "error_cm": gen["avg_affordance_errors"][i], "model": name})
        csv_header.append(f"{name}_error_cm")
        for i, s in enumerate(gen["step_indices"]):
            csv_rows_map.setdefault(s + 1, [s + 1])
            csv_rows_map[s + 1].append(f"{gen['avg_affordance_errors'][i]:.4f}")

    if not rows:
        plt.close(fig)
        return

    df = pd.DataFrame(rows)
    sns.lineplot(data=df, x="step", y="error_cm", hue="model", palette=palette,
                 linewidth=1.5, marker="o", markersize=4, ax=ax)
    for line, name in zip(ax.get_lines(), list(palette.keys())):
        _, data_tag = _parse_model_name(name)
        marker = DATA_MARKERS.get(data_tag, "o")
        ms = DATA_MARKER_SIZE.get(data_tag, 4)
        line.set_marker(marker)
        line.set_markersize(ms)
        if data_tag == "220k":
            line.set_linestyle('none')

    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Affordance Error (cm)')
    ax.set_ylim(bottom=0)
    source_label = SPLIT_LABELS.get(source, source)
    ax.set_title(f'{GEN_LABELS[category]} — Open-Loop Affordance Prediction Error ({source_label})')
    _build_arch_data_legend(ax, list(palette.keys()))

    _save_fig(fig, output_path)
    _save_csv(
        pathlib.Path(output_path).with_suffix('.csv'),
        csv_header,
        [csv_rows_map[s] for s in sorted(csv_rows_map)],
    )


# ── One-step summary bar chart ──────────────────────────────────────────────

def create_comparison_one_step_bar(all_model_results, source, output_path):
    """
    Grouped bar chart: x-axis = models, groups = seen/unseen, y = 1-step drift.
    """
    cats_with_data = []
    for cat in GEN_CATEGORIES:
        if any(
            model.get("windowed_by_source", {}).get(source, {}).get(cat) is not None
            for model in all_model_results
        ):
            cats_with_data.append(cat)

    if not cats_with_data:
        return

    rows = []
    for model in all_model_results:
        for cat in cats_with_data:
            w = model.get("windowed_by_source", {}).get(source, {}).get(cat)
            if w is not None:
                rows.append({"model": model["model_name"], "category": GEN_LABELS[cat],
                            "error_cm": w["avg_aff_drift_cm"][0]})
    if not rows:
        return

    df = pd.DataFrame(rows)
    model_order = [m["model_name"] for m in all_model_results if m["model_name"] in df["model"].values]
    palette = {GEN_LABELS[c]: GEN_COLORS[c] for c in cats_with_data}
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
    sns.barplot(data=df, x="model", y="error_cm", hue="category", palette=palette,
                order=model_order, hue_order=[GEN_LABELS[c] for c in cats_with_data],
                edgecolor="black", linewidth=0.5, alpha=0.85, ax=ax)

    for container in ax.containers:
        for bar in container:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('1-Step Error (cm)')
    source_label = SPLIT_LABELS.get(source, source)
    ax.set_title(f'Single-Step Affordance Prediction Error ({source_label})')
    ax.set_ylim(bottom=0)
    _mono_legend(ax)

    _save_fig(fig, output_path)


# ── Orchestrator ────────────────────────────────────────────────────────────

def create_comparison_phase_plot(
    all_model_results, category, source, output_dir, window_steps=18,
):
    """
    Phase-based comparison: 3-panel figure (Reach | Grasp | Pull).
    Each panel shows models as separate lines for the given (source, category).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharey=False)

    any_data = False
    csv_all = {}

    for phase_idx, phase_name in enumerate(PHASE_NAMES):
        ax = axes[phase_idx]
        csv_header = ["step"]
        csv_rows = {k + 1: [k + 1] for k in range(window_steps)}
        rows = []
        palette = {}

        for model_idx, model in enumerate(all_model_results):
            phase_data = (
                model.get("phase_by_source", {})
                .get(source, {}).get(category, {}).get(phase_name)
            )
            if phase_data is None:
                continue
            any_data = True
            name = model["model_name"]
            color, marker, ms = _get_model_style(name, model_idx)
            palette[name] = color
            for k, s in enumerate(phase_data["step_indices"]):
                rows.append({"step": s, "drift_cm": phase_data["avg_aff_drift_cm"][k], "model": name})
            csv_header.append(f"{name}_drift_cm")
            for k, s in enumerate(phase_data["step_indices"]):
                csv_rows.setdefault(s, [s])
                csv_rows[s].append(f"{phase_data['avg_aff_drift_cm'][k]:.4f}")

        if rows:
            df = pd.DataFrame(rows)
            sns.lineplot(data=df, x="step", y="drift_cm", hue="model", palette=palette,
                         linewidth=1.5, marker="o", markersize=4, ax=ax)
            for line, name in zip(ax.get_lines(), list(palette.keys())):
                _, data_tag = _parse_model_name(name)
                mk = DATA_MARKERS.get(data_tag, "o")
                msz = DATA_MARKER_SIZE.get(data_tag, 4)
                line.set_marker(mk)
                line.set_markersize(msz)
                if data_tag == "220k":
                    line.set_linestyle('none')
        ax.set_xlabel('Prediction Step')
        ax.set_ylabel('Affordance Error (cm)')
        ax.set_ylim(bottom=0)
        ax.set_title(PHASE_LABELS[phase_name])
        _build_arch_data_legend(ax, list(palette.keys()), fontsize=7, loc='upper left')

        csv_all[phase_name] = (csv_header, csv_rows)

    if not any_data:
        plt.close(fig)
        return

    source_label = SPLIT_LABELS.get(source, source)
    fig.suptitle(
        f'{GEN_LABELS[category]} — Phase-Based Affordance Prediction Error ({source_label})',
        fontsize=12,
    )
    fig.tight_layout()

    output_dir = pathlib.Path(output_dir)
    _save_fig(fig, output_dir / f"phase_eval_{category}.pdf")

    for phase_name, (header, rows) in csv_all.items():
        sorted_steps = sorted(rows.keys())
        _save_csv(
            output_dir / f"phase_{phase_name}_{category}.csv",
            header,
            [rows[s] for s in sorted_steps],
        )


# ── Combined: all models × seen/unseen on one plot ────────────────────────

def _merge_drift_across_sources(model, cat):
    """Weighted-average full-horizon drift for a gen category across all sources."""
    gen_all = model.get("gen_avg_results", {})
    parts = []
    for source in ["expert", "new_config", "replay"]:
        r = gen_all.get(source, {}).get(cat)
        if r is not None:
            parts.append(r)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    # Equal weight per source (clip to min episodes, 50/50 contribution)
    min_len = min(len(p["avg_affordance_errors"]) for p in parts)
    min_eps = min(p["num_episodes"] for p in parts)
    n_sources = len(parts)
    merged_errors = np.zeros(min_len)
    for p in parts:
        merged_errors += np.array(p["avg_affordance_errors"][:min_len]) / n_sources
    out = {
        "step_indices": parts[0]["step_indices"][:min_len],
        "avg_affordance_errors": merged_errors.tolist(),
        "num_episodes": min_eps * n_sources,  # effective count at equal proportion
    }
    if all(p.get("avg_agent_errors") is not None for p in parts):
        merged_agent = np.zeros(min_len)
        for p in parts:
            merged_agent += np.array(p["avg_agent_errors"][:min_len]) / n_sources
        out["avg_agent_errors"] = merged_agent.tolist()
    return out


def _merge_windowed_across_sources(model, cat):
    """Weighted-average windowed drift for a gen category across all sources."""
    w_all = model.get("windowed_by_source", {})
    parts = []
    for source in ["expert", "new_config", "replay"]:
        r = w_all.get(source, {}).get(cat)
        if r is not None:
            parts.append(r)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    # Equal weight per source (clip to min windows, 50/50 contribution)
    n_steps = len(parts[0]["avg_aff_drift_cm"])
    min_wins = min(p["num_windows"] for p in parts)
    n_sources = len(parts)
    merged_drift = np.zeros(n_steps)
    for p in parts:
        merged_drift += np.array(p["avg_aff_drift_cm"][:n_steps]) / n_sources
    out = {
        "step_indices": parts[0]["step_indices"][:n_steps],
        "avg_aff_drift_cm": merged_drift.tolist(),
        "num_windows": min_wins * n_sources,  # effective count at equal proportion
    }
    if all(p.get("avg_agent_drift_cm") is not None for p in parts):
        merged_agent = np.zeros(n_steps)
        for p in parts:
            merged_agent += np.array(p["avg_agent_drift_cm"][:n_steps]) / n_sources
        out["avg_agent_drift_cm"] = merged_agent.tolist()
    return out


_MONO_FONT = {"family": "monospace"}


def _pad_names(names):
    """Pad names so '/' separators align (for monospace legends)."""
    parts = [n.split("/", 1) for n in names]
    if all(len(p) == 2 for p in parts):
        max_left = max(len(p[0].rstrip()) for p in parts)
        return [f"{p[0].rstrip():<{max_left}} /{p[1]}" for p in parts]
    return list(names)


def _mono_legend(ax, **kwargs):
    """Re-create the current legend with padded model names and monospace font."""
    handles, labels = ax.get_legend_handles_labels()
    padded = _pad_names(labels)
    defaults = dict(frameon=False, fontsize=8, prop=_MONO_FONT, ncol=len(handles))
    defaults.update(kwargs)
    ax.legend(handles, padded, **defaults)


def _unique_archs_and_data(model_names):
    """Return (ordered unique archs with colors, set of data tags) from model names."""
    from collections import OrderedDict
    arch_colors = OrderedDict()
    data_tags = set()
    for name in model_names:
        arch, data_tag = _parse_model_name(name)
        if arch not in arch_colors:
            color, _, _ = _get_model_style(name, 0)
            arch_colors[arch] = color
        if data_tag:
            data_tags.add(data_tag)
    return arch_colors, data_tags


def _build_arch_data_legend(ax, model_names, **kwargs):
    """Two legends side by side: left = all models (one column), right = 110k, 220k (one column)."""
    from matplotlib.lines import Line2D
    arch_colors, data_tags = _unique_archs_and_data(model_names)
    arch_handles = [Line2D([0], [0], color=c, linewidth=2, label=a) for a, c in arch_colors.items()]
    leg1 = ax.legend(handles=arch_handles, ncol=1, frameon=False, fontsize=8, prop=_MONO_FONT, loc="upper left")
    ax.add_artist(leg1)
    if len(data_tags) > 1:
        data_handles = [
            Line2D([0], [0], color="gray", linewidth=1.5, marker="o", markersize=4, label="110k"),
            Line2D([0], [0], color="gray", linewidth=0, marker="D", markersize=5, label="2× data"),
        ]
        leg2 = ax.legend(handles=data_handles, ncol=1, frameon=False, fontsize=8, prop=_MONO_FONT,
                         loc="upper left", bbox_to_anchor=(0.35, 1.0))


def _build_factored_legend(ax, model_names, model_colors):
    """Build a factored legend: color = architecture, linestyle = seen/unseen, marker = data size."""
    from matplotlib.lines import Line2D
    arch_colors, data_tags = _unique_archs_and_data(model_names)
    handles = []
    # Architecture entries (colored solid lines)
    for arch, color in arch_colors.items():
        handles.append(Line2D([0], [0], color=color, linewidth=2, label=arch))
    # Separator
    handles.append(Line2D([0], [0], linewidth=0, label=""))
    # Seen / Unseen entries (black, different linestyles)
    handles.append(Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="Seen objects"))
    handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Unseen objects"))
    # Data-size entries (only when both sizes present): line+marker = 110k, marker-only = 2× data
    if len(data_tags) > 1:
        handles.append(Line2D([0], [0], linewidth=0, label=""))
        handles.append(Line2D([0], [0], color="gray", linewidth=1.5, marker="o", markersize=4, label="110k"))
        handles.append(Line2D([0], [0], color="gray", linewidth=0, marker="D", markersize=5, label="2× data"))
    ncol = len(arch_colors)  # model colors in row 1, rest below
    ax.legend(handles=handles, frameon=False, fontsize=8,
              prop={"family": "monospace"}, ncol=ncol)


def _build_bar_legend(ax, model_names, **kwargs):
    """Two legends side by side: left = all models (one column), right = 110k, 220k (one column)."""
    arch_colors, data_tags = _unique_archs_and_data(model_names)
    arch_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=a)
                    for a, c in arch_colors.items()]
    leg1 = ax.legend(handles=arch_handles, ncol=1, frameon=False, fontsize=8, loc="upper left")
    ax.add_artist(leg1)
    if len(data_tags) > 1:
        data_handles = [
            Patch(facecolor="gray", edgecolor="black", linewidth=0.5, label="110k"),
            Patch(facecolor="gray", edgecolor="black", linewidth=0.5, hatch="///", label="2× data"),
        ]
        ax.legend(handles=data_handles, ncol=1, frameon=False, fontsize=8,
                  loc="upper left", bbox_to_anchor=(0.35, 1.0))


def _draw_seen_unseen_bar_plot(ax, plotted_names, plotted_colors, get_val_fn, ylabel, title):
    """
    Draw Seen/Unseen bar plot with proper spacing: Seen block, gap, Unseen block.
    110k = solid color, 220k = same color with stripes. Each model gets its own bar.
    get_val_fn(name, cat_label) -> float.
    """
    n_models = len(plotted_names)
    bar_width = 0.8
    gap = 1.5
    x_seen = np.arange(n_models)
    x_unseen = np.arange(n_models) + n_models + gap

    for i, name in enumerate(plotted_names):
        color = plotted_colors[i]
        _, data_tag = _parse_model_name(name)
        hatch = "///" if data_tag == "220k" else None
        seen_val = get_val_fn(name, "Seen")
        unseen_val = get_val_fn(name, "Unseen")
        ax.bar(x_seen[i], seen_val, bar_width, color=color,
               edgecolor="black", linewidth=0.5, hatch=hatch)
        ax.bar(x_unseen[i], unseen_val, bar_width, color=color,
               edgecolor="black", linewidth=0.5, hatch=hatch)

    ax.set_xticks([(n_models - 1) / 2, n_models + gap + (n_models - 1) / 2])
    ax.set_xticklabels(["Seen", "Unseen"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)


def create_comparison_combined_drift(all_model_results, output_path, source=None):
    """
    All models × seen/unseen on one full-horizon drift plot.
    If source is None: pool expert + interaction per gen category.
    If source is set: use only that source's data (expert or new_config).
    Color = model, solid = seen, dashed = unseen.
    """
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)

    csv_header = ["step"]
    csv_rows_map = {}
    rows = []
    plotted_names = []
    plotted_colors = []
    model_auc = {}  # (name, cat_label) -> area under curve

    for idx, model in enumerate(all_model_results):
        name = model["model_name"]
        color, marker, ms = _get_model_style(name, idx)
        plotted_names.append(name)
        plotted_colors.append(color)

        for cat in GEN_CATEGORIES:
            if source is None:
                merged = _merge_drift_across_sources(model, cat)
            else:
                merged = model.get("gen_avg_results", {}).get(source, {}).get(cat)
            if merged is None:
                continue
            cat_label = "Seen" if cat == "seen_obj" else "Unseen"
            steps = merged["step_indices"]
            errors = merged["avg_affordance_errors"]
            x = np.array([s + 1 for s in steps], dtype=float)
            err = np.asarray(errors, dtype=float)
            if len(x) > 1:
                model_auc[(name, cat_label)] = float(np.trapz(err, x))
            else:
                model_auc[(name, cat_label)] = 0.0

            for i, s in enumerate(steps):
                rows.append({"step": s + 1, "error_cm": errors[i], "model": name,
                             "category": cat_label})

            col_name = f"{name}_{cat_label}_error_cm"
            csv_header.append(col_name)
            for i, s in enumerate(steps):
                csv_rows_map.setdefault(s + 1, [s + 1])
                csv_rows_map[s + 1].append(f"{errors[i]:.4f}")

    if not rows:
        plt.close(fig)
        return

    df = pd.DataFrame(rows)
    palette = {name: c for name, c in zip(plotted_names, plotted_colors)}
    sns.lineplot(data=df, x="step", y="error_cm", hue="model", style="category",
                 palette=palette, dashes={"Seen": "", "Unseen": (2, 2)},
                 linewidth=1.5, marker="o", markersize=4, ax=ax)
    model_for_each_line = [m for m in plotted_names for _ in range(2)]  # seen+unseen per model
    for line, name in zip(ax.get_lines(), model_for_each_line):
        _, data_tag = _parse_model_name(name)
        mk = DATA_MARKERS.get(data_tag, "o")
        msz = DATA_MARKER_SIZE.get(data_tag, 4)
        line.set_marker(mk)
        line.set_markersize(msz)
        if data_tag == "220k":
            line.set_linestyle('none')

    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Affordance Error (cm)')
    ax.set_ylim(bottom=0)
    title = 'Open-Loop Affordance Prediction Error'
    if source is not None:
        title += f" ({SPLIT_LABELS.get(source, source)})"
    ax.set_title(title)
    _build_factored_legend(ax, plotted_names, plotted_colors)

    _save_fig(fig, output_path)
    if csv_rows_map:
        _save_csv(
            pathlib.Path(output_path).with_suffix('.csv'),
            csv_header,
            [csv_rows_map[s] for s in sorted(csv_rows_map)],
        )

    # Area-under-curve bar plot (Seen block, gap, Unseen block)
    if model_auc and plotted_names:
        fig_bar, ax_bar = plt.subplots(figsize=SINGLE_FIG_SIZE)
        get_val = lambda name, cat: model_auc.get((name, cat), 0)
        title = "Cumulative Affordance Prediction Error"
        if source is not None:
            title += f" ({SPLIT_LABELS.get(source, source)})"
        _draw_seen_unseen_bar_plot(ax_bar, plotted_names, plotted_colors, get_val,
                                   "Cumulative Error (cm·step)", title)
        _build_bar_legend(ax_bar, plotted_names)
        bar_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + "_bar" + pathlib.Path(output_path).suffix)
        _save_fig(fig_bar, bar_path)
        plt.close(fig_bar)


def create_comparison_combined_drift_agent(all_model_results, output_path, source=None):
    """
    Same as combined drift but for agent state error (full-horizon curve + AUC bar).
    Only runs when at least one model has avg_agent_errors.
    """
    plotted_names = []
    plotted_colors = []
    model_auc = {}
    rows = []

    for idx, model in enumerate(all_model_results):
        name = model["model_name"]
        color, marker, ms = _get_model_style(name, idx)
        plotted_names.append(name)
        plotted_colors.append(color)

        for cat in GEN_CATEGORIES:
            if source is None:
                merged = _merge_drift_across_sources(model, cat)
            else:
                merged = model.get("gen_avg_results", {}).get(source, {}).get(cat)
            if merged is None or merged.get("avg_agent_errors") is None:
                continue
            cat_label = "Seen" if cat == "seen_obj" else "Unseen"
            steps = merged["step_indices"]
            errors = merged["avg_agent_errors"]
            x = np.array([s + 1 for s in steps], dtype=float)
            err = np.asarray(errors, dtype=float)
            model_auc[(name, cat_label)] = float(np.trapz(err, x)) if len(x) > 1 else 0.0
            for i, s in enumerate(steps):
                rows.append({"step": s + 1, "error_cm": errors[i], "model": name,
                             "category": cat_label})

    if not model_auc:
        return

    # Curve plot
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)
    df = pd.DataFrame(rows)
    palette = {name: c for name, c in zip(plotted_names, plotted_colors)}
    sns.lineplot(data=df, x="step", y="error_cm", hue="model", style="category",
                 palette=palette, dashes={"Seen": "", "Unseen": (2, 2)},
                 linewidth=1.5, marker="o", markersize=4, ax=ax)
    model_for_each_line = [m for m in plotted_names for _ in range(2)]
    for line, name in zip(ax.get_lines(), model_for_each_line):
        _, data_tag = _parse_model_name(name)
        mk = DATA_MARKERS.get(data_tag, "o")
        msz = DATA_MARKER_SIZE.get(data_tag, 4)
        line.set_marker(mk)
        line.set_markersize(msz)
        if data_tag == "220k":
            line.set_linestyle('none')
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("Agent State Error (cm)")
    ax.set_ylim(bottom=0)
    title = "Open-Loop Agent State Prediction Error"
    if source is not None:
        title += f" ({SPLIT_LABELS.get(source, source)})"
    ax.set_title(title)
    _build_factored_legend(ax, plotted_names, plotted_colors)
    agent_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + "_agent" + pathlib.Path(output_path).suffix)
    _save_fig(fig, agent_path)
    plt.close(fig)

    # AUC bar plot
    fig_bar, ax_bar = plt.subplots(figsize=SINGLE_FIG_SIZE)
    get_val = lambda name, cat: model_auc.get((name, cat), 0)
    title = "Cumulative Agent State Prediction Error"
    if source is not None:
        title += f" ({SPLIT_LABELS.get(source, source)})"
    _draw_seen_unseen_bar_plot(ax_bar, plotted_names, plotted_colors, get_val,
                               "Cumulative Error (cm·step)", title)
    _build_bar_legend(ax_bar, plotted_names)
    bar_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + "_agent_bar" + pathlib.Path(output_path).suffix)
    _save_fig(fig_bar, bar_path)
    plt.close(fig_bar)


def create_comparison_combined_windowed(all_model_results, output_path, window_steps=6, source=None):
    """
    All models × seen/unseen on one windowed N-step plot.
    If source is None: pool expert + interaction per gen category.
    If source is set: use only that source's data (expert or new_config).
    Color = model, solid = seen, dashed = unseen.
    """
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)

    csv_header = ["step"]
    csv_data = {k + 1: [k + 1] for k in range(window_steps)}
    plotted_names = []
    plotted_colors = []
    agent_vals = {}  # (name, "Seen"|"Unseen") -> step-6 value (float)
    rows = []

    for idx, model in enumerate(all_model_results):
        name = model["model_name"]
        color, marker, ms = _get_model_style(name, idx)
        plotted_names.append(name)
        plotted_colors.append(color)

        for cat in GEN_CATEGORIES:
            if source is None:
                merged = _merge_windowed_across_sources(model, cat)
            else:
                merged = model.get("windowed_by_source", {}).get(source, {}).get(cat)
            if merged is None:
                continue
            cat_label = "Seen" if cat == "seen_obj" else "Unseen"
            for k, s in enumerate(merged["step_indices"]):
                rows.append({"step": s, "drift_cm": merged["avg_aff_drift_cm"][k],
                            "model": name, "category": cat_label})

            col_name = f"{name}_{cat_label}_drift_cm"
            csv_header.append(col_name)
            for k, s in enumerate(merged["step_indices"]):
                csv_data.setdefault(s, [s])
                csv_data[s].append(f"{merged['avg_aff_drift_cm'][k]:.4f}")

            if merged.get("avg_agent_drift_cm") is not None and len(merged["avg_agent_drift_cm"]) >= window_steps:
                agent_vals[(name, cat_label)] = merged["avg_agent_drift_cm"][window_steps - 1]

    if not rows:
        plt.close(fig)
        return

    df = pd.DataFrame(rows)
    palette = {name: c for name, c in zip(plotted_names, plotted_colors)}
    sns.lineplot(data=df, x="step", y="drift_cm", hue="model", style="category",
                 palette=palette, dashes={"Seen": "", "Unseen": (2, 2)},
                 linewidth=1.5, marker="o", markersize=4, ax=ax)
    model_for_each_line = [m for m in plotted_names for _ in range(2)]
    for line, name in zip(ax.get_lines(), model_for_each_line):
        _, data_tag = _parse_model_name(name)
        mk = DATA_MARKERS.get(data_tag, "o")
        msz = DATA_MARKER_SIZE.get(data_tag, 4)
        line.set_marker(mk)
        line.set_markersize(msz)
        if data_tag == "220k":
            line.set_linestyle('none')

    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Affordance Error (cm)')
    ax.set_ylim(bottom=0)
    title = f'Windowed {window_steps}-Step Affordance Prediction Error'
    if source is not None:
        title += f" ({SPLIT_LABELS.get(source, source)})"
    ax.set_title(title)
    ax.set_xticks(list(range(1, window_steps + 1)))
    _build_factored_legend(ax, plotted_names, plotted_colors)

    _save_fig(fig, output_path)
    _save_csv(
        pathlib.Path(output_path).with_suffix('.csv'),
        csv_header,
        [csv_data[k + 1] for k in range(window_steps)],
    )

    # 6-step error table: rows = models, columns = Seen, Unseen (same base name, .txt)
    step6_row = csv_data.get(window_steps)
    if step6_row is not None and len(csv_header) > 1:
        # csv_header = ["step", "Name_Seen_drift_cm", "Name_Unseen_drift_cm", ...]; step6_row = [6, v1, v2, ...]
        model_vals = {}
        for col_idx, col_name in enumerate(csv_header[1:], start=1):
            if col_name.endswith("_Seen_drift_cm"):
                name = col_name.replace("_Seen_drift_cm", "")
                model_vals.setdefault(name, {})["Seen"] = step6_row[col_idx]
            elif col_name.endswith("_Unseen_drift_cm"):
                name = col_name.replace("_Unseen_drift_cm", "")
                model_vals.setdefault(name, {})["Unseen"] = step6_row[col_idx]
        if model_vals:
            # Display name: "ModelName (12.5M)" when num_params available
            num_params_by_name = {m["model_name"]: m.get("num_params") for m in all_model_results}
            def display_name(name):
                n = num_params_by_name.get(name)
                if n is not None:
                    return f"{name} ({n/1e6:.1f}M)"
                return name
            names_display = [display_name(name) for name in plotted_names]
            txt_path = pathlib.Path(output_path).with_suffix('.txt')
            col_w = max(12, max(len(d) for d in names_display))
            lines = [
                f"{'Model':<{col_w}}  {'Seen (cm)':>10}  {'Unseen (cm)':>10}",
                "-" * (col_w + 26),
            ]
            for name, disp in zip(plotted_names, names_display):
                row = model_vals.get(name, {})
                seen_s = row.get("Seen", "")
                unseen_s = row.get("Unseen", "")
                lines.append(f"{disp:<{col_w}}  {str(seen_s):>10}  {str(unseen_s):>10}")
            with open(txt_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            # Bar plot: Seen block, gap, Unseen block
            def get_val(name, cat):
                val = model_vals.get(name, {}).get(cat, "")
                return float(val) if val != "" else 0
            fig_bar, ax_bar = plt.subplots(figsize=SINGLE_FIG_SIZE)
            title = f"Windowed {window_steps}-Step Affordance Prediction Error"
            if source is not None:
                title += f" ({SPLIT_LABELS.get(source, source)})"
            _draw_seen_unseen_bar_plot(ax_bar, plotted_names, plotted_colors, get_val,
                                       "Affordance Error (cm)", title)
            _build_bar_legend(ax_bar, plotted_names)
            bar_path = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + "_bar" + pathlib.Path(output_path).suffix)
            _save_fig(fig_bar, bar_path)
            plt.close(fig_bar)

    # Agent state: 6-step table + bar (when available)
    if agent_vals and plotted_names:
        num_params_by_name = {m["model_name"]: m.get("num_params") for m in all_model_results}
        def display_name(name):
            n = num_params_by_name.get(name)
            if n is not None:
                return f"{name} ({n/1e6:.1f}M)"
            return name
        names_display = [display_name(name) for name in plotted_names]
        base = pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + "_agent")
        col_w = max(12, max(len(d) for d in names_display))
        lines = [
            f"{'Model':<{col_w}}  {'Seen (cm)':>10}  {'Unseen (cm)':>10}",
            "-" * (col_w + 26),
        ]
        seen_vals_agent = []
        unseen_vals_agent = []
        for name in plotted_names:
            row = {"Seen": agent_vals.get((name, "Seen")), "Unseen": agent_vals.get((name, "Unseen"))}
            seen_s = f"{row['Seen']:.4f}" if row["Seen"] is not None else ""
            unseen_s = f"{row['Unseen']:.4f}" if row["Unseen"] is not None else ""
            lines.append(f"{display_name(name):<{col_w}}  {seen_s:>10}  {unseen_s:>10}")
        with open(base.with_suffix(".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        def get_val_agent(name, cat):
            val = agent_vals.get((name, cat), 0)
            return val if val is not None else 0
        fig_bar, ax_bar = plt.subplots(figsize=SINGLE_FIG_SIZE)
        title = f"Windowed {window_steps}-Step Agent State Prediction Error"
        if source is not None:
            title += f" ({SPLIT_LABELS.get(source, source)})"
        _draw_seen_unseen_bar_plot(ax_bar, plotted_names, plotted_colors, get_val_agent,
                                   "Agent State Error (cm)", title)
        _build_bar_legend(ax_bar, plotted_names)
        _save_fig(fig_bar, pathlib.Path(output_path).parent / (pathlib.Path(output_path).stem + "_agent_bar" + pathlib.Path(output_path).suffix))
        plt.close(fig_bar)


def create_comparison_config_line_plot(
    all_model_results, output_path, window_steps=6, source=None, metric="affordance",
):
    """
    Config-ordered line plot: X = architecture (by param count), Y = error.
    Four lines with factored encoding: color = seen/unseen, marker = data size.
    Blue = seen, orange = unseen. Circle = 110k steps, triangle = 220k steps.
    """
    from matplotlib.lines import Line2D

    # Build arch -> {110k: (seen, unseen), 220k: (seen, unseen)} with num_params from 110k
    arch_data = {}
    for model in all_model_results:
        name = model["model_name"]
        arch, data_tag = _parse_model_name(name)
        num_params = model.get("num_params") or 0

        seen_err, unseen_err = None, None
        for cat in GEN_CATEGORIES:
            if source is None:
                merged = _merge_windowed_across_sources(model, cat)
            else:
                merged = model.get("windowed_by_source", {}).get(source, {}).get(cat)
            if merged is None:
                continue
            drift = merged["avg_aff_drift_cm"]
            agent_drift = merged.get("avg_agent_drift_cm")
            if metric == "agent":
                val = agent_drift[window_steps - 1] if agent_drift and len(agent_drift) >= window_steps else None
            else:
                val = drift[window_steps - 1] if len(drift) >= window_steps else None
            if val is not None:
                if cat == "seen_obj":
                    seen_err = val
                else:
                    unseen_err = val

        if arch not in arch_data:
            arch_data[arch] = {"num_params": num_params, "110k": None, "220k": None}
        if seen_err is not None or unseen_err is not None:
            pair = (seen_err or 0, unseen_err or 0)
            if data_tag == "110k":
                arch_data[arch]["110k"] = pair
                arch_data[arch]["num_params"] = num_params
            elif data_tag == "220k":
                arch_data[arch]["220k"] = pair

    arch_data = {a: v for a, v in arch_data.items() if v["110k"] is not None}
    if not arch_data:
        return

    archs = sorted(arch_data.keys(), key=lambda a: arch_data[a]["num_params"])
    x = np.arange(len(archs))

    y_seen_110k = np.array([arch_data[a]["110k"][0] if arch_data[a]["110k"] else np.nan for a in archs])
    y_unseen_110k = np.array([arch_data[a]["110k"][1] if arch_data[a]["110k"] else np.nan for a in archs])
    y_seen_220k = np.array([arch_data[a]["220k"][0] if arch_data[a]["220k"] else np.nan for a in archs])
    y_unseen_220k = np.array([arch_data[a]["220k"][1] if arch_data[a]["220k"] else np.nan for a in archs])

    fig, ax = plt.subplots(figsize=(max(8, len(archs) * 1.2), 4.5))

    # Color = seen vs unseen; marker = data size (factored encoding)
    COLOR_SEEN = "#0072B2"    # blue
    COLOR_UNSEEN = "#D55E00"  # vermillion/orange

    ax.plot(x, y_seen_110k, color=COLOR_SEEN, linestyle='-', linewidth=2)
    ax.scatter(x, y_seen_110k, marker='o', c=COLOR_SEEN, s=60, zorder=5, edgecolors='white', linewidths=1)
    ax.scatter(x, y_unseen_110k, marker='o', c=COLOR_UNSEEN, s=60, zorder=5, edgecolors='white', linewidths=1)

    has_220k = np.isfinite(y_seen_220k)
    if np.any(has_220k):
        ax.plot(x, y_seen_220k, color=COLOR_SEEN, linestyle='-', linewidth=2)
        ax.scatter(x[has_220k], y_seen_220k[has_220k], marker='^', c=COLOR_SEEN, s=60, zorder=5, edgecolors='white', linewidths=1)
        ax.scatter(x[has_220k], y_unseen_220k[has_220k], marker='^', c=COLOR_UNSEEN, s=60, zorder=5, edgecolors='white', linewidths=1)

    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=0, ha='center')
    ax.set_ylabel("Affordance Error (cm)" if metric == "affordance" else "Agent State Error (cm)")
    ax.set_xlabel("Model config (ordered by param count)")
    title = f"Windowed {window_steps}-Step {'Affordance' if metric == 'affordance' else 'Agent State'} Error by Config"
    if source is not None:
        title += f" ({SPLIT_LABELS.get(source, source)})"
    ax.set_title(title)

    # Factored legend: 2 dimensions, 4 items total
    # Row 1: marker = data size; Row 2: color = seen/unseen
    leg_handles = [
        Line2D([0], [0], color='gray', marker='o', linestyle='none', markersize=8, label='110k steps'),
        Line2D([0], [0], color='gray', marker='^', linestyle='none', markersize=8, label='220k steps'),
        Line2D([0], [0], color=COLOR_SEEN, linestyle='-', linewidth=2, label='Seen'),
        Line2D([0], [0], color=COLOR_UNSEEN, linestyle='-', linewidth=2, label='Unseen'),
    ]
    ax.legend(handles=leg_handles, ncol=2, frameon=False, fontsize=8, loc='upper left')

    _save_fig(fig, output_path)
    plt.close(fig)


def create_all_cross_model_comparisons(
    all_model_results, output_dir, window_steps=6, phase_eval_steps=18,
):
    """
    Create all cross-model comparison plots organised by category.

    Directory layout:
        output_dir/
            {source}/
                windowed_{cat}.pdf
                drift_{cat}.pdf
                one_step_summary.pdf
                phase_eval_{cat}.pdf
                phase_{phase}_{cat}.csv
    """
    output_dir = pathlib.Path(output_dir)

    for source in ["expert", "new_config", "replay"]:
        has_data = any(
            model.get("windowed_by_source", {}).get(source)
            or model.get("gen_avg_results", {}).get(source)
            for model in all_model_results
        )
        has_phase = any(
            model.get("phase_by_source", {}).get(source)
            for model in all_model_results
        )
        if not has_data and not has_phase:
            continue

        source_dir = output_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        cprint(f"  → {source_dir}", "cyan")

        source_label = SPLIT_LABELS.get(source, source)
        cprint(f"\nCross-model comparison -- {source_label}", "magenta")

        if has_data:
            for cat in GEN_CATEGORIES:
                create_comparison_windowed_plot(
                    all_model_results, cat, source,
                    output_path=source_dir / f"windowed_{cat}.pdf",
                    window_steps=window_steps,
                )
                create_comparison_drift_plot(
                    all_model_results, cat, source,
                    output_path=source_dir / f"drift_{cat}.pdf",
                )

            create_comparison_one_step_bar(
                all_model_results, source,
                output_path=source_dir / "one_step_summary.pdf",
            )

            # Same combined-style plots (all models × seen/unseen) but for this source only
            create_comparison_combined_drift(
                all_model_results,
                output_path=source_dir / "combined_drift_full_horizon.pdf",
                source=source,
            )
            create_comparison_combined_drift_agent(
                all_model_results,
                output_path=source_dir / "combined_drift_full_horizon.pdf",
                source=source,
            )
            create_comparison_combined_windowed(
                all_model_results,
                output_path=source_dir / f"combined_windowed_{window_steps}step.pdf",
                window_steps=window_steps,
                source=source,
            )
            create_comparison_config_line_plot(
                all_model_results,
                output_path=source_dir / f"config_line_{window_steps}step.pdf",
                window_steps=window_steps,
                source=source,
                metric="affordance",
            )
            create_comparison_config_line_plot(
                all_model_results,
                output_path=source_dir / f"config_line_{window_steps}step_agent.pdf",
                window_steps=window_steps,
                source=source,
                metric="agent",
            )

        if has_phase:
            for cat in GEN_CATEGORIES:
                create_comparison_phase_plot(
                    all_model_results, cat, source,
                    output_dir=source_dir,
                    window_steps=phase_eval_steps,
                )

    # ── Combined plots: pool expert + interaction, split by seen/unseen ──
    cprint("\nCross-model comparison -- Combined (all sources pooled)", "magenta")
    create_comparison_combined_drift(
        all_model_results,
        output_path=output_dir / "combined_drift_full_horizon.pdf",
    )
    create_comparison_combined_drift_agent(
        all_model_results,
        output_path=output_dir / "combined_drift_full_horizon.pdf",
    )
    create_comparison_combined_windowed(
        all_model_results,
        output_path=output_dir / f"combined_windowed_{window_steps}step.pdf",
        window_steps=window_steps,
    )
    create_comparison_config_line_plot(
        all_model_results,
        output_path=output_dir / f"config_line_{window_steps}step.pdf",
        window_steps=window_steps,
        metric="affordance",
    )
    create_comparison_config_line_plot(
        all_model_results,
        output_path=output_dir / f"config_line_{window_steps}step_agent.pdf",
        window_steps=window_steps,
        metric="agent",
    )


# ── Cache loading ───────────────────────────────────────────────────────────

def resolve_output_dir(checkpoint_path, output_dir_cfg):
    """Resolve the per-model output directory (mirrors evaluate_single_checkpoint logic)."""
    cp = pathlib.Path(checkpoint_path)
    checkpoint_dir = cp.parent
    run_dir = checkpoint_dir.parent
    date_dir = run_dir.parent
    run_identifier = f"{date_dir.name}_{run_dir.name}"

    if output_dir_cfg is None:
        return checkpoint_dir / "eval_final_results"
    return pathlib.Path(output_dir_cfg) / run_identifier


def _model_name_from_checkpoint(checkpoint_path):
    """Extract model_name from checkpoint path (mirrors evaluate_single_checkpoint)."""
    run_dir = pathlib.Path(checkpoint_path).parent.parent
    parts = run_dir.name.split("_", 1)
    return parts[1] if len(parts) > 1 else run_dir.name


def _load_single_cache(cp_path, output_dir_cfg, test_set_hash=None):
    """Load a single model's cached evaluation_summary.json.

    When test_set_hash is provided, looks for hash-specific cache file first
    (evaluation_summary_{hash[:8]}.json), then falls back to generic file.

    Returns:
        dict with model results, or None if cache is missing.
    """
    out_dir = resolve_output_dir(cp_path, output_dir_cfg)

    # Try hash-specific cache first, then generic fallback
    candidates = []
    if test_set_hash:
        candidates.append(out_dir / f"evaluation_summary_{test_set_hash[:8]}.json")
    candidates.append(out_dir / "evaluation_summary.json")

    summary_path = None
    for candidate in candidates:
        if candidate.exists():
            summary_path = candidate
            break

    if summary_path is None:
        cprint(f"Cache miss: {out_dir} (no evaluation_summary found)", "yellow")
        return None

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    model_name = summary.get("model_name") or _model_name_from_checkpoint(cp_path)
    num_params = summary.get("num_params")

    # Reconstruct gen_avg_results: {source: {cat: result_dict}}
    gen_avg_results = {}
    for source, source_data in summary.get("generalization", {}).items():
        gen_avg_results[source] = {}
        for label, result in source_data.get("full_episode_rollouts", {}).items():
            cat = _LABEL_TO_CAT.get(label)
            if cat is not None:
                gen_avg_results[source][cat] = result

    # Reconstruct windowed_by_source: {source: {cat: result_dict}}
    windowed_by_source = {}
    for source, source_data in summary.get("windowed_eval", {}).items():
        windowed_by_source[source] = {}
        for label, result in source_data.items():
            cat = _LABEL_TO_CAT.get(label)
            if cat is not None:
                windowed_by_source[source][cat] = result

    # Reconstruct split_avg_results
    split_avg_results = {}
    for split_name, result in summary.get("full_episode_rollouts", {}).items():
        split_avg_results[split_name] = result

    # Reconstruct phase_by_source: {source: {cat: {phase: result_dict}}}
    phase_by_source = {}
    for source, source_data in summary.get("generalization", {}).items():
        phase_data = source_data.get("phase_eval", {})
        if phase_data:
            phase_by_source[source] = {}
            for label, cat_phases in phase_data.items():
                cat = _LABEL_TO_CAT.get(label)
                if cat is not None:
                    phase_by_source[source][cat] = cat_phases

    test_set_hash = summary.get("test_set_hash")

    cprint(f"Loaded cache: {model_name} ({summary_path})", "green")

    return {
        "model_name": model_name,
        "num_params": num_params,
        "checkpoint_path": cp_path,
        "split_avg_results": split_avg_results,
        "gen_avg_results": gen_avg_results,
        "windowed_by_source": windowed_by_source,
        "phase_by_source": phase_by_source,
        "test_set_hash": test_set_hash,
    }


def load_cached_model_results(checkpoint_paths, output_dir_cfg, test_set_hash=None):
    """
    Load evaluation_summary.json for each checkpoint and reconstruct
    the all_model_results list expected by the comparison plots.

    Returns:
        list of model result dicts, or None if any cache is missing.
    """
    all_results = []

    for cp_path in checkpoint_paths:
        result = _load_single_cache(cp_path, output_dir_cfg, test_set_hash=test_set_hash)
        if result is None:
            return None
        all_results.append(result)

    cprint(f"\nLoaded {len(all_results)} model(s) from cache", "green")
    return all_results
