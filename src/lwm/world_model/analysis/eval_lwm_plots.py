"""
Evaluation script for Local World Model (LWM) (scaling figures).

Thin orchestration layer; all logic lives in ``lwm.world_model.analysis``:
  constants.py         — colors, labels, object IDs
  checkpoint.py        — model + config loading
  episode_discovery.py — episode scanning, classification, ReplayBuffer
  core.py              — rollout evaluation, windowed N-step eval
  plots.py             — all matplotlib plotting
  summary.py           — JSON summary + printing helpers

Usage:
  python -m lwm.world_model.analysis.eval_lwm_plots --config-name eval/model_size_ablation_large
"""

import os
import sys
import pathlib
import pickle

import torch
import numpy as np
from termcolor import cprint

import hydra
from omegaconf import OmegaConf

from lwm.paths import CONFIG_DIR
from lwm.world_model.analysis.constants import GEN_CATEGORIES, GEN_LABELS, PHASE_NAMES, PHASE_LABELS, SPLIT_LABELS
from lwm.world_model.analysis.checkpoint import load_checkpoint, load_episode_splits
from lwm.world_model.analysis.episode_discovery import (
    find_test_episodes, load_eval_replay_buffer,
    compute_test_intersection_exclusions, compute_test_set_hash,
)
from lwm.world_model.analysis.comparison_utils import (
    create_all_cross_model_comparisons,
    load_cached_model_results,
    _load_single_cache,
)
from lwm.world_model.analysis.core import (
    evaluate_episode,
    evaluate_episodes_cached, aggregate_avg_error_from_cache,
    evaluate_windowed, aggregate_windowed_results,
    evaluate_phase_based, aggregate_phase_results,
)
from lwm.world_model.evaluation_utils import preload_episodes
from lwm.world_model.analysis.plots import (
    create_combined_plot,
    create_combined_avg_error_plot,
    create_generalization_avg_error_plot,
    create_windowed_eval_plot,
    create_one_step_summary_plot,
    create_combined_4way_drift_plot,
    create_combined_4way_windowed_plot,
)
from lwm.world_model.analysis.summary import save_evaluation_summary, print_windowed_summary
from lwm.world_model.evaluation_utils import compose_affordance_with_9dof_delta


# =============================================================================
# Helpers
# =============================================================================

def _resolve_output_dir(checkpoint_path, output_dir_cfg):
    """Resolve per-model output directory from checkpoint path."""
    cp = pathlib.Path(checkpoint_path)
    checkpoint_dir = cp.parent
    run_dir = checkpoint_dir.parent
    date_dir = run_dir.parent
    run_identifier = f"{date_dir.name}_{run_dir.name}"
    if output_dir_cfg is None:
        return checkpoint_dir / "eval_final_results"
    return pathlib.Path(output_dir_cfg) / run_identifier


def _model_name_from_checkpoint(checkpoint_path):
    """Extract short model name from checkpoint path."""
    run_dir = pathlib.Path(checkpoint_path).parent.parent
    parts = run_dir.name.split("_", 1)
    return parts[1] if len(parts) > 1 else run_dir.name


def _print_test_set_summary(split_indices, gen_indices_by_source):
    """Print a clear, compact summary of the test set."""
    for split_name, indices in split_indices.items():
        if indices:
            cprint(f"  {split_name}: {len(indices)} episodes", "cyan")
    for source in gen_indices_by_source:
        source_label = SPLIT_LABELS.get(source, source)
        parts = []
        for cat in GEN_CATEGORIES:
            n = len(gen_indices_by_source[source].get(cat, []))
            if n:
                parts.append(f"{GEN_LABELS[cat]}={n}")
        if parts:
            cprint(f"  {source_label} generalization: {', '.join(parts)}", "cyan")


def _compute_shared_val_new_config(checkpoint_paths, rollout_dataset_path):
    """Find new_config episodes that are in the validation set of ALL models.

    These were never trained on by any model, so they're fair game for evaluation.
    Returns dict: gen_category -> list of episode paths.
    """
    from lwm.world_model.analysis.checkpoint import load_episode_splits
    from lwm.world_model.analysis.episode_discovery import (
        classify_episode, classify_generalization, collect_episode_paths, episode_key)

    val_sets = []
    for cp_path in checkpoint_paths:
        splits = load_episode_splits(cp_path)
        if splits is None:
            return {}
        val_keys = {episode_key(ep["episode_path"]) for ep in splits.get("val_episodes", [])}
        val_sets.append(val_keys)

    # Intersection: episodes in val set of EVERY model
    shared_val = val_sets[0]
    for vs in val_sets[1:]:
        shared_val &= vs

    # Map the machine-independent keys back to local episode paths
    local_by_key = {episode_key(p): p for p in collect_episode_paths(rollout_dataset_path)}
    shared_val = {local_by_key[k] for k in shared_val if k in local_by_key}

    # Filter to new_config only, classify by generalization category
    result = {}
    for path in sorted(shared_val):
        if classify_episode(path) == "new_config":
            cat = classify_generalization(path)
            result.setdefault(cat, []).append(path)

    total = sum(len(v) for v in result.values())
    if total > 0:
        parts = [f"{GEN_LABELS.get(c, c)}={len(v)}" for c, v in result.items()]
        cprint(f"  Shared-val new_config: {total} episodes ({', '.join(parts)})", "green")
    return result


def _load_phase_cache(cache_path, expected_params):
    """Load per-episode phase cache; return empty if missing or params changed."""
    cache_path = pathlib.Path(cache_path)
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        if cache.get("params") == expected_params:
            n = len(cache.get("episodes", {}))
            cprint(f"  Phase cache: {n} episodes loaded from {cache_path.name}", "green")
            return cache
        cprint("  Phase cache params changed — invalidating", "yellow")
    return {"params": expected_params, "episodes": {}}


def _save_phase_cache(cache_path, phase_cache):
    """Persist phase cache to disk."""
    with open(cache_path, 'wb') as f:
        pickle.dump(phase_cache, f)
    n = len(phase_cache.get("episodes", {}))
    cprint(f"  Phase cache saved: {n} episodes → {cache_path}", "green")


# =============================================================================
# Main
# =============================================================================

@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIR / "lwm"),
    config_name="eval/model_size_ablation_large",
)
def main(cfg):
    """Main evaluation function using Hydra config."""

    checkpoint_paths = cfg.get("checkpoint_paths", None)
    checkpoint_path_single = cfg.get("checkpoint_path", None)

    if checkpoint_paths is None and checkpoint_path_single is None:
        raise ValueError("Either checkpoint_paths (list) or checkpoint_path (single) is required")

    if checkpoint_paths is None:
        checkpoint_paths = [checkpoint_path_single]
    else:
        checkpoint_paths = OmegaConf.to_container(checkpoint_paths, resolve=True)
        if not isinstance(checkpoint_paths, list):
            checkpoint_paths = [checkpoint_paths]

    use_cache = cfg.get("use_cache", False)
    output_dir_cfg = cfg.get("output_dir", None)
    test_intersection = cfg.get("test_intersection", False)
    seed = cfg.get("seed", 42)

    # Optional display names for models (same order as checkpoint_paths)
    model_names_cfg = cfg.get("model_names", None)
    if model_names_cfg is not None:
        model_names_cfg = OmegaConf.to_container(model_names_cfg, resolve=True)

    # Extract config name for comparison folder naming
    config_name = "comparison"
    try:
        from hydra.core.hydra_config import HydraConfig
        config_name = HydraConfig.get().job.config_name
    except Exception:
        pass

    def _get_model_name(idx):
        if model_names_cfg and idx < len(model_names_cfg):
            return model_names_cfg[idx]
        return _model_name_from_checkpoint(checkpoint_paths[idx])

    n_models = len(checkpoint_paths)
    cprint(f"\n{'='*80}", "cyan")
    cprint(f"LWM Evaluation — {n_models} model(s)", "cyan")
    for i, cp in enumerate(checkpoint_paths):
        # Quick param count from state_dict (no model instantiation)
        try:
            sd = torch.load(cp, map_location="cpu")["model_state_dict"]
            n_params = sum(v.numel() for v in sd.values())
            param_str = f" — {n_params/1e6:.1f}M params"
            del sd
        except Exception:
            param_str = ""
        cprint(f"  [{i+1}] {_get_model_name(i)}{param_str}", "cyan")
    cprint(f"{'='*80}\n", "cyan")

    # -----------------------------------------------------------------
    # Shared test set (intersection mode)
    # Computed BEFORE cache loading so we can use the hash to find
    # the correct cache file for this eval config.
    # -----------------------------------------------------------------
    shared_eval_data = None
    test_set_hash = None
    # Deferred: eval_episode_paths / gen_episodes_by_source are kept
    # around so the replay buffer can be loaded lazily (only if needed).
    _deferred_eval_episode_paths = None
    _deferred_gen_episodes_by_source = None

    if test_intersection and n_models > 1:
        additional_exclusions = compute_test_intersection_exclusions(checkpoint_paths)
        if additional_exclusions is not None:
            expert_path = cfg.get("expert_dataset_path", None)
            rollout_path = cfg.get("rollout_dataset_path", None)
            # Use empty episode_splits — the exclusions already cover everything
            dummy_splits = {"train_episodes": [], "val_episodes": []}
            eval_episode_paths, gen_episodes_by_source = find_test_episodes(
                dummy_splits, expert_path, rollout_path,
                additional_exclusions=additional_exclusions,
            )
            # Include shared-val new_config episodes (val in ALL models → never trained on)
            if cfg.get("include_shared_val_new_config", False):
                shared_val_by_cat = _compute_shared_val_new_config(checkpoint_paths, rollout_path)
                existing_new_config = set(eval_episode_paths.get("new_config", []))
                for cat, paths in shared_val_by_cat.items():
                    for p in paths:
                        if p not in existing_new_config:
                            eval_episode_paths.setdefault("new_config", []).append(p)
                            gen_episodes_by_source.setdefault("new_config", {}).setdefault(cat, []).append(p)

            total_eval = sum(len(v) for v in eval_episode_paths.values())
            if total_eval == 0:
                cprint("No test episodes in intersection — aborting", "red")
                return

            # Compute test set hash for cache validation
            test_set_hash = compute_test_set_hash(eval_episode_paths)
            cprint(f"Test set hash: {test_set_hash}", "magenta")

            # Defer replay buffer loading — only needed if we have uncached models
            _deferred_eval_episode_paths = eval_episode_paths
            _deferred_gen_episodes_by_source = gen_episodes_by_source
        else:
            cprint("Could not compute intersection — evaluating per-model\n", "yellow")

    # -----------------------------------------------------------------
    # Fast path: use cached evaluation_summary.json files
    # Cache files are hash-specific so different eval configs coexist.
    # -----------------------------------------------------------------
    cached_results = {}  # idx -> result dict (populated when use_cache=True)
    if use_cache:
        cprint("use_cache=true — loading from evaluation_summary.json", "magenta")
        # Try loading each model's cache individually
        for i, cp in enumerate(checkpoint_paths):
            result = _load_single_cache(cp, output_dir_cfg, test_set_hash=test_set_hash)
            if result is not None:
                # Validate hash matches (covers generic fallback files without hash in name)
                cached_hash = result.get("test_set_hash")
                if test_set_hash and cached_hash != test_set_hash:
                    reason = "hash mismatch" if cached_hash else "no hash stored in cache"
                    cprint(f"Cache invalidated for [{i+1}] {_get_model_name(i)}: {reason}", "yellow")
                    continue
                result["model_name"] = _get_model_name(i)
                cached_results[i] = result

    # -----------------------------------------------------------------
    # Now check cache status (after hash validation may have invalidated some)
    # -----------------------------------------------------------------
    if use_cache:
        n_cached = len(cached_results)
        if n_cached == n_models:
            # All cached (and hash-validated) — skip evaluation entirely
            all_cached = [cached_results[i] for i in range(n_models)]
            _run_comparison_plots(cfg, all_cached, checkpoint_paths, output_dir_cfg, config_name)
            return
        elif n_cached > 0:
            cprint(f"Partial cache: {n_cached}/{n_models} models cached, evaluating {n_models - n_cached} remaining\n", "yellow")
        else:
            cprint("No valid cached results — running full evaluation\n", "yellow")

    # -----------------------------------------------------------------
    # Load shared replay buffer (deferred from intersection computation)
    # -----------------------------------------------------------------
    if _deferred_eval_episode_paths is not None:
        replay_buffer, split_indices, gen_indices_by_source = load_eval_replay_buffer(
            _deferred_eval_episode_paths,
            gen_episodes_by_source=_deferred_gen_episodes_by_source,
        )

        # Shuffle once — same order for all models
        rng = np.random.RandomState(seed)
        for split_name in split_indices:
            rng.shuffle(split_indices[split_name])
        for source in gen_indices_by_source:
            for cat in gen_indices_by_source[source]:
                rng.shuffle(gen_indices_by_source[source][cat])

        cprint("Shared test set (intersection of all models):", "green")
        _print_test_set_summary(split_indices, gen_indices_by_source)
        cprint("", "green")

        shared_eval_data = {
            "replay_buffer": replay_buffer,
            "split_indices": split_indices,
            "gen_indices_by_source": gen_indices_by_source,
        }

    # -----------------------------------------------------------------
    # Per-model evaluation
    # -----------------------------------------------------------------
    all_model_results = []

    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
        # Use cached result if available
        if checkpoint_idx in cached_results:
            cprint(f"\n[{checkpoint_idx+1}/{n_models}] {_get_model_name(checkpoint_idx)} — using cache", "green")
            all_model_results.append(cached_results[checkpoint_idx])
            continue

        display_name = _get_model_name(checkpoint_idx)
        cprint(f"\n{'='*80}", "yellow")
        cprint(f"[{checkpoint_idx+1}/{n_models}] {display_name}", "yellow")
        cprint(f"{'='*80}", "yellow")

        model_results = evaluate_single_checkpoint(
            cfg, checkpoint_path, shared_eval_data=shared_eval_data,
            model_name_override=display_name,
            test_set_hash=test_set_hash,
        )
        if model_results is not None:
            all_model_results.append(model_results)

    # -----------------------------------------------------------------
    # Cross-model comparison plots
    # -----------------------------------------------------------------
    _run_comparison_plots(cfg, all_model_results, checkpoint_paths, output_dir_cfg, config_name)

    cprint(f"\n{'='*80}", "green")
    cprint("All evaluations complete!", "green")
    cprint(f"{'='*80}\n", "green")


def _run_comparison_plots(cfg, all_model_results, checkpoint_paths, output_dir_cfg, config_name="comparison"):
    """Create cross-model comparison plots if multiple models were evaluated."""
    if len(all_model_results) <= 1:
        return

    cprint(f"\n{'='*80}", "cyan")
    cprint(f"Creating cross-model comparison plots ({len(all_model_results)} models) → {config_name}/", "cyan")
    cprint(f"{'='*80}\n", "cyan")

    if output_dir_cfg is None:
        comparison_dir = pathlib.Path(checkpoint_paths[0]).parent.parent / "eval_final_results" / config_name
    else:
        comparison_dir = pathlib.Path(output_dir_cfg) / config_name

    comparison_dir.mkdir(parents=True, exist_ok=True)

    windowed_eval_steps = cfg.get("windowed_eval_steps", 6)
    phase_eval_steps = cfg.get("phase_eval_steps", 18)
    create_all_cross_model_comparisons(
        all_model_results, comparison_dir,
        window_steps=windowed_eval_steps,
        phase_eval_steps=phase_eval_steps,
    )


# =============================================================================
# Single-checkpoint evaluation
# =============================================================================

def evaluate_single_checkpoint(cfg, checkpoint_path, shared_eval_data=None, model_name_override=None, test_set_hash=None):
    """
    Evaluate a single checkpoint on held-out test episodes.

    Args:
        shared_eval_data: Optional dict with preloaded replay_buffer, split_indices,
            gen_indices_by_source.  When provided (intersection mode), skips episode
            discovery and buffer loading — uses the shared data directly.
        model_name_override: Optional display name for this model (from config).
        test_set_hash: Optional hash of the test episode paths, stored in the
            evaluation summary JSON for cache validation.
    """

    # Setup device
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Setup output directory
    checkpoint_path_obj = pathlib.Path(checkpoint_path)
    checkpoint_dir = checkpoint_path_obj.parent
    run_dir = checkpoint_dir.parent
    date_dir = run_dir.parent
    run_identifier = f"{date_dir.name}_{run_dir.name}"

    run_dir_name = run_dir.name
    parts = run_dir_name.split("_", 1)
    model_name = model_name_override or (parts[1] if len(parts) > 1 else run_dir_name)

    output_dir_cfg = cfg.get("output_dir", None)
    if output_dir_cfg is None:
        output_dir = checkpoint_dir / "eval_final_results"
    else:
        output_dir = pathlib.Path(output_dir_cfg) / run_identifier
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for split in ["expert", "new_config"]:
        (output_dir / "individual_episodes" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "windowed_eval").mkdir(parents=True, exist_ok=True)
    (output_dir / "generalization").mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    lwm, checkpoint, normalization_stats, training_config = load_checkpoint(checkpoint_path, device)
    predict_agent_state = lwm.predict_agent_state

    # Detect if model is 9-DoF (v8)
    affordance_output_dim = getattr(lwm, 'affordance_output_dim', 3)
    is_9dof = (affordance_output_dim == 9)
    compose_affordance_fn = compose_affordance_with_9dof_delta if is_9dof else None
    if is_9dof:
        cprint("  9-DoF model (v8) — rigid body affordance composition "
               f"(target frame: {getattr(lwm, 'affordance_target_frame', 'world')}, "
               f"input frame: {getattr(lwm, 'input_frame', 'world')})", "yellow")

    if normalization_stats is None:
        cprint("Error: No normalization stats available", "red")
        return None

    # -----------------------------------------------------------------
    # Get test data — either shared (intersection mode) or per-model
    # -----------------------------------------------------------------
    if shared_eval_data is not None:
        replay_buffer = shared_eval_data["replay_buffer"]
        split_indices = shared_eval_data["split_indices"]
        gen_indices_by_source = shared_eval_data["gen_indices_by_source"]
        cprint("  Using shared test set", "green")
    else:
        # Per-model episode discovery
        episode_splits = load_episode_splits(checkpoint_path)
        if episode_splits is None:
            cprint("Error: No episode_splits.json found — cannot determine train/val split. "
                   "Refusing to evaluate without it (risk of testing on training data).", "red")
            return None

        expert_path = cfg.get("expert_dataset_path", None)
        rollout_path = cfg.get("rollout_dataset_path", None)
        if expert_path is None and training_config is not None:
            expert_path = training_config.get("expert_dataset_path", None)
        if rollout_path is None and training_config is not None:
            rollout_path = training_config.get("rollout_dataset_path", None)
        if expert_path is None:
            cprint("Error: No expert_dataset_path configured", "red")
            return None

        eval_episode_paths, gen_episodes_by_source = find_test_episodes(
            episode_splits, expert_path, rollout_path,
        )
        total_eval = sum(len(v) for v in eval_episode_paths.values())
        if total_eval == 0:
            cprint("No test episodes found!", "red")
            return None

        replay_buffer, split_indices, gen_indices_by_source = load_eval_replay_buffer(
            eval_episode_paths, gen_episodes_by_source=gen_episodes_by_source
        )
        if replay_buffer is None:
            cprint("Failed to load evaluation episodes into ReplayBuffer", "red")
            return None

        # Shuffle
        seed = cfg.get("seed", 42)
        rng = np.random.RandomState(seed)
        for split_name in split_indices:
            rng.shuffle(split_indices[split_name])
        for source in gen_indices_by_source:
            for cat in gen_indices_by_source[source]:
                rng.shuffle(gen_indices_by_source[source][cat])

        cprint("  Test set:", "green")
        _print_test_set_summary(split_indices, gen_indices_by_source)

    # Get evaluation settings
    num_expert_plots = cfg.get("num_expert_plots", 1)
    num_new_config_plots = cfg.get("num_new_config_plots", 1)
    max_steps = cfg.get("max_steps", 256)
    seed = cfg.get("seed", 42)
    compute_avg_error_vs_step = cfg.get("compute_avg_error_vs_step", True)
    num_episodes_for_avg_error = cfg.get("num_episodes_for_avg_error", None)

    # Windowed eval settings
    compute_windowed_eval = cfg.get("compute_windowed_eval", True)
    windowed_eval_steps = cfg.get("windowed_eval_steps", 6)
    windows_per_episode = cfg.get("windows_per_episode", 5)
    num_episodes_for_windowed = cfg.get("num_episodes_for_windowed", 100)

    # =========================================================================
    # Preload only the episode data we'll actually use (respecting limits)
    # =========================================================================
    compute_phase_eval = cfg.get("compute_phase_eval", False)
    num_episodes_for_phase = cfg.get("num_episodes_for_phase_eval", 100) if compute_phase_eval else 0

    all_episode_indices = set()

    # Split indices: individual plots + per-split avg error
    for split_name in split_indices:
        indices = split_indices[split_name]
        needed = 0
        # Individual plots
        if split_name == "expert":
            needed = max(needed, num_expert_plots)
        elif split_name == "new_config":
            needed = max(needed, num_new_config_plots)
        # Avg error vs step
        if compute_avg_error_vs_step:
            limit = num_episodes_for_avg_error if num_episodes_for_avg_error is not None else len(indices)
            needed = max(needed, limit)
        all_episode_indices.update(indices[:needed])

    # Generalization indices: windowed + phase + gen avg error
    for source in gen_indices_by_source:
        for cat in gen_indices_by_source[source]:
            indices = gen_indices_by_source[source][cat]
            needed = 0
            if compute_windowed_eval:
                limit = num_episodes_for_windowed if num_episodes_for_windowed is not None else len(indices)
                needed = max(needed, limit)
            if compute_phase_eval:
                limit = num_episodes_for_phase if num_episodes_for_phase is not None else len(indices)
                needed = max(needed, limit)
            if compute_avg_error_vs_step:
                limit = num_episodes_for_avg_error if num_episodes_for_avg_error is not None else len(indices)
                needed = max(needed, limit)
            all_episode_indices.update(indices[:needed])

    preloaded_data = preload_episodes(replay_buffer, sorted(all_episode_indices), device)

    # =========================================================================
    # Individual episode plots
    # =========================================================================
    split_plot_counts = {
        "expert": num_expert_plots,
        "new_config": num_new_config_plots,
    }

    for split_name, n_plots in split_plot_counts.items():
        indices = split_indices.get(split_name, [])
        if n_plots > 0 and indices:
            for i in range(min(n_plots, len(indices))):
                ep_idx = indices[i]
                result = evaluate_episode(
                    lwm, replay_buffer, ep_idx, device,
                    normalization_stats=normalization_stats,
                    max_steps=max_steps,
                    predict_agent_state=predict_agent_state,
                    compose_affordance_fn=compose_affordance_fn,
                    preloaded_data=preloaded_data,
                )
                if result is not None:
                    plot_img = create_combined_plot(result, split_name, ep_idx, predict_agent_state)
                    plot_path = output_dir / "individual_episodes" / split_name / f"episode_{ep_idx}.png"
                    plot_img.save(plot_path)

    # =========================================================================
    # Per-split average cumulative error vs step (full-horizon rollout from t=0)
    # =========================================================================
    split_avg_results = {}
    rollout_cache = {}

    if compute_avg_error_vs_step:
        for split_name in ["expert", "new_config"]:
            indices = split_indices.get(split_name, [])
            if not indices:
                continue

            episodes_to_use = indices
            if num_episodes_for_avg_error is not None:
                episodes_to_use = indices[:num_episodes_for_avg_error]

            cprint(f"  Full-horizon: {len(episodes_to_use)} {split_name} episodes", "cyan")
            evaluate_episodes_cached(
                lwm, replay_buffer, episodes_to_use, device,
                normalization_stats=normalization_stats,
                max_steps=max_steps,
                predict_agent_state=predict_agent_state,
                compose_affordance_fn=compose_affordance_fn,
                cache=rollout_cache,
                preloaded_data=preloaded_data,
            )
            split_avg_results[split_name] = aggregate_avg_error_from_cache(
                rollout_cache, episodes_to_use, data_type=split_name,
                predict_agent_state=predict_agent_state,
            )

    # =========================================================================
    # Windowed N-step evaluation (random starting points, real GT state)
    # =========================================================================
    all_windowed_by_source = {}

    if compute_windowed_eval:
        for source, gen_indices in gen_indices_by_source.items():
            has_gen_episodes = any(len(gen_indices.get(cat, [])) > 0 for cat in GEN_CATEGORIES)
            if not has_gen_episodes:
                continue

            source_label = SPLIT_LABELS.get(source, source)
            source_dir = output_dir / "windowed_eval" / source
            source_dir.mkdir(parents=True, exist_ok=True)

            # Collect all episodes for this source (limited per category)
            all_windowed_indices = set()
            limited_gen_indices = {}
            for cat in GEN_CATEGORIES:
                indices = gen_indices.get(cat, [])
                if num_episodes_for_windowed is not None:
                    indices = indices[:num_episodes_for_windowed]
                limited_gen_indices[cat] = indices
                all_windowed_indices.update(indices)

            if not all_windowed_indices:
                continue

            cprint(f"  Windowed {windowed_eval_steps}-step ({source_label}): "
                   f"{len(all_windowed_indices)} episodes, {windows_per_episode} windows/ep", "cyan")

            windowed_results = evaluate_windowed(
                lwm, replay_buffer, sorted(all_windowed_indices), device,
                normalization_stats=normalization_stats,
                window_steps=windowed_eval_steps,
                windows_per_episode=windows_per_episode,
                predict_agent_state=predict_agent_state,
                compose_affordance_fn=compose_affordance_fn,
                seed=seed,
                preloaded_data=preloaded_data,
            )

            # Aggregate by generalization category
            windowed_by_cat = {}
            for cat in GEN_CATEGORIES:
                indices = limited_gen_indices.get(cat, [])
                if not indices:
                    continue
                result = aggregate_windowed_results(
                    windowed_results, episode_indices=indices,
                    window_steps=windowed_eval_steps,
                )
                if result is not None:
                    windowed_by_cat[cat] = result

            if windowed_by_cat:
                cprint(f"  Windowed {windowed_eval_steps}-step summary ({source_label}):", "green")
                print_windowed_summary(windowed_by_cat, window_steps=windowed_eval_steps)

                create_windowed_eval_plot(
                    windowed_by_cat,
                    output_path=source_dir / f"windowed_{windowed_eval_steps}step.pdf",
                    window_steps=windowed_eval_steps,
                )

                create_one_step_summary_plot(
                    windowed_by_cat,
                    output_path=source_dir / "one_step_summary.pdf",
                )

            all_windowed_by_source[source] = windowed_by_cat

    # =========================================================================
    # Phase-based evaluation (reach / grasp / pull)
    # =========================================================================
    all_phase_by_source = {}

    compute_phase_eval = cfg.get("compute_phase_eval", False)
    if compute_phase_eval:
        phase_eval_steps = cfg.get("phase_eval_steps", 18)
        phase_proximity_cm = cfg.get("phase_proximity_threshold_cm", 1.0)
        phase_pull_cm = cfg.get("phase_pull_movement_threshold_cm", 0.5)
        num_episodes_for_phase = cfg.get("num_episodes_for_phase_eval", 100)

        # Per-model phase cache (persists across runs, keyed by episode path)
        phase_cache_path = output_dir / "phase_cache.pkl"
        phase_cache_params = {
            "window_steps": phase_eval_steps,
            "proximity_threshold_cm": phase_proximity_cm,
            "pull_movement_threshold_cm": phase_pull_cm,
        }
        phase_cache = _load_phase_cache(phase_cache_path, phase_cache_params)

        for source, gen_indices in gen_indices_by_source.items():
            has_gen_episodes = any(len(gen_indices.get(cat, [])) > 0 for cat in GEN_CATEGORIES)
            if not has_gen_episodes:
                continue

            source_label = SPLIT_LABELS.get(source, source)

            # Collect all episodes for this source (limited per category)
            all_phase_indices = set()
            limited_gen_indices = {}
            for cat in GEN_CATEGORIES:
                indices = gen_indices.get(cat, [])
                if num_episodes_for_phase is not None:
                    indices = indices[:num_episodes_for_phase]
                limited_gen_indices[cat] = indices
                all_phase_indices.update(indices)

            if not all_phase_indices:
                continue

            cprint(f"  Phase eval {phase_eval_steps}-step ({source_label}): "
                   f"{len(all_phase_indices)} episodes", "cyan")

            phase_results, phase_counts = evaluate_phase_based(
                lwm, replay_buffer, sorted(all_phase_indices), device,
                normalization_stats=normalization_stats,
                proximity_threshold_cm=phase_proximity_cm,
                window_steps=phase_eval_steps,
                pull_movement_threshold_cm=phase_pull_cm,
                predict_agent_state=predict_agent_state,
                compose_affordance_fn=compose_affordance_fn,
                seed=seed,
                phase_cache=phase_cache,
                preloaded_data=preloaded_data,
            )

            cprint(f"  Phase counts ({source_label}): "
                   f"reach={phase_counts['reach']}, "
                   f"grasp={phase_counts['grasp']}, "
                   f"pull={phase_counts['pull']}", "green")

            # Aggregate by generalization category and phase
            phase_by_cat = {}
            for cat in GEN_CATEGORIES:
                indices = limited_gen_indices.get(cat, [])
                if not indices:
                    continue
                cat_phases = {}
                for phase_name in PHASE_NAMES:
                    result = aggregate_phase_results(
                        phase_results, phase_name,
                        episode_indices=indices,
                        window_steps=phase_eval_steps,
                    )
                    if result is not None:
                        cat_phases[phase_name] = result
                        cprint(f"    {GEN_LABELS[cat]} / {PHASE_LABELS[phase_name]}: "
                               f"{result['num_windows']} windows", "white")
                if cat_phases:
                    phase_by_cat[cat] = cat_phases

            all_phase_by_source[source] = phase_by_cat

        # Save updated cache (accumulates across runs)
        _save_phase_cache(phase_cache_path, phase_cache)

    # =========================================================================
    # Generalization breakdown: full-horizon rollout (seen vs unseen)
    # =========================================================================
    all_gen_avg_results = {}

    if compute_avg_error_vs_step:
        for source, gen_indices in gen_indices_by_source.items():
            has_gen_episodes = any(len(gen_indices.get(cat, [])) > 0 for cat in GEN_CATEGORIES)
            if not has_gen_episodes:
                continue

            source_label = SPLIT_LABELS.get(source, source)
            source_dir = output_dir / "generalization" / source
            source_dir.mkdir(parents=True, exist_ok=True)

            gen_avg_results = {}
            for cat in GEN_CATEGORIES:
                indices = gen_indices.get(cat, [])
                if not indices:
                    continue
                episodes_to_use = indices
                if num_episodes_for_avg_error is not None:
                    episodes_to_use = indices[:num_episodes_for_avg_error]

                cprint(f"  Generalization ({source_label}, {GEN_LABELS[cat]}): "
                       f"{len(episodes_to_use)} episodes", "cyan")
                evaluate_episodes_cached(
                    lwm, replay_buffer, episodes_to_use, device,
                    normalization_stats=normalization_stats,
                    max_steps=max_steps,
                    predict_agent_state=predict_agent_state,
                    compose_affordance_fn=compose_affordance_fn,
                    cache=rollout_cache,
                    preloaded_data=preloaded_data,
                )
                gen_avg_results[cat] = aggregate_avg_error_from_cache(
                    rollout_cache, episodes_to_use, data_type=GEN_LABELS[cat],
                    predict_agent_state=predict_agent_state,
                )

            if any(v is not None for v in gen_avg_results.values()):
                create_generalization_avg_error_plot(
                    gen_avg_results,
                    output_path=source_dir / "drift_full_horizon.pdf",
                    title_suffix=f"Full Horizon -- {source_label}",
                )

            all_gen_avg_results[source] = gen_avg_results

    # =========================================================================
    # Combined 4-way plots (expert + interaction × seen + unseen)
    # =========================================================================
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    if compute_avg_error_vs_step and all_gen_avg_results:
        # Check that at least 2 sources have data
        sources_with_data = [s for s in ["expert", "new_config"]
                             if any(all_gen_avg_results.get(s, {}).get(c) is not None for c in GEN_CATEGORIES)]
        if len(sources_with_data) >= 1:
            create_combined_4way_drift_plot(
                all_gen_avg_results,
                output_path=combined_dir / "drift_combined_full_horizon.pdf",
            )

    if compute_windowed_eval and all_windowed_by_source:
        sources_with_data = [s for s in ["expert", "new_config"]
                             if any(all_windowed_by_source.get(s, {}).get(c) is not None for c in GEN_CATEGORIES)]
        if len(sources_with_data) >= 1:
            create_combined_4way_windowed_plot(
                all_windowed_by_source,
                output_path=combined_dir / f"windowed_{windowed_eval_steps}step_combined.pdf",
                window_steps=windowed_eval_steps,
            )

    # =========================================================================
    # Save evaluation summary JSON
    # =========================================================================
    num_params = sum(p.numel() for p in lwm.parameters())
    save_evaluation_summary(
        output_dir,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        num_params=num_params,
        split_avg_results=split_avg_results,
        gen_avg_results=all_gen_avg_results,
        windowed_by_source=all_windowed_by_source,
        phase_by_source=all_phase_by_source,
        test_set_hash=test_set_hash,
    )

    cprint(f"  Done — {output_dir}", "green")

    return {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "split_avg_results": split_avg_results,
        "gen_avg_results": all_gen_avg_results,
        "windowed_by_source": all_windowed_by_source,
        "phase_by_source": all_phase_by_source,
    }


if __name__ == "__main__":
    # Support --config <name> as alias for Hydra's --config-name=<name>
    # e.g. python -m lwm.world_model.analysis.eval_lwm_plots --config eval/no_agent_state.yaml
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            name = sys.argv[idx + 1].replace(".yaml", "")
            sys.argv.pop(idx + 1)
            sys.argv[idx] = f"--config-name={name}"
    main()
