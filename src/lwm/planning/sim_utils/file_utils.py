"""Utilities for file and directory operations."""
import os
import glob
import json
import yaml
import pickle
import numpy as np
from omegaconf import OmegaConf
from termcolor import cprint
from typing import Dict, List, Optional, Set, Tuple

# Goal-based object categories (must match robogen_dataset.py)
# Each category has different valid goal percentage ranges
GOAL_BASED_OBJECT_CATEGORIES = {
    "door": {
        "ids": ["45443", "35059", "45526", "45249", "46092", "45166", "46417", "47388"],
        "max_goal": 0.4,  # Valid range: 0.0 to 0.4
        "description": "Cabinet doors (sideways + downward)"
    },
    "drawer": {
        "ids": ["44853", "44817", "45092", "45132", "45135", "45243", "45290", "45374"],
        "max_goal": 0.8,  # Valid range: 0.0 to 0.8
        "description": "Drawer cabinets"
    }
}

# Build reverse lookup: object_id -> category
OBJECT_ID_TO_CATEGORY = {}
for category, info in GOAL_BASED_OBJECT_CATEGORIES.items():
    for obj_id in info["ids"]:
        OBJECT_ID_TO_CATEGORY[obj_id] = category


def get_object_category_by_id(object_id: str) -> Optional[str]:
    """Get the category of an object by its ID."""
    return OBJECT_ID_TO_CATEGORY.get(str(object_id))


def ensure_dir(path):
    """Ensure directory exists."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_object_name_from_config(config_file):
    """Extract object name from config file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    for config_dict in config:
        if "name" in config_dict:
            return config_dict["name"]
    return None


def _migrate_old_session_to_new_path(old_session_folder, new_session_folder):
    """
    Migrate a session folder from old path structure to new path structure.

    Old: save_path/{object_id}/experiment/{checkpoint_name}/{session_name}/
    New: save_path/{split}/{object_id}/experiment/{checkpoint_name}/{session_name}/
    """
    import shutil
    if os.path.exists(old_session_folder) and not os.path.exists(new_session_folder):
        os.makedirs(os.path.dirname(new_session_folder), exist_ok=True)
        shutil.move(old_session_folder, new_session_folder)
        return True
    return False


def _migrate_old_object_results_to_new_path(old_object_folder, new_object_folder, session_name=None):
    """
    Migrate opened_joint_angles.json entries for a specific session from old path to new path.
    Only migrates the specific session's data, not the entire file.

    Returns True if migration occurred.
    """
    import shutil
    old_json_path = os.path.join(old_object_folder, "opened_joint_angles.json")
    new_json_path = os.path.join(new_object_folder, "opened_joint_angles.json")

    if not os.path.exists(old_json_path):
        return False

    # Ensure new directory exists
    os.makedirs(new_object_folder, exist_ok=True)

    try:
        with open(old_json_path, "r") as f:
            old_results = json.load(f)

        if not isinstance(old_results, list):
            return False

        # Find entries matching the session_name
        entries_to_migrate = []
        entries_to_keep = []
        for entry in old_results:
            if session_name is None or entry.get("session_name") == session_name:
                entries_to_migrate.append(entry)
            else:
                entries_to_keep.append(entry)

        if not entries_to_migrate:
            return False

        # Load or create new results file
        if os.path.exists(new_json_path):
            with open(new_json_path, "r") as f:
                new_results = json.load(f)
            if not isinstance(new_results, list):
                new_results = []
        else:
            new_results = []

        # Add migrated entries (avoid duplicates)
        existing_configs = {r.get("config_file") for r in new_results}
        for entry in entries_to_migrate:
            if entry.get("config_file") not in existing_configs:
                new_results.append(entry)

        # Write updated new results
        with open(new_json_path, "w") as f:
            json.dump(new_results, f, indent=4)

        # Update old file to remove migrated entries
        if entries_to_keep:
            with open(old_json_path, "w") as f:
                json.dump(entries_to_keep, f, indent=4)
        else:
            # All entries migrated, remove old file
            os.remove(old_json_path)

        return True
    except (json.JSONDecodeError, IOError):
        return False


def check_completed_sessions(all_session_args_list, save_path, checkpoint_name):
    """
    Check which sessions are already completed by looking for results in opened_joint_angles.json files.
    Also checks legacy path structure (without split) and migrates if found.

    Matching is done by session_name first (more robust across different mount points),
    with config_file as a secondary check for disambiguation.

    Args:
        all_session_args_list: List of session argument dictionaries, each containing:
            - object_id: str
            - split: str
            - session_name: str
            - config_file: str
        save_path: Base save path for evaluation results
        checkpoint_name: Checkpoint name used in the save path

    Returns:
        completed_sessions: Set of tuples (object_id, session_name) for completed sessions
    """
    completed_sessions = set()
    migrated_count = 0

    # DEBUG: Track what we're checking
    debug_not_found = []
    debug_found = []
    debug_json_files_checked = set()

    for session_args in all_session_args_list:
        object_id = session_args["object_id"]
        split = session_args["split"]
        session_name = session_args["session_name"]
        config_file = session_args["config_file"]

        # New path: save_path/{split}/{object_id}/experiment/{checkpoint_name}/
        new_object_folder = os.path.join(save_path, split, str(object_id), "experiment", checkpoint_name)
        new_json_path = os.path.join(new_object_folder, "opened_joint_angles.json")

        # Old path (legacy): save_path/{object_id}/experiment/{checkpoint_name}/
        old_object_folder = os.path.join(save_path, str(object_id), "experiment", checkpoint_name)
        old_json_path = os.path.join(old_object_folder, "opened_joint_angles.json")

        # Check new path first
        found_in_new = False
        if os.path.exists(new_json_path):
            debug_json_files_checked.add(new_json_path)
            try:
                with open(new_json_path, "r") as f:
                    opened_joint_angles = json.load(f)
                if isinstance(opened_joint_angles, list):
                    # DEBUG: Log all session names in this JSON file (only once per file)
                    existing_sessions_in_file = [r.get("session_name", "NO_SESSION_NAME") for r in opened_joint_angles]

                    for existing_result in opened_joint_angles:
                        # Match by session_name first (more robust - handles different mount points)
                        # config_file paths can differ between runs (e.g. different storage roots)
                        existing_session = existing_result.get("session_name")
                        existing_config = existing_result.get("config_file", "")
                        if existing_session == session_name:
                            completed_sessions.add((object_id, session_name))
                            found_in_new = True
                            debug_found.append((object_id, session_name, "session_name match"))
                            break
                        # Fallback: match by config_file for backward compatibility
                        elif existing_config == config_file:
                            completed_sessions.add((object_id, session_name))
                            found_in_new = True
                            debug_found.append((object_id, session_name, "config_file match"))
                            break

                    if not found_in_new:
                        # DEBUG: Session not found in this file
                        debug_not_found.append({
                            "object_id": object_id,
                            "session_name": session_name,
                            "split": split,
                            "json_path": new_json_path,
                            "sessions_in_file": existing_sessions_in_file[:5],  # First 5
                            "total_in_file": len(existing_sessions_in_file),
                        })
            except json.JSONDecodeError:
                pass
        else:
            # JSON file doesn't exist
            debug_not_found.append({
                "object_id": object_id,
                "session_name": session_name,
                "split": split,
                "json_path": new_json_path,
                "reason": "JSON file does not exist",
            })

        # If not found in new path, check old path and migrate if found
        if not found_in_new and os.path.exists(old_json_path):
            try:
                with open(old_json_path, "r") as f:
                    opened_joint_angles = json.load(f)
                if isinstance(opened_joint_angles, list):
                    for existing_result in opened_joint_angles:
                        existing_session = existing_result.get("session_name")
                        existing_config = existing_result.get("config_file", "")
                        # Match by session_name or config_file
                        if existing_session == session_name or existing_config == config_file:
                            # Found in old path - migrate to new path
                            old_session_folder = os.path.join(old_object_folder, session_name)
                            new_session_folder = os.path.join(new_object_folder, session_name)

                            # Migrate session folder
                            _migrate_old_session_to_new_path(old_session_folder, new_session_folder)

                            # Migrate opened_joint_angles.json entry
                            _migrate_old_object_results_to_new_path(
                                old_object_folder, new_object_folder, session_name
                            )

                            completed_sessions.add((object_id, session_name))
                            migrated_count += 1
                            break
            except json.JSONDecodeError:
                pass

    if migrated_count > 0:
        from termcolor import cprint
        cprint(f"[Cache Migration] Migrated {migrated_count} sessions from old path structure to new split-based structure", "cyan")

    # Print cache summary (only when some sessions were found in cache)
    from termcolor import cprint
    if completed_sessions:
        cprint(f"[Cache] {len(completed_sessions)}/{len(all_session_args_list)} sessions found in cache, "
               f"{len(debug_not_found)} will be re-evaluated", "cyan")

    cprint("=" * 80, "yellow")

    return completed_sessions


def _extract_result_from_json(opened_joint_angles, config_file, session_name=None):
    """Helper to extract result dict from opened_joint_angles list.

    Matches by session_name first (more robust), then falls back to config_file.
    """
    if isinstance(opened_joint_angles, list):
        for existing_result in opened_joint_angles:
            existing_session = existing_result.get("session_name")
            existing_config = existing_result.get("config_file", "")
            # Match by session_name first (handles different mount points)
            # Then fallback to config_file for backward compatibility
            if (session_name is not None and existing_session == session_name) or existing_config == config_file:
                result = {
                    "final_door_joint_angle": existing_result["final_door_joint_angle"],
                    "expert_door_joint_angle": existing_result["expert_door_joint_angle"],
                    "initial_joint_angle": existing_result["initial_joint_angle"],
                    "ik_failure": existing_result["ik_failure"],
                    "grasped_handle": existing_result["grasped_handle"],
                }
                # Include goal-based metrics if available
                if "goal_static_affordance" in existing_result:
                    result["goal_static_affordance"] = existing_result["goal_static_affordance"]
                if "goal_joint_angle" in existing_result:
                    result["goal_joint_angle"] = existing_result["goal_joint_angle"]
                if "goal_percentage" in existing_result:
                    result["goal_percentage"] = existing_result["goal_percentage"]
                # Handle both key names: "final_affordance_point" and "affordance_point" (legacy)
                if "final_affordance_point" in existing_result:
                    result["final_affordance_point"] = existing_result["final_affordance_point"]
                elif "affordance_point" in existing_result:
                    result["final_affordance_point"] = existing_result["affordance_point"]
                if "distance_to_goal_cm" in existing_result:
                    result["distance_to_goal_cm"] = existing_result["distance_to_goal_cm"]
                return result
    return None


def load_cached_result(object_id, session_name, config_file, split, save_path, checkpoint_name):
    """
    Load a cached result from opened_joint_angles.json for a completed session.
    Also checks legacy path structure (without split) as fallback.

    Args:
        object_id: Object ID
        session_name: Session name
        config_file: Config file path
        split: Split name (train/val/unseen/unknown)
        save_path: Base save path for evaluation results
        checkpoint_name: Checkpoint name used in the save path

    Returns:
        Dictionary with cached result metrics, or None if not found
    """
    # New path: save_path/{split}/{object_id}/experiment/{checkpoint_name}/
    new_object_folder = os.path.join(save_path, split, str(object_id), "experiment", checkpoint_name)
    new_json_path = os.path.join(new_object_folder, "opened_joint_angles.json")

    # Old path (legacy): save_path/{object_id}/experiment/{checkpoint_name}/
    old_object_folder = os.path.join(save_path, str(object_id), "experiment", checkpoint_name)
    old_json_path = os.path.join(old_object_folder, "opened_joint_angles.json")

    # Check new path first
    if os.path.exists(new_json_path):
        try:
            with open(new_json_path, "r") as f:
                opened_joint_angles = json.load(f)
            result = _extract_result_from_json(opened_joint_angles, config_file, session_name=session_name)
            if result is not None:
                return result
        except (json.JSONDecodeError, IOError):
            pass

    # Fallback to old path (migration should have happened in check_completed_sessions,
    # but this handles edge cases)
    if os.path.exists(old_json_path):
        try:
            with open(old_json_path, "r") as f:
                opened_joint_angles = json.load(f)
            result = _extract_result_from_json(opened_joint_angles, config_file, session_name=session_name)
            if result is not None:
                return result
        except (json.JSONDecodeError, IOError):
            pass

    return None


def extract_minimal_worker_config(cfg):
    """Extract only the config values needed by worker processes to reduce serialization overhead.
    
    Returns a minimal dict with only the necessary config values instead of the full cfg object.
    This significantly reduces memory when using spawn multiprocessing.
    
    Args:
        cfg: Full OmegaConf config object
        
    Returns:
        Minimal OmegaConf config with only the values needed by worker processes
    """
    # Extract only the values actually used by workers
    minimal_cfg = {
        "n_obs_steps": cfg.n_obs_steps,
        "n_action_steps": cfg.get("n_action_steps", 4),
        "policy": {
            "use_static_affordance": cfg.policy.get("use_static_affordance", False),
            "use_wm": cfg.policy.get("use_wm", False),
            "n_future_steps": cfg.policy.get("n_future_steps", 0),
        },
        "task": {
            "env_runner": {
                "num_point_in_pc": cfg.task.env_runner.get("num_point_in_pc", None),
                "observation_mode": cfg.task.env_runner.observation_mode,
            },
            "dataset": {
                "use_part_point_cloud": cfg.task.dataset.get("use_part_point_cloud", False),
            },
        },
    }
    
    # Handle lwm_encoder_cfg (new) or act3d_encoder_cfg (backward compatibility) if it exists
    encoder_cfg = getattr(cfg.policy, "lwm_encoder_cfg", None) or getattr(cfg.policy, "act3d_encoder_cfg", None)
    if encoder_cfg is not None:
        minimal_cfg["policy"]["lwm_encoder_cfg"] = {
            "use_static_affordance": encoder_cfg.get("use_static_affordance", False),
            "use_goal_static_affordance": encoder_cfg.get("use_goal_static_affordance", False),
            "use_wm": encoder_cfg.get("use_wm", False),
            "n_future_steps": encoder_cfg.get("n_future_steps", 0),
        }

    # Add goal-based config from dataset
    minimal_cfg["task"]["dataset"]["use_goal_static_affordance"] = cfg.task.dataset.get("use_goal_static_affordance", False)

    # Convert to OmegaConf for compatibility with existing code that uses .get()
    return OmegaConf.create(minimal_cfg)


def load_goal_from_extracted_dataset(extracted_dataset_path, object_id, session_name, experiment_name="open_cabinet_goal_based"):
    """
    Load goal_static_affordance from the extracted dataset's last pickle.

    For goal-based episodes, the goal is the static_affordance from the final timestep
    of the demonstration trajectory.

    Args:
        extracted_dataset_path: Base path to extracted dataset (e.g., .../datasets/extracted/final_goal_based)
        object_id: Object ID (e.g., "44853")
        session_name: Session name, which for goal-based datasets includes the goal
                     (e.g., "44853_2025-12-21-14-37-42_goal_008")
        experiment_name: Experiment name subdirectory (default: "open_cabinet_goal_based")

    Returns:
        Dictionary with:
            - goal_static_affordance: np.ndarray of shape (3,) - the target affordance position
            - goal_percentage: float - the goal opening percentage (e.g., 0.08)
            - goal_joint_angle: float - the target joint angle (if available)
        Returns None if goal cannot be loaded.
    """
    # Construct episode directory path
    episode_dir = os.path.join(extracted_dataset_path, str(object_id), experiment_name, session_name)

    if not os.path.isdir(episode_dir):
        cprint(f"[GoalLoader] Episode directory not found: {episode_dir}", "yellow")
        return None

    # Find all pickle files and get the last one (highest index)
    pkl_files = glob.glob(os.path.join(episode_dir, "*.pkl"))
    if not pkl_files:
        cprint(f"[GoalLoader] No pickle files found in {episode_dir}", "yellow")
        return None

    # Extract indices and find the maximum
    pkl_indices = []
    for pkl_file in pkl_files:
        basename = os.path.basename(pkl_file)
        try:
            idx = int(basename.replace(".pkl", ""))
            pkl_indices.append((idx, pkl_file))
        except ValueError:
            continue

    if not pkl_indices:
        cprint(f"[GoalLoader] No valid pickle files found in {episode_dir}", "yellow")
        return None

    # Get the last pickle (highest index = goal state)
    last_idx, last_pkl_path = max(pkl_indices, key=lambda x: x[0])

    try:
        with open(last_pkl_path, "rb") as f:
            last_data = pickle.load(f)
    except Exception as e:
        cprint(f"[GoalLoader] Error loading {last_pkl_path}: {e}", "red")
        return None

    # Extract goal_static_affordance (last timestep's static_affordance)
    if "static_affordance" not in last_data:
        cprint(f"[GoalLoader] No static_affordance in {last_pkl_path}", "yellow")
        return None

    goal_static_affordance = last_data["static_affordance"]
    # Normalize shape to (3,)
    if hasattr(goal_static_affordance, "shape"):
        if len(goal_static_affordance.shape) > 1:
            goal_static_affordance = goal_static_affordance.squeeze()
        goal_static_affordance = goal_static_affordance.astype(np.float32)

    # Extract additional goal metadata from first pickle (where it's stored)
    goal_percentage = None
    goal_joint_angle = None
    first_pkl_path = os.path.join(episode_dir, "0.pkl")
    if os.path.exists(first_pkl_path):
        try:
            with open(first_pkl_path, "rb") as f:
                first_data = pickle.load(f)
            goal_percentage = first_data.get("goal_percentage")
            goal_joint_angle = first_data.get("goal_joint_angle")
        except Exception as e:
            cprint(f"[GoalLoader] Warning: Could not load metadata from {first_pkl_path}: {e}", "yellow")

    return {
        "goal_static_affordance": goal_static_affordance,
        "goal_percentage": goal_percentage,
        "goal_joint_angle": goal_joint_angle,
    }


def parse_goal_from_session_name(session_name):
    """
    Parse goal percentage from session name for goal-based episodes.

    Goal-based episode names follow the pattern:
        {object_id}_{timestamp}_goal_{goal_percentage_int}
    Example:
        "44853_2025-12-21-14-37-42_goal_008" -> goal_percentage = 0.08

    Args:
        session_name: Session name string

    Returns:
        goal_percentage (float) or None if not a goal-based episode
    """
    if "_goal_" not in session_name:
        return None

    try:
        # Extract the part after "_goal_"
        goal_part = session_name.split("_goal_")[-1]
        # Parse as integer (e.g., "008" -> 8) and convert to percentage
        goal_int = int(goal_part)
        goal_percentage = goal_int / 100.0
        return goal_percentage
    except (ValueError, IndexError):
        return None


def is_goal_based_episode(session_name):
    """Check if a session name indicates a goal-based episode."""
    return "_goal_" in session_name


def extract_base_session_name(session_name):
    """
    Extract the base session name (timestamp) from a session name.

    Handles multiple formats:
    1. Goal-based: {object_id}_{timestamp}_goal_{goal_percentage_int}
       Example: "44853_2025-12-21-14-37-42_goal_008" -> "2025-12-21-14-37-42"
    2. Non-goal-based with prefix: {object_id}_{timestamp}
       Example: "35059_2026-01-12-14-34-17" -> "2026-01-12-14-34-17"
    3. Plain timestamp: {timestamp}
       Example: "2026-01-12-14-34-17" -> "2026-01-12-14-34-17"

    Args:
        session_name: Session name string

    Returns:
        Base session name (timestamp portion)
    """
    # Handle goal-based sessions first
    if "_goal_" in session_name:
        # Remove the _goal_XXX suffix
        base_with_prefix = session_name.split("_goal_")[0]
    else:
        base_with_prefix = session_name

    # Remove the object_id prefix if present (format: {object_id}_{timestamp})
    # Object ID is numeric, timestamp starts with year (2025 or 2026)
    parts = base_with_prefix.split("_", 1)  # Split only on first underscore
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 5:
        # Object IDs are 5-digit numbers, timestamp follows after first underscore
        return parts[1]

    return base_with_prefix


def load_episodes_from_checkpoint_splits(checkpoint_dir, num_train=None, num_val=None, num_unseen=None, seed=42):
    """
    Load evaluation episodes from a checkpoint's episode_splits.json file.

    This allows evaluating on the exact same train/val split used during training,
    without needing to manually create an eval_episodes.json file.

    Args:
        checkpoint_dir: Path to the checkpoint directory (containing episode_splits.json
                       either directly or in a 'checkpoints' subdirectory)
        num_train: Number of training episodes to sample (None = all)
        num_val: Number of validation episodes to sample (None = all)
        num_unseen: Number of unseen episodes to sample (None = all, 0 = disabled)
        seed: Random seed for sampling (for reproducibility)

    Returns:
        Tuple of (eval_episodes_set, split_lookup, metadata) matching the format
        returned by load_eval_episodes(), or (None, None, None) if loading fails.

        - eval_episodes_set: Set of (object_id, session_name) tuples
        - split_lookup: Dict mapping (object_id, session_name) -> split
        - metadata: Dict with dataset info
    """
    import random

    # Find episode_splits.json
    splits_path = None
    for candidate in [
        os.path.join(checkpoint_dir, "episode_splits.json"),
        os.path.join(checkpoint_dir, "checkpoints", "episode_splits.json"),
    ]:
        if os.path.exists(candidate):
            splits_path = candidate
            break

    if splits_path is None:
        cprint(f"[EpisodeSplits] No episode_splits.json found in {checkpoint_dir}", "yellow")
        return None, None, None

    cprint(f"[EpisodeSplits] Loading from {splits_path}", "cyan")

    with open(splits_path, "r") as f:
        splits_data = json.load(f)

    train_episodes = splits_data.get("train_episodes", [])
    val_episodes = splits_data.get("val_episodes", [])
    unseen_episodes = splits_data.get("unseen_episodes", [])

    cprint(f"[EpisodeSplits] Found {len(train_episodes)} train, {len(val_episodes)} val, {len(unseen_episodes)} unseen episodes", "cyan")

    # Sample if requested
    rng = random.Random(seed)
    if num_train is not None and num_train < len(train_episodes):
        train_episodes = rng.sample(train_episodes, num_train)
        cprint(f"[EpisodeSplits] Sampled {num_train} train episodes", "cyan")
    if num_val is not None and num_val < len(val_episodes):
        val_episodes = rng.sample(val_episodes, num_val)
        cprint(f"[EpisodeSplits] Sampled {num_val} val episodes", "cyan")

    # Handle unseen episodes (only include if num_unseen > 0)
    if num_unseen is not None and num_unseen > 0:
        if num_unseen < len(unseen_episodes):
            unseen_episodes = rng.sample(unseen_episodes, num_unseen)
            cprint(f"[EpisodeSplits] Sampled {num_unseen} unseen episodes", "cyan")
        elif len(unseen_episodes) > 0:
            cprint(f"[EpisodeSplits] Using all {len(unseen_episodes)} unseen episodes", "cyan")
    else:
        # num_unseen is None or 0, disable unseen evaluation
        unseen_episodes = []

    # Build eval_episodes_set and split_lookup
    eval_episodes_set = set()
    split_lookup = {}

    def extract_session_name(episode_path):
        """Extract session name from episode path."""
        # episode_path is like "/tmp/training_data/open_cabinet/44817_2025-12-21-13-35-32"
        # or "/path/to/extracted/44817/open_cabinet_goal_based/44817_2025-12-21-13-35-32_goal_008"
        return os.path.basename(episode_path)

    for ep in train_episodes:
        object_id = str(ep["object_id"])
        session_name = extract_session_name(ep["episode_path"])
        eval_episodes_set.add((object_id, session_name))
        split_lookup[(object_id, session_name)] = "train"

    for ep in val_episodes:
        object_id = str(ep["object_id"])
        session_name = extract_session_name(ep["episode_path"])
        eval_episodes_set.add((object_id, session_name))
        split_lookup[(object_id, session_name)] = "val"

    # Handle unseen episodes (different format: uses "path" key instead of "episode_path")
    for ep in unseen_episodes:
        object_id = str(ep["object_id"])
        # unseen_episodes use "path" key instead of "episode_path"
        episode_path = ep.get("path") or ep.get("episode_path", "")
        session_name = extract_session_name(episode_path)
        eval_episodes_set.add((object_id, session_name))
        split_lookup[(object_id, session_name)] = "unseen"

    # Build metadata from splits_data
    metadata = {
        "source": "checkpoint_splits",
        "splits_path": splits_path,
        "num_train_episodes": len(train_episodes),
        "num_val_episodes": len(val_episodes),
        "num_unseen_episodes": len(unseen_episodes),
        "train_objects": splits_data.get("train_objects", []),
        "val_objects": splits_data.get("val_objects", []),
        "unseen_objects": splits_data.get("unseen_objects", []),
        "unseen_goal_percentages": splits_data.get("unseen_goal_percentages", []),
    }

    # Try to infer dataset info from episode paths
    if train_episodes:
        sample_path = train_episodes[0]["episode_path"]
        # Try to extract experiment_name from path
        # Path might be: /path/to/extracted/object_id/experiment_name/session_name
        parts = sample_path.split(os.sep)
        for i, part in enumerate(parts):
            if part in ["open_cabinet", "open_cabinet_goal_based", "open_door"]:
                metadata["experiment_name"] = part
                break

    cprint(f"[EpisodeSplits] Loaded {len(eval_episodes_set)} episodes for evaluation", "green")
    return eval_episodes_set, split_lookup, metadata


def auto_discover_episodes_from_dataset(
    extracted_dataset_path,
    num_train=None,
    num_val=None,
    num_unseen=None,
    val_ratio=0.2,
    seed=42,
    experiment_name=None,
):
    """
    Auto-discover episodes from an extracted dataset directory.

    This scans the dataset directory structure to find all available episodes,
    then splits them into train/val by object (to avoid data leakage) and
    samples the requested number of episodes.

    Note: Unseen episodes are not available in auto-discovery mode because they
    are specifically filtered during training based on goal percentages.

    Args:
        extracted_dataset_path: Path to extracted dataset (e.g., /path/to/extracted/final_goal_based)
        num_train: Number of train episodes to sample (None = all)
        num_val: Number of val episodes to sample (None = all)
        num_unseen: Number of unseen episodes (ignored in auto-discovery, unseen requires checkpoint splits)
        val_ratio: Fraction of objects to use for validation (default: 0.2)
        seed: Random seed for reproducibility
        experiment_name: Experiment name subdirectory (auto-detected if None)

    Returns:
        Tuple of (eval_episodes_set, split_lookup, metadata) or (None, None, None) if failed
    """
    import random

    if not os.path.isdir(extracted_dataset_path):
        cprint(f"[AutoDiscover] Dataset path not found: {extracted_dataset_path}", "red")
        return None, None, None

    # Warn if unseen episodes are requested (not available in auto-discovery)
    if num_unseen is not None and num_unseen > 0:
        cprint(f"[AutoDiscover] Warning: Unseen episodes not available in auto-discovery mode (requires checkpoint's episode_splits.json)", "yellow")

    rng = random.Random(seed)

    # Find all object directories
    object_dirs = []
    for entry in os.listdir(extracted_dataset_path):
        entry_path = os.path.join(extracted_dataset_path, entry)
        if os.path.isdir(entry_path) and entry.isdigit():
            object_dirs.append(entry)

    if not object_dirs:
        cprint(f"[AutoDiscover] No object directories found in {extracted_dataset_path}", "red")
        return None, None, None

    object_dirs = sorted(object_dirs)
    cprint(f"[AutoDiscover] Found {len(object_dirs)} objects: {object_dirs}", "cyan")

    # Auto-detect experiment_name from first object if not provided
    if experiment_name is None:
        first_obj_path = os.path.join(extracted_dataset_path, object_dirs[0])
        for entry in os.listdir(first_obj_path):
            if os.path.isdir(os.path.join(first_obj_path, entry)):
                experiment_name = entry
                break
        if experiment_name is None:
            cprint(f"[AutoDiscover] Could not auto-detect experiment_name", "red")
            return None, None, None
        cprint(f"[AutoDiscover] Auto-detected experiment_name: {experiment_name}", "cyan")

    # Split objects into train/val
    rng.shuffle(object_dirs)
    num_val_objects = max(1, int(len(object_dirs) * val_ratio))
    val_objects = set(object_dirs[:num_val_objects])
    train_objects = set(object_dirs[num_val_objects:])

    cprint(f"[AutoDiscover] Train objects ({len(train_objects)}): {sorted(train_objects)}", "cyan")
    cprint(f"[AutoDiscover] Val objects ({len(val_objects)}): {sorted(val_objects)}", "cyan")

    # Collect all episodes per object
    train_episodes = []  # List of (object_id, session_name)
    val_episodes = []

    # Non-episode directories to skip
    non_episode_dirs = {
        'distance_to_goal_plots', 'trajectory_visualizations',
        'contact_point_images', 'demo_rgbs', 'affordance_visualizations',
        'metadata', '__pycache__', '.ipynb_checkpoints', 'checkpoints',
        'step_visualizations', 'wandb', 'logs', '.git', 'cache'
    }

    for obj_id in sorted(object_dirs):
        exp_dir = os.path.join(extracted_dataset_path, obj_id, experiment_name)
        if not os.path.isdir(exp_dir):
            continue

        # Filter sessions: must be directories with pickle files, not system/viz directories
        sessions = []
        for s in os.listdir(exp_dir):
            session_path = os.path.join(exp_dir, s)
            if not os.path.isdir(session_path):
                continue
            if s in non_episode_dirs or s.startswith('.'):
                continue
            # Validate that this is an actual episode directory (contains pickle files)
            pkl_files = glob.glob(os.path.join(session_path, "*.pkl"))
            if pkl_files:
                sessions.append(s)

        if obj_id in val_objects:
            val_episodes.extend([(obj_id, s) for s in sessions])
        else:
            train_episodes.extend([(obj_id, s) for s in sessions])

    cprint(f"[AutoDiscover] Found {len(train_episodes)} train episodes, {len(val_episodes)} val episodes", "cyan")

    # Sample if requested
    if num_train is not None and num_train < len(train_episodes):
        # Sample evenly across objects
        train_episodes = _sample_balanced(train_episodes, num_train, rng)
        cprint(f"[AutoDiscover] Sampled {len(train_episodes)} train episodes", "cyan")

    if num_val is not None and num_val < len(val_episodes):
        val_episodes = _sample_balanced(val_episodes, num_val, rng)
        cprint(f"[AutoDiscover] Sampled {len(val_episodes)} val episodes", "cyan")

    # Build output
    eval_episodes_set = set(train_episodes + val_episodes)
    split_lookup = {}
    for obj_id, session in train_episodes:
        split_lookup[(obj_id, session)] = "train"
    for obj_id, session in val_episodes:
        split_lookup[(obj_id, session)] = "val"

    metadata = {
        "source": "auto_discover",
        "extracted_dataset_path": extracted_dataset_path,
        "experiment_name": experiment_name,
        "num_train_episodes": len(train_episodes),
        "num_val_episodes": len(val_episodes),
        "num_unseen_episodes": 0,  # Unseen not available in auto-discovery
        "train_objects": sorted(train_objects),
        "val_objects": sorted(val_objects),
        "unseen_objects": [],
    }

    cprint(f"[AutoDiscover] Ready to evaluate {len(eval_episodes_set)} episodes", "green")
    return eval_episodes_set, split_lookup, metadata


def _sample_balanced(episodes, num_samples, rng):
    """Sample episodes while trying to balance across objects."""
    from collections import defaultdict

    # Group by object
    by_object = defaultdict(list)
    for obj_id, session in episodes:
        by_object[obj_id].append((obj_id, session))

    # Round-robin sample from objects
    sampled = []
    obj_ids = list(by_object.keys())
    rng.shuffle(obj_ids)

    while len(sampled) < num_samples and any(by_object.values()):
        for obj_id in obj_ids:
            if by_object[obj_id] and len(sampled) < num_samples:
                sampled.append(by_object[obj_id].pop())

    return sampled


def find_goal_based_episode_path(extracted_dataset_path, object_id, base_session_name, goal_percentage, experiment_name="open_cabinet_goal_based"):
    """
    Find the full episode path for a goal-based episode given base session and goal percentage.

    This is useful when you have a base session name (timestamp) and want to find
    the corresponding goal-based episode directory.

    Args:
        extracted_dataset_path: Base path to extracted dataset
        object_id: Object ID
        base_session_name: Base session name (timestamp, e.g., "2025-12-21-14-37-42")
        goal_percentage: Goal percentage (e.g., 0.08)
        experiment_name: Experiment name subdirectory

    Returns:
        Full session name if found (e.g., "44853_2025-12-21-14-37-42_goal_008"), or None
    """
    # Convert goal_percentage to int representation (e.g., 0.08 -> "008")
    goal_int = int(goal_percentage * 100)
    goal_str = f"{goal_int:03d}"

    # Construct expected session name
    expected_session = f"{object_id}_{base_session_name}_goal_{goal_str}"
    expected_path = os.path.join(extracted_dataset_path, str(object_id), experiment_name, expected_session)

    if os.path.isdir(expected_path):
        return expected_session

    # Try finding any matching episode in the directory
    experiment_dir = os.path.join(extracted_dataset_path, str(object_id), experiment_name)
    if os.path.isdir(experiment_dir):
        for entry in os.listdir(experiment_dir):
            if base_session_name in entry and f"_goal_{goal_str}" in entry:
                return entry

    return None


# =============================================================================
# 4-Way Split Utilities (val, seen_unseen, unseen_seen, unseen_unseen)
# =============================================================================
#
# For official evaluation results, we use ONLY validation and unseen data:
# - val (seen_seen): Validation episodes with seen object + seen goal
# - seen_unseen: Seen object + unseen goal (filtered during training)
# - unseen_seen: Unseen object + seen goal (filtered during training)
# - unseen_unseen: Unseen object + unseen goal (filtered during training)
#
# NOTE: Training data is NEVER included in official results.

class FourWaySplitClassifier:
    """
    Classifier for 4-way episode splits based on object and goal visibility.

    Split definitions (for evaluation - excludes training data):
    - seen_seen: VALIDATION episodes where object AND goal were in training
    - seen_unseen: Object was in training BUT goal was NOT in training
    - unseen_seen: Object was NOT in training BUT goal type was in training
    - unseen_unseen: Object was NOT in training AND goal was NOT in training
    """

    SPLIT_NAMES = ["seen_seen", "seen_unseen", "unseen_seen", "unseen_unseen"]

    def __init__(
        self,
        seen_objects: Set[str],
        train_episodes: Set[Tuple[str, str]],  # Episodes used for training (to exclude)
        seen_seen_episodes: Set[Tuple[str, str]],    # Validation episodes (seen object + seen goal)
        allowed_goal_percentages_door: Optional[List[float]] = None,
        allowed_goal_percentages_drawer: Optional[List[float]] = None,
        allowed_goal_percentages: Optional[List[float]] = None,
        goal_percentage_tolerance: float = 0.02,
    ):
        """
        Initialize the 4-way split classifier.

        Args:
            seen_objects: Set of object IDs that were in training (train + val objects)
            train_episodes: Set of (object_id, session_name) tuples for training episodes (to exclude)
            seen_seen_episodes: Set of (object_id, session_name) tuples for validation episodes (seen_seen)
            allowed_goal_percentages_door: List of goal percentages allowed for door objects during training
            allowed_goal_percentages_drawer: List of goal percentages allowed for drawer objects during training
            allowed_goal_percentages: Global list of allowed goal percentages (fallback)
            goal_percentage_tolerance: Tolerance for matching goal percentages
        """
        self.seen_objects = set(str(obj_id) for obj_id in seen_objects)
        self.train_episodes = train_episodes
        self.seen_seen_episodes = seen_seen_episodes
        self.allowed_goal_percentages_door = allowed_goal_percentages_door
        self.allowed_goal_percentages_drawer = allowed_goal_percentages_drawer
        self.allowed_goal_percentages = allowed_goal_percentages
        self.goal_percentage_tolerance = goal_percentage_tolerance

    def is_goal_seen(self, goal_percentage: Optional[float], object_id: str) -> bool:
        """
        Check if a goal percentage was seen during training for the given object.

        Args:
            goal_percentage: The goal percentage (e.g., 0.08)
            object_id: The object ID to determine category

        Returns:
            True if this goal was in training, False otherwise
        """
        if goal_percentage is None:
            # Non-goal-based episode - treat as "seen" since no goal filtering was applied
            return True

        # Get object category
        category = get_object_category_by_id(object_id)

        # Determine which allowed list to use
        allowed_pcts = None
        if category == "door" and self.allowed_goal_percentages_door is not None:
            allowed_pcts = self.allowed_goal_percentages_door
        elif category == "drawer" and self.allowed_goal_percentages_drawer is not None:
            allowed_pcts = self.allowed_goal_percentages_drawer
        elif self.allowed_goal_percentages is not None:
            allowed_pcts = self.allowed_goal_percentages

        if allowed_pcts is None:
            # No goal filtering was applied during training - all goals are "seen"
            return True

        # Check if goal matches any allowed percentage
        for allowed_pct in allowed_pcts:
            if abs(goal_percentage - allowed_pct) <= self.goal_percentage_tolerance:
                return True

        return False

    def is_object_seen(self, object_id: str) -> bool:
        """Check if an object was seen during training."""
        return str(object_id) in self.seen_objects

    def is_train_episode(self, object_id: str, session_name: str) -> bool:
        """Check if an episode was used for training (should be excluded from official results)."""
        return (str(object_id), session_name) in self.train_episodes

    def is_seen_seen_episode(self, object_id: str, session_name: str) -> bool:
        """Check if an episode was in the seen_seen set (validation: seen object + seen goal)."""
        return (str(object_id), session_name) in self.seen_seen_episodes

    def classify(self, object_id: str, session_name: str, goal_percentage: Optional[float]) -> Optional[str]:
        """
        Classify an episode into one of the 4 splits.

        Args:
            object_id: Object ID
            session_name: Session name
            goal_percentage: Goal percentage (can be None for non-goal-based episodes)

        Returns:
            One of: "seen_seen", "seen_unseen", "unseen_seen", "unseen_unseen"
            Returns None if this is a training episode (should be excluded)
        """
        # Exclude training episodes from official results
        if self.is_train_episode(object_id, session_name):
            return None  # Skip training data

        # Check if this is a seen_seen episode (validation: seen object + seen goal)
        if self.is_seen_seen_episode(object_id, session_name):
            return "seen_seen"

        # For other episodes (filtered out during training), classify by object/goal
        obj_seen = self.is_object_seen(object_id)
        goal_seen = self.is_goal_seen(goal_percentage, object_id)

        if obj_seen and not goal_seen:
            return "seen_unseen"
        elif not obj_seen and goal_seen:
            return "unseen_seen"
        elif not obj_seen and not goal_seen:
            return "unseen_unseen"
        else:
            # seen object + seen goal but not in seen_seen_episodes
            # This could happen if new episodes were added after training
            return "seen_seen"

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> Optional["FourWaySplitClassifier"]:
        """
        Create a classifier from a checkpoint directory.

        Loads:
        - episode_splits.json for seen objects and train/val episodes
        - .hydra/config.yaml for allowed goal percentages

        Args:
            checkpoint_dir: Path to checkpoint directory

        Returns:
            FourWaySplitClassifier instance, or None if loading fails
        """
        # Find episode_splits.json
        splits_path = None
        for candidate in [
            os.path.join(checkpoint_dir, "episode_splits.json"),
            os.path.join(checkpoint_dir, "checkpoints", "episode_splits.json"),
        ]:
            if os.path.exists(candidate):
                splits_path = candidate
                break

        if splits_path is None:
            cprint(f"[FourWaySplit] No episode_splits.json found in {checkpoint_dir}", "yellow")
            return None

        with open(splits_path, "r") as f:
            splits_data = json.load(f)

        # Get seen objects from expert training only (ignore MPC data augmentation)
        # Use expert_train_objects if available, fallback to train_objects for backwards compat
        expert_train_objects = splits_data.get("expert_train_objects")
        if expert_train_objects is not None:
            seen_objects = set(str(o) for o in expert_train_objects)
        else:
            # Backwards compat: derive expert objects by filtering MPC from train_episodes
            seen_objects = set()
            for ep in splits_data.get("train_episodes", []):
                path = ep.get("episode_path", "")
                if ep.get("is_mpc", False) or "/mpc_results/" in path:
                    continue
                seen_objects.add(str(ep["object_id"]))
            if not seen_objects:
                # Ultimate fallback: use all train_objects
                seen_objects = set(str(o) for o in splits_data.get("train_objects", []))
        seen_objects.update(str(o) for o in splits_data.get("val_objects", []))

        # Build train and seen_seen episode sets (exclude MPC from train exclusion)
        train_episodes = set()
        seen_seen_episodes = set()
        mpc_skipped = 0

        for ep in splits_data.get("train_episodes", []):
            path = ep.get("episode_path", "")
            if ep.get("is_mpc", False) or "/mpc_results/" in path:
                mpc_skipped += 1
                continue  # MPC episodes don't count as training data for eval splits
            obj_id = str(ep["object_id"])
            session_name = os.path.basename(path)
            train_episodes.add((obj_id, session_name))

        for ep in splits_data.get("val_episodes", []):
            obj_id = str(ep["object_id"])
            session_name = os.path.basename(ep["episode_path"])
            seen_seen_episodes.add((obj_id, session_name))

        cprint(f"[FourWaySplit] Expert train episodes to exclude: {len(train_episodes)} (skipped {mpc_skipped} MPC episodes)", "cyan")
        cprint(f"[FourWaySplit] Seen_seen episodes: {len(seen_seen_episodes)}", "cyan")

        # Load training config for goal percentages
        config_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")
        allowed_goal_percentages_door = None
        allowed_goal_percentages_drawer = None
        allowed_goal_percentages = None
        goal_percentage_tolerance = 0.02

        if os.path.exists(config_path):
            try:
                cfg = OmegaConf.load(config_path)
                dataset_cfg = cfg.task.dataset
                allowed_goal_percentages_door = dataset_cfg.get("allowed_goal_percentages_door", None)
                allowed_goal_percentages_drawer = dataset_cfg.get("allowed_goal_percentages_drawer", None)
                allowed_goal_percentages = dataset_cfg.get("allowed_goal_percentages", None)
                goal_percentage_tolerance = dataset_cfg.get("goal_percentage_tolerance", 0.02)

                # Convert OmegaConf lists to Python lists
                if allowed_goal_percentages_door is not None:
                    allowed_goal_percentages_door = list(allowed_goal_percentages_door)
                if allowed_goal_percentages_drawer is not None:
                    allowed_goal_percentages_drawer = list(allowed_goal_percentages_drawer)
                if allowed_goal_percentages is not None:
                    allowed_goal_percentages = list(allowed_goal_percentages)

            except Exception as e:
                cprint(f"[FourWaySplit] Warning: Could not load config from {config_path}: {e}", "yellow")

        cprint(f"[FourWaySplit] Loaded classifier with {len(seen_objects)} seen objects", "cyan")
        if allowed_goal_percentages_door:
            cprint(f"[FourWaySplit] Door goals (seen): {allowed_goal_percentages_door}", "cyan")
        if allowed_goal_percentages_drawer:
            cprint(f"[FourWaySplit] Drawer goals (seen): {allowed_goal_percentages_drawer}", "cyan")

        return cls(
            seen_objects=seen_objects,
            train_episodes=train_episodes,
            seen_seen_episodes=seen_seen_episodes,
            allowed_goal_percentages_door=allowed_goal_percentages_door,
            allowed_goal_percentages_drawer=allowed_goal_percentages_drawer,
            allowed_goal_percentages=allowed_goal_percentages,
            goal_percentage_tolerance=goal_percentage_tolerance,
        )


def load_episodes_with_4way_splits(
    checkpoint_dir: str,
    extracted_dataset_path: str,
    num_seen_seen: Optional[int] = None,
    num_seen_unseen: Optional[int] = None,
    num_unseen_seen: Optional[int] = None,
    num_unseen_unseen: Optional[int] = None,
    seed: int = 42,
    experiment_name: str = "open_cabinet_goal_based",
) -> Tuple[Optional[Set[Tuple[str, str]]], Optional[Dict[Tuple[str, str], str]], Optional[Dict]]:
    """
    Load evaluation episodes with 4-way split classification.

    IMPORTANT: This function excludes training data - only validation and unseen
    episodes are returned for official evaluation results.

    Splits:
    - seen_seen: Validation episodes (seen object + seen goal, held out during training)
    - seen_unseen: Seen object + unseen goal (filtered during training)
    - unseen_seen: Unseen object + seen goal (filtered during training)
    - unseen_unseen: Unseen object + unseen goal (filtered during training)

    Args:
        checkpoint_dir: Path to checkpoint directory
        extracted_dataset_path: Path to extracted dataset (e.g., /path/to/extracted/final_goal_based)
        num_seen_seen: Number of seen_seen episodes to sample (None = all)
        num_seen_unseen: Number of seen_unseen episodes to sample (None = all)
        num_unseen_seen: Number of unseen_seen episodes to sample (None = all)
        num_unseen_unseen: Number of unseen_unseen episodes to sample (None = all)
        seed: Random seed for reproducibility
        experiment_name: Experiment name subdirectory

    Returns:
        Tuple of (eval_episodes_set, split_lookup, metadata) or (None, None, None) if failed
        - eval_episodes_set: Set of (object_id, session_name) tuples
        - split_lookup: Dict mapping (object_id, session_name) -> 4-way split name
        - metadata: Dict with split statistics and configuration
    """
    import random

    # Create classifier from checkpoint
    classifier = FourWaySplitClassifier.from_checkpoint(checkpoint_dir)
    if classifier is None:
        return None, None, None

    if not os.path.isdir(extracted_dataset_path):
        cprint(f"[4WaySplit] Dataset path not found: {extracted_dataset_path}", "red")
        return None, None, None

    rng = random.Random(seed)

    # Collect all episodes from extracted dataset and classify them
    episodes_by_split = {split: [] for split in FourWaySplitClassifier.SPLIT_NAMES}
    skipped_train = 0

    # Try nested structure first: <object_id>/<experiment_name>/<session>
    found_nested = False
    for obj_entry in os.listdir(extracted_dataset_path):
        obj_path = os.path.join(extracted_dataset_path, obj_entry)
        if not os.path.isdir(obj_path) or not obj_entry.isdigit():
            continue

        found_nested = True
        object_id = obj_entry
        exp_dir = os.path.join(obj_path, experiment_name)

        if not os.path.isdir(exp_dir):
            continue

        # Find all sessions
        for session_name in os.listdir(exp_dir):
            session_path = os.path.join(exp_dir, session_name)
            if not os.path.isdir(session_path):
                continue

            # Skip non-episode directories (visualizations, metadata, system files, etc.)
            non_episode_dirs = {
                'distance_to_goal_plots', 'trajectory_visualizations',
                'contact_point_images', 'demo_rgbs', 'affordance_visualizations',
                'metadata', '__pycache__', '.ipynb_checkpoints', 'checkpoints',
                'step_visualizations', 'wandb', 'logs', '.git', 'cache'
            }
            if session_name in non_episode_dirs or session_name.startswith('.'):
                continue

            # Validate that this is an actual episode directory (contains pickle files)
            pkl_files = glob.glob(os.path.join(session_path, "*.pkl"))
            if not pkl_files:
                continue

            # Extract goal percentage from session name
            goal_percentage = parse_goal_from_session_name(session_name)

            # Classify the episode
            split = classifier.classify(object_id, session_name, goal_percentage)

            if split is None:
                # Training episode - skip
                skipped_train += 1
                continue

            episodes_by_split[split].append((object_id, session_name, goal_percentage))

    # If no nested structure found, try flat structure: <experiment_name>/<objectid_timestamp>
    if not found_nested:
        exp_dir = os.path.join(extracted_dataset_path, experiment_name)
        if os.path.isdir(exp_dir):
            cprint(f"[4WaySplit] Using flat directory structure: {exp_dir}", "cyan")
            for entry in os.listdir(exp_dir):
                entry_path = os.path.join(exp_dir, entry)
                if not os.path.isdir(entry_path):
                    continue

                # Skip non-episode directories (visualizations, metadata, system files, etc.)
                non_episode_dirs = {
                    'distance_to_goal_plots', 'trajectory_visualizations',
                    'contact_point_images', 'demo_rgbs', 'affordance_visualizations',
                    'metadata', '__pycache__', '.ipynb_checkpoints', 'checkpoints',
                    'step_visualizations', 'wandb', 'logs', '.git', 'cache'
                }
                if entry in non_episode_dirs or entry.startswith('.'):
                    continue

                # Validate that this is an actual episode directory (contains pickle files)
                pkl_files = glob.glob(os.path.join(entry_path, "*.pkl"))
                if not pkl_files:
                    continue

                # Parse objectid_timestamp format (e.g., "35059_2026-01-12-14-34-17")
                if "_" in entry:
                    parts = entry.split("_", 1)
                    if parts[0].isdigit():
                        object_id = parts[0]
                        session_name = entry  # Use full entry as session name for matching

                        # Extract goal percentage from session name
                        goal_percentage = parse_goal_from_session_name(session_name)

                        # Classify the episode
                        split = classifier.classify(object_id, session_name, goal_percentage)

                        if split is None:
                            # Training episode - skip
                            skipped_train += 1
                            continue

                        episodes_by_split[split].append((object_id, session_name, goal_percentage))

    # Log statistics
    cprint(f"[4WaySplit] Skipped {skipped_train} training episodes", "cyan")
    cprint(f"[4WaySplit] Episode counts by split:", "cyan")
    for split in FourWaySplitClassifier.SPLIT_NAMES:
        cprint(f"  {split}: {len(episodes_by_split[split])}", "cyan")

    # Sample from each split
    split_counts = {
        "seen_seen": num_seen_seen,
        "seen_unseen": num_seen_unseen,
        "unseen_seen": num_unseen_seen,
        "unseen_unseen": num_unseen_unseen,
    }

    sampled_episodes = {split: [] for split in FourWaySplitClassifier.SPLIT_NAMES}

    for split, count in split_counts.items():
        available = episodes_by_split[split]
        if count is None or count >= len(available):
            sampled_episodes[split] = available
        elif count == 0:
            sampled_episodes[split] = []
        else:
            # Balanced sampling across objects
            sampled_episodes[split] = _sample_balanced_4way(available, count, rng)

        if count is not None and count > 0 and len(sampled_episodes[split]) < count:
            cprint(f"[4WaySplit] Warning: Only {len(sampled_episodes[split])} episodes available for {split} (requested {count})", "yellow")

    # Build output
    eval_episodes_set = set()
    split_lookup = {}

    for split in FourWaySplitClassifier.SPLIT_NAMES:
        for obj_id, session_name, goal_pct in sampled_episodes[split]:
            eval_episodes_set.add((obj_id, session_name))
            split_lookup[(obj_id, session_name)] = split

    # Build metadata
    metadata = {
        "source": "4way_split",
        "checkpoint_dir": checkpoint_dir,
        "extracted_dataset_path": extracted_dataset_path,
        "experiment_name": experiment_name,
        "total_available": {split: len(episodes_by_split[split]) for split in FourWaySplitClassifier.SPLIT_NAMES},
        "sampled_counts": {split: len(sampled_episodes[split]) for split in FourWaySplitClassifier.SPLIT_NAMES},
        "skipped_train_episodes": skipped_train,
        "seen_objects": sorted(classifier.seen_objects),
        "allowed_goal_percentages_door": classifier.allowed_goal_percentages_door,
        "allowed_goal_percentages_drawer": classifier.allowed_goal_percentages_drawer,
    }

    cprint(f"[4WaySplit] Ready to evaluate {len(eval_episodes_set)} episodes (excluding training data):", "green")
    for split in FourWaySplitClassifier.SPLIT_NAMES:
        cprint(f"  {split}: {len(sampled_episodes[split])}", "green")

    return eval_episodes_set, split_lookup, metadata


def _sample_balanced_4way(episodes: List[Tuple[str, str, Optional[float]]], num_samples: int, rng) -> List[Tuple[str, str, Optional[float]]]:
    """Sample episodes while trying to balance across objects."""
    from collections import defaultdict

    # Group by object
    by_object = defaultdict(list)
    for ep in episodes:
        by_object[ep[0]].append(ep)

    # Shuffle within each object
    for obj_id in by_object:
        rng.shuffle(by_object[obj_id])

    # Round-robin sample from objects
    sampled = []
    obj_ids = list(by_object.keys())
    rng.shuffle(obj_ids)

    while len(sampled) < num_samples and any(by_object.values()):
        for obj_id in obj_ids:
            if by_object[obj_id] and len(sampled) < num_samples:
                sampled.append(by_object[obj_id].pop())

    return sampled


def get_4way_split_from_legacy(
    legacy_split: str,
    object_id: str,
    session_name: str,
    goal_percentage: Optional[float],
    classifier: FourWaySplitClassifier,
) -> Optional[str]:
    """
    Convert a legacy split (train/val/unseen) to a 4-way split.

    This is useful for post-processing existing results.

    Args:
        legacy_split: Original split name (train, val, unseen, unknown)
        object_id: Object ID
        session_name: Session name
        goal_percentage: Goal percentage (can be None)
        classifier: FourWaySplitClassifier instance

    Returns:
        4-way split name, or None if training episode (should be excluded)
    """
    return classifier.classify(object_id, session_name, goal_percentage)

