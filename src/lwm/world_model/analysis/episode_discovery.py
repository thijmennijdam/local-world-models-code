"""Episode scanning, classification, and test episode discovery."""

import hashlib
import os
import re

from termcolor import cprint

from lwm.world_model.analysis.constants import (
    SEEN_OBJECT_IDS, OBJECT_CATEGORIES, GEN_CATEGORIES, GEN_LABELS,
)
from lwm.world_model.analysis.checkpoint import load_episode_splits
from lwm.world_model import packed
from lwm.world_model.replay_buffer import ReplayBuffer


def _is_episode_dir(dir_path):
    """Check if a directory contains step pickle files (0.pkl, 1.pkl, ...)."""
    if not os.path.isdir(dir_path):
        return False
    for f in os.listdir(dir_path):
        if re.match(r'^\d+\.pkl$', f):
            return True
    return False


def collect_episode_paths(dataset_path):
    """
    Recursively find all episode directories under a dataset path.
    Standalone version of LWMDataset._collect_episode_paths. Accepts packed datasets
    (index.json; virtual episode paths, see lwm.world_model.packed).
    """
    if dataset_path is None or not os.path.isdir(dataset_path):
        return []

    exclude_names = {"contact_point_images", "demo_rgbs", "trajectory_visualizations"}
    if packed.is_packed_dir(dataset_path):
        return packed.episode_paths(dataset_path, exclude_names=exclude_names, max_depth=5)

    def find_episodes_recursive(dir_path, max_depth=5, current_depth=0):
        if current_depth > max_depth:
            return []
        if _is_episode_dir(dir_path):
            return [dir_path]
        episode_paths = []
        try:
            subdirs = sorted(os.listdir(dir_path))
        except PermissionError:
            return []
        subdirs = [f for f in subdirs if f not in exclude_names and not f.startswith('.')]
        for subdir in subdirs:
            subdir_path = os.path.join(dir_path, subdir)
            if os.path.isdir(subdir_path):
                episode_paths.extend(find_episodes_recursive(subdir_path, max_depth, current_depth + 1))
        return episode_paths

    return find_episodes_recursive(dataset_path, max_depth=5, current_depth=0)


def classify_episode(episode_path):
    """Classify an episode into one of: 'expert', 'replay', 'new_config'."""
    path_lower = episode_path.lower()
    # Interaction (rollout) episodes: under a "rollout" dir on the training cluster, under
    # "epoch-<k>/" in the released Hugging Face data.
    is_rollout = "rollout" in path_lower or os.path.basename(os.path.dirname(episode_path.rstrip("/"))).startswith("epoch-")
    if not is_rollout:
        return "expert"
    name = os.path.basename(episode_path)
    if "_new_config_" in name:
        return "new_config"
    return "replay"


def extract_object_id(episode_path):
    """Extract object ID from episode directory name (format: {obj_id}_{session})."""
    name = os.path.basename(episode_path)
    parts = name.split('_')
    if parts[0] == 'obj':
        return parts[1]
    return parts[0]


def get_object_category(object_id):
    """Return 'door' or 'drawer' for an object ID."""
    for cat_name, cat_info in OBJECT_CATEGORIES.items():
        if str(object_id) in cat_info["ids"]:
            return cat_name
    return None


def classify_generalization(episode_path):
    """
    Classify an episode as seen or unseen object.

    Returns 'seen_obj' or 'unseen_obj'.
    """
    obj_id = extract_object_id(episode_path)
    return "seen_obj" if str(obj_id) in SEEN_OBJECT_IDS else "unseen_obj"


def episode_key(path):
    """Machine-independent episode id: "<epoch-dir>/<name>" for interaction episodes, else "<name>".

    episode_splits.json stores absolute paths from the training machine; matching on this key
    keeps train/val exclusion correct wherever the data is extracted.
    """
    path = str(path).rstrip("/")
    parent = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)
    return f"{parent}/{name}" if parent.startswith("epoch-") else name


def compute_test_intersection_exclusions(checkpoint_paths):
    """
    Compute the union of train+val episodes across all checkpoints.

    The intersection of test sets equals: all_episodes - union(all known paths).
    By passing this union as the exclusion set to find_test_episodes, every
    model gets evaluated on the exact same test episodes.

    Returns:
        set of episode paths to exclude, or None if any checkpoint lacks splits.
    """
    union_known = set()
    per_model_counts = []

    for cp_path in checkpoint_paths:
        splits = load_episode_splits(cp_path)
        if splits is None:
            cprint(f"Warning: No episode_splits.json for {cp_path}, "
                   "cannot compute intersection", "yellow")
            return None

        model_known = set()
        for split_key in ["train_episodes", "val_episodes"]:
            for ep in splits.get(split_key, []):
                model_known.add(episode_key(ep["episode_path"]))
        per_model_counts.append(len(model_known))
        union_known |= model_known

    cprint(f"Intersection mode: excluding {len(union_known)} episodes "
           f"(union of {len(per_model_counts)} models' train+val)", "magenta")

    return union_known


def compute_test_set_hash(eval_episode_paths):
    """Compute a deterministic hash of the test episode paths.

    Used to verify that cached evaluation results were computed on the
    same test set as the current intersection configuration.

    Args:
        eval_episode_paths: dict of {split_name: [episode_path, ...]}

    Returns:
        16-character hex digest string.
    """
    all_paths = []
    for split_paths in eval_episode_paths.values():
        all_paths.extend(split_paths)
    all_paths.sort()
    content = "\n".join(all_paths)
    return hashlib.sha256(content.encode()).hexdigest()


def find_test_episodes(episode_splits, expert_dataset_path, rollout_dataset_path,
                       additional_exclusions=None):
    """
    Find episodes NOT in train or val splits (true held-out test set).

    Args:
        additional_exclusions: Optional set of extra episode keys (see episode_key) to exclude.
            Used by test_intersection mode to exclude episodes that are in
            ANY model's train/val set, ensuring identical test sets across models.

    Returns:
        tuple of (test_episodes_by_split, gen_episodes_by_source)
    """
    known_paths = set()
    for split_key in ["train_episodes", "val_episodes"]:
        for ep in episode_splits.get(split_key, []):
            known_paths.add(episode_key(ep["episode_path"]))

    if additional_exclusions is not None:
        known_paths = known_paths | additional_exclusions

    all_expert = collect_episode_paths(expert_dataset_path)
    all_rollout = collect_episode_paths(rollout_dataset_path)

    test_episodes_by_split = {"expert": [], "replay": [], "new_config": []}

    for p in all_expert:
        if episode_key(p) not in known_paths:
            test_episodes_by_split["expert"].append(p)
    for p in all_rollout:
        if episode_key(p) not in known_paths:
            split = classify_episode(p)
            test_episodes_by_split[split].append(p)

    # Generalization: seen vs unseen object, per data source
    gen_episodes_by_source = {
        "expert": {cat: [] for cat in GEN_CATEGORIES},
        "new_config": {cat: [] for cat in GEN_CATEGORIES},
        "replay": {cat: [] for cat in GEN_CATEGORIES},
    }

    for source_key in ["expert", "new_config", "replay"]:
        for p in test_episodes_by_split.get(source_key, []):
            gen_cat = classify_generalization(p)
            gen_episodes_by_source[source_key][gen_cat].append(p)

    return test_episodes_by_split, gen_episodes_by_source


def load_eval_replay_buffer(episode_paths_by_split, gen_episodes_by_source=None, is_pickle=True):
    """
    Load evaluation episodes into a ReplayBuffer.

    Returns:
        Tuple of (ReplayBuffer, split_indices, gen_indices_by_source)
    """
    all_paths = []
    split_assignments = []

    for split_name in ["expert", "replay", "new_config"]:
        paths = episode_paths_by_split.get(split_name, [])
        for p in paths:
            all_paths.append(p)
            split_assignments.append(split_name)

    # Add gen-only paths not already included (e.g. val expert for seen+seen)
    path_set = set(all_paths)
    if gen_episodes_by_source is not None:
        for source in gen_episodes_by_source:
            for cat in GEN_CATEGORIES:
                for p in gen_episodes_by_source[source].get(cat, []):
                    if p not in path_set:
                        all_paths.append(p)
                        split_assignments.append(f"val_{source}")
                        path_set.add(p)

    if not all_paths:
        return None, {}, {}

    cprint(f"  Loading {len(all_paths)} evaluation episodes into ReplayBuffer...", "yellow")

    keys = ["state", "action", "part_point_cloud", "static_affordance"]
    replay_buffer = ReplayBuffer.copy_from_multiple_path(
        all_paths,
        keys=keys,
        load_per_step=True,
        is_pickle=is_pickle,
        target_action="action",
    )

    split_indices = {"expert": [], "replay": [], "new_config": []}
    for idx, split_name in enumerate(split_assignments):
        if split_name in split_indices:
            split_indices[split_name].append(idx)

    gen_indices_by_source = {}
    if gen_episodes_by_source is not None:
        path_to_idx = {p: i for i, p in enumerate(all_paths)}
        for source in gen_episodes_by_source:
            gen_indices_by_source[source] = {cat: [] for cat in GEN_CATEGORIES}
            for cat in GEN_CATEGORIES:
                for p in gen_episodes_by_source[source].get(cat, []):
                    if p in path_to_idx:
                        gen_indices_by_source[source][cat].append(path_to_idx[p])

    return replay_buffer, split_indices, gen_indices_by_source
