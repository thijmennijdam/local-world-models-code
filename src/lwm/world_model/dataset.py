"""
Standalone dataset for LWM training.

For each sequence in an episode:
- Creates training pairs for each timestep t where we can predict horizon steps ahead
- Each pair contains:
  - Current observation at timestep t
  - Action horizon (actions from t to t+horizon-1)
  - Target: future static_affordance and agent_pos (from t+1 to t+horizon)
"""

import os
import hashlib
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from termcolor import cprint
from tqdm import tqdm

from lwm.world_model import packed
from lwm.world_model.replay_buffer import ReplayBuffer
from lwm.world_model.sampler import SequenceSampler
from lwm.world_model.targets import (
    check_frames,
    encode_affordance_delta,
    sample_scene_transform,
    transform_world_delta,
)


def resolve_dataset_path(path):
    """Return ``path`` unchanged; fail clearly if it looks like a HuggingFace repo ID.

    Datasets are not downloaded automatically. Download them first (see README),
    then pass the local directory.
    """
    if "/" in path and not os.path.exists(path):
        parts = path.split("/")
        if len(parts) == 2 and not os.path.isabs(path):
            raise FileNotFoundError(
                f"Dataset path '{path}' does not exist and looks like a HuggingFace repo ID. "
                "Download the dataset first (see README, 'Data') and pass the local directory."
            )
    return path


# ---------------------------------------------------------------------------
# Object categories (mirrors robogen_dataset.py for consistent filtering)
# ---------------------------------------------------------------------------
GOAL_BASED_OBJECT_CATEGORIES = {
    "door": {
        "ids": ["45443", "35059", "45526", "45249", "46092", "45166", "46417", "47388"],
        "max_goal": 0.4,
        "description": "Cabinet doors (sideways + downward)",
    },
    "drawer": {
        "ids": ["44853", "44817", "45092", "45132", "45135", "45243", "45290", "45374"],
        "max_goal": 0.8,
        "description": "Drawer cabinets",
    },
}


def get_object_category(path_or_object_id):
    """Return 'door', 'drawer', or None for an object ID or path containing one."""
    for cat_name, cat_info in GOAL_BASED_OBJECT_CATEGORIES.items():
        for oid in cat_info["ids"]:
            if oid in str(path_or_object_id):
                return cat_name
    return None


def rotation_transfer_6D_to_matrix(orient):
    """
    Convert 6D rotation representation to a 3x3 rotation matrix.
    Copied from `lwm/sim/utils.py` to avoid cross-package imports.
    """
    if isinstance(orient, (list, tuple)):
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rotation_transfer_matrix_to_6D(rotate_matrix):
    """
    Convert a 3x3 rotation matrix to 6D rotation representation.
    Copied from `lwm/sim/utils.py` to avoid cross-package imports.
    """
    if isinstance(rotate_matrix, (list, tuple)):
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)

    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]
    return np.concatenate([a1, a2])


def compute_pcd_rigid_body_delta_9dof(pcd_current, pcd_future):
    """
    Compute 9-DoF rigid body transform from two corresponding point clouds
    using the Kabsch algorithm (SVD-based optimal alignment).

    Finds R, t such that pcd_future ≈ R @ pcd_current + t.

    Returns [9] vector: [rot_6d(6), translation(3)]
    - rot_6d: first two columns of rotation matrix (Gram-Schmidt / Zhou et al. 2019)
    - translation: 3D translation vector

    Args:
        pcd_current: [N, 3] point cloud at time t (same point indices as pcd_future)
        pcd_future:  [N, 3] point cloud at time t+1
    Returns:
        [9] float32 vector: [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z, tx, ty, tz]
    """
    # Centroids
    p_bar = pcd_current.mean(axis=0)  # [3]
    q_bar = pcd_future.mean(axis=0)   # [3]

    # Center the point clouds
    P = pcd_current - p_bar  # [N, 3]
    Q = pcd_future - q_bar   # [N, 3]

    # Cross-covariance matrix
    H = P.T @ Q  # [3, 3]

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1, not reflection)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T  # [3, 3]

    # Translation
    t = q_bar - R @ p_bar  # [3]

    # 6D rotation representation: first two columns of R
    rot_6d = rotation_transfer_matrix_to_6D(R)  # [6]

    return np.concatenate([rot_6d, t]).astype(np.float32)


def compute_proper_state_delta(current_state: np.ndarray, future_states: np.ndarray) -> np.ndarray:
    """
    Compute proper state deltas, handling rotation correctly.
    
    State format: [pos(3), ori_6d(6), finger(1)] = 10D
    
    For position and finger: simple subtraction (delta = future - current)
    For orientation: proper rotation composition (delta = current.T @ future in matrix form, then to 6D)
    
    Args:
        current_state: [state_dim] or [n_obs_steps, state_dim] current state(s)
        future_states: [horizon, state_dim] future states
        
    Returns:
        state_deltas: [horizon, state_dim] properly computed deltas
    """
    # Handle case where current_state is [n_obs_steps, state_dim] - use the last (most recent) observation
    if current_state.ndim == 2:
        current_state = current_state[-1]  # [state_dim] - use most recent observation
    
    horizon = future_states.shape[0]
    state_dim = current_state.shape[0]
    state_deltas = np.zeros((horizon, state_dim), dtype=np.float32)
    
    # Position deltas (indices 0:3) - simple subtraction
    state_deltas[:, :3] = future_states[:, :3] - current_state[:3]
    
    # Finger delta (index 9) - simple subtraction
    state_deltas[:, 9] = future_states[:, 9] - current_state[9]
    
    # Orientation deltas (indices 3:9) - proper rotation composition
    # Convert current orientation 6D to rotation matrix
    current_ori_6d = current_state[3:9]
    current_rot_mat = rotation_transfer_6D_to_matrix(current_ori_6d)
    
    for i in range(horizon):
        future_ori_6d = future_states[i, 3:9]
        future_rot_mat = rotation_transfer_6D_to_matrix(future_ori_6d)
        
        # Compute relative rotation: delta_R = R_current.T @ R_future
        # This gives the rotation needed to go from current to future orientation
        delta_rot_mat = current_rot_mat.T @ future_rot_mat
        
        # Convert back to 6D representation
        delta_ori_6d = rotation_transfer_matrix_to_6D(delta_rot_mat)
        state_deltas[i, 3:9] = delta_ori_6d
    
    return state_deltas


class LWMDataset(Dataset):
    """
    Dataset for training Local World Model.
    
    Loads sequences from zarr files and creates training pairs:
    - Input: current observation + action horizon
    - Target: future static_affordance + agent_pos (horizon steps ahead)
    """
    
    def __init__(
        self,
        expert_dataset_path,
        rollout_dataset_path=None,
        horizon=8,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.1,
        train_ratio=0.9,
        is_pickle=True,
        n_obs_steps=1,
        data_splits=None,
        use_part_pointcloud=False,
        k_nearest=0,
        use_pcd_transform_target=True,
        skip_rotation_normalization=False,
        include_object_ids=None,
        include_object_ids_new_config=None,
        allowed_goal_percentages_door=None,
        allowed_goal_percentages_drawer=None,
        goal_percentage_tolerance=0.02,
        load_unseen_val=False,
        affordance_target_frame="world",
        augment_yaw=0.0,
        augment_translation=0.0,
    ):
        """
        Args:
            expert_dataset_path: Path to expert dataset or HuggingFace dataset ID
            rollout_dataset_path: Optional path to rollout dataset (same structure as expert)
            horizon: Number of future steps to predict (also action horizon length)
            pad_before: Padding before sequence
            pad_after: Padding after sequence
            seed: Random seed
            val_ratio: Validation ratio
            train_ratio: Training ratio
            is_pickle: Whether data is pickled
            n_obs_steps: Number of observation steps to use (default: 1)
            data_splits: Dict with keys 'expert', 'new_config', 'replay', each having
                         'enabled' (bool) and 'step_budget' (int or None).
                         Required — controls which data splits are used for training.
            use_part_pointcloud: If True, load 'part_point_cloud' (512 pts) instead of
                                 'point_cloud' (4500 pts). Used by v4 per-point transformer.
            k_nearest: Number of nearest points to affordance to predict displacement for.
                       0 = disabled (default, backward compatible). When > 0, dataset returns
                       knn_points [K, 3] in observation and knn_point_delta [horizon, K, 3] in target.
            use_pcd_transform_target: If True, predict 9-DoF rigid body transform computed from
                                      part point cloud correspondences via SVD (Kabsch algorithm).
                                      Returns affordance_delta as [horizon, 9] (rot_6d[6] + translation[3]).
                                      Requires use_part_pointcloud=True and re-extracted data with
                                      consistent point indices across timesteps.
            include_object_ids: Optional list of object ID strings to include.
                                Applied to expert and replay episodes (by path).
            include_object_ids_new_config: Optional list of object ID strings to include
                                           for new_config episodes. None = no filtering (include all).
            allowed_goal_percentages_door: Optional list of allowed goal percentages for door objects.
                                           Applied to expert (goal_percentage) and replay (source_goal_percentage).
            allowed_goal_percentages_drawer: Optional list of allowed goal percentages for drawer objects.
                                             Applied to expert (goal_percentage) and replay (source_goal_percentage).
            goal_percentage_tolerance: Tolerance for goal percentage matching (default: 0.02).
            affordance_target_frame: "world" (default) or "centroid": frame of the 9-D
                part-motion target, see lwm.world_model.targets.
            augment_yaw: Max yaw (degrees) of the training-only scene rotation about the
                vertical axis through the affordance point a_t. 0 = off.
            augment_translation: Max xy shift (metres) of the training-only scene
                translation (z is never shifted). 0 = off.
        """
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.k_nearest = k_nearest
        self.use_pcd_transform_target = use_pcd_transform_target
        self.skip_rotation_normalization = skip_rotation_normalization
        check_frames(affordance_target_frame)
        if affordance_target_frame != "world" and not use_pcd_transform_target:
            raise ValueError("affordance_target_frame='centroid' requires use_pcd_transform_target=True")
        self.affordance_target_frame = affordance_target_frame
        # Augmentation is applied by __getitem__ of the training dataset only; the
        # validation datasets (get_validation_dataset) switch it off.
        self.augment_yaw = float(augment_yaw or 0.0)
        self.augment_translation = float(augment_translation or 0.0)
        if use_pcd_transform_target and not use_part_pointcloud:
            cprint("Warning: use_pcd_transform_target requires part point clouds. Enabling use_part_pointcloud.", "yellow")
            use_part_pointcloud = True
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.data_splits = data_splits

        # Filtering parameters
        self.include_object_ids = set(include_object_ids) if include_object_ids else None
        self.include_object_ids_new_config = set(include_object_ids_new_config) if include_object_ids_new_config else None
        self.allowed_goal_percentages_door = allowed_goal_percentages_door
        self.allowed_goal_percentages_drawer = allowed_goal_percentages_drawer
        self.goal_percentage_tolerance = goal_percentage_tolerance
        has_goal_filter = (allowed_goal_percentages_door is not None or
                           allowed_goal_percentages_drawer is not None)

        cprint(f"\n{'='*70}", "green")
        cprint("LWM Dataset Loading", "green")
        cprint(f"{'='*70}", "green")

        if self.include_object_ids is not None:
            cprint(f"[Filter] Object IDs (expert/replay): {sorted(self.include_object_ids)}", "cyan")
        else:
            cprint(f"[Filter] Object IDs (expert/replay): all (no filter)", "cyan")
        if self.include_object_ids_new_config is not None:
            cprint(f"[Filter] Object IDs (new_config): {sorted(self.include_object_ids_new_config)}", "cyan")
        else:
            cprint(f"[Filter] Object IDs (new_config): all (no filter)", "cyan")
        if has_goal_filter:
            cprint(f"[Filter] Goal % door: {allowed_goal_percentages_door}", "cyan")
            cprint(f"[Filter] Goal % drawer: {allowed_goal_percentages_drawer}", "cyan")
        else:
            cprint(f"[Filter] Goal %: all (no filter)", "cyan")

        if data_splits is None:
            raise ValueError("data_splits is required. Provide a dict with keys 'expert', 'new_config', 'replay'.")

        rng = np.random.RandomState(seed)

        # Resolve expert dataset path (fails clearly for HuggingFace repo IDs)
        if not os.path.isdir(expert_dataset_path):
            expert_dataset_path = resolve_dataset_path(expert_dataset_path)

        # --- Collect and classify episodes by split ---
        cprint(f"\n--- Episode Collection ---", "yellow")
        expert_split_cfg = data_splits.get("expert", {})
        new_config_split_cfg = data_splits.get("new_config", {})
        replay_split_cfg = data_splits.get("replay", {})

        # Expert episodes
        expert_episode_paths = []
        if expert_split_cfg.get("enabled", True):
            all_expert = self._collect_episode_paths(expert_dataset_path, is_pickle)
            cprint(f"Found {len(all_expert)} valid episodes in expert dataset", "green")
            # Filter expert by object ID and goal percentage
            if self.include_object_ids is not None or has_goal_filter:
                pre_filter = len(all_expert)
                all_expert = self._filter_expert_episodes(all_expert, is_pickle)
                cprint(f"  After object/goal filter: {len(all_expert)}/{pre_filter} expert episodes", "cyan")
            expert_episode_paths = self._apply_step_budget(
                all_expert, expert_split_cfg.get("step_budget"), rng
            )
            if expert_split_cfg.get("step_budget") is not None:
                cprint(f"  After step budget ({expert_split_cfg['step_budget']}): {len(expert_episode_paths)} expert episodes", "yellow")
        else:
            cprint("Expert split disabled", "yellow")

        # Rollout episodes (classify into new_config and replay)
        new_config_episode_paths = []
        replay_episode_paths = []
        if rollout_dataset_path is not None and (new_config_split_cfg.get("enabled", True) or replay_split_cfg.get("enabled", True)):
            all_rollout = self._collect_episode_paths(rollout_dataset_path, is_pickle)
            cprint(f"Found {len(all_rollout)} valid episodes in rollout dataset", "green")

            # Classify rollout episodes
            all_new_config = []
            all_replay = []
            for ep_path in all_rollout:
                split_type = self._classify_rollout_episode(ep_path)
                if split_type == "new_config":
                    all_new_config.append(ep_path)
                else:
                    all_replay.append(ep_path)
            cprint(f"  Classified: {len(all_new_config)} new_config, {len(all_replay)} replay", "cyan")

            # Apply object ID filter and budget for new_config
            if new_config_split_cfg.get("enabled", True):
                if self.include_object_ids_new_config is not None:
                    pre_filter = len(all_new_config)
                    all_new_config = [ep for ep in all_new_config
                                      if self._extract_object_id(ep) in self.include_object_ids_new_config]
                    cprint(f"  After object filter: {len(all_new_config)}/{pre_filter} new_config episodes", "cyan")
                new_config_episode_paths = self._apply_step_budget(
                    all_new_config, new_config_split_cfg.get("step_budget"), rng
                )
                if new_config_split_cfg.get("step_budget") is not None:
                    cprint(f"  After step budget ({new_config_split_cfg['step_budget']}): {len(new_config_episode_paths)} new_config episodes", "yellow")
            else:
                cprint("New config split disabled", "yellow")

            # Filter replay by object ID and source_goal_percentage, then apply budget
            if replay_split_cfg.get("enabled", True):
                if self.include_object_ids is not None or has_goal_filter:
                    pre_filter = len(all_replay)
                    all_replay = self._filter_replay_episodes(all_replay, is_pickle)
                    cprint(f"  After object/goal filter: {len(all_replay)}/{pre_filter} replay episodes", "cyan")
                replay_episode_paths = self._apply_step_budget(
                    all_replay, replay_split_cfg.get("step_budget"), rng
                )
                if replay_split_cfg.get("step_budget") is not None:
                    cprint(f"  After step budget ({replay_split_cfg['step_budget']}): {len(replay_episode_paths)} replay episodes", "yellow")
            else:
                cprint("Replay split disabled", "yellow")

        # Collect unseen object episodes (val-only) from expert + new_config datasets
        unseen_episode_paths = []
        num_unseen_expert = 0
        num_unseen_new_config = 0
        self.load_unseen_val = load_unseen_val
        if load_unseen_val and self.include_object_ids is not None:
            # Derive unseen object IDs: all known objects minus seen ones
            all_known_ids = set()
            for cat_info in GOAL_BASED_OBJECT_CATEGORIES.values():
                all_known_ids.update(cat_info["ids"])
            unseen_ids = all_known_ids - self.include_object_ids
            if unseen_ids:
                cprint(f"\n--- Unseen Object Validation ---", "yellow")
                cprint(f"  Unseen object IDs: {sorted(unseen_ids)}", "cyan")
                # Collect expert episodes for unseen objects
                all_expert_unfiltered = self._collect_episode_paths(expert_dataset_path, is_pickle)
                unseen_expert = []
                for ep_path in all_expert_unfiltered:
                    obj_id = self._extract_object_id(ep_path)
                    if obj_id in unseen_ids:
                        unseen_expert.append(ep_path)
                cprint(f"  Found {len(unseen_expert)} unseen expert episodes", "cyan")
                # Collect new_config episodes for unseen objects from rollout dataset
                unseen_new_config = []
                if rollout_dataset_path is not None:
                    all_rollout_unfiltered = self._collect_episode_paths(rollout_dataset_path, is_pickle)
                    for ep_path in all_rollout_unfiltered:
                        obj_id = self._extract_object_id(ep_path)
                        if obj_id in unseen_ids and self._classify_rollout_episode(ep_path) == "new_config":
                            unseen_new_config.append(ep_path)
                    cprint(f"  Found {len(unseen_new_config)} unseen new_config episodes", "cyan")
                # Cap each split separately so we have both expert unseen and new_config unseen
                max_unseen_per_split = 100
                rng_unseen = np.random.RandomState(seed + 999)
                if len(unseen_expert) > max_unseen_per_split:
                    unseen_expert = [str(p) for p in rng_unseen.choice(unseen_expert, max_unseen_per_split, replace=False)]
                if len(unseen_new_config) > max_unseen_per_split:
                    unseen_new_config = [str(p) for p in rng_unseen.choice(unseen_new_config, max_unseen_per_split, replace=False)]
                unseen_episode_paths = unseen_expert + unseen_new_config
                num_unseen_expert = len(unseen_expert)
                num_unseen_new_config = len(unseen_new_config)
                cprint(f"  Loaded {num_unseen_expert} unseen expert + {num_unseen_new_config} unseen new_config episodes (val-only)", "green")

        # Track counts for masks
        num_expert = len(expert_episode_paths)
        num_new_config = len(new_config_episode_paths)
        num_replay = len(replay_episode_paths)
        num_unseen = len(unseen_episode_paths)

        # Merge all episode paths: expert + new_config + replay + unseen
        episode_paths = expert_episode_paths + new_config_episode_paths + replay_episode_paths + unseen_episode_paths
        cprint(f"\n--- Final Episode Counts ---", "yellow")
        cprint(f"  Expert:     {num_expert:>5}", "green")
        cprint(f"  New config: {num_new_config:>5}", "green")
        cprint(f"  Replay:     {num_replay:>5}", "green")
        if num_unseen > 0:
            cprint(f"  Unseen (expert):     {num_unseen_expert:>5} (val-only)", "green")
            cprint(f"  Unseen (new_config): {num_unseen_new_config:>5} (val-only)", "green")
        cprint(f"  TOTAL:      {len(episode_paths):>5}", "green")
        
        # Point cloud loading configuration
        self.use_part_pointcloud = use_part_pointcloud
        self.num_points = 512  # Part point cloud size (fixed at extraction time)
        if use_part_pointcloud:
            self.pcd_key = "part_point_cloud"
        else:
            self.pcd_key = "point_cloud"
        keys = ["state", "action", self.pcd_key, "static_affordance"]

        # Check if precomputed pcd_rigid_body_delta_9dof is available in ALL episodes.
        # Must check every episode — sampling misses scattered missing keys.
        self.has_precomputed_pcd_delta = False
        if use_pcd_transform_target and episode_paths:
            try:
                cprint(f"[9-DoF] Checking {len(episode_paths)} episodes for precomputed pcd_rigid_body_delta_9dof...", "cyan")
                missing_episodes = []
                for ep_path in episode_paths:
                    if packed.lookup(ep_path) is not None:
                        if not packed.has_field(ep_path, 'pcd_rigid_body_delta_9dof'):
                            missing_episodes.append(ep_path)
                        continue
                    probe_path = os.path.join(ep_path, "0.pkl")
                    with open(probe_path, 'rb') as f:
                        probe_data = pickle.load(f)
                    if 'pcd_rigid_body_delta_9dof' not in probe_data:
                        missing_episodes.append(ep_path)

                if missing_episodes:
                    cprint(f"[9-DoF] {len(missing_episodes)}/{len(episode_paths)} episodes missing pcd_rigid_body_delta_9dof", "yellow")
                    for mp in missing_episodes[:5]:
                        cprint(f"  Missing: {os.path.basename(mp)}", "yellow")
                    if len(missing_episodes) > 5:
                        cprint(f"  ... and {len(missing_episodes) - 5} more", "yellow")
                    cprint(f"  Will compute SVD at runtime (slower). Re-extract missing episodes to fix.", "yellow")
                else:
                    keys.append('pcd_rigid_body_delta_9dof')
                    self.has_precomputed_pcd_delta = True
                    cprint(f"[9-DoF] All {len(episode_paths)} episodes have precomputed delta — using direct lookup", "green")
            except Exception as e:
                cprint(f"[9-DoF] Could not probe for precomputed delta: {e} — will compute SVD at runtime", "yellow")

        self.replay_buffer = ReplayBuffer.copy_from_multiple_path(
            episode_paths,
            keys=keys,
            load_per_step=True,
            is_pickle=is_pickle,
            target_action="action",
        )
        
        # Create train/val split
        num_episodes = len(self.replay_buffer.episode_ends)

        # Track which episodes are from each split
        self.expert_episode_mask = np.zeros(num_episodes, dtype=bool)
        self.new_config_episode_mask = np.zeros(num_episodes, dtype=bool)
        self.replay_episode_mask = np.zeros(num_episodes, dtype=bool)
        self.unseen_episode_mask = np.zeros(num_episodes, dtype=bool)
        self.unseen_expert_episode_mask = np.zeros(num_episodes, dtype=bool)
        self.unseen_new_config_episode_mask = np.zeros(num_episodes, dtype=bool)
        self.expert_episode_mask[:num_expert] = True
        self.new_config_episode_mask[num_expert:num_expert + num_new_config] = True
        num_seen = num_expert + num_new_config + num_replay
        self.replay_episode_mask[num_expert + num_new_config:num_seen] = True
        self.unseen_episode_mask[num_seen:] = True
        self.unseen_expert_episode_mask[num_seen:num_seen + num_unseen_expert] = True
        self.unseen_new_config_episode_mask[num_seen + num_unseen_expert:num_seen + num_unseen] = True

        # Train/val split: only over seen episodes; unseen are always val-only
        rng = np.random.RandomState(seed)
        perm = rng.permutation(num_seen)
        n_val = int(num_seen * val_ratio)
        n_train = num_seen - n_val

        train_mask = np.zeros(num_episodes, dtype=bool)
        val_mask = np.zeros(num_episodes, dtype=bool)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        # Force unseen episodes into val only
        val_mask[num_seen:] = True

        cprint(f"\n--- Train/Val Split ---", "yellow")
        cprint(f"  Train: {train_mask.sum():>5}  |  Val: {val_mask.sum():>5}", "cyan")
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        
        # Create sequence sampler for training
        # Need horizon+1 for targets, and n_obs_steps for observations
        sequence_length = max(horizon + 1, n_obs_steps)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        
        # Pre-compute all valid training pairs
        # For each sequence, we can create pairs starting at t=0 to t=sequence_length-horizon-1
        self._build_pair_indices()

        cprint(f"\n--- Dataset Ready ---", "yellow")
        cprint(f"  Training pairs: {len(self):,}  (from {len(self.sampler):,} sequences)", "green")
        cprint(f"{'='*70}\n", "green")

        # ------------------------------------------------------------------
        # Compute normalization statistics (mean and std) over dataset
        # ------------------------------------------------------------------
        self._compute_action_normalization_stats()
        self._compute_delta_normalization_stats()
        if self.augment_yaw > 0:
            self._pool_xy_std_for_yaw_augmentation()

    def _pool_xy_std_for_yaw_augmentation(self):
        """Use one shared std for the x and y components of every translation-like quantity.

        The stats come from un-augmented data, where parts move mostly along x (e.g. part
        centroid std x 1.9 mm vs y 0.28 mm). A training-time yaw rotates x-motion into y, so
        y-targets would land many stds out. The pooled std sqrt((sx^2 + sy^2) / 2) is
        invariant under a yaw. Dims whose std is set to 1.0 (not a real spread) are
        left out of the pool. Applied after loading or computing stats, so caches are unaffected.
        """
        pairs = [("action_std", (0, 1)), ("agent_state_delta_std", (0, 1))]
        if self.use_pcd_transform_target and self.affordance_delta_std.shape[0] == 9:
            pairs.append(("affordance_delta_std", (6, 7)))   # part translation / centroid shift
        for name, (ix, iy) in pairs:
            std = getattr(self, name).copy()
            real = [s for s in (std[ix], std[iy]) if s < 0.5]
            if not real:
                continue
            pooled = float(np.sqrt(np.mean(np.square(real))))
            before = (float(std[ix]), float(std[iy]))
            std[ix] = std[iy] = pooled
            setattr(self, name, std.astype(np.float32))
            cprint(f"  [yaw augmentation] {name} dims ({ix},{iy}) std {before[0]:.2e},{before[1]:.2e} "
                   f"-> pooled {pooled:.2e}", "yellow")

    def _collect_episode_paths(self, dataset_path, is_pickle):
        """Collect episode paths from a dataset directory.

        Supports multiple directory structures:
        1. Flat structure: dataset_path/{object_id}_{session_name}/
        2. Epoch subdirectory structure: dataset_path/epoch-{N}/{object_id}_{session_name}/
        3. Goal-based structure: dataset_path/{object_id}/{task_name}/{episode_folders}/
        4. Any nested structure with episode folders at arbitrary depth
        5. A packed dataset (index.json, see lwm.world_model.packed): virtual episode
           paths in the same order as the pickle directory walk
        """
        cprint(f"Loading dataset from: {dataset_path}", "yellow")

        # Files/folders to exclude (not episodes)
        exclude_names = [
            "action_dist", "demo_rgbs", "all_demo_path.txt", "meta_info.json",
            "example_pointcloud", ".zgroup", "metadata.json", "filtered_demos.txt",
            "articulation_points.json", ".articulation_points_lock", ".extraction_lock"
        ]

        if packed.is_packed_dir(dataset_path):
            episode_paths = packed.episode_paths(dataset_path, exclude_names=exclude_names, max_depth=5)
            cprint(f"Found {len(episode_paths)} episodes in packed dataset {dataset_path}", "cyan")
            return episode_paths

        def is_episode_dir(dir_path):
            """Check if a directory is an episode directory (contains data files)."""
            if not os.path.isdir(dir_path):
                return False
            if is_pickle:
                files = [f for f in os.listdir(dir_path) if f.endswith(".pkl")]
            else:
                # For zarr format, check for subdirectories (step folders)
                files = [f for f in os.listdir(dir_path)
                        if os.path.isdir(os.path.join(dir_path, f))]
            return len(files) > 0

        def find_episodes_recursive(dir_path, max_depth=5, current_depth=0):
            """Recursively find all episode directories up to max_depth."""
            if current_depth > max_depth:
                return []

            if not os.path.isdir(dir_path):
                return []

            # Check if current directory is an episode
            if is_episode_dir(dir_path):
                return [dir_path]

            # Otherwise, recurse into subdirectories
            episode_paths = []
            try:
                subdirs = os.listdir(dir_path)
            except PermissionError:
                return []

            subdirs = [f for f in subdirs if f not in exclude_names and not f.startswith('.')]
            subdirs = sorted(subdirs)

            for subdir in subdirs:
                subdir_path = os.path.join(dir_path, subdir)
                if os.path.isdir(subdir_path):
                    episode_paths.extend(find_episodes_recursive(subdir_path, max_depth, current_depth + 1))

            return episode_paths

        # Use recursive search to find all episodes
        episode_paths = find_episodes_recursive(dataset_path, max_depth=5, current_depth=0)

        if len(episode_paths) == 0:
            cprint(f"Warning: No episodes found in {dataset_path}", "yellow")
        else:
            cprint(f"Found {len(episode_paths)} episodes in {dataset_path}", "cyan")

        return episode_paths
    
    def _classify_rollout_episode(self, episode_path):
        """Classify a rollout episode as 'new_config' or 'replay' based on directory name."""
        name = os.path.basename(episode_path)
        if "_new_config_" in name:
            return "new_config"
        return "replay"

    def _apply_step_budget(self, episode_paths, step_budget, rng):
        """Select episodes until cumulative steps >= step_budget.

        Args:
            episode_paths: List of episode directory paths
            step_budget: Max total steps to include (None = use all)
            rng: numpy RandomState for deterministic shuffling

        Returns:
            Filtered list of episode paths
        """
        if step_budget is None:
            return episode_paths
        step_budget = float(step_budget)
        # Count steps per episode (number of .pkl files)
        ep_steps = []
        for path in episode_paths:
            n = packed.num_steps(path)
            if n is None:
                n = len([f for f in os.listdir(path) if f.endswith('.pkl')])
            ep_steps.append(n)
        # Shuffle and accumulate until budget is met
        indices = list(range(len(episode_paths)))
        rng.shuffle(indices)
        selected = []
        total = 0
        for i in indices:
            selected.append(episode_paths[i])
            total += ep_steps[i]
            if total >= step_budget:
                break
        cprint(f"    Step budget: selected {len(selected)}/{len(episode_paths)} episodes, {total} steps (budget={step_budget})", "cyan")
        return selected

    def _extract_object_id(self, episode_path):
        """Extract object ID from episode path (format: {object_id}_{session} or obj_{object_id}_{session})"""
        episode_name = os.path.basename(episode_path)
        parts = episode_name.split('_')
        if parts[0] == 'obj':
            return parts[1]
        else:
            return parts[0]

    def _goal_matches(self, goal_pct, object_id):
        """Check if a goal percentage is in the allowed set for the object's category."""
        category = get_object_category(object_id)
        if category == "door" and self.allowed_goal_percentages_door is not None:
            allowed = self.allowed_goal_percentages_door
        elif category == "drawer" and self.allowed_goal_percentages_drawer is not None:
            allowed = self.allowed_goal_percentages_drawer
        else:
            return True  # No filter for this category
        return any(abs(goal_pct - a) <= self.goal_percentage_tolerance for a in allowed)

    def _filter_expert_episodes(self, episode_paths, is_pickle):
        """Filter expert episodes by object ID and goal_percentage.

        - include_object_ids: whitelist of object IDs (from path)
        - allowed_goal_percentages_door/drawer: read goal_percentage from 0.pkl
        """
        filtered = []
        stats = {"object_filtered": 0, "goal_filtered": 0, "goal_missing": 0}
        for ep_path in episode_paths:
            obj_id = self._extract_object_id(ep_path)
            # Object ID filter
            if self.include_object_ids is not None and obj_id not in self.include_object_ids:
                stats["object_filtered"] += 1
                continue
            # Goal percentage filter
            if self.allowed_goal_percentages_door is not None or self.allowed_goal_percentages_drawer is not None:
                goal_pct = self._read_goal_percentage(ep_path, key="goal_percentage")
                if goal_pct is None:
                    stats["goal_missing"] += 1
                    continue
                if not self._goal_matches(goal_pct, obj_id):
                    stats["goal_filtered"] += 1
                    continue
            filtered.append(ep_path)
        if any(v > 0 for v in stats.values()):
            cprint(f"    Expert filter: object={stats['object_filtered']}, goal={stats['goal_filtered']}, "
                   f"goal_missing={stats['goal_missing']} filtered out", "yellow")
        return filtered

    def _filter_replay_episodes(self, episode_paths, is_pickle):
        """Filter replay episodes by object ID and source_goal_percentage.

        - include_object_ids: whitelist of object IDs (from path)
        - allowed_goal_percentages_door/drawer: uses source_goal_percentage from 0.pkl
          (the goal of the expert demo being replayed, NOT the inference goal)
        """
        filtered = []
        stats = {"object_filtered": 0, "goal_filtered": 0, "goal_missing": 0}
        for ep_path in episode_paths:
            obj_id = self._extract_object_id(ep_path)
            # Object ID filter
            if self.include_object_ids is not None and obj_id not in self.include_object_ids:
                stats["object_filtered"] += 1
                continue
            # Source goal percentage filter
            if self.allowed_goal_percentages_door is not None or self.allowed_goal_percentages_drawer is not None:
                source_goal_pct = self._read_goal_percentage(ep_path, key="source_goal_percentage")
                if source_goal_pct is None:
                    stats["goal_missing"] += 1
                    continue
                if not self._goal_matches(source_goal_pct, obj_id):
                    stats["goal_filtered"] += 1
                    continue
            filtered.append(ep_path)
        if any(v > 0 for v in stats.values()):
            cprint(f"    Replay filter: object={stats['object_filtered']}, goal={stats['goal_filtered']}, "
                   f"goal_missing={stats['goal_missing']} filtered out", "yellow")
        return filtered

    @staticmethod
    def _read_goal_percentage(episode_path, key="goal_percentage"):
        """Read a goal percentage field from the first pickle of an episode."""
        meta = packed.episode_meta(episode_path)
        if meta is not None:
            val = meta.get(key)
            return None if val is None else float(val)
        first_pkl = os.path.join(episode_path, "0.pkl")
        if not os.path.exists(first_pkl):
            return None
        try:
            with open(first_pkl, "rb") as f:
                data = pickle.load(f)
            val = data.get(key)
            if val is not None:
                return float(val)
        except Exception:
            pass
        return None

    def get_episode_splits_metadata(self):
        """
        Return metadata about train/val episode splits.
        Tracks which episodes and objects are in train vs val.
        
        Returns:
            {
                "train_episodes": [
                    {"episode_idx": 0, "episode_path": "...", "object_id": "44817"},
                    ...
                ],
                "val_episodes": [...],
                "train_objects": ["44817", "45123", ...],
                "val_objects": ["45678", ...],
                "seed": 42,
                "val_ratio": 0.1
            }
        """
        train_episodes = []
        val_episodes = []
        train_objects = set()
        val_objects = set()
        
        for ep_idx in range(len(self.replay_buffer.episode_ends)):
            episode_path = self.replay_buffer.all_path_list[ep_idx]
            object_id = self._extract_object_id(episode_path)
            
            episode_info = {
                "episode_idx": ep_idx,
                "episode_path": episode_path,
                "object_id": object_id,
            }
            
            if self.train_mask[ep_idx]:
                train_episodes.append(episode_info)
                train_objects.add(object_id)
            elif self.val_mask[ep_idx]:
                val_episodes.append(episode_info)
                val_objects.add(object_id)
        
        return {
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "train_objects": sorted(list(train_objects)),
            "val_objects": sorted(list(val_objects)),
            "seed": self.seed,
            "val_ratio": self.val_ratio,
        }

    def _compute_action_normalization_stats(self):
        """
        Compute per-dimension mean and std for actions over training episodes only.
        Using only train data avoids leakage from validation into normalization.
        Actions will be standardized as: (action - mean) / std
        """
        action_sum = None
        action_sq_sum = None
        action_min = None
        action_max = None
        total_count = 0

        for ep_idx, zarr_path in enumerate(self.replay_buffer.all_path_list):
            if not self.train_mask[ep_idx]:
                continue
            ep_len = int(self.replay_buffer.episode_lengths[ep_idx])
            if ep_len <= 0:
                continue

            data = self.replay_buffer._get_data_single_file(zarr_path, 0, ep_len)
            actions = data["action"].astype(np.float32)  # [T, action_dim]

            if actions.size == 0:
                continue

            if action_sum is None:
                action_sum = actions.sum(axis=0)
                action_sq_sum = (actions ** 2).sum(axis=0)
                action_min = actions.min(axis=0)
                action_max = actions.max(axis=0)
            else:
                action_sum += actions.sum(axis=0)
                action_sq_sum += (actions ** 2).sum(axis=0)
                action_min = np.minimum(action_min, actions.min(axis=0))
                action_max = np.maximum(action_max, actions.max(axis=0))

            total_count += actions.shape[0]

        if total_count == 0:
            raise ValueError("No actions found while computing normalization statistics.")

        action_mean = action_sum / float(total_count)
        action_var = action_sq_sum / float(total_count) - action_mean ** 2
        # Numerical safety: variance cannot be negative
        action_var = np.maximum(action_var, 0.0)
        action_std = np.sqrt(action_var)

        # Avoid division by zero: for near-constant dimensions, fall back to 1.0
        # Threshold 1.3e-3 catches quaternion w-components and rotation matrix
        # diagonals (std ~0.0002-0.0012) while preserving real small-variance
        # dimensions like translation deltas (std >= 0.0013).
        eps = 1.3e-3
        action_std[action_std < eps] = 1.0

        self.action_mean = action_mean.astype(np.float32)
        self.action_std = action_std.astype(np.float32)
        self.action_min = action_min.astype(np.float32)
        self.action_max = action_max.astype(np.float32)

        # Skip normalization for 6D rotation components (indices 3:9) in actions.
        # 6D rotation representations are inherently bounded in [-1, 1] and Z-score
        # normalization warps their geometric properties (see Zhou et al. 2019).
        if self.skip_rotation_normalization:
            self.action_mean[3:9] = 0.0
            self.action_std[3:9] = 1.0
            cprint("  [skip_rotation_normalization] Action rotation dims (3:9) set to mean=0, std=1", "yellow")

        cprint(
            f"Computed action normalization stats over {total_count} action steps (train episodes only). "
            f"Mean range [{self.action_mean.min():.4f}, {self.action_mean.max():.4f}], "
            f"Std range [{self.action_std.min():.4f}, {self.action_std.max():.4f}]",
            "cyan",
        )
    
    def _get_norm_cache_path(self):
        """Get path for cached normalization stats, keyed by training data config."""
        train_paths = sorted([
            self.replay_buffer.all_path_list[i]
            for i in range(len(self.train_mask))
            if self.train_mask[i]
        ])
        cache_key_str = (
            str(train_paths)
            + f"|pcd_target={self.use_pcd_transform_target}"
            + f"|k_nearest={self.k_nearest}"
            + f"|skip_rot_norm={self.skip_rotation_normalization}"
        )
        # Only non-default settings change the key, so caches of existing runs stay valid.
        if getattr(self, "affordance_target_frame", "world") != "world":
            cache_key_str += f"|aff_frame={self.affordance_target_frame}"
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:12]
        cache_dir = os.path.expanduser("~/.cache/lwm_norm_stats")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"norm_stats_{cache_hash}.npz")

    def _compute_delta_normalization_stats(self):
        """
        Compute per-dimension mean and std for target deltas over training episodes.
        Uses episode-level iteration (fast) and disk caching.
        """
        # Try loading from disk cache
        cache_path = self._get_norm_cache_path()
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=True)
                self.agent_state_delta_mean = cached["agent_state_delta_mean"]
                self.agent_state_delta_std = cached["agent_state_delta_std"]
                self.affordance_delta_mean = cached["affordance_delta_mean"]
                self.affordance_delta_std = cached["affordance_delta_std"]
                self.knn_point_delta_mean = cached["knn_point_delta_mean"].item() if cached["knn_point_delta_mean"].shape == () else cached["knn_point_delta_mean"]
                self.knn_point_delta_std = cached["knn_point_delta_std"].item() if cached["knn_point_delta_std"].shape == () else cached["knn_point_delta_std"]
                cprint(
                    f"Loaded cached delta normalization stats from {cache_path}\n"
                    f"  Agent delta mean range [{self.agent_state_delta_mean.min():.6f}, {self.agent_state_delta_mean.max():.6f}], "
                    f"std range [{self.agent_state_delta_std.min():.6f}, {self.agent_state_delta_std.max():.6f}]\n"
                    f"  Affordance delta mean range [{self.affordance_delta_mean.min():.6f}, {self.affordance_delta_mean.max():.6f}], "
                    f"std range [{self.affordance_delta_std.min():.6f}, {self.affordance_delta_std.max():.6f}]",
                    "green",
                )
                return
            except Exception as e:
                cprint(f"Failed to load norm stats cache ({e}), recomputing...", "yellow")

        # Episode-level computation (much faster than pair-level)
        agent_delta_sum = None
        agent_delta_sq_sum = None
        affordance_delta_sum = None
        affordance_delta_sq_sum = None
        knn_delta_sum = None
        knn_delta_sq_sum = None
        total_agent_count = 0
        total_aff_count = 0
        knn_total_count = 0

        rb = self.replay_buffer
        for ep_idx, zarr_path in enumerate(tqdm(rb.all_path_list, desc="Computing delta normalization stats (episode-level)")):
            if not self.train_mask[ep_idx]:
                continue
            ep_len = int(rb.episode_lengths[ep_idx])
            if ep_len < 2:
                continue

            data = rb._get_data_single_file(zarr_path, 0, ep_len)
            states = data["state"].astype(np.float32)  # [T, 10]

            # Agent state deltas: consecutive t -> t+1
            for t in range(ep_len - 1):
                delta = compute_proper_state_delta(states[t], states[t+1:t+2])  # [1, 10]
                if agent_delta_sum is None:
                    agent_delta_sum = delta.sum(axis=0)
                    agent_delta_sq_sum = (delta ** 2).sum(axis=0)
                else:
                    agent_delta_sum += delta.sum(axis=0)
                    agent_delta_sq_sum += (delta ** 2).sum(axis=0)
                total_agent_count += 1

            # Affordance deltas
            if self.use_pcd_transform_target and self.has_precomputed_pcd_delta:
                # Direct from precomputed: [T, 9], last step is zeros — exclude it
                pcd_deltas = data['pcd_rigid_body_delta_9dof'].astype(np.float32)
                pcd_deltas = pcd_deltas[:ep_len - 1]  # [T-1, 9]
                if self.affordance_target_frame != "world":
                    pcd_deltas = self._encode_frame(pcd_deltas, data[self.pcd_key][:ep_len - 1])
                if affordance_delta_sum is None:
                    affordance_delta_sum = pcd_deltas.sum(axis=0)
                    affordance_delta_sq_sum = (pcd_deltas ** 2).sum(axis=0)
                else:
                    affordance_delta_sum += pcd_deltas.sum(axis=0)
                    affordance_delta_sq_sum += (pcd_deltas ** 2).sum(axis=0)
                total_aff_count += pcd_deltas.shape[0]
            elif self.use_pcd_transform_target:
                # Compute SVD on the fly (fallback, still episode-level)
                pcd_key = self.pcd_key
                if pcd_key in data:
                    point_cloud = data[pcd_key].astype(np.float32)  # [T, N, 3]
                    for t in range(ep_len - 1):
                        delta = compute_pcd_rigid_body_delta_9dof(point_cloud[t], point_cloud[t + 1])
                        if self.affordance_target_frame != "world":
                            delta = self._encode_frame(delta[None], point_cloud[t:t + 1])[0]
                        if affordance_delta_sum is None:
                            affordance_delta_sum = delta.copy()
                            affordance_delta_sq_sum = delta ** 2
                        else:
                            affordance_delta_sum += delta
                            affordance_delta_sq_sum += delta ** 2
                        total_aff_count += 1
            else:
                # Standard 3D affordance delta
                static_aff = data["static_affordance"].astype(np.float32)  # [T, 3]
                aff_deltas = static_aff[1:] - static_aff[:-1]  # [T-1, 3]
                if affordance_delta_sum is None:
                    affordance_delta_sum = aff_deltas.sum(axis=0)
                    affordance_delta_sq_sum = (aff_deltas ** 2).sum(axis=0)
                else:
                    affordance_delta_sum += aff_deltas.sum(axis=0)
                    affordance_delta_sq_sum += (aff_deltas ** 2).sum(axis=0)
                total_aff_count += aff_deltas.shape[0]

            # KNN point delta stats (if enabled)
            if self.k_nearest > 0 and self.pcd_key in data:
                point_cloud = data[self.pcd_key].astype(np.float32)
                static_aff = data["static_affordance"].astype(np.float32)
                for t in range(ep_len - 1):
                    aff_pt = static_aff[t]
                    dists = np.linalg.norm(point_cloud[t] - aff_pt, axis=-1)
                    knn_idx = np.argsort(dists)[:self.k_nearest]
                    knn_delta = point_cloud[t + 1][knn_idx] - point_cloud[t][knn_idx]  # [K, 3]
                    knn_flat = knn_delta.reshape(-1)  # [K*3]
                    if knn_delta_sum is None:
                        knn_delta_sum = knn_flat.copy()
                        knn_delta_sq_sum = knn_flat ** 2
                    else:
                        knn_delta_sum += knn_flat
                        knn_delta_sq_sum += knn_flat ** 2
                    knn_total_count += 1

        if total_agent_count == 0 or total_aff_count == 0:
            raise ValueError("No deltas found while computing normalization statistics.")

        agent_delta_mean = agent_delta_sum / float(total_agent_count)
        agent_delta_var = agent_delta_sq_sum / float(total_agent_count) - agent_delta_mean ** 2
        agent_delta_var = np.maximum(agent_delta_var, 0.0)
        agent_delta_std = np.sqrt(agent_delta_var)

        affordance_delta_mean = affordance_delta_sum / float(total_aff_count)
        affordance_delta_var = affordance_delta_sq_sum / float(total_aff_count) - affordance_delta_mean ** 2
        affordance_delta_var = np.maximum(affordance_delta_var, 0.0)
        affordance_delta_std = np.sqrt(affordance_delta_var)

        # See action normalization comment for threshold rationale
        eps = 1.3e-3
        agent_delta_std[agent_delta_std < eps] = 1.0
        affordance_delta_std[affordance_delta_std < eps] = 1.0

        # Skip normalization for 6D rotation components if requested.
        # 6D rotation representations are inherently bounded in [-1, 1] and Z-score
        # normalization warps their geometric properties (see Zhou et al. 2019).
        if self.skip_rotation_normalization:
            # Agent state delta: indices 3:9 are 6D rotation
            agent_delta_mean[3:9] = 0.0
            agent_delta_std[3:9] = 1.0
            # Affordance delta (9D mode): indices 0:6 are 6D rotation
            if self.use_pcd_transform_target and affordance_delta_mean.shape[0] == 9:
                affordance_delta_mean[:6] = 0.0
                affordance_delta_std[:6] = 1.0
            cprint(
                "  [skip_rotation_normalization] Rotation dims set to mean=0, std=1:\n"
                "    Agent state delta indices 3:9, Affordance delta indices 0:6 (9D mode)",
                "yellow",
            )

        self.agent_state_delta_mean = agent_delta_mean.astype(np.float32)
        self.agent_state_delta_std = agent_delta_std.astype(np.float32)
        self.affordance_delta_mean = affordance_delta_mean.astype(np.float32)
        self.affordance_delta_std = affordance_delta_std.astype(np.float32)

        # KNN point delta normalization stats
        if self.k_nearest > 0 and knn_total_count > 0:
            knn_mean = knn_delta_sum / float(knn_total_count)
            knn_var = knn_delta_sq_sum / float(knn_total_count) - knn_mean ** 2
            knn_var = np.maximum(knn_var, 0.0)
            knn_std = np.sqrt(knn_var)
            knn_std[knn_std < eps] = 1.0
            self.knn_point_delta_mean = knn_mean.astype(np.float32)
            self.knn_point_delta_std = knn_std.astype(np.float32)
            cprint(
                f"KNN point delta (K={self.k_nearest}) mean range [{knn_mean.min():.6f}, {knn_mean.max():.6f}], "
                f"std range [{knn_std.min():.6f}, {knn_std.max():.6f}]",
                "cyan",
            )
        else:
            self.knn_point_delta_mean = None
            self.knn_point_delta_std = None

        # Save to disk cache
        try:
            np.savez(
                cache_path,
                agent_state_delta_mean=self.agent_state_delta_mean,
                agent_state_delta_std=self.agent_state_delta_std,
                affordance_delta_mean=self.affordance_delta_mean,
                affordance_delta_std=self.affordance_delta_std,
                knn_point_delta_mean=self.knn_point_delta_mean if self.knn_point_delta_mean is not None else np.array(None),
                knn_point_delta_std=self.knn_point_delta_std if self.knn_point_delta_std is not None else np.array(None),
            )
            cprint(f"Saved delta normalization stats cache to {cache_path}", "green")
        except Exception as e:
            cprint(f"Warning: could not save norm stats cache: {e}", "yellow")

        cprint(
            f"Computed delta normalization stats over {total_agent_count} agent / {total_aff_count} affordance steps (episode-level). "
            f"Agent delta mean range [{agent_delta_mean.min():.6f}, {agent_delta_mean.max():.6f}], "
            f"std range [{agent_delta_std.min():.6f}, {agent_delta_std.max():.6f}]. "
            f"Affordance delta mean range [{affordance_delta_mean.min():.6f}, {affordance_delta_mean.max():.6f}], "
            f"std range [{affordance_delta_std.min():.6f}, {affordance_delta_std.max():.6f}]",
            "cyan",
        )
    
    def _encode_frame(self, delta_world, pcds):
        """World-form 9-D deltas [H, 9] -> self.affordance_target_frame, using the
        measured centroids of the clouds the steps start from (pcds [H, N, 3])."""
        centroids = pcds.astype(np.float64).mean(axis=1)  # [H, 3]
        out = encode_affordance_delta(
            torch.from_numpy(np.asarray(delta_world, dtype=np.float64)),
            torch.from_numpy(centroids),
            frame=self.affordance_target_frame,
        )
        return out.numpy().astype(np.float32)

    def _augment_sequence(self, sequence, t):
        """Apply a random scene map x -> Rz x + b to a sampled sequence (training only).

        Rz is a yaw about the vertical axis through the affordance point a_t, b adds an
        xy shift. Transformed: point clouds, affordance points, EE position (Rz p + b),
        EE orientation (Rz R_ee), action translation (Rz da; the sim applies it in the
        world frame: pos' = pos + a[:3]). Not transformed: the action rotation (the sim
        applies it in the EE frame: R' = R_ee @ dR, so dR is invariant under a world
        rotation) and the finger. Agent-state and part-motion targets are computed from
        the transformed sequence afterwards; a precomputed world-form part-motion delta
        is transformed by transform_world_delta.

        Returns (new sequence dict, (Rz, b)).
        """
        pivot = sequence["static_affordance"][t]
        Rz, b = sample_scene_transform(np.random, self.augment_yaw, self.augment_translation, pivot)
        Rz32, b32 = Rz.astype(np.float32), b.astype(np.float32)

        def pts(x):
            return (x.astype(np.float32) @ Rz32.T + b32).astype(x.dtype)

        def vecs(x):
            return (x.astype(np.float32) @ Rz32.T).astype(x.dtype)

        out = dict(sequence)
        out["static_affordance"] = pts(sequence["static_affordance"])
        out[self.pcd_key] = pts(sequence[self.pcd_key])
        state = sequence["state"].copy()
        state[:, 0:3] = pts(state[:, 0:3])
        state[:, 3:6] = vecs(state[:, 3:6])   # 6D = first two columns of R_ee
        state[:, 6:9] = vecs(state[:, 6:9])
        out["state"] = state
        action = sequence["action"].copy()
        action[:, 0:3] = vecs(action[:, 0:3])
        out["action"] = action
        return out, (Rz, b)

    def _build_pair_indices(self):
        """Build indices mapping flat index to (sequence_idx, timestep) pair"""
        self.pair_indices = []
        sequence_length = self.sampler.sequence_length

        # For each sequence, create pairs for each timestep where we can predict horizon steps ahead
        # We need at least horizon+1 timesteps in the sequence (current + horizon future)
        max_t = sequence_length - self.horizon

        for seq_idx in range(len(self.sampler)):
            for t in range(max_t):
                self.pair_indices.append((seq_idx, t))

    def get_balanced_sample_weights(self):
        """
        Compute per-sample weights for WeightedRandomSampler so that expert and
        interaction (rollout) episodes are sampled with equal probability.

        Returns:
            torch.DoubleTensor of shape [len(self)] with per-sample weights.
        """
        import torch

        episode_ends = self.replay_buffer.episode_ends[:]

        # Map each pair_index → episode_idx → expert or interaction
        is_expert = np.zeros(len(self.pair_indices), dtype=bool)
        for i, (seq_idx, _t) in enumerate(self.pair_indices):
            buffer_start_idx = self.sampler.indices[seq_idx][0]
            episode_idx = np.searchsorted(episode_ends, buffer_start_idx, side='right')
            is_expert[i] = self.expert_episode_mask[episode_idx]

        n_expert = is_expert.sum()
        n_interaction = len(is_expert) - n_expert

        cprint(f"  Balanced sampling: {n_expert:,} expert pairs, "
               f"{n_interaction:,} interaction pairs", "cyan")

        weights = np.zeros(len(self.pair_indices), dtype=np.float64)
        if n_expert > 0:
            weights[is_expert] = 1.0 / (2.0 * n_expert)
        if n_interaction > 0:
            weights[~is_expert] = 1.0 / (2.0 * n_interaction)

        return torch.from_numpy(weights)

    def __len__(self):
        return len(self.pair_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            {
                "observation": {
                    "agent_state": [agent_state_dim] or [n_obs_steps, agent_state_dim],
                    "static_affordance": [3] or [n_obs_steps, 3],
                    "pointcloud": [N, 3] or [n_obs_steps, N, 3],
                    "action_horizon": [horizon, action_dim]
                },
                "target": {
                    "agent_state_delta": [horizon, agent_state_dim],
                    "affordance_delta": [horizon, 3] or [horizon, 9]
                }
            }

        When use_pcd_transform_target=True, affordance_delta contains the 9-DoF rigid
        body transform from SVD: [rot_6d(6), tx, ty, tz].
        """
        seq_idx, t = self.pair_indices[idx]
        
        # Sample sequence from replay buffer
        sequence = self.sampler.sample_sequence(seq_idx)

        # Training-only augmentation (validation datasets have augment_* = 0)
        scene_transform = None
        if getattr(self, "augment_yaw", 0.0) > 0 or getattr(self, "augment_translation", 0.0) > 0:
            sequence, scene_transform = self._augment_sequence(sequence, t)
        
        # Extract data
        agent_pos = sequence["state"]  # [T, agent_state_dim]
        static_affordance = sequence["static_affordance"]  # [T, 3]
        point_cloud = sequence[self.pcd_key]  # [T, N, 3]
        action = sequence["action"]  # [T, action_dim]
        
        # Get observation steps: from t-n_obs_steps+1 to t (inclusive)
        # If n_obs_steps=1, this is just timestep t
        obs_start = max(0, t - self.n_obs_steps + 1)
        obs_end = t + 1
        
        if self.n_obs_steps == 1:
            # Single timestep: [agent_state_dim]
            current_agent_state = agent_pos[t]  # [agent_state_dim]
            current_pointcloud = point_cloud[t]  # [N, 3]
            current_affordance = static_affordance[t]  # [3]
        else:
            # Multiple timesteps: [n_obs_steps, agent_state_dim]
            # Pad with first timestep if we don't have enough history
            if obs_start > 0:
                # We have enough history, use it
                current_agent_state = agent_pos[obs_start:obs_end]  # [n_obs_steps, agent_state_dim]
                current_pointcloud = point_cloud[obs_start:obs_end]  # [n_obs_steps, N, 3]
                current_affordance = static_affordance[obs_start:obs_end]  # [n_obs_steps, 3]
            else:
                # Not enough history, pad with first timestep
                current_agent_state = agent_pos[:obs_end]  # [obs_end, agent_state_dim]
                current_pointcloud = point_cloud[:obs_end]  # [obs_end, N, 3]
                current_affordance = static_affordance[:obs_end]  # [obs_end, 3]
                # Pad to n_obs_steps
                if current_agent_state.shape[0] < self.n_obs_steps:
                    padding_size = self.n_obs_steps - current_agent_state.shape[0]
                    first_state = agent_pos[0:1]  # [1, agent_state_dim]
                    first_pc = point_cloud[0:1]  # [1, N, 3]
                    first_aff = static_affordance[0:1]  # [1, 3]
                    current_agent_state = np.concatenate([first_state.repeat(padding_size, axis=0), current_agent_state], axis=0)
                    current_pointcloud = np.concatenate([first_pc.repeat(padding_size, axis=0), current_pointcloud], axis=0)
                    current_affordance = np.concatenate([first_aff.repeat(padding_size, axis=0), current_affordance], axis=0)
        
        # Action horizon: actions from t to t+horizon-1
        action_horizon = action[t:t+self.horizon]  # [horizon, action_dim]
        
        # Future states (targets): from t+1 to t+horizon
        future_agent_states = agent_pos[t+1:t+1+self.horizon]  # [horizon, agent_state_dim]
        future_affordances = static_affordance[t+1:t+1+self.horizon]  # [horizon, 3]
        
        # Compute displacements (deltas) from current state
        # Use proper rotation composition for orientation dimensions
        agent_state_delta = compute_proper_state_delta(current_agent_state, future_agent_states)  # [horizon, agent_state_dim]
        
        # Compute affordance-space target (3D delta or 9-DoF PCD transform)
        if self.use_pcd_transform_target:
            # 9-DoF consecutive step-to-step rigid body transforms: [horizon, 9]
            # Step h predicts transform from t+h to t+h+1
            if self.has_precomputed_pcd_delta:
                # Direct lookup — no SVD at runtime
                pcd_deltas = sequence['pcd_rigid_body_delta_9dof']  # [T, 9]
                affordance_delta = pcd_deltas[t:t+self.horizon]  # [horizon, 9]
                if scene_transform is not None:
                    affordance_delta = transform_world_delta(affordance_delta, *scene_transform).astype(np.float32)
            else:
                # Fallback: compute SVD on-the-fly (consecutive deltas)
                affordance_delta = np.zeros((self.horizon, 9), dtype=np.float32)
                for h in range(self.horizon):
                    pcd_t = point_cloud[t + h]      # [N, 3]
                    pcd_t1 = point_cloud[t + 1 + h]  # [N, 3]
                    affordance_delta[h] = compute_pcd_rigid_body_delta_9dof(pcd_t, pcd_t1)
            if getattr(self, "affordance_target_frame", "world") != "world":
                # [rot6d(R_h), centroid(pcd_{t+h+1}) - centroid(pcd_{t+h})]
                affordance_delta = self._encode_frame(affordance_delta, point_cloud[t:t + self.horizon])
        else:
            # Standard 3D affordance delta
            if self.n_obs_steps == 1:
                current_affordance_for_delta = current_affordance  # [3]
            else:
                current_affordance_for_delta = current_affordance[-1]  # [3]
            affordance_delta = future_affordances - current_affordance_for_delta  # [horizon, 3]

        # Compute contact label: binary indicator of whether affordance moved at each step
        # contact[h] = 1 if ||aff_{t+h+1} - aff_{t+h}|| > threshold, else 0
        contact_threshold = getattr(self, 'contact_threshold', 1e-4)
        consecutive_aff = static_affordance[t:t+self.horizon+1]  # [horizon+1, 3]
        aff_deltas_norm = np.linalg.norm(
            consecutive_aff[1:] - consecutive_aff[:-1], axis=-1
        )  # [horizon]
        contact_label = (aff_deltas_norm > contact_threshold).astype(np.float32)  # [horizon]

        # Normalize action horizon using dataset-wide statistics
        # Standardization: (action - mean) / std
        normalized_action_horizon = (action_horizon - self.action_mean) / self.action_std

        # Normalize target deltas using dataset-wide statistics
        # Standardization: (delta - mean) / std
        normalized_agent_state_delta = (agent_state_delta - self.agent_state_delta_mean) / self.agent_state_delta_std
        normalized_affordance_delta = (affordance_delta - self.affordance_delta_mean) / self.affordance_delta_std

        # Convert to torch tensors
        observation = {
            "agent_state": torch.from_numpy(current_agent_state).float(),
            "static_affordance": torch.from_numpy(current_affordance).float(),
            "pointcloud": torch.from_numpy(current_pointcloud).float(),
            "action_horizon": torch.from_numpy(normalized_action_horizon).float(),
        }

        target = {
            "agent_state_delta": torch.from_numpy(normalized_agent_state_delta).float(),
            "affordance_delta": torch.from_numpy(normalized_affordance_delta).float(),
            "contact": torch.from_numpy(contact_label).float().unsqueeze(-1),  # [horizon, 1]
        }

        # KNN points and their deltas (for v6 K-point prediction)
        if self.k_nearest > 0:
            # current_pointcloud: [N, 3] (n_obs_steps=1) or [n_obs_steps, N, 3]
            if self.n_obs_steps == 1:
                pcd_for_knn = current_pointcloud  # [N, 3]
                aff_for_knn = current_affordance   # [3]
            else:
                pcd_for_knn = current_pointcloud[-1]  # [N, 3] - most recent
                aff_for_knn = current_affordance[-1]   # [3]

            dists = np.linalg.norm(pcd_for_knn - aff_for_knn, axis=-1)  # [N]
            knn_indices = np.argsort(dists)[:self.k_nearest]  # [K]
            knn_points = pcd_for_knn[knn_indices]  # [K, 3]

            # Compute KNN point deltas: future_pcd[knn_indices] - current_pcd[knn_indices]
            future_pcds = point_cloud[t+1:t+1+self.horizon]  # [horizon, N, 3]
            knn_point_delta = future_pcds[:, knn_indices, :] - knn_points  # [horizon, K, 3]

            # Normalize KNN point delta
            knn_mean = self.knn_point_delta_mean.reshape(self.k_nearest, 3)
            knn_std = self.knn_point_delta_std.reshape(self.k_nearest, 3)
            normalized_knn_point_delta = (knn_point_delta - knn_mean) / knn_std

            observation["knn_points"] = torch.from_numpy(knn_points).float()
            target["knn_point_delta"] = torch.from_numpy(normalized_knn_point_delta).float()

        return {
            "observation": observation,
            "target": target,
        }
    
    def get_validation_dataset(self):
        """Create validation dataset with same structure"""
        val_dataset = LWMDataset.__new__(LWMDataset)
        val_dataset.horizon = self.horizon
        val_dataset.n_obs_steps = self.n_obs_steps
        val_dataset.use_pcd_transform_target = self.use_pcd_transform_target
        val_dataset.skip_rotation_normalization = self.skip_rotation_normalization
        val_dataset.affordance_target_frame = getattr(self, "affordance_target_frame", "world")
        val_dataset.augment_yaw = 0.0          # never augment validation data
        val_dataset.augment_translation = 0.0
        val_dataset.has_precomputed_pcd_delta = self.has_precomputed_pcd_delta
        val_dataset.pad_before = self.pad_before
        val_dataset.pad_after = self.pad_after
        val_dataset.replay_buffer = self.replay_buffer
        val_dataset.train_mask = self.val_mask
        val_dataset.val_mask = self.train_mask  # Swap for val
        val_dataset.expert_episode_mask = self.expert_episode_mask
        val_dataset.new_config_episode_mask = self.new_config_episode_mask
        val_dataset.replay_episode_mask = self.replay_episode_mask
        val_dataset.unseen_episode_mask = self.unseen_episode_mask
        val_dataset.unseen_expert_episode_mask = getattr(self, 'unseen_expert_episode_mask', np.zeros_like(self.unseen_episode_mask))
        val_dataset.unseen_new_config_episode_mask = getattr(self, 'unseen_new_config_episode_mask', np.zeros_like(self.unseen_episode_mask))
        val_dataset.use_part_pointcloud = self.use_part_pointcloud
        val_dataset.num_points = self.num_points
        val_dataset.pcd_key = self.pcd_key
        # Share normalization statistics
        val_dataset.action_mean = self.action_mean
        val_dataset.action_std = self.action_std
        val_dataset.action_min = self.action_min
        val_dataset.action_max = self.action_max
        val_dataset.agent_state_delta_mean = self.agent_state_delta_mean
        val_dataset.agent_state_delta_std = self.agent_state_delta_std
        val_dataset.affordance_delta_mean = self.affordance_delta_mean
        val_dataset.affordance_delta_std = self.affordance_delta_std
        val_dataset.k_nearest = self.k_nearest
        val_dataset.knn_point_delta_mean = self.knn_point_delta_mean
        val_dataset.knn_point_delta_std = self.knn_point_delta_std
        val_dataset.contact_threshold = getattr(self, 'contact_threshold', 1e-4)

        # Create sequence sampler for validation
        val_dataset.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + 1,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_dataset.train_mask,
        )
        
        # Build pair indices for validation
        val_dataset._build_pair_indices()
        
        return val_dataset
    
    def get_expert_validation_dataset(self):
        """Create validation dataset with only expert episodes."""
        return self._get_split_validation_dataset(self.expert_episode_mask)
    
    def _get_split_validation_dataset(self, split_mask):
        """Create validation dataset over the held-out (val-split) episodes of one split.

        Uses self.val_mask (the val split of episode_splits.json). The returned dataset's
        train_mask/val_mask both hold the episodes it iterates.
        """
        val_dataset = self.get_validation_dataset()
        split_val_mask = self.val_mask & split_mask

        val_dataset.val_mask = split_val_mask.copy()
        val_dataset.train_mask = split_val_mask.copy()

        val_dataset.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + 1,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=split_val_mask,
        )
        val_dataset._build_pair_indices()
        return val_dataset

    def get_new_config_validation_dataset(self):
        """Create validation dataset with only new_config episodes."""
        return self._get_split_validation_dataset(self.new_config_episode_mask)

    def get_replay_validation_dataset(self):
        """Create validation dataset with only replay episodes."""
        return self._get_split_validation_dataset(self.replay_episode_mask)

    def get_unseen_validation_dataset(self):
        """Create validation dataset with all unseen object episodes (expert + new_config)."""
        return self._get_unseen_split_validation_dataset(self.unseen_episode_mask)

    def get_unseen_expert_validation_dataset(self):
        """Create validation dataset with only unseen expert episodes (unseen objects, expert demos)."""
        mask = getattr(self, 'unseen_expert_episode_mask', np.zeros_like(self.unseen_episode_mask))
        return self._get_unseen_split_validation_dataset(mask)

    def get_unseen_new_config_validation_dataset(self):
        """Create validation dataset with only unseen new_config episodes (unseen objects, rollout new_config)."""
        mask = getattr(self, 'unseen_new_config_episode_mask', np.zeros_like(self.unseen_episode_mask))
        return self._get_unseen_split_validation_dataset(mask)

    def _get_unseen_split_validation_dataset(self, unseen_mask):
        """Create validation dataset filtered to the given unseen episode mask."""
        if not unseen_mask.any():
            return self._get_split_validation_dataset(unseen_mask)

        val_dataset = self.get_validation_dataset()
        val_dataset.val_mask = unseen_mask.copy()
        val_dataset.train_mask = unseen_mask.copy()  # sampler iterates over this

        val_dataset.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + 1,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=unseen_mask,
        )
        val_dataset._build_pair_indices()
        return val_dataset
