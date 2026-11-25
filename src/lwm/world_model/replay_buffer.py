"""
Custom ReplayBuffer for LWM training.
Simplified version that only includes what's needed.
"""

import os
import numpy as np
import zarr
import pickle
from collections import defaultdict
from termcolor import cprint
from tqdm import tqdm

from lwm.world_model import packed


def farthest_point_sampling(points, n_samples):
    """Deterministic FPS: downsample points to n_samples.

    Always starts from index 0 and uses argmax (deterministic tie-breaking),
    matching the FPS used in extraction (object_geometry.py).

    Args:
        points: [N, 3] numpy array
        n_samples: number of points to select
    Returns:
        indices: [n_samples] sorted numpy array of selected indices
    """
    n_points = points.shape[0]
    if n_points <= n_samples:
        return np.arange(n_points)

    selected = [0]
    distances = np.full(n_points, np.inf)

    for _ in range(n_samples - 1):
        last = points[selected[-1]]
        dist_to_last = np.linalg.norm(points - last, axis=1)
        distances = np.minimum(distances, dist_to_last)
        selected.append(np.argmax(distances))

    return np.sort(selected)


class ReplayBuffer:
    """
    Simplified ReplayBuffer for loading data from zarr/pickle files.
    Only supports what's needed for LWM training.
    """
    
    def __init__(
        self,
        all_path_list,
        episode_lengths,
        accumulated_episode_lengths,
        keys,
        load_per_step=True,
        is_pickle=True,
        pcd_subsample=None,
    ):
        self.all_path_list = all_path_list
        self.episode_lengths = episode_lengths
        self.accumulated_episode_lengths = accumulated_episode_lengths
        self.keys_ = keys
        self.load_per_step = load_per_step
        self.is_pickle = is_pickle
        self.pcd_subsample = pcd_subsample or {}
        self._fps_cache = {}  # (episode_path, key) -> FPS indices
    
    @classmethod
    def copy_from_multiple_path(
        cls,
        path_list,
        keys=None,
        load_per_step=True,
        is_pickle=True,
        target_action="action",
        pcd_subsample=None,
    ):
        """
        Load data from multiple paths and create a ReplayBuffer.
        """
        cprint(f"Loading data from {len(path_list)} path(s)...", "yellow")
        
        episode_lengths = []
        valid_path_list = []
        
        # Load episode lengths and filter out empty episodes
        for zarr_path in tqdm(path_list):
            zarr_path = os.fsdecode(zarr_path)  # ensure str (handles bytes & numpy.str_)
            n_packed = packed.num_steps(zarr_path)
            if n_packed is not None:
                # Episode of a packed dataset (lwm.world_model.packed)
                all_substeps = range(n_packed)
            elif is_pickle:
                all_substeps = [f for f in os.listdir(zarr_path) if f.endswith(".pkl")]
                all_substeps = sorted(all_substeps, key=lambda x: int(x.split(".")[0]))
            else:
                all_substeps = sorted([f for f in os.listdir(zarr_path) if os.path.isdir(os.path.join(zarr_path, f))], 
                                     key=lambda x: int(x))
            
            if len(all_substeps) > 0:
                episode_lengths.append(len(all_substeps))
                valid_path_list.append(zarr_path)
            else:
                cprint(f"Skipping empty episode: {zarr_path}", "yellow")
        
        if len(valid_path_list) == 0:
            raise ValueError("No valid episodes found in dataset (all episodes are empty)")
        
        episode_lengths = np.array(episode_lengths)
        accumulated_episode_lengths = np.cumsum(episode_lengths)
        
        cprint(f"Loaded {len(valid_path_list)} valid episodes (skipped {len(path_list) - len(valid_path_list)} empty)", "green")
        
        return cls(
            all_path_list=valid_path_list,
            episode_lengths=episode_lengths,
            accumulated_episode_lengths=accumulated_episode_lengths,
            keys=keys,
            load_per_step=load_per_step,
            is_pickle=is_pickle,
            pcd_subsample=pcd_subsample,
        )
    
    def get_data_disk(self, start_idx, end_idx):
        """
        Get data from disk for the given index range.
        Matches the implementation from replay_buffer_disk.py
        """
        # Clamp end_idx to valid range
        total_length = self.accumulated_episode_lengths[-1] if len(self.accumulated_episode_lengths) > 0 else 0
        if end_idx > total_length:
            end_idx = total_length
        
        # Find the corresponding episode/file
        start_episode_idx = np.searchsorted(
            self.accumulated_episode_lengths, start_idx, side="right"
        )
        end_episode_idx = np.searchsorted(
            self.accumulated_episode_lengths, end_idx, side="right"
        )
        
        # Clamp indices to valid range (searchsorted can return len(array) if value >= max)
        max_episode_idx = len(self.accumulated_episode_lengths) - 1
        if start_episode_idx > max_episode_idx:
            start_episode_idx = max_episode_idx
        if end_episode_idx > max_episode_idx:
            end_episode_idx = max_episode_idx
        
        if start_episode_idx == end_episode_idx:
            # Only one episode/file
            # Convert to local indices within the episode
            episode_start = self.accumulated_episode_lengths[start_episode_idx - 1] if start_episode_idx > 0 else 0
            local_start = start_idx - episode_start
            local_end = end_idx - episode_start
            
            # Handle edge case where start_idx == end_idx (empty range)
            if local_start >= local_end and packed.lookup(self.all_path_list[start_episode_idx]) is not None:
                return self._get_data_single_file(self.all_path_list[start_episode_idx], local_start, local_start)
            if local_start >= local_end:
                # Return empty arrays with correct shape
                # Load one sample to get the shape
                if self.is_pickle:
                    all_files = sorted([f for f in os.listdir(self.all_path_list[start_episode_idx]) if f.endswith(".pkl")], 
                                     key=lambda x: int(x.split(".")[0]))
                else:
                    all_files = sorted([f for f in os.listdir(self.all_path_list[start_episode_idx]) if os.path.isdir(os.path.join(self.all_path_list[start_episode_idx], f))], 
                                      key=lambda x: int(x))
                
                if len(all_files) == 0:
                    raise ValueError(f"Episode {start_episode_idx} has no files")
                
                # Get shape from first file
                first_file = all_files[0]
                if self.is_pickle:
                    step_path = os.path.join(self.all_path_list[start_episode_idx], first_file)
                    with open(step_path, "rb") as f:
                        sample_data = pickle.load(f)
                else:
                    step_path = os.path.join(self.all_path_list[start_episode_idx], first_file)
                    group = zarr.open(step_path, "r")
                    src_store = group.store
                    src_root = zarr.group(src_store)
                    sample_data = {key: src_root["data"][key][:] for key in self.keys_}
                
                # Return empty arrays with correct shape (respecting pcd_subsample)
                ret_data = {}
                for key in self.keys_:
                    shape = list(sample_data[key].shape)
                    if key in self.pcd_subsample and len(shape) >= 2:
                        shape[-2] = min(shape[-2], self.pcd_subsample[key])
                    ret_data[key] = np.zeros((0,) + tuple(shape[1:]), dtype=sample_data[key].dtype)
                return ret_data
            
            return self._get_data_single_file(
                self.all_path_list[start_episode_idx], local_start, local_end
            )
        
        # Multiple files - concatenate data from multiple episodes
        ret_data = None
        # Ensure we don't go beyond valid episode indices
        max_valid_ep_idx = len(self.accumulated_episode_lengths) - 1
        for ep_idx in range(start_episode_idx, min(end_episode_idx + 1, max_valid_ep_idx + 1)):
            ep_start = self.accumulated_episode_lengths[ep_idx - 1] if ep_idx > 0 else 0
            # ep_end is the accumulated length at this episode index
            ep_end = self.accumulated_episode_lengths[ep_idx]
            
            if ep_idx == start_episode_idx:
                local_start = start_idx - ep_start
                local_end = None
            elif ep_idx == end_episode_idx:
                local_start = 0
                local_end = end_idx - ep_start
            else:
                local_start = 0
                local_end = None
            
            ep_data = self._get_data_single_file(
                self.all_path_list[ep_idx], local_start, local_end
            )
            
            # Subsample point clouds using deterministic FPS to ensure consistent
            # sizes and stable point indices across timesteps
            ep_path = self.all_path_list[ep_idx]
            for key in self.keys_:
                if key in self.pcd_subsample:
                    target_n = self.pcd_subsample[key]
                    arr = ep_data[key]
                    if arr.ndim >= 2 and arr.shape[0] > 0 and arr.shape[-2] > target_n:
                        cache_key = (ep_path, key)
                        if cache_key not in self._fps_cache:
                            self._fps_cache[cache_key] = farthest_point_sampling(arr[0], target_n)
                        ep_data[key] = arr[..., self._fps_cache[cache_key], :]

            if ret_data is None:
                ret_data = ep_data
            else:
                for key in self.keys_:
                    ret_data[key] = np.concatenate([ret_data[key], ep_data[key]], axis=0)
        
        return ret_data
    
    def _get_data_single_file(self, zarr_path, start_idx, end_idx):
        """Get data from a single file (or a packed-dataset episode)."""
        if packed.lookup(zarr_path) is not None:
            if self.pcd_subsample:
                raise NotImplementedError("pcd_subsample is not supported for packed datasets")
            return packed.read_episode(zarr_path, self.keys_, start_idx, end_idx)
        return self._get_data_per_step(zarr_path, start_idx, end_idx)
    
    def _get_data_per_step(self, zarr_path, start_idx, end_idx):
        """Get data when load_per_step=True."""
        ret_data = defaultdict(list)
        
        # Get sorted list of files
        if self.is_pickle:
            all_files = sorted([f for f in os.listdir(zarr_path) if f.endswith(".pkl")], 
                             key=lambda x: int(x.split(".")[0]))
        else:
            all_files = sorted([f for f in os.listdir(zarr_path) if os.path.isdir(os.path.join(zarr_path, f))], 
                              key=lambda x: int(x))
        
        all_steps = len(all_files)
        
        start_idx = start_idx + all_steps if start_idx < 0 else start_idx
        end_idx = all_steps if end_idx is None else (end_idx + all_steps if end_idx < 0 else end_idx)
        
        # Handle empty range case
        if start_idx >= end_idx:
            # Return empty arrays with correct shape
            if len(all_files) == 0:
                raise ValueError(f"Episode has no files: {zarr_path}")
            
            # Get shape from first file
            first_file = all_files[0]
            if self.is_pickle:
                step_path = os.path.join(zarr_path, first_file)
                file_size = os.path.getsize(step_path)
                if file_size == 0:
                    raise ValueError(f"Pickle file is empty (0 bytes): {step_path}")
                with open(step_path, "rb") as f:
                    sample_data = pickle.load(f)
            else:
                step_path = os.path.join(zarr_path, first_file)
                group = zarr.open(step_path, "r")
                src_store = group.store
                src_root = zarr.group(src_store)
                sample_data = {key: src_root["data"][key][:] for key in self.keys_}
            
            # Return empty arrays with correct shape (respecting pcd_subsample)
            ret_data = {}
            for key in self.keys_:
                shape = list(sample_data[key].shape)
                if key in self.pcd_subsample and len(shape) >= 2:
                    shape[-2] = min(shape[-2], self.pcd_subsample[key])
                ret_data[key] = np.zeros((0,) + tuple(shape[1:]), dtype=sample_data[key].dtype)
            return ret_data

        for step_idx in range(start_idx, end_idx):
            file_name = all_files[step_idx]
            if self.is_pickle:
                step_path = os.path.join(zarr_path, file_name)
                file_size = os.path.getsize(step_path)
                if file_size == 0:
                    raise ValueError(f"Pickle file is empty (0 bytes): {step_path}")
                with open(step_path, "rb") as f:
                    data = pickle.load(f)
                for key in self.keys_:
                    val = data[key][:]
                    if key in self.pcd_subsample and val.ndim >= 2:
                        target_n = self.pcd_subsample[key]
                        n_pts = val.shape[-2]
                        if n_pts > target_n:
                            cache_key = (zarr_path, key)
                            if cache_key not in self._fps_cache:
                                pts_2d = val.reshape(-1, val.shape[-1]) if val.ndim > 2 else val
                                self._fps_cache[cache_key] = farthest_point_sampling(pts_2d, target_n)
                            val = val[..., self._fps_cache[cache_key], :]
                    ret_data[key].append(val)
            else:
                step_path = os.path.join(zarr_path, file_name)
                group = zarr.open(step_path, "r")
                src_store = group.store
                src_root = zarr.group(src_store)
                for key in self.keys_:
                    ret_data[key].append(src_root["data"][key][:])
        
        for key in self.keys_:
            if len(ret_data[key]) == 0:
                raise ValueError(
                    f"No data collected for key '{key}'. start_idx={start_idx}, end_idx={end_idx}, "
                    f"all_steps={all_steps}, zarr_path={zarr_path}"
                )
            ret_data[key] = np.concatenate(ret_data[key], axis=0)
        
        return ret_data
    
    
    @property
    def episode_ends(self):
        """Return accumulated episode lengths (episode end indices)."""
        return self.accumulated_episode_lengths
    
    def keys(self):
        """Return list of data keys."""
        return self.keys_
