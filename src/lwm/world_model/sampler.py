"""
Custom SequenceSampler for LWM training.
Simplified version that only includes what's needed.
"""

import numpy as np
from typing import Optional


def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    """
    Create indices for sequence sampling.
    Simplified version without numba for easier maintenance.
    """
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)
    
    indices = []
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            
            # Skip if empty range (shouldn't happen, but handle gracefully)
            if buffer_start_idx >= buffer_end_idx:
                continue
            
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    
    return np.array(indices, dtype=np.int64)


class SequenceSampler:
    """
    Simplified SequenceSampler for LWM training.
    """
    
    def __init__(
        self,
        replay_buffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        episode_mask: Optional[np.ndarray] = None,
    ):
        assert sequence_length >= 1
        
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
        
        if np.any(episode_mask):
            indices = create_indices(
                episode_ends,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
            )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)
        
        self.indices = indices
        self.keys = list(keys)
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
    
    def __len__(self):
        return len(self.indices)
    
    def sample_sequence(self, idx):
        """
        Sample a sequence from the replay buffer.
        """
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]
        
        # Get data from replay buffer
        data_from_buffer = self.replay_buffer.get_data_disk(
            start_idx=buffer_start_idx, end_idx=buffer_end_idx
        )
        
        result = dict()
        for key in self.keys:
            input_arr = data_from_buffer[key]
            sample = input_arr[:]
            
            # Handle padding
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            
            result[key] = data
        
        return result
