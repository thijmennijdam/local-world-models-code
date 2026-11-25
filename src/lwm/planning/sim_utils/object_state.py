"""Utilities for extracting and managing object state."""
import numpy as np
import pickle as pkl


def _find_object_name(state, object_name):
    """Find actual object name in state dict."""
    if object_name in state:
        return object_name
    object_name_lower = object_name.lower()
    if object_name_lower in state:
        return object_name_lower
    object_keys = list(state.keys())
    if len(object_keys) > 0:
        return object_keys[0]
    return None


def extract_object_state(state, object_name):
    """Extract object state from state file."""
    actual_name = _find_object_name(state["object_base_position"], object_name)
    if actual_name is None:
        raise ValueError("No object found in state file")
    
    base_pos = np.array(state["object_base_position"][actual_name], dtype=np.float32)
    base_quat = np.array(state["object_base_orientation"][actual_name], dtype=np.float32)
    joint_angles = state["object_joint_angle_dicts"][actual_name]
    joint_names = state["object_joint_name_dicts"][actual_name]
    joint_dict = {str(name): float(angle) for name, angle in zip(joint_names, joint_angles)}
    object_scale = float(state.get("object_sizes", {}).get(actual_name, 1.0))
    
    return {
        "base_pos": base_pos,
        "base_quat": base_quat,
        "joint_dict": joint_dict,
        "object_scale": object_scale,
    }


def get_object_scale_from_state(state_file_path, object_name):
    """Get object scale from state file."""
    with open(state_file_path, "rb") as f:
        state = pkl.load(f)
    actual_name = _find_object_name(state.get("object_sizes", {}), object_name)
    return float(state.get("object_sizes", {}).get(actual_name, 1.0)) if actual_name else 1.0


def get_object_scale_from_state_dict(state_dict, object_name):
    """Get object scale from state dictionary (already loaded).

    Args:
        state_dict: Dictionary containing state data
        object_name: Name of the object to look up

    Returns:
        Object scale as float, defaults to 1.0 if not found
    """
    if state_dict is None:
        return 1.0
    actual_name = _find_object_name(state_dict.get("object_sizes", {}), object_name)
    return float(state_dict.get("object_sizes", {}).get(actual_name, 1.0)) if actual_name else 1.0

