"""Utilities for loading and using articulation points during evaluation."""
import numpy as np
import os
import json
import random
import pybullet as p
from lwm.sim.utils import get_joint_id_from_name


# Global cache for articulation points JSON
_articulation_points_cache = {}


def load_articulation_points_json(dataset_path):
    """
    Load articulation points JSON file from dataset directory.

    Searches for articulation_points.json at the root level first, then
    recursively in subdirectories (e.g. per-object or per-epoch dirs).
    All found files are merged into a single dict.

    Parameters
    ----------
    dataset_path : str
        Path to the extracted dataset directory

    Returns
    -------
    dict or None
        Dictionary with structure {object_id: {handle_joint_id: [points]}}, or None if not found
    """
    if dataset_path in _articulation_points_cache:
        return _articulation_points_cache[dataset_path]

    # Try root-level file first
    json_path = os.path.join(dataset_path, "articulation_points.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        _articulation_points_cache[dataset_path] = data
        return data

    # Search subdirectories and merge all found files
    merged = {}
    for root, _dirs, files in os.walk(dataset_path):
        if "articulation_points.json" in files:
            with open(os.path.join(root, "articulation_points.json"), "r") as f:
                sub_data = json.load(f)
            for obj_id, joints in sub_data.items():
                if obj_id not in merged:
                    merged[obj_id] = {}
                for joint_key, points in joints.items():
                    if joint_key not in merged[obj_id]:
                        merged[obj_id][joint_key] = []
                    merged[obj_id][joint_key].extend(points)

    if not merged:
        _articulation_points_cache[dataset_path] = None
        return None

    _articulation_points_cache[dataset_path] = merged
    return merged


def sample_articulation_point(articulation_points, object_id, handle_joint_id):
    """
    Sample a random articulation point from saved points.
    
    Parameters
    ----------
    articulation_points : dict
        Dictionary with structure {object_id: {handle_joint_id: [points]}}
    object_id : str
        Object ID
    handle_joint_id : int
        Handle joint ID
        
    Returns
    -------
    np.ndarray or None
        Randomly sampled point in handle-local coordinates, or None if not found
    """
    if articulation_points is None:
        return None
    
    if object_id not in articulation_points:
        return None
    
    handle_key = str(handle_joint_id)
    obj_data = articulation_points[object_id]
    if handle_key not in obj_data:
        # Fallback: use points from any other joint on the same object.
        # Articulation points are in handle-local frame, so they transfer
        # across joints with the same handle geometry (e.g. drawers).
        all_points = [pt for pts in obj_data.values() for pt in pts]
        if not all_points:
            return None
        point = random.choice(all_points)
        return np.array(point, dtype=np.float32)

    points = obj_data[handle_key]
    if len(points) == 0:
        return None

    # Randomly sample one point
    point = random.choice(points)
    return np.array(point, dtype=np.float32)


def get_handle_link_transform_from_joint(env, object_name, handle_joint_id, mobility_info):
    """
    Get the handle link transform (parent link of the handle joint) in world frame.
    
    Parameters
    ----------
    env : object
        Environment object with urdf_ids, urdf_paths, id attributes
    object_name : str
        Name of the object
    handle_joint_id : int
        PyBullet joint ID of the handle
    mobility_info : list
        Mobility information from mobility_v2.json
        
    Returns
    -------
    T_body_to_world : np.ndarray
        4x4 transformation matrix from handle link frame to world frame
    """
    obj_name = object_name.lower()
    obj_id = env.urdf_ids[obj_name]
    
    # Find the joint info in mobility_info
    parent_joint_idx = None
    for joint_info in mobility_info:
        joint_name = "joint_{}".format(joint_info["id"])
        joint_idx = get_joint_id_from_name(env, obj_name, joint_name)
        if joint_idx == handle_joint_id:
            parent_joint_name = "joint_{}".format(joint_info["parent"])
            parent_joint_idx = get_joint_id_from_name(env, obj_name, parent_joint_name)
            break
    
    if parent_joint_idx is None:
        raise ValueError(f"Could not find parent joint for handle joint {handle_joint_id}")
    
    # Get parent link state (this is the handle link)
    parent_link_state = p.getLinkState(
        obj_id, parent_joint_idx, physicsClientId=env.id
    )
    link_urdf_world_pos, link_urdf_world_orn = (
        parent_link_state[0],
        parent_link_state[1],
    )
    
    # Build transformation matrix from handle link frame to world frame
    T_body_to_world = np.eye(4)
    T_body_to_world[:3, :3] = np.array(
        p.getMatrixFromQuaternion(link_urdf_world_orn)
    ).reshape(3, 3)
    T_body_to_world[:3, 3] = link_urdf_world_pos
    
    return T_body_to_world


def transform_handle_local_to_world(point_handle_local, T_body_to_world):
    """
    Transform a point from handle-local frame to world frame.
    
    Parameters
    ----------
    point_handle_local : np.ndarray
        Point in handle-local frame, shape (3,)
    T_body_to_world : np.ndarray
        4x4 transformation matrix from handle link frame to world frame
        
    Returns
    -------
    point_world : np.ndarray
        Point in world frame, shape (3,)
    """
    R_handle = T_body_to_world[:3, :3]
    handle_pos_world = T_body_to_world[:3, 3]
    point_world = handle_pos_world + R_handle @ point_handle_local
    return point_world
