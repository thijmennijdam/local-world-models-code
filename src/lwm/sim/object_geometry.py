import os
import numpy as np
import pybullet as p
import trimesh
import json
from contextlib import contextmanager
import sys
import warnings
import logging


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, "w")

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def transform_points(points: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    R = quat_to_matrix(quat)
    return points @ R.T + pos[None, :]


class ArticulatedObjectGeometry:
    """Create articulated object geometry and call vertices from different parts of the .obj files."""

    def __init__(self, urdf_path: str, object_scale=1.0):
        """Initialize the articulated object geometry."""
        import time as _time
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.object_scale = object_scale
        self.urdf_path = urdf_path
        # Initialize PyBullet
        _t = _time.time()
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(os.path.dirname(self.urdf_path), physicsClientId=self.client_id)
        print(f"    [geom] p.connect(DIRECT): {_time.time() - _t:.2f}s")

        # Load URDF
        _t = _time.time()
        with suppress_stdout():
            self.body = p.loadURDF(
                self.urdf_path,
                useFixedBase=False,
                physicsClientId=self.client_id,
                globalScaling=self.object_scale,
            )
        print(f"    [geom] loadURDF: {_time.time() - _t:.2f}s")

        # Gather visual information
        _t = _time.time()
        self.visuals = self._gather_visuals()
        self._joint_info = self._get_joint_info()
        print(f"    [geom] gather visuals + joint info: {_time.time() - _t:.2f}s")
        # Cache for local-frame points to ensure point-to-point correspondence across timesteps
        self._local_points_cache = {}
        self._handle_points_cache = {}

    def _gather_visuals(self):
        """Gather visual (mesh) information and local transforms for all links.

        Stores, per link, a list of dicts with keys: 'mesh_file', 'local_pos', 'local_quat'.
        """
        visuals = {}
        vis_info = p.getVisualShapeData(self.body, physicsClientId=self.client_id)

        for item in vis_info:
            # PyBullet visual shape tuple structure:
            # (bodyUniqueId, linkIndex, visualGeometryType, dimensions, filename,
            #  localVisualFramePosition, localVisualFrameOrientation, rgbaColor, ...)
            _, link_index, _, _, mesh_file, local_pos, local_quat, _rgba = item[:8]
            mesh_file = mesh_file.decode("utf-8") if isinstance(mesh_file, (bytes, bytearray)) else str(mesh_file)
            mesh_file = os.path.normpath(mesh_file)
            visuals.setdefault(link_index, []).append(
                {
                    "mesh_file": mesh_file,
                    "local_pos": np.array(local_pos, dtype=np.float32),
                    "local_quat": np.array(local_quat, dtype=np.float32),
                }
            )
        return visuals

    def _get_joint_info(self):
        """Get joint name to index mapping."""
        joint_info = {}
        for j in range(p.getNumJoints(self.body, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.body, j, physicsClientId=self.client_id)
            jname = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            joint_info[jname] = j
        return joint_info

    def _load_mesh_points(self, mesh_path, n_samples=350, method="surface_sampling"):
        """Load points from a mesh file."""
        # Suppress trimesh texture warnings (they use logging, not warnings)
        trimesh_logger = logging.getLogger("trimesh")
        original_level = trimesh_logger.level
        trimesh_logger.setLevel(logging.ERROR)
        mesh = trimesh.load(mesh_path, force="mesh")
        trimesh_logger.setLevel(original_level)
        if method == "surface_sampling":
            verts, _ = trimesh.sample.sample_surface(mesh, n_samples)
        elif method == "vertex_sampling":
            verts = mesh.vertices
        else:
            raise ValueError(f"Invalid method: {method}")
        verts = np.array(verts, dtype=np.float32)
        return verts

    def set_state(self, base_pos, base_quat, joint_angles):
        """Set the object state."""
        p.resetBasePositionAndOrientation(
            self.body, base_pos.tolist(), base_quat.tolist(), physicsClientId=self.client_id
        )

        # Set joint angles
        for joint_name, angle in joint_angles.items():
            if joint_name in self._joint_info:
                p.resetJointState(self.body, self._joint_info[joint_name], float(angle), physicsClientId=self.client_id)

    def get_handle_ids(self):
        object_dir = os.path.dirname(self.urdf_path)
        json_path = os.path.join(object_dir, "result_after_merging.json")
        results_after_merging = json.load(open(json_path, "r"))
        handle_ids = set()

        def walk(node):
            if isinstance(node, dict):
                # If this node names a handle, collect its objs
                if node.get("name", "").lower() == "handle" and "objs" in node:
                    for obj_name in node["objs"]:
                        # obj_name like 'original-69' (without extension)
                        base = os.path.basename(obj_name)
                        handle_ids.add(base)
                # Recurse into children
                if "children" in node:
                    for child in node["children"]:
                        walk(child)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(results_after_merging)

        return list(handle_ids)

    def farthest_point_sampling(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Farthest Point Sampling (FPS) for point cloud downsampling.
        Iteratively selects points that are maximally distant from already selected points.

        Args:
            points: Input point cloud of shape (N, 3)
            n_samples: Number of points to sample

        Returns:
            Sampled points of shape (n_samples, 3)
        """
        N = points.shape[0]
        if n_samples >= N:
            return points

        # Initialize with a random point
        selected_indices = [0]
        distances = np.full(N, np.inf, dtype=np.float32)

        for _ in range(n_samples - 1):
            # Get last selected point
            last_point = points[selected_indices[-1]]

            # Update distances: for each point, keep minimum distance to any selected point
            dist_to_last = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, dist_to_last)

            # Select point with maximum distance to the set of selected points
            farthest_idx = np.argmax(distances)
            selected_indices.append(farthest_idx)

        return points[selected_indices]

    def get_part_point_cloud(self, part_index, total_points=512, reuse_local_points=True):
        """Get point cloud for a specific part.

        When reuse_local_points=True (default), caches the local-frame points
        (mesh sampling + FPS) on the first call and reuses them for subsequent
        calls. Only the link world transform is reapplied each time. This ensures
        1:1 point index correspondence across timesteps — the same index always
        refers to the same physical point on the mesh.
        """
        cache_key = (part_index, total_points)

        if reuse_local_points and cache_key in self._local_points_cache:
            points_local = self._local_points_cache[cache_key]
            handle_points_local = self._handle_points_cache[cache_key]
        else:
            # Sample mesh surface points, apply local mesh transforms, then FPS
            points_all = []
            handle_ids = self.get_handle_ids()
            handle_points_list = []

            for vis in self.visuals[part_index]:
                mesh_path = vis["mesh_file"]
                base = os.path.basename(mesh_path).lower()

                pts = self._load_mesh_points(mesh_path, method="surface_sampling")
                pts = transform_points(pts * self.object_scale, vis["local_pos"], vis["local_quat"])
                points_all.append(pts)

                if base.split(".")[0] in handle_ids:
                    hpts = self._load_mesh_points(mesh_path, method="vertex_sampling")
                    hpts = transform_points(hpts * self.object_scale, vis["local_pos"], vis["local_quat"])
                    handle_points_list.append(hpts)

            points_local = self.farthest_point_sampling(np.concatenate(points_all, axis=0), total_points)

            if len(handle_points_list) > 0:
                handle_points_local = np.concatenate(handle_points_list, axis=0)
            else:
                raise ValueError("No handle points found")

            if reuse_local_points:
                self._local_points_cache[cache_key] = points_local.copy()
                self._handle_points_cache[cache_key] = handle_points_local.copy()

        # Transform cached local-frame points to world frame using current link pose
        ls = p.getLinkState(self.body, part_index, physicsClientId=self.client_id)
        link_world_pos = np.array(ls[4], dtype=np.float32)
        link_world_quat = np.array(ls[5], dtype=np.float32)

        points_world = transform_points(points_local, link_world_pos, link_world_quat)
        handle_points_world = transform_points(handle_points_local, link_world_pos, link_world_quat)

        return points_world.astype(np.float32), handle_points_world.astype(np.float32)

    def get_full_object_point_cloud(self, total_points_per_link: int = 350, total_points: int = 2048) -> np.ndarray:
        """Get a sampled point cloud covering the full object across all links.

        This samples surface points from each link's meshes, transforms them
        into world coordinates using the current simulated state, concatenates
        them, and downsamples using farthest point sampling.
        """
        points_all_links = []

        for link_index, visuals in self.visuals.items():
            if link_index == -1:
                base_pos, base_quat = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client_id)
                link_world_pos = np.array(base_pos, dtype=np.float32)
                link_world_quat = np.array(base_quat, dtype=np.float32)
            else:
                ls = p.getLinkState(self.body, link_index, physicsClientId=self.client_id)
                link_world_pos = np.array(ls[4], dtype=np.float32)
                link_world_quat = np.array(ls[5], dtype=np.float32)

            link_points = []
            for vis in visuals:
                mesh_path = vis["mesh_file"]
                # Use raw mesh vertices; simple and fast
                pts = self._load_mesh_points(mesh_path, n_samples=total_points_per_link, method="vertex_sampling")
                # Apply visual local transform before link/world
                pts = transform_points(pts * self.object_scale, vis["local_pos"], vis["local_quat"])
                link_points.append(pts)

            if len(link_points) == 0:
                continue

            link_points = np.concatenate(link_points, axis=0)
            link_points = transform_points(link_points, link_world_pos, link_world_quat)
            points_all_links.append(link_points)

        if len(points_all_links) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        full_points = np.concatenate(points_all_links, axis=0)
        full_points = self.farthest_point_sampling(full_points, total_points)
        return full_points.astype(np.float32)

    def close(self):
        """Close the PyBullet connection and clean up resources."""
        p.disconnect(self.client_id)
