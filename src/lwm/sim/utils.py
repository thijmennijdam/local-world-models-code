import os
import yaml
import numpy as np
from PIL import Image
import os.path as osp
import pybullet as p
import os
import json
import multiprocessing
from multiprocessing import Process
import multiprocessing.pool
import pickle
import copy
import importlib
from PIL import Image, ImageSequence
import multiprocessing
import trimesh
import random
import scipy
from lwm.sim.object_dicts import (
    partnet_mobility_dict,
    sapaien_cannot_vhacd_part_dict,
)
from typing import List, Optional
from scipy import ndimage

from scipy.spatial.transform import Rotation as R

from lwm.paths import PROJECT_DIR

# PartNet-Mobility dataset root: PARTNET_DATASET_DIR, else <repo>/data/partnet-mobility
if "PARTNET_DATASET_DIR" in os.environ:
    data_dir = os.path.abspath(os.environ["PARTNET_DATASET_DIR"])
else:
    data_dir = os.path.abspath(os.path.join(PROJECT_DIR, "data", "partnet-mobility"))

# Stored states/configs hold absolute URDF paths from the machine that generated them.
# Everything after one of these directory names is re-rooted onto data_dir.
_PARTNET_ROOT_MARKERS = ("/partnet-mobility/", "/partnet_dataset/", "/ArticuBot/data/dataset/")


def reroot_partnet_path(urdf_path):
    """Map a stored PartNet-Mobility URDF path onto the local dataset root (data_dir).

    Paths that exist locally are returned unchanged.
    """
    if not os.path.exists(urdf_path):
        for marker in _PARTNET_ROOT_MARKERS:
            if marker in urdf_path:
                return os.path.join(data_dir, urdf_path.split(marker, 1)[1])
    return urdf_path


default_config = {
    "gui": False,
}


def load_gif(gif_path):
    img = Image.open(gif_path)
    # Extract each frame from the GIF and convert to RGB
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(img)]
    # Convert each frame to a numpy array
    frames_arrays = [np.array(frame) for frame in frames]
    return frames_arrays


def build_up_env(
    task_config=None,
    solution_path=None,
    task_name=None,
    restore_state_file=None,
    return_env_class=False,
    render=False,
    randomize=False,
    obj_id=0,
    env_name=None,
    action_space="delta-translation",
    random_object_translation=None,
    partnet_dataset_dir=None,
    **kwargs,
):
    # Parse config YAML to extract object metadata
    config = yaml.safe_load(open(task_config, "r"))
    link_name = "link_0"
    init_angle = None
    object_name = None
    for config_dict in config:
        if "name" in config_dict:
            object_name = config_dict["name"].lower()
        if "link_name" in config_dict:
            link_name = config_dict["link_name"]
        if "init_angle" in config_dict:
            init_angle = config_dict["init_angle"]

    save_config = copy.deepcopy(default_config)
    save_config["config_path"] = task_config
    save_config["task_name"] = task_name
    save_config["object_name"] = object_name
    save_config["link_name"] = link_name
    save_config["init_angle"] = init_angle
    save_config["restore_state_file"] = restore_state_file
    save_config["gui"] = render
    save_config["randomize"] = randomize
    save_config["obj_id"] = obj_id
    save_config["translation_mode"] = action_space
    save_config["random_object_translation"] = random_object_translation
    save_config["partnet_dataset_dir"] = partnet_dataset_dir
    for key, value in kwargs.items():
        save_config[key] = value

    if env_name is not None:
        module = importlib.import_module("lwm.sim.envs.{}".format(env_name))
        env_class = getattr(module, env_name)
    elif solution_path is not None:
        # "manipulation" is the former name of the lwm.sim package (it may appear in
        # stored configs).
        if solution_path.replace("/", ".") == "manipulation":
            solution_path = "lwm.sim"
        module = importlib.import_module(
            "{}.{}".format(solution_path.replace("/", "."), task_name)
        )
        if solution_path == "lwm.sim" and task_name == "sim":
            env_class = getattr(module, "SimpleEnv")
        else:
            env_class = getattr(module, task_name)
    else:
        raise ValueError("Either env_name or solution_path must be provided to build_up_env")

    env = env_class(**save_config)

    if not return_env_class:
        return env, save_config
    else:
        return env, save_config, env_class


class NonDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""

            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc


def plot_contact_point(
    handle_points,
    gripper_points,
    contact_point_idx,
    save_path=None,
    title="Contact Point Visualization",
):
    """Plot gripper points, handle points, and highlight the contact point."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(handle_points[:, 0], handle_points[:, 1], handle_points[:, 2],
               c="blue", s=5, alpha=0.4, label="Handle Points", zorder=1)

    if 0 <= contact_point_idx < len(handle_points):
        contact_point = handle_points[contact_point_idx]
        ax.scatter(contact_point[0], contact_point[1], contact_point[2],
                   c="red", s=500, marker="*", edgecolors="darkred", linewidths=3,
                   label="Contact Point", zorder=10000)

    gripper_labels = ["Right Hand", "Right Finger", "Left Finger", "End Effector"]
    gripper_colors = ["green", "orange", "orange", "purple"]
    gripper_markers = ["o", "s", "s", "D"]
    gripper_sizes = [100, 150, 150, 200]

    for i, (point, label, color, marker, size) in enumerate(
        zip(gripper_points, gripper_labels, gripper_colors, gripper_markers, gripper_sizes)
    ):
        ax.scatter(point[0], point[1], point[2], c=color, s=size, marker=marker,
                   edgecolors="black", linewidths=1.5,
                   label=label if i == 0 or i == 3 else None, alpha=0.8, zorder=5)

    if len(gripper_points) >= 4:
        for j in [1, 2]:
            ax.plot([gripper_points[0, 0], gripper_points[j, 0]],
                    [gripper_points[0, 1], gripper_points[j, 1]],
                    [gripper_points[0, 2], gripper_points[j, 2]], "g-", linewidth=2, alpha=0.5)
        ax.plot([gripper_points[0, 0], gripper_points[3, 0]],
                [gripper_points[0, 1], gripper_points[3, 1]],
                [gripper_points[0, 2], gripper_points[3, 2]], "g-", linewidth=2, alpha=0.5)

    if 0 <= contact_point_idx < len(handle_points):
        eef_pos = gripper_points[3]
        ax.plot([eef_pos[0], contact_point[0]], [eef_pos[1], contact_point[1]],
                [eef_pos[2], contact_point[2]], "r--", linewidth=2, alpha=0.7, label="Contact Distance")

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_zlabel("Z (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    max_range = np.array([handle_points[:, 0].ptp(), handle_points[:, 1].ptp(),
                          handle_points[:, 2].ptp()]).max() / 2.0
    mid = handle_points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """
    # Lazy import to avoid ffmpeg check at module import time
    from moviepy.editor import ImageSequenceClip

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + ".gif"

    # convert to uint8 if needed to avoid lossy conversion warnings
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip


def obj_to_urdf(
    obj_file_path, scale=1, vhacd=True, normalized=True, obj_name="material"
):
    header = """<?xml version="1.0" ?>
<robot name="cube.urdf">
  <link name="baseLink">
    <contact>
      <lateral_friction value="1.0"/>
      <rolling_friction value="0.0"/>
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="0.0 0.02 0.0"/>
       <mass value=".1"/>
       <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
"""

    all_files = os.listdir(obj_file_path)
    png_file = None
    for x in all_files:
        if x.endswith(".png"):
            png_file = x
            break

    if png_file is not None:
        material = """
         <material name="texture">
        <texture filename="{}"/>
      </material>""".format(
            osp.join(obj_file_path, png_file)
        )
    else:
        material = """
        <material name="yellow">
            <color rgba="1 1 0.4 1"/>
        </material>
        """

    obj_file = (
        "{}.obj".format(obj_name)
        if not normalized
        else "{}_normalized.obj".format(obj_name)
    )
    visual = """
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{}" scale="{} {} {}"/>
      </geometry>
      {}
    </visual>
    """.format(
        osp.join(obj_file_path, obj_file), scale, scale, scale, material
    )

    if normalized:
        collision_file = (
            "{}_normalized_vhacd.obj".format(obj_name)
            if vhacd
            else "{}_normalized.obj".format(obj_name)
        )
    else:
        collision_file = (
            "{}_vhacd.obj".format(obj_name) if vhacd else "{}.obj".format(obj_name)
        )
    collision = """
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
             <mesh filename="{}" scale="{} {} {}"/>
      </geometry>
    </collision>
  </link>
  </robot>
  """.format(
        osp.join(obj_file_path, collision_file), scale, scale, scale
    )

    urdf = "".join([header, visual, collision])
    with open(osp.join(obj_file_path, "{}.urdf".format(obj_name)), "w") as f:
        f.write(urdf)


def run_vhacd(input_obj_file_path, normalized=True, obj_name="material"):
    p.connect(p.DIRECT)
    if normalized:
        name_in = os.path.join(
            input_obj_file_path, "{}_normalized.obj".format(obj_name)
        )
        name_out = os.path.join(
            input_obj_file_path, "{}_normalized_vhacd.obj".format(obj_name)
        )
        name_log = os.path.join(input_obj_file_path, "log.txt")
    else:
        name_in = os.path.join(input_obj_file_path, "{}.obj".format(obj_name))
        name_out = os.path.join(input_obj_file_path, "{}_vhacd.obj".format(obj_name))
        name_log = os.path.join(input_obj_file_path, "log.txt")
    p.vhacd(name_in, name_out, name_log)


def parse_center(center):
    if center.startswith("(") or center.startswith("["):
        center = center[1:-1]

    center = center.split(",")
    center = [float(x) for x in center]
    return np.array(center)


def run_vhacd_with_timeout(args):
    name_in, name_out, name_log, urdf_file_path, obj_file_name = args
    id = p.connect(p.DIRECT)
    proc = Process(target=p.vhacd, args=(name_in, name_out, name_log))

    proc.start()

    # Wait for 10 seconds or until process finishes
    proc.join(200)

    # If thread is still active
    if proc.is_alive():
        print("running too long... let's kill it...")

        # Terminate
        proc.kill()
        proc.join()

        if urdf_file_path not in sapaien_cannot_vhacd_part_dict.keys():
            sapaien_cannot_vhacd_part_dict[urdf_file_path] = []
        sapaien_cannot_vhacd_part_dict[urdf_file_path].append(obj_file_name)

        p.disconnect(id)
        return False

    else:
        print("process finished")
        p.disconnect(id)
        return True


def preprocess_urdf(urdf_file_path, num_processes=6):
    new_lines = []
    with open(urdf_file_path, "r") as f:
        lines = f.readlines()

    num_lines = len(lines)
    l_idx = 0
    to_process_args = []
    while l_idx < num_lines:
        line_1 = lines[l_idx]

        if "<collision>" in line_1:
            new_lines.append(line_1)

            for l_idx_2 in range(l_idx + 1, num_lines):
                line_2 = lines[l_idx_2]

                if ".obj" in line_2:
                    start_idx = line_2.find('filename="') + len('filename="')
                    end_idx = line_2.find(".obj") + len(".obj")
                    obj_file_name = line_2[start_idx:end_idx]
                    obj_file_path = osp.join(osp.dirname(urdf_file_path), obj_file_name)
                    # import pdb; pdb.set_trace()
                    name_in = obj_file_path
                    name_out = obj_file_path[:-4] + "_vhacd.obj"
                    name_log = obj_file_path[:-4] + "_log.txt"

                    if not osp.exists(
                        name_out
                    ) and obj_file_name not in sapaien_cannot_vhacd_part_dict.get(
                        urdf_file_path, []
                    ):
                        to_process_args.append(
                            [name_in, name_out, name_log, urdf_file_path, obj_file_name]
                        )
                        new_lines.append("to_be_processed, {}".format(line_2))
                    else:
                        new_name = line_2.replace(
                            obj_file_name, obj_file_name[:-4] + "_vhacd.obj"
                        )
                        new_lines.append(new_name)

                elif "</collision>" in line_2:
                    new_lines.append(line_2)
                    l_idx = l_idx_2
                    break

                else:
                    new_lines.append(line_2)

        else:
            new_lines.append(line_1)

        l_idx += 1

    # do vhacd in parallel, each has a timeout of 200 seconds
    with NonDaemonPool(processes=num_processes) as pool:
        results = pool.map(run_vhacd_with_timeout, to_process_args)

    processed_idx = 0
    for l_idx in range(len(new_lines)):
        if "to_be_processed" in new_lines[l_idx]:
            if results[processed_idx]:
                new_name = new_lines[l_idx].replace("to_be_processed, ", "")
                new_name = new_name.replace(".obj", "_vhacd.obj")
                new_lines[l_idx] = new_name
            else:
                new_name = new_lines[l_idx].replace("to_be_processed, ", "")
                new_lines[l_idx] = new_name
            processed_idx += 1

    new_path = urdf_file_path.replace(".urdf", "_vhacd.urdf")
    with open(new_path, "w") as f:
        f.writelines("".join(new_lines))

    with open(f"{data_dir}/sapien_cannot_vhacd_part.json", "w") as f:
        json.dump(sapaien_cannot_vhacd_part_dict, f, indent=4)

    return new_path


def parse_config(config, use_bard=True, obj_id=None, use_gpt_size=True, use_vhacd=True, partnet_dataset_dir=None):
    urdf_paths = []
    urdf_sizes = []
    urdf_locations = []
    urdf_orientations = []
    urdf_names = []
    urdf_types = []
    urdf_on_tables = []
    urdf_movables = []
    urdf_crop_sizes = []
    use_table = False
    articulated_joint_angles = {}
    spatial_relationships = []
    distractor_config_path = None

    robot_initial_joint_angles = [0.0, 0.0, 0.0, -0.4, 0.0, 0.4, 0.0]

    for obj in config:
        if "use_table" in obj.keys():
            use_table = obj["use_table"]

        if "set_joint_angle_object_name" in obj.keys():
            new_obj = copy.deepcopy(obj)
            new_obj.pop("set_joint_angle_object_name")
            articulated_joint_angles[obj["set_joint_angle_object_name"]] = new_obj

        if "spatial_relationships" in obj.keys():
            spatial_relationships = obj["spatial_relationships"]

        if "task_name" in obj.keys() or "task_description" in obj.keys():
            continue

        if "distractor_config_path" in obj.keys():
            distractor_config_path = obj["distractor_config_path"]

        if "initial_joint_angles" in obj.keys():
            initial_joint_angles = obj["initial_joint_angles"]
            initial_joint_angles = parse_center(initial_joint_angles)
            robot_initial_joint_angles = initial_joint_angles

        if "type" not in obj.keys():
            continue

        if obj["type"] == "mesh":
            if "uid" not in obj.keys():
                continue
            if obj_id is None:
                uid = obj["uid"][np.random.randint(len(obj["uid"]))]
            else:
                uid = obj["uid"][obj_id]

            urdf_file_path = osp.join(
                "objaverse_utils/data/obj", "{}".format(uid), "material.urdf"
            )
            if not os.path.exists(urdf_file_path):
                # Objaverse mesh objects are not used in this project (the original
                # code downloaded them here).
                raise FileNotFoundError(f"Mesh object URDF not found: {urdf_file_path}")

            if not use_vhacd:
                new_urdf_file_path = urdf_file_path.replace(
                    "material.urdf", "material_non_vhacd.urdf"
                )
                new_urdf_lines = []
                with open(urdf_file_path, "r") as f:
                    urdf_lines = f.readlines()
                for line in urdf_lines:
                    if "vhacd" in line:
                        new_line = line.replace("_vhacd", "")
                        new_urdf_lines.append(new_line)
                    else:
                        new_urdf_lines.append(line)
                with open(new_urdf_file_path, "w") as f:
                    f.writelines(new_urdf_lines)
                urdf_file_path = new_urdf_file_path

            urdf_paths.append(urdf_file_path)
            urdf_types.append("mesh")
            urdf_movables.append(True)  # all mesh objects are movable

        elif obj["type"] == "urdf":
            if "reward_asset_path" in obj:
                obj_path = obj["reward_asset_path"]
            else:
                try:
                    category = obj["lang"]
                    possible_obj_path = partnet_mobility_dict[category]
                except:
                    category = obj["name"]
                    if category == "Computer display":
                        category = "Display"
                    possible_obj_path = partnet_mobility_dict[category]
                obj_path = np.random.choice(possible_obj_path)
                if category == "Toaster":
                    obj_path = str(103486)
                if category == "Microwave":
                    obj_path = str(7310)
                if category == "Oven":
                    obj_path = str(101808)
                if category == "Refrigerator":
                    obj_path = str(10638)

            if "urdf" not in obj_path:
                # Use partnet_dataset_dir if provided, otherwise fall back to data_dir
                base_dir = partnet_dataset_dir if partnet_dataset_dir else data_dir
                urdf_file_path = osp.join(base_dir, obj_path, "mobility.urdf")
                if use_vhacd:
                    new_urdf_file_path = urdf_file_path.replace(
                        "mobility.urdf", "mobility_vhacd.urdf"
                    )
                    if not osp.exists(new_urdf_file_path):
                        new_urdf_file_path = preprocess_urdf(urdf_file_path)
                    urdf_paths.append(new_urdf_file_path)
                else:
                    urdf_paths.append(urdf_file_path)
            else:
                urdf_paths.append(obj_path)

            urdf_types.append("urdf")
            urdf_movables.append(
                obj.get("movable", False)
            )  # by default, urdf objects are not movable, unless specified

        urdf_sizes.append(obj["size"])
        urdf_locations.append(parse_center(obj["center"]))
        ori = obj.get("orientation", [0, 0, 0, 1])
        if type(ori) == str:
            ori = parse_center(ori)
        urdf_orientations.append(ori)
        urdf_names.append(obj["name"])
        urdf_on_tables.append(obj.get("on_table", False))
        urdf_crop_sizes.append(obj.get("is_crop_size", True))
    return (
        urdf_paths,
        urdf_sizes,
        urdf_locations,
        urdf_orientations,
        urdf_names,
        urdf_types,
        urdf_on_tables,
        use_table,
        urdf_crop_sizes,
        articulated_joint_angles,
        spatial_relationships,
        distractor_config_path,
        urdf_movables,
        robot_initial_joint_angles,
    )


def take_round_images(
    env,
    center,
    distance,
    elevation=30,
    azimuth_interval=30,
    camera_width=640,
    camera_height=480,
    return_camera_matrices=False,
    z_near=0.01,
    z_far=10,
    save_path=None,
):
    camera_target = center
    delta_z = distance * np.sin(np.deg2rad(elevation))
    xy_distance = distance * np.cos(np.deg2rad(elevation))

    env_prev_view_matrix, env_prev_projection_matrix = (
        env.view_matrix,
        env.projection_matrix,
    )

    rgbs = []
    depths = []
    view_camera_matrices = []
    project_camera_matrices = []
    for azimuth in range(0, 360, azimuth_interval):
        delta_x = xy_distance * np.cos(np.deg2rad(azimuth))
        delta_y = xy_distance * np.sin(np.deg2rad(azimuth))
        camera_position = [
            camera_target[0] + delta_x,
            camera_target[1] + delta_y,
            camera_target[2] + delta_z,
        ]
        env.setup_camera(
            camera_position,
            camera_target,
            camera_width=camera_width,
            camera_height=camera_height,
        )

        rgb, depth = env.render(return_depth=True)
        rgbs.append(rgb)
        depths.append(depth)
        view_camera_matrices.append(env.view_matrix)
        project_camera_matrices.append(env.projection_matrix)

    env.view_matrix, env.projection_matrix = (
        env_prev_view_matrix,
        env_prev_projection_matrix,
    )

    if not return_camera_matrices:
        return rgbs, depths
    else:
        return rgbs, depths, view_camera_matrices, project_camera_matrices


def take_round_images_around_object(
    env,
    object_name,
    distance=None,
    save_path=None,
    azimuth_interval=30,
    elevation=30,
    return_camera_matrices=False,
    camera_width=640,
    camera_height=480,
    only_object=False,
):
    if only_object:
        ### make all other objects invisiable
        prev_rgbas = []
        object_id = env.urdf_ids[object_name]
        for obj_name, obj_id in env.urdf_ids.items():
            if obj_name != object_name:
                num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
                for link_idx in range(-1, num_links):
                    prev_rgba = p.getVisualShapeData(
                        obj_id, link_idx, physicsClientId=env.id
                    )[0][14:18]
                    prev_rgbas.append(prev_rgba)
                    p.changeVisualShape(
                        obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=env.id
                    )

    obj_id = env.urdf_ids[object_name]
    min_aabb, max_aabb = env.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    if distance is None:
        distance = np.linalg.norm(max_aabb - min_aabb) * 1.1

    res = take_round_images(
        env,
        camera_target,
        distance,
        elevation=elevation,
        azimuth_interval=azimuth_interval,
        camera_width=camera_width,
        camera_height=camera_height,
        save_path=save_path,
        return_camera_matrices=return_camera_matrices,
    )

    if only_object:
        cnt = 0
        object_id = env.urdf_ids[object_name]
        for obj_name, obj_id in env.urdf_ids.items():
            if obj_name != object_name:
                num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
                for link_idx in range(-1, num_links):
                    p.changeVisualShape(
                        obj_id,
                        link_idx,
                        rgbaColor=prev_rgbas[cnt],
                        physicsClientId=env.id,
                    )
                    cnt += 1

    return res


def center_camera_at_object(
    env,
    object_name,
    distance=None,
    elevation=30,
    azimuth=0,
    camera_width=640,
    camera_height=480,
):
    obj_id = env.urdf_ids[object_name]
    min_aabb, max_aabb = env.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    if distance is None:
        distance = np.linalg.norm(max_aabb - min_aabb) * 1.1

    delta_z = distance * np.sin(np.deg2rad(elevation))
    xy_distance = distance * np.cos(np.deg2rad(elevation))

    delta_x = xy_distance * np.cos(np.deg2rad(azimuth))
    delta_y = xy_distance * np.sin(np.deg2rad(azimuth))
    camera_position = [
        camera_target[0] + delta_x,
        camera_target[1] + delta_y,
        camera_target[2] + delta_z,
    ]
    env.setup_camera(
        camera_position,
        camera_target,
        camera_width=camera_width,
        camera_height=camera_height,
    )


def get_pc(proj_matrix, view_matrix, depth, width, height, mask_infinite=False):
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1 : 1 : 2 / height, -1 : 1 : 2 / width]
    y *= -1.0
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    if mask_infinite:
        pixels = pixels[z < 0.9999]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3:4]
    points = points[:, :3]

    return points


def get_pixel_location(proj_matrix, view_matrix, point_3d, width, height):
    # Ensure matrices are in the correct shape
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")

    # Combine the projection and view matrices
    tran_world_pix = np.matmul(proj_matrix, view_matrix)

    # Add homogeneous coordinate to the 3D point
    point_3d_h = np.append(point_3d, 1.0)

    # Transform the 3D point to pixel coordinates
    pixel_h = np.matmul(tran_world_pix, point_3d_h)

    # Normalize by the homogeneous coordinate
    pixel_h /= pixel_h[3]

    # Convert from normalized device coordinates to pixel coordinates
    x_ndc, y_ndc, z_ndc = pixel_h[:3]

    # Transform normalized device coordinates to image coordinates
    x_img = (x_ndc * 0.5 + 0.5) * width
    y_img = (1.0 - (y_ndc * 0.5 + 0.5)) * height  # Note: y-axis is inverted

    return int(x_img), int(y_img), z_ndc


def get_pc_in_camera_frame(
    proj_matrix, view_matrix, depth, width, height, mask_infinite=False
):
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1 : 1 : 2 / height, -1 : 1 : 2 / width]
    y *= -1.0
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    if mask_infinite:
        pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1
    # turn pixels to camera cooridnates
    points = np.matmul(np.linalg.inv(proj_matrix), pixels.T).T
    points /= points[:, 3:4]
    points = points[:, :3]
    return points


def setup_camera_ben(
    client_id,
    camera_eye=[0.5, -0.75, 1.5],
    camera_target=[-0.2, 0, 0.75],
    camera_width=1920 // 4,
    camera_height=1080 // 4,
    z_near=0.01,
    z_far=100,
):
    view_matrix = p.computeViewMatrix(
        camera_eye, camera_target, [0, 0, 1], physicsClientId=client_id
    )
    focal_length = 450  # CAMERA_INTRINSICS[0, 0]
    fov = (np.arctan((camera_height / 2) / focal_length) * 2 / np.pi) * 180
    projection_matrix = p.computeProjectionMatrixFOV(
        fov, camera_width / camera_height, z_near, z_far, physicsClientId=client_id
    )
    return view_matrix, projection_matrix


def get_pc_ben(depth, view_matrix, projection_matrix, znear, zfar):
    height, width = depth.shape
    CAMERA_INTRINSICS = np.array(
        [
            [450, 0, width / 2],
            [0, 450, height / 2],
            [0, 0, 1],
        ]
    )

    T_CAMGL_2_CAM = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    depth = zfar + znear - (2.0 * depth - 1.0) * (zfar - znear)
    depth = (2.0 * znear * zfar) / depth

    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - CAMERA_INTRINSICS[0, 2]) * (depth / CAMERA_INTRINSICS[0, 0])
    py = (py - CAMERA_INTRINSICS[1, 2]) * (depth / CAMERA_INTRINSICS[1, 1])
    P_cam = np.float32([px, py, depth]).transpose(1, 2, 0).reshape(-1, 3)

    T_camgl2world = np.asarray(view_matrix).reshape(4, 4).T
    T_world2camgl = np.linalg.inv(T_camgl2world)
    T_world2cam = T_world2camgl @ T_CAMGL_2_CAM

    Ph_cam = np.concatenate([P_cam, np.ones((len(P_cam), 1))], axis=1)
    Ph_world = (T_world2cam @ Ph_cam.T).T
    P_world = Ph_world[:, :3]

    return P_world


def save_env(env, save_path=None, simplified=False):
    object_joint_angle_dicts = {}
    object_joint_name_dicts = {}
    object_link_name_dicts = {}
    robot_name = env.robot_name
    for obj_name, obj_id in env.urdf_ids.items():
        num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
        object_joint_angle_dicts[obj_name] = []
        object_joint_name_dicts[obj_name] = []
        object_link_name_dicts[obj_name] = []
        for link_idx in range(0, num_links):
            joint_angle = p.getJointState(obj_id, link_idx, physicsClientId=env.id)[0]
            object_joint_angle_dicts[obj_name].append(joint_angle)
            joint_name = p.getJointInfo(obj_id, link_idx, physicsClientId=env.id)[
                1
            ].decode("utf-8")
            object_joint_name_dicts[obj_name].append(joint_name)
            link_name = p.getJointInfo(obj_id, link_idx, physicsClientId=env.id)[
                12
            ].decode("utf-8")
            object_link_name_dicts[obj_name].append(link_name)

    object_base_position = {}
    for obj_name, obj_id in env.urdf_ids.items():
        object_base_position[obj_name] = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=env.id
        )[0]

    object_base_orientation = {}
    for obj_name, obj_id in env.urdf_ids.items():
        object_base_orientation[obj_name] = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=env.id
        )[1]

    state = {
        "object_joint_angle_dicts": object_joint_angle_dicts,
        "object_joint_name_dicts": object_joint_name_dicts,
        "object_link_name_dicts": object_link_name_dicts,
        "object_base_position": object_base_position,
        "object_base_orientation": object_base_orientation,
    }
    if not simplified:
        state["urdf_paths"] = copy.deepcopy(env.urdf_paths)
        state["object_sizes"] = env.simulator_sizes
        state["robot_name"] = env.robot_name
        state["grasped_handle"] = env.grasped_handle

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

    return state


def load_env(env, load_path=None, state=None):
    if load_path is not None:
        with open(load_path, "rb") as f:
            state = pickle.load(f)

    ### set env to stored object position and orientation
    for obj_name, obj_id in env.urdf_ids.items():
        if obj_name not in state["object_base_position"].keys():
            continue
        p.resetBasePositionAndOrientation(
            obj_id,
            state["object_base_position"][obj_name],
            state["object_base_orientation"][obj_name],
            physicsClientId=env.id,
        )

    ### set env to stored object joint angles
    for obj_name, obj_id in env.urdf_ids.items():
        if obj_name not in state["object_joint_angle_dicts"].keys():
            continue

        num_links = p.getNumJoints(obj_id, physicsClientId=env.id)

        for link_idx in range(0, num_links):
            joint_angle = state["object_joint_angle_dicts"][obj_name][link_idx]
            p.resetJointState(obj_id, link_idx, joint_angle, physicsClientId=env.id)

    if "urdf_paths" in state:
        urdf_paths = state["urdf_paths"]
        for urdf_name, urdf_path in urdf_paths.items():
            urdf_path = reroot_partnet_path(urdf_path)
            urdf_paths[urdf_name] = urdf_path
        env.urdf_paths = urdf_paths
        # env.urdf_paths = state["urdf_paths"]

    if "object_sizes" in state:
        env.simulator_sizes = state["object_sizes"]

    if "robot_name" in state:
        env.robot_name = state["robot_name"]

    if "grasped_handle" in state:
        env.grasped_handle = state["grasped_handle"]

    return state


### get handle utility functions
def load_obj(fn):
    fin = open(fn, "r")
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith("f "):
            faces.append(np.int32([item.split("/")[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)

    return v, f


def get_link_handle(all_handle_pos, handle_joint_id, link_pc, threshold=0.02):
    handle_median_points = np.array(
        [np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]
    ).reshape(-1, 3)
    distance_handle_median_to_link_pc = scipy.spatial.distance.cdist(
        handle_median_points, link_pc
    )
    min_distance = np.min(distance_handle_median_to_link_pc, axis=1)
    min_distance_handle_idx = np.argmin(min_distance)

    # Note: debug prints removed here; selection follows training branch logic.

    handle_joint_id = handle_joint_id[min_distance_handle_idx]
    handle_pc = all_handle_pos[min_distance_handle_idx]
    handle_median = handle_median_points[min_distance_handle_idx]
    # Use KDTree for memory-efficient nearest neighbor search instead of full cdist
    from scipy.spatial import cKDTree
    tree = cKDTree(handle_pc)
    pc_to_handle_distance, _ = tree.query(link_pc, k=1)
    handle_pc = link_pc[pc_to_handle_distance < threshold]
    # use the pointcloud of link instead of the handle itself. (partially occluded)
    return handle_pc, handle_joint_id, handle_median, min_distance_handle_idx


def find_nearest_point_on_line(line_pt1, line_pt2, target_pt):
    line_pt1 = np.array(line_pt1).reshape(-1, 3)
    line_pt2 = np.array(line_pt2).reshape(-1, 3)
    target_pt = np.array(target_pt).reshape(-1, 3)

    # Step 1: Compute the vector along the line
    line_vec = line_pt2 - line_pt1

    # Step 2: Compute the vector from line_pt1 to target_pt
    pt_vec = target_pt - line_pt1

    # Step 3: Project pt_vec onto line_vec to find the projection scalar
    # dot_product(pt_vec, line_vec) / dot_product(line_vec, line_vec) gives the scalar
    # by which to multiply line_vec to get the projection vector.
    projection_scalar = np.sum(pt_vec * line_vec, axis=1) / np.sum(line_vec * line_vec)

    # Step 4: Find the nearest point on the line by scaling line_vec and adding it to line_pt1
    nearest_pt = line_pt1 + projection_scalar.reshape(-1, 1) * line_vec.repeat(
        len(projection_scalar), axis=0
    )

    return nearest_pt  # (-1, 3)


def rotate_point_around_axis(pt, ax, theta_rad):
    """
    Rotate a point around a given axis by theta radiance.

    :param pt: The point to rotate (3D coordinates).
    :param ax: The rotation axis (3D unit vector).
    :param theta: The rotation angle in radians.
    :return: The rotated point's coordinates.
    """
    # Ensure ax is a unit vector
    ax = ax / np.linalg.norm(ax)
    ax = ax.reshape(-1, 3)

    # Rodrigues' rotation formula
    v_rot = (
        pt * np.cos(theta_rad)
        + np.cross(ax, pt) * np.sin(theta_rad)
        + ax
        * np.sum(ax.repeat(pt.shape[0], axis=0) * pt, axis=1, keepdims=True)
        * (1 - np.cos(theta_rad))
    )

    return v_rot


def add_sphere(position, radius=0.05, rgba=[0, 1, 1, 1]):
    sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba
    )
    mass = 0.1
    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=sphere_collision,
        baseVisualShapeIndex=sphere_visual,
        basePosition=position,
    )
    return body


def rotation_transfer_6D_to_matrix(orient):
    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T

    return rotate_matrix


def rotation_transfer_matrix_to_6D(rotate_matrix):
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)

    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient


def rotation_transfer_6D_to_matrix_batch(orient):
    # orient shape = (B, 6)
    # return shape = (3, B * 3)

    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    assert orient.shape[-1] == 6

    orient = orient.reshape(-1, 2, 3)
    a1 = orient[:, 0]
    a2 = orient[:, 1]

    b1 = a1 / np.linalg.norm(a1, axis=-1).reshape(-1, 1)
    b2 = a2 - (np.sum(a2 * b1, axis=-1).reshape(-1, 1) * b1)
    b2 = b2 / np.linalg.norm(b2, axis=-1).reshape(-1, 1)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.hstack((b1, b2, b3))
    rotate_matrix = rotate_matrix.reshape(-1, 3).T

    return rotate_matrix


def rotation_transfer_matrix_to_6D_batch(rotate_matrix):
    # rotate_matrix.shape = (B, 9) or (B x 3, 3) rotation transpose (i.e., row vectors instead of column vectors)
    # return shape = (B, 6)

    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(-1, 9)
    rotate_matrix = rotate_matrix.reshape(-1, 9)

    return rotate_matrix[:, :6]


###########################################


# [Chialiang]: The following functions are from my previous projects
def xyzw2wxyz(quat: np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f"quaternion size must be 4, got {len(quat)}"
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])


def wxyz2xyzw(quat: np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f"quaternion size must be 4, got {len(quat)}"
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])


def pose_6d_to_7d(pose) -> np.ndarray:
    assert len(pose) == 6, f"pose must contain 6 elements, but got {len(pose)}"
    pos = np.asarray(pose[:3])
    rot = R.from_rotvec(pose[3:]).as_quat()
    pose = list(pos) + list(rot)

    return np.array(pose)


def pose_6d_to_7d(pose) -> np.ndarray:
    if len(pose) == 7:
        return np.array(pose)
    pos = np.asarray(pose[:3])
    rot = R.from_rotvec(pose[3:]).as_quat()
    pose_ret = list(pos) + list(rot)

    return np.array(pose_ret)


def pose_7d_to_6d(pose) -> np.ndarray:
    if len(pose) == 6:
        return np.array(pose)
    pos = np.asarray(pose[:3])
    rot = R.from_quat(pose[3:]).as_rotvec()
    pose_ret = list(pos) + list(rot)

    return np.array(pose_ret)


def get_matrix_from_pose(pose) -> np.ndarray:
    assert (
        len(pose) == 6 or len(pose) == 7 or len(pose) == 9
    ), f"pose must contain 6 or 7 elements, but got {len(pose)}"
    pos_m = np.asarray(pose[:3])
    rot_m = np.identity(3)

    if len(pose) == 6:
        rot_m = R.from_rotvec(pose[3:]).as_matrix()
    elif len(pose) == 7:
        rot_m = R.from_quat(pose[3:]).as_matrix()
    elif len(pose) == 9:
        rot_xy = pose[3:].reshape(2, 3)
        rot_m = np.vstack((rot_xy, np.cross(rot_xy[0], rot_xy[1]))).T

    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m

    return ret_m


def rot_6d_to_3d(rot) -> np.ndarray:
    rot_xy = np.asarray(rot)

    assert rot_xy.shape == (
        6,
    ), f"dimension of rot should be (6,), but got {rot_xy.shape}"

    rot_xy = rot_xy.reshape(2, 3)
    rot_mat = np.vstack((rot_xy, np.cross(rot_xy[0], rot_xy[1]))).T

    return R.from_matrix(rot_mat).as_rotvec()


def get_pose_from_matrix(matrix, pose_size: int = 7) -> np.ndarray:
    mat = np.array(matrix)
    assert mat.shape == (4, 4), f"pose must contain 4 x 4 elements, but got {mat.shape}"

    pos = matrix[:3, 3]
    rot = None

    if pose_size == 6:
        rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    elif pose_size == 7:
        rot = R.from_matrix(matrix[:3, :3]).as_quat()
    elif pose_size == 9:
        rot = (matrix[:3, :2].T).reshape(-1)

    pose = list(pos) + list(rot)

    return np.array(pose)


def get_matrix_from_pos_rot(pos, rot) -> np.ndarray:
    assert (len(pos) == 3 and len(rot) == 4) or (len(pos) == 3 and len(rot) == 3)
    pos_m = np.asarray(pos)
    if len(rot) == 3:
        rot_m = R.from_rotvec(rot).as_matrix()
        # rot_m = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rot))).reshape((3, 3))
    elif len(rot) == 4:  # x, y, z, w
        rot_m = R.from_quat(rot).as_matrix()
        # rot_m = np.asarray(p.getMatrixFromQuaternion(rot)).reshape((3, 3))
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m
    return ret_m


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def get_pos_rot_from_matrix(pose: np.ndarray) -> np.ndarray:
    assert pose.shape == (4, 4)
    pos = pose[:3, 3]
    rot = R.from_matrix(pose[:3, :3]).as_quat()
    return pos, rot


def get_projmat_and_intrinsic(width, height, fx, fy, far, near):
    cx = width / 2
    cy = height / 2
    fov = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi

    project_matrix = p.computeProjectionMatrixFOV(
        fov=fov, aspect=width / height, nearVal=near, farVal=far
    )

    intrinsic = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
    )

    return project_matrix, intrinsic


def get_viewmat_and_extrinsic(cameraEyePosition, cameraTargetPosition, cameraUpVector):
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cameraEyePosition,
        cameraTargetPosition=cameraTargetPosition,
        cameraUpVector=cameraUpVector,
    )

    # rotation vector extrinsic
    z = np.asarray(cameraTargetPosition) - np.asarray(cameraEyePosition)
    norm = np.linalg.norm(z, ord=2)
    assert norm > 0, f"cameraTargetPosition and cameraEyePosition is at same location"
    z /= norm

    y = -np.asarray(cameraUpVector)
    y -= (np.dot(z, y)) * z
    norm = np.linalg.norm(y, ord=2)
    assert norm > 0, f"cameraUpVector is parallel to z axis"
    y /= norm

    x = cross(y, z)

    # extrinsic
    extrinsic = np.identity(4)
    extrinsic[:3, 0] = x
    extrinsic[:3, 1] = y
    extrinsic[:3, 2] = z
    extrinsic[:3, 3] = np.asarray(cameraEyePosition)

    return view_matrix, extrinsic


def draw_coordinate(
    pose, size, color: np.ndarray = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
):
    assert (
        (type(pose) == np.ndarray and pose.shape == (4, 4))
        or (len(pose) == 7)
        or (len(pose) == 6)
    )

    if len(pose) == 7 or len(pose) == 6:
        pose = get_matrix_from_pose(pose)

    origin = pose[:3, 3]
    x = origin + pose[:3, 0] * size
    y = origin + pose[:3, 1] * size
    z = origin + pose[:3, 2] * size
    p.addUserDebugLine(origin, x, color[0], 2, 0)
    p.addUserDebugLine(origin, y, color[1], 2, 0)
    p.addUserDebugLine(origin, z, color[2], 2, 0)


def draw_bbox(start, end):
    assert (
        len(start) == 3 and len(end) == 3
    ), f"infeasible size of position, len(position) must be 3"

    points_bb = [
        [start[0], start[1], start[2]],
        [end[0], start[1], start[2]],
        [end[0], end[1], start[2]],
        [start[0], end[1], start[2]],
        [start[0], start[1], end[2]],
        [end[0], start[1], end[2]],
        [end[0], end[1], end[2]],
        [start[0], end[1], end[2]],
    ]

    for i in range(4):
        p.addUserDebugLine(points_bb[i], points_bb[(i + 1) % 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i + 4], points_bb[(i + 1) % 4 + 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i], points_bb[i + 4], [1, 0, 0])


def piecewise_uniform_sample(low: float, high: float) -> float:
    """
    Samples from a piece-wise uniform distribution of [low,high]+[-high, -low]
    """
    is_negative = np.random.uniform(0, 1) <= 0.5
    if is_negative:
        return np.random.uniform(-high, -low)
    else:
        return np.random.uniform(low, high)


def radial_shift(x_coord: float, y_coord: float, noise_bounds: List[float]):
    theta = np.arctan2(y_coord, x_coord)
    theta_noise = np.random.uniform(-0.1, 0.1)
    dist = np.linalg.norm([x_coord, y_coord])
    dist_noise = np.random.uniform(noise_bounds[0], noise_bounds[1])
    theta += theta_noise
    dist += dist_noise
    perturbed_x = dist * np.cos(theta)
    perturbed_y = dist * np.sin(theta)
    return perturbed_x, perturbed_y


def sample_point_inside_triangle(v1, v2, v3):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    while r1 + r2 >= 1:
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
    r3 = 1 - r1 - r2

    # Calculate the point using barycentric coordinates
    x = r1 * v1[0] + r2 * v2[0] + r3 * v3[0]
    y = r1 * v1[1] + r2 * v2[1] + r3 * v3[1]
    z = r1 * v1[2] + r2 * v2[2] + r3 * v3[2]
    return [x, y, z]


def in_bbox(pos, bbox_min, bbox_max):
    if (
        pos[0] <= bbox_max[0]
        and pos[0] >= bbox_min[0]
        and pos[1] <= bbox_max[1]
        and pos[1] >= bbox_min[1]
        and pos[2] <= bbox_max[2]
        and pos[2] >= bbox_min[2]
    ):
        return True
    return False


def get_joint_id_from_name(simulator, object_name, joint_name):
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[1].decode("utf-8") == joint_name:
            joint_index = i
            break

    return joint_index


def get_handle_pos(
    simulator,
    obj_name,
    return_median=True,
    handle_pts_obj_frame=None,
    mobility_info=None,
    return_info=False,
    target_object="handle",
):
    obj_name = obj_name.lower()
    scaling = simulator.simulator_sizes[obj_name]

    # get the parent frame of the revolute joint.
    obj_id = simulator.urdf_ids[obj_name]

    # axis in parent frame, transform everything to world frame
    if mobility_info is None:
        urdf_path = simulator.urdf_paths[obj_name]
        parent_dir = os.path.dirname(urdf_path)
        mobility_info = json.load(open(f"{parent_dir}/mobility_v2.json", "r"))

    # return a list of handle points in world frame
    ret_handle_pt_list = []
    ret_joint_idx_list = []

    handle_idx = 0
    all_handle_pts_object_frame = []
    for joint_info in mobility_info:
        all_parts = [part["name"] for part in joint_info["parts"]]
        if target_object in all_parts:
            all_ids = [part["id"] for part in joint_info["parts"]]
            index = all_parts.index(target_object)
            handle_id = all_ids[index]
            joint_name = "joint_{}".format(joint_info["id"])
            parent_joint_name = "joint_{}".format(joint_info["parent"])
            joint_data = joint_info["jointData"]
            axis_body = np.array(joint_data["axis"]["origin"]) * scaling
            axis_dir_body = np.array(joint_data["axis"]["direction"])
            joint_limit = joint_data["limit"]
            if joint_limit["a"] > joint_limit["b"]:
                axis_dir_body = -axis_dir_body

            joint_idx = get_joint_id_from_name(
                simulator, obj_name, joint_name
            )  # this is the joint id in pybullet
            parent_joint_idx = get_joint_id_from_name(
                simulator, obj_name, parent_joint_name
            )  # this is the joint id in pybullet

            parent_link_state = p.getLinkState(
                obj_id, parent_joint_idx, physicsClientId=simulator.id
            )  # NOTE: the handle link id should be dependent on the object urdf.
            link_urdf_world_pos, link_urdf_world_orn = (
                parent_link_state[0],
                parent_link_state[1],
            )
            # this is the transformation from the parent frame to the world frame.
            T_body_to_world = np.eye(
                4
            )  # transformation from the parent body frame to the world frame
            T_body_to_world[:3, :3] = np.array(
                p.getMatrixFromQuaternion(link_urdf_world_orn)
            ).reshape(3, 3)
            T_body_to_world[:3, 3] = link_urdf_world_pos

            axis_world = T_body_to_world[:3, :3] @ axis_body + T_body_to_world[:3, 3]
            axis_pt2_body = np.array(axis_body) + axis_dir_body
            axis_end_world = (
                T_body_to_world[:3, :3] @ axis_pt2_body + T_body_to_world[:3, 3]
            )
            axis_dir_world = axis_end_world - axis_world

            # get the handle points in world frame
            if handle_pts_obj_frame is None:
                handle_obj_path = f"{parent_dir}/parts_render/{handle_id}{target_object}.obj"
                handle_pts, handle_faces = load_obj(
                    handle_obj_path
                )  # this is in object frame

                handle_pts = handle_pts * scaling
                # add more dense points around handle
                added_points = []
                for f in handle_faces:
                    v1, v2, v3 = f
                    v1 = handle_pts[v1 - 1]
                    v2 = handle_pts[v2 - 1]
                    v3 = handle_pts[v3 - 1]
                    a = np.linalg.norm(v1 - v2)
                    b = np.linalg.norm(v2 - v3)
                    c = np.linalg.norm(v3 - v1)
                    s = (a + b + c) / 2
                    temp = max(0, s * (s - a) * (s - b) * (s - c))
                    surface = np.sqrt(temp)
                    num_points = surface * 1e6
                    num_points = int(num_points)
                    num_points = np.clip(num_points, 0, 5)
                    added_points.extend(
                        [
                            sample_point_inside_triangle(v1, v2, v3)
                            for _ in range(num_points)
                        ]
                    )

                if added_points != []:
                    added_points = np.array(added_points)
                    handle_pts = np.concatenate((handle_pts, added_points), axis=0)

                all_handle_pts_object_frame.append(handle_pts)

            else:
                handle_pts = handle_pts_obj_frame[handle_idx]

            # transform this to the world frame using the object *base*'s position and orientation
            handle_points_world = T_body_to_world[
                :3, :3
            ] @ handle_pts.T + T_body_to_world[:3, 3].reshape(
                3, 1
            )  # 3 x N
            if return_median:
                handle_point_median = np.median(handle_points_world, axis=1)
            else:
                handle_point_median = handle_points_world.T

            # find the projection of the handle point to the rotation axis, in world frame.
            project_on_rotation_axis = find_nearest_point_on_line(
                axis_world, axis_end_world, handle_point_median
            )

            # TODO: parse the mobility.json to get the joint name.
            joint_info_pb = p.getJointInfo(
                obj_id, joint_idx, physicsClientId=simulator.id
            )
            joint_type = joint_info_pb[2]

            if joint_type == p.JOINT_REVOLUTE:
                rotation_angle = p.getJointState(
                    obj_id, joint_idx, physicsClientId=simulator.id
                )[0]
                rotated_handle_pt_local = rotate_point_around_axis(
                    handle_point_median - project_on_rotation_axis,
                    axis_dir_world,
                    rotation_angle,
                )
                rotated_handle_pt = project_on_rotation_axis + rotated_handle_pt_local
            elif joint_type == p.JOINT_PRISMATIC:
                translation = p.getJointState(
                    obj_id, joint_idx, physicsClientId=simulator.id
                )[0]
                rotated_handle_pt = handle_point_median + axis_dir_world * translation
            else:
                rotated_handle_pt = handle_point_median

            if return_median:
                ret_handle_pt_list.append(rotated_handle_pt.flatten())
            else:
                ret_handle_pt_list.append(rotated_handle_pt)
            ret_joint_idx_list.append(joint_idx)

            handle_idx += 1

    if return_info:
        return (
            ret_handle_pt_list,
            ret_joint_idx_list,
            all_handle_pts_object_frame,
            mobility_info,
        )

    return ret_handle_pt_list, ret_joint_idx_list


def get_link_pc(simulator, object_name, custom_link_name):
    object_name = object_name.lower()
    urdf_link_name = custom_link_name
    _, all_pc = render_to_get_link_com(simulator, object_name, urdf_link_name)

    return all_pc


def render_to_get_link_com(simulator, object_name, urdf_link_name):
    # make all other objects invisible
    prev_rgbas = []
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                prev_rgba = p.getVisualShapeData(
                    obj_id, link_idx, physicsClientId=simulator.id
                )[0][14:18]
                prev_rgbas.append(prev_rgba)
                p.changeVisualShape(
                    obj_id,
                    link_idx,
                    rgbaColor=[0, 0, 0, 0],
                    physicsClientId=simulator.id,
                )

    # center camera to the target object
    env_prev_view_matrix, env_prev_projection_matrix = (
        simulator.view_matrix,
        simulator.projection_matrix,
    )
    camera_width = 640
    camera_height = 480
    obj_id = object_id
    min_aabb, max_aabb = simulator.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    distance = np.linalg.norm(max_aabb - min_aabb) * 1.2
    elevation = 30

    # get a round of images of the target object
    rgbs, depths, view_matrices, projection_matrices = take_round_images(
        simulator,
        camera_target,
        distance,
        elevation,
        camera_width=camera_width,
        camera_height=camera_height,
        z_near=0.01,
        z_far=10,
        return_camera_matrices=True,
    )

    # make the target link invisible
    link_id = get_link_id_from_name(simulator, object_name, urdf_link_name)
    prev_link_rgba = p.getVisualShapeData(
        obj_id, link_id, physicsClientId=simulator.id
    )[0][14:18]
    p.changeVisualShape(
        obj_id, link_id, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id
    )

    # get a round of images of the target object with link invisible
    rgbs_link_invisible, depths_link_invisible, _, _ = take_round_images(
        simulator,
        camera_target,
        distance,
        elevation,
        camera_width=camera_width,
        camera_height=camera_height,
        z_near=0.01,
        z_far=10,
        return_camera_matrices=True,
    )

    # use subtraction to get the link mask
    max_num_diff_pixels = 0
    best_idx = 0
    for idx, (depth, depth_) in enumerate(zip(depths, depths_link_invisible)):
        diff_image = np.abs(depth - depth_)
        diff_pixels = np.sum(diff_image > 0)
        if diff_pixels > max_num_diff_pixels:
            max_num_diff_pixels = diff_pixels
            best_idx = idx
    best_mask = np.abs(depths[best_idx] - depths_link_invisible[best_idx]) > 0

    # get the link mask center
    center = ndimage.measurements.center_of_mass(best_mask)
    center = [int(center[0]), int(center[1])]

    # back project the link mask center to get the link com in 3d coordinate
    best_pc = get_pc(
        projection_matrices[best_idx],
        view_matrices[best_idx],
        depths[best_idx],
        camera_width,
        camera_height,
    )

    pt_idx = center[0] * camera_width + center[1]
    link_com = best_pc[pt_idx]
    best_pc = best_pc.reshape((camera_height, camera_width, 3))
    all_pc = best_pc[best_mask]

    # reset the object and link rgba to previous values, and the simulator view matrix and projection matrix
    p.changeVisualShape(
        obj_id, link_id, rgbaColor=prev_link_rgba, physicsClientId=simulator.id
    )

    cnt = 0
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                p.changeVisualShape(
                    obj_id,
                    link_idx,
                    rgbaColor=prev_rgbas[cnt],
                    physicsClientId=simulator.id,
                )
                cnt += 1

    simulator.view_matrix, simulator.projection_matrix = (
        env_prev_view_matrix,
        env_prev_projection_matrix,
    )

    # add a safety check here in case the rendering fails
    bounding_box = get_bounding_box_link(simulator, object_name, urdf_link_name)
    if not in_bbox(link_com, bounding_box[0], bounding_box[1]):
        link_com = (bounding_box[0] + bounding_box[1]) / 2

    return link_com, all_pc


def get_bounding_box_link(simulator, object_name, link_name):
    object_name = object_name.lower()
    link_id = get_link_id_from_name(simulator, object_name, link_name)
    object_id = simulator.urdf_ids[object_name]
    return simulator.get_aabb_link(object_id, link_id)


def get_link_id_from_name(simulator, object_name, link_name):
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[12].decode("utf-8") == link_name:
            joint_index = i
            break

    return joint_index


def plot_rollout_observations(
    observations,
    output_path,
    frame_skip=5,
    fps=5,
    title="Rollout Observations",
):
    """
    Plot observations from a rollout episode and save as GIF.
    
    This function visualizes the observations collected during a rollout,
    including point clouds, gripper positions, agent states, and affordances.
    
    Parameters
    ----------
    observations : list of dict
        List of observation dictionaries from a rollout episode.
        Each observation dict should contain:
        - 'point_cloud': np.ndarray, shape (N, 3) or (1, N, 3) - Object point cloud
        - 'gripper_pcd': np.ndarray, shape (4, 3) or (1, 4, 3) - Gripper point cloud
        - 'agent_pos': np.ndarray, shape (10,) or (1, 10) - Agent position/state
        - 'static_affordance': optional, np.ndarray, shape (3,) or (1, 3) - Static affordance point
    output_path : str
        Path where to save the GIF
    frame_skip : int
        Render every Nth frame (default: 5)
    fps : int
        Frames per second for the GIF (default: 5)
    title : str
        Title for the visualization
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from PIL import Image
    import tqdm
    
    if not observations:
        raise ValueError("No observations provided")
    
    # Convert observations to consistent format
    processed_obs = []
    for obs in observations:
        processed = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # Remove batch dimension if present
                if value.ndim > 1 and value.shape[0] == 1:
                    value = value.squeeze(0)
                processed[key] = value
            else:
                processed[key] = value
        processed_obs.append(processed)
    
    # Extract all point clouds to compute bounds
    # Prefer part_point_cloud over point_cloud if available
    all_points = []
    for obs in processed_obs:
        # Use part_point_cloud if available, otherwise use point_cloud
        if "part_point_cloud" in obs:
            part_pc = obs["part_point_cloud"]
            # Ensure 2D shape (N, 3)
            if part_pc.ndim == 3:
                part_pc = part_pc.reshape(-1, 3)
            elif part_pc.ndim == 1:
                part_pc = part_pc.reshape(1, -1)
            if part_pc.shape[0] > 0 and part_pc.shape[1] >= 3:
                all_points.append(part_pc[:, :3])
        elif "point_cloud" in obs:
            pc = obs["point_cloud"]
            # Ensure 2D shape (N, 3)
            if pc.ndim == 3:
                pc = pc.reshape(-1, 3)
            elif pc.ndim == 1:
                pc = pc.reshape(1, -1)
            if pc.shape[0] > 0 and pc.shape[1] >= 3:
                all_points.append(pc[:, :3])
        
        if "gripper_pcd" in obs:
            gp = obs["gripper_pcd"]
            # Ensure 2D shape (N, 3)
            if gp.ndim == 3:
                gp = gp.reshape(-1, 3)
            elif gp.ndim == 1:
                gp = gp.reshape(1, -1)
            if gp.shape[0] > 0 and gp.shape[1] >= 3:
                all_points.append(gp[:, :3])
        
        # Only include static_affordance in bounds if it exists
        if "static_affordance" in obs:
            aff = obs["static_affordance"]
            # Ensure 2D shape (1, 3)
            if aff.ndim == 0:
                continue
            elif aff.ndim == 1:
                if aff.shape[0] >= 3:
                    all_points.append(aff[:3].reshape(1, -1))
            elif aff.ndim == 2:
                if aff.shape[0] > 0 and aff.shape[1] >= 3:
                    all_points.append(aff[:, :3])
    
    if not all_points:
        raise ValueError("No valid point cloud data found in observations")
    
    # Ensure all arrays are 2D before stacking
    all_points_2d = []
    for pts in all_points:
        if pts.ndim == 2:
            all_points_2d.append(pts)
        else:
            all_points_2d.append(pts.reshape(-1, 3))
    
    combined_points = np.vstack(all_points_2d)
    
    # Compute scene bounds with padding
    padding = 0.1
    x_min, x_max = (
        combined_points[:, 0].min() - padding,
        combined_points[:, 0].max() + padding,
    )
    y_min, y_max = (
        combined_points[:, 1].min() - padding,
        combined_points[:, 1].max() + padding,
    )
    z_min, z_max = (
        combined_points[:, 2].min() - padding,
        combined_points[:, 2].max() + padding,
    )
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
    
    def render_frame(frame_idx, total_frames):
        """Render a single frame with observations."""
        fig = plt.figure(figsize=(24, 8), dpi=100)
        fig.patch.set_facecolor("white")
        
        def draw_scene(ax, elev, azim, subtitle):
            """Draw a 3D scene in the given axes."""
            ax.set_facecolor("white")
            obs = processed_obs[frame_idx]
            
            # Render point cloud - show part_point_cloud if available, otherwise show regular point_cloud
            has_part_pc = "part_point_cloud" in obs
            if has_part_pc:
                part_pc = obs["part_point_cloud"]
                # Ensure 2D shape (N, 3)
                if part_pc.ndim == 3:
                    part_pc = part_pc.reshape(-1, 3)
                elif part_pc.ndim == 1:
                    part_pc = part_pc.reshape(1, -1)
                if part_pc.shape[0] > 0 and part_pc.shape[1] >= 3:
                    ax.scatter(
                        part_pc[:, 0],
                        part_pc[:, 1],
                        part_pc[:, 2],
                        c="cyan",
                        s=5,
                        alpha=0.8,
                        label="Part Point Cloud",
                        zorder=3,
                        edgecolors="blue",
                        linewidths=0.5,
                    )
            elif "point_cloud" in obs:
                pc = obs["point_cloud"]
                # Ensure 2D shape (N, 3)
                if pc.ndim == 3:
                    pc = pc.reshape(-1, 3)
                elif pc.ndim == 1:
                    pc = pc.reshape(1, -1)
                if pc.shape[0] > 0 and pc.shape[1] >= 3:
                    ax.scatter(
                        pc[:, 0],
                        pc[:, 1],
                        pc[:, 2],
                        c="saddlebrown",
                        s=1,
                        alpha=0.35,
                        label="Object Point Cloud",
                        zorder=1,
                    )
            
            # Render gripper point cloud
            if "gripper_pcd" in obs:
                gp = obs["gripper_pcd"]
                # Ensure 2D shape (N, 3)
                if gp.ndim == 3:
                    gp = gp.reshape(-1, 3)
                elif gp.ndim == 1:
                    gp = gp.reshape(1, -1)
                if gp.shape[0] >= 4 and gp.shape[1] >= 3:
                    # First 3 points are gripper fingers
                    ax.scatter(
                        gp[0:3, 0],
                        gp[0:3, 1],
                        gp[0:3, 2],
                        c="green",
                        s=50,
                        alpha=1.0,
                        marker="o",
                        label="Gripper",
                        zorder=4,
                        edgecolors="darkgreen",
                        linewidth=1,
                    )
                    # 4th point is end effector
                    ax.scatter(
                        gp[3:4, 0],
                        gp[3:4, 1],
                        gp[3:4, 2],
                        c="darkgreen",
                        s=100,
                        alpha=1.0,
                        marker="*",
                        label="End Effector",
                        zorder=4,
                        edgecolors="black",
                        linewidth=1,
                    )
            
            # Render static affordance point
            if "static_affordance" in obs:
                aff = obs["static_affordance"]
                # Handle different shapes
                if aff.ndim == 0:
                    pass  # Skip scalar
                elif aff.ndim == 1:
                    if aff.shape[0] >= 3:
                        ax.scatter(
                            aff[0],
                            aff[1],
                            aff[2],
                            c="red",
                            s=200,
                            alpha=1.0,
                            marker="*",
                            label="Static Affordance",
                            edgecolors="darkred",
                            linewidths=2,
                            zorder=10000,
                        )
                elif aff.ndim == 2:
                    if aff.shape[0] > 0 and aff.shape[1] >= 3:
                        ax.scatter(
                            aff[0, 0],
                            aff[0, 1],
                            aff[0, 2],
                            c="red",
                            s=200,
                            alpha=1.0,
                            marker="*",
                            label="Static Affordance",
                            edgecolors="darkred",
                            linewidths=2,
                            zorder=10000,
                        )
            
            # Set bounds and labels
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.view_init(elev=elev, azim=azim)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(subtitle, fontsize=10)
            
            # Add agent_pos info as text if available
            if "agent_pos" in obs:
                agent_pos = obs["agent_pos"]
                if agent_pos.shape == (10,):
                    info_text = f"Agent Pos: [{agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f}]"
                    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                             fontsize=8, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Create title
        base_title = f"{title} | Frame {frame_idx}/{total_frames}"
        
        # View 1
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        draw_scene(ax1, elev=20, azim=45, subtitle=base_title + " | View 1")
        
        # View 2
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        draw_scene(ax2, elev=20, azim=225, subtitle=base_title + " | View 2")
        
        # View 3
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        draw_scene(ax3, elev=20, azim=135, subtitle=base_title + " | View 3")
        
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(buf, mode="RGBA").convert("RGB")
        plt.close(fig)
        return img
    
    # Render frames
    images = []
    total_frames = len(processed_obs)
    frame_indices = range(0, total_frames, frame_skip)
    
    for i in frame_indices:
        img = render_frame(i, total_frames)
        images.append(img)
    
    # Save the GIF
    if not images:
        raise ValueError("No images to save!")
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def save_gif_from_trajectory_data(
    trajectory_data,
    output_path,
    frame_skip=5,
    fps=5,
    title="Trajectory Visualization",
    show_goal_gripper=True,
):
    """
    Create and save a GIF from trajectory data dictionary.

    Parameters
    ----------
    trajectory_data : dict
        Dictionary containing trajectory data with keys:
        - 'point_cloud': list of np.ndarray, shape (N, 3) for each timestep
        - 'gripper_pcd': list of np.ndarray, shape (4, 3) for each timestep
        - 'goal_gripper_pcd': optional, list of np.ndarray, shape (4, 3) for each timestep
        - 'static_affordance': optional, list of np.ndarray, shape (3,) for each timestep
        - 'part_point_cloud': optional, list of np.ndarray, shape (M, 3) for each timestep
        - 'displacement_gripper_to_object': optional, list of np.ndarray, shape (4, 3) for each timestep
        - 'agent_pos': optional, list of np.ndarray, shape (10,) for each timestep
        - 'future_point_cloud': optional, list of np.ndarray, shape (n_future_steps, N, 3) for each timestep
        - 'future_static_affordance': optional, list of np.ndarray, shape (n_future_steps, 3) for each timestep
    output_path : str
        Path where to save the GIF
    frame_skip : int
        Render every Nth frame (default: 5)
    fps : int
        Frames per second for the GIF (default: 5)
    title : str
        Title for the visualization
    show_goal_gripper : bool
        Whether to show goal gripper visualization (default: True)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from PIL import Image
    from mpl_toolkits.mplot3d import Axes3D
    import tqdm

    # Extract data
    point_clouds = trajectory_data.get("point_cloud", [])
    gripper_pcds = trajectory_data.get("gripper_pcd", [])
    goal_gripper_pcds = trajectory_data.get("goal_gripper_pcd", None)
    static_affordances = trajectory_data.get("static_affordance", None)
    part_point_clouds = trajectory_data.get("part_point_cloud", None)
    displacement_gripper_to_object = trajectory_data.get(
        "displacement_gripper_to_object", None
    )
    agent_pos = trajectory_data.get("agent_pos", None)
    future_point_clouds = trajectory_data.get("future_point_cloud", None)
    future_static_affordances = trajectory_data.get("future_static_affordance", None)

    if not point_clouds:
        raise ValueError("No point cloud data found in trajectory_data")
    
    
    # Convert to numpy arrays
    point_clouds = [np.asarray(pc) for pc in point_clouds]
    gripper_pcds = [np.asarray(gp) for gp in gripper_pcds]
    if goal_gripper_pcds is not None:
        goal_gripper_pcds = [np.asarray(gp) for gp in goal_gripper_pcds]
    if static_affordances is not None:
        static_affordances = [np.asarray(sa) for sa in static_affordances]
    if part_point_clouds is not None:
        part_point_clouds = [np.asarray(ppc) for ppc in part_point_clouds]
    if future_point_clouds is not None:
        future_point_clouds = [np.asarray(fpc) for fpc in future_point_clouds]
    if future_static_affordances is not None:
        future_static_affordances = [
            np.asarray(fsa) for fsa in future_static_affordances
        ]
        
        
    # Compute bounds from all data
    all_points = []
    for i in range(len(point_clouds)):
        all_points.append(point_clouds[i])
        all_points.append(gripper_pcds[i])
        if goal_gripper_pcds is not None:
            all_points.append(goal_gripper_pcds[i])
        if static_affordances is not None and len(static_affordances[i].shape) == 1:
            all_points.append(static_affordances[i].reshape(1, -1))
        if part_point_clouds is not None and i < len(part_point_clouds):
            all_points.append(part_point_clouds[i])
        # Include future data in bounds
        if future_point_clouds is not None and i < len(future_point_clouds):
            # future_point_clouds[i] has shape (n_future_steps, N, 3)
            for future_step in range(future_point_clouds[i].shape[0]):
                all_points.append(future_point_clouds[i][future_step])
        if future_static_affordances is not None and i < len(future_static_affordances):
            # future_static_affordances[i] has shape (n_future_steps, 3)
            all_points.append(future_static_affordances[i])
    combined_points = np.vstack(all_points)

    # Compute scene bounds with padding
    padding = 0.1
    x_min, x_max = (
        combined_points[:, 0].min() - padding,
        combined_points[:, 0].max() + padding,
    )
    y_min, y_max = (
        combined_points[:, 1].min() - padding,
        combined_points[:, 1].max() + padding,
    )
    z_min, z_max = (
        combined_points[:, 2].min() - padding,
        combined_points[:, 2].max() + padding,
    )
    bounds = (x_min, x_max, y_min, y_max, z_min, z_max)

    def render_frame(frame_idx, total_frames):
        """Render a single frame with three side-by-side viewpoints."""
        fig = plt.figure(figsize=(24, 8), dpi=100)
        fig.patch.set_facecolor("white")

        def draw_scene(ax, elev, azim, subtitle):
            """Draw a 3D scene in the given axes."""
            ax.set_facecolor("white")

            # Render point cloud
            if point_clouds[frame_idx].shape[0] > 0:
                ax.scatter(
                    point_clouds[frame_idx][:, 0],
                    point_clouds[frame_idx][:, 1],
                    point_clouds[frame_idx][:, 2],
                    c="saddlebrown",
                    s=1,
                    alpha=0.35,
                    label="Object Point Cloud",
                    zorder=1,
                )

            # Render part point cloud
            if part_point_clouds is not None and frame_idx < len(part_point_clouds):
                part_pc = part_point_clouds[frame_idx]
                if part_pc.shape[0] > 0:
                    ax.scatter(
                        part_pc[:, 0],
                        part_pc[:, 1],
                        part_pc[:, 2],
                        c="red",
                        s=1,
                        alpha=0.2,
                        label="Part Point Cloud",
                        zorder=2,
                    )

            # Render static affordance point
            if static_affordances is not None and frame_idx < len(static_affordances):
                affordance_pt = static_affordances[frame_idx]
                if affordance_pt.shape == (3,):
                    ax.scatter(
                        affordance_pt[0],
                        affordance_pt[1],
                        affordance_pt[2],
                        c="red",
                        s=200,
                        alpha=1.0,
                        marker="*",
                        label="Static Affordance",
                        edgecolors="darkred",
                        linewidths=2,
                        zorder=10000,
                    )

            # Render future point clouds
            if future_point_clouds is not None and frame_idx < len(future_point_clouds):
                future_pcs = future_point_clouds[
                    frame_idx
                ]  # Shape: (n_future_steps, N, 3)
                if future_pcs is not None and len(future_pcs.shape) >= 2:
                    n_future_steps = future_pcs.shape[0]
                    # Use a colormap to show progression through future steps
                    cmap = cm.get_cmap("plasma")
                    for future_step in range(n_future_steps):
                        future_pc = future_pcs[future_step]  # Shape: (N, 3)
                        if future_pc.shape[0] > 0:
                            # Subsample for performance if too many points
                            max_points = 500
                            if future_pc.shape[0] > max_points:
                                indices = np.random.choice(
                                    future_pc.shape[0], max_points, replace=False
                                )
                                future_pc = future_pc[indices]
                            color = cmap(future_step / max(1, n_future_steps - 1))
                            ax.scatter(
                                future_pc[:, 0],
                                future_pc[:, 1],
                                future_pc[:, 2],
                                c=[color],
                                s=0.5,
                                alpha=0.3,
                                label=(
                                    "Future Point Cloud" if future_step == 0 else None
                                ),
                                zorder=1,
                            )

            # Render future static affordance path - just a line
            if future_static_affordances is not None and frame_idx < len(
                future_static_affordances
            ):
                future_affs = future_static_affordances[
                    frame_idx
                ]  # Shape: (n_future_steps, 3)
                if future_affs is not None and len(future_affs.shape) >= 2:
                    n_future_steps = future_affs.shape[0]
                    # Plot path as a line
                    if n_future_steps > 1:
                        ax.plot(
                            future_affs[:, 0],
                            future_affs[:, 1],
                            future_affs[:, 2],
                            color="orange",
                            linewidth=2,
                            alpha=0.8,
                            linestyle="-",
                            label="Future Affordance Path",
                            zorder=10000,
                        )
                    # Optionally show very small dots at each point
                    for future_step in range(n_future_steps):
                        future_aff = future_affs[future_step]  # Shape: (3,)
                        ax.scatter(
                            future_aff[0],
                            future_aff[1],
                            future_aff[2],
                            c="orange",
                            s=1,
                            alpha=0.5,
                            marker="o",
                            zorder=10001,
                        )

            # Render goal gripper points
            if show_goal_gripper and goal_gripper_pcds is not None and frame_idx < len(goal_gripper_pcds):
                goal_gp = goal_gripper_pcds[frame_idx]
                if goal_gp.shape[0] >= 4:
                    ax.scatter(
                        goal_gp[0:3, 0],
                        goal_gp[0:3, 1],
                        goal_gp[0:3, 2],
                        c="lightgreen",
                        s=50,
                        alpha=0.6,
                        marker="o",
                        label="Goal Gripper",
                        zorder=3,
                    )
                    ax.scatter(
                        goal_gp[3:4, 0],
                        goal_gp[3:4, 1],
                        goal_gp[3:4, 2],
                        c="lime",
                        s=100,
                        alpha=0.6,
                        marker="*",
                        label="Goal End Effector",
                        zorder=3,
                    )

            # Render current gripper points
            if frame_idx < len(gripper_pcds):
                gp = gripper_pcds[frame_idx]
                if gp.shape[0] >= 4:
                    ax.scatter(
                        gp[0:3, 0],
                        gp[0:3, 1],
                        gp[0:3, 2],
                        c="green",
                        s=50,
                        alpha=1.0,
                        marker="o",
                        label="Gripper",
                        zorder=4,
                        edgecolors="darkgreen",
                        linewidth=1,
                    )
                    ax.scatter(
                        gp[3:4, 0],
                        gp[3:4, 1],
                        gp[3:4, 2],
                        c="darkgreen",
                        s=100,
                        alpha=1.0,
                        marker="*",
                        label="End Effector",
                        zorder=4,
                        edgecolors="black",
                        linewidth=1,
                    )

            # Set bounds and labels
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.view_init(elev=elev, azim=azim)
            # Only show legend for items that were actually plotted
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="upper right", fontsize=8)
            ax.set_title(subtitle, fontsize=10)

        # Create title
        base_title = f"{title} | Frame {frame_idx}/{total_frames}"

        # View 1
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        draw_scene(ax1, elev=20, azim=45, subtitle=base_title + " | View 1")

        # View 2
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        draw_scene(ax2, elev=20, azim=225, subtitle=base_title + " | View 2")

        # View 3
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        draw_scene(ax3, elev=20, azim=135, subtitle=base_title + " | View 3")

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(buf, mode="RGBA").convert("RGB")
        plt.close(fig)
        return img

    # Render frames
    images = []
    total_frames = len(point_clouds)
    frame_indices = range(0, total_frames, frame_skip)

    for i in frame_indices:
        img = render_frame(i, total_frames)
        images.append(img)

    # Save the GIF
    if not images:
        raise ValueError("No images to save!")

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def extract_ee_trajectory_from_actions(actions, start_ee_pos):
    """
    Extract end-effector position trajectory from action deltas.
    
    Actions are deltas: [pos_delta(3), rot_6d_delta(6), gripper_delta(1)]
    We accumulate position deltas starting from start_ee_pos.
    
    Parameters
    ----------
    actions : np.ndarray
        Shape (horizon, 10) - predicted actions
    start_ee_pos : np.ndarray
        Shape (3,) - starting end-effector position (from gripper_pcd[3])
    
    Returns
    -------
    ee_trajectory : np.ndarray
        Shape (horizon+1, 3) - EE positions including start position
    """
    if len(actions.shape) == 1:
        actions = actions.reshape(1, -1)
    
    horizon = actions.shape[0]
    ee_trajectory = np.zeros((horizon + 1, 3), dtype=np.float32)
    ee_trajectory[0] = start_ee_pos
    
    current_pos = start_ee_pos.copy()
    for i in range(horizon):
        pos_delta = actions[i, :3]
        current_pos = current_pos + pos_delta
        ee_trajectory[i + 1] = current_pos
    
    return ee_trajectory


def _get_observation_bounds(obs_dict, ee_trajectory=None, future_affordance=None):
    """Helper function to compute bounds for visualization."""
    all_points = []
    
    # Add point cloud
    if "part_point_cloud" in obs_dict:
        pc = obs_dict["part_point_cloud"]
        if pc.ndim == 3:
            pc = pc.reshape(-1, 3)
        elif pc.ndim == 1:
            pc = pc.reshape(1, -1)
        if pc.shape[0] > 0 and pc.shape[1] >= 3:
            all_points.append(pc[:, :3])
    elif "point_cloud" in obs_dict:
        pc = obs_dict["point_cloud"]
        if pc.ndim == 3:
            pc = pc.reshape(-1, 3)
        elif pc.ndim == 1:
            pc = pc.reshape(1, -1)
        if pc.shape[0] > 0 and pc.shape[1] >= 3:
            all_points.append(pc[:, :3])
    
    # Add gripper points
    if "gripper_pcd" in obs_dict:
        gp = obs_dict["gripper_pcd"]
        if gp.ndim == 3:
            gp = gp.reshape(-1, 3)
        elif gp.ndim == 1:
            gp = gp.reshape(1, -1)
        if gp.shape[0] > 0 and gp.shape[1] >= 3:
            all_points.append(gp[:, :3])
    
    # Add static affordance
    if "static_affordance" in obs_dict:
        aff = obs_dict["static_affordance"]
        if aff.ndim == 1 and aff.shape[0] >= 3:
            all_points.append(aff[:3].reshape(1, -1))
        elif aff.ndim == 2 and aff.shape[0] > 0 and aff.shape[1] >= 3:
            all_points.append(aff[:, :3])
    
    # Add EE trajectory
    if ee_trajectory is not None:
        if ee_trajectory.shape[0] > 0 and ee_trajectory.shape[1] >= 3:
            all_points.append(ee_trajectory[:, :3])
    
    # Add future affordance trajectory
    if future_affordance is not None:
        if future_affordance.ndim == 2 and future_affordance.shape[0] > 0 and future_affordance.shape[1] >= 3:
            all_points.append(future_affordance[:, :3])
        elif future_affordance.ndim == 1 and future_affordance.shape[0] >= 3:
            all_points.append(future_affordance[:3].reshape(1, -1))
    
    if not all_points:
        # Default bounds
        return (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    
    combined_points = np.vstack(all_points)
    padding = 0.1
    x_min = combined_points[:, 0].min() - padding
    x_max = combined_points[:, 0].max() + padding
    y_min = combined_points[:, 1].min() - padding
    y_max = combined_points[:, 1].max() + padding
    z_min = combined_points[:, 2].min() - padding
    z_max = combined_points[:, 2].max() + padding
    
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def _draw_scene_3d(ax, obs_dict, bounds, elev=20, azim=45, ee_trajectory=None, future_affordance=None):
    """Helper function to draw a 3D scene."""
    ax.set_facecolor("white")
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    # Draw point cloud
    if "part_point_cloud" in obs_dict:
        part_pc = obs_dict["part_point_cloud"]
        if part_pc.ndim == 3:
            part_pc = part_pc.reshape(-1, 3)
        elif part_pc.ndim == 1:
            part_pc = part_pc.reshape(1, -1)
        if part_pc.shape[0] > 0 and part_pc.shape[1] >= 3:
            ax.scatter(
                part_pc[:, 0], part_pc[:, 1], part_pc[:, 2],
                c="cyan", s=5, alpha=0.8, label="Part Point Cloud",
                zorder=3, edgecolors="blue", linewidths=0.5
            )
    elif "point_cloud" in obs_dict:
        pc = obs_dict["point_cloud"]
        if pc.ndim == 3:
            pc = pc.reshape(-1, 3)
        elif pc.ndim == 1:
            pc = pc.reshape(1, -1)
        if pc.shape[0] > 0 and pc.shape[1] >= 3:
            ax.scatter(
                pc[:, 0], pc[:, 1], pc[:, 2],
                c="saddlebrown", s=1, alpha=0.35, label="Object Point Cloud",
                zorder=1
            )
    
    # Draw gripper
    if "gripper_pcd" in obs_dict:
        gp = obs_dict["gripper_pcd"]
        if gp.ndim == 3:
            gp = gp.reshape(-1, 3)
        elif gp.ndim == 1:
            gp = gp.reshape(1, -1)
        if gp.shape[0] >= 4 and gp.shape[1] >= 3:
            # First 3 points are gripper fingers
            ax.scatter(
                gp[0:3, 0], gp[0:3, 1], gp[0:3, 2],
                c="green", s=50, alpha=1.0, marker="o",
                label="Gripper", zorder=4, edgecolors="darkgreen", linewidth=1
            )
            # 4th point is end effector
            ax.scatter(
                gp[3:4, 0], gp[3:4, 1], gp[3:4, 2],
                c="darkgreen", s=100, alpha=1.0, marker="*",
                label="End Effector", zorder=4, edgecolors="black", linewidth=1
            )
    
    # Draw static affordance
    if "static_affordance" in obs_dict:
        aff = obs_dict["static_affordance"]
        if aff.ndim == 1 and aff.shape[0] >= 3:
            ax.scatter(
                aff[0], aff[1], aff[2],
                c="red", s=200, alpha=1.0, marker="*",
                label="Static Affordance", edgecolors="darkred",
                linewidths=2, zorder=10000
            )
        elif aff.ndim == 2 and aff.shape[0] > 0 and aff.shape[1] >= 3:
            ax.scatter(
                aff[0, 0], aff[0, 1], aff[0, 2],
                c="red", s=200, alpha=1.0, marker="*",
                label="Static Affordance", edgecolors="darkred",
                linewidths=2, zorder=10000
            )
    
    # Draw EE trajectory
    if ee_trajectory is not None and ee_trajectory.shape[0] > 0:
        ax.plot(
            ee_trajectory[:, 0], ee_trajectory[:, 1], ee_trajectory[:, 2],
            c="orange", linewidth=2, alpha=0.8, label="Predicted EE Trajectory",
            zorder=5
        )
        ax.scatter(
            ee_trajectory[:, 0], ee_trajectory[:, 1], ee_trajectory[:, 2],
            c="orange", s=30, alpha=0.6, zorder=5
        )
    
    # Draw future affordance trajectory
    if future_affordance is not None:
        if future_affordance.ndim == 2 and future_affordance.shape[0] > 0:
            ax.plot(
                future_affordance[:, 0], future_affordance[:, 1], future_affordance[:, 2],
                c="magenta", linewidth=2, alpha=0.8, label="Future Static Affordance",
                zorder=6, linestyle="--"
            )
            ax.scatter(
                future_affordance[:, 0], future_affordance[:, 1], future_affordance[:, 2],
                c="magenta", s=50, alpha=0.6, marker="s", zorder=6
            )
        elif future_affordance.ndim == 1 and future_affordance.shape[0] >= 3:
            ax.scatter(
                future_affordance[0], future_affordance[1], future_affordance[2],
                c="magenta", s=100, alpha=0.8, marker="s",
                label="Future Static Affordance", zorder=6
            )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right", fontsize=8)


def plot_static_timestep_visualization(obs, predicted_actions, output_path, title="Static Timestep"):
    """
    Visualize static model timestep with observation and predicted EE trajectory.
    
    Parameters
    ----------
    obs : dict
        Observation dictionary with point_cloud/part_point_cloud, gripper_pcd, static_affordance
    predicted_actions : np.ndarray
        Shape (horizon, 10) - predicted actions
    output_path : str
        Path to save GIF
    title : str
        Title for visualization
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from PIL import Image
    
    # Extract EE trajectory
    if "gripper_pcd" in obs:
        gp = obs["gripper_pcd"]
        if gp.ndim == 3:
            gp = gp.reshape(-1, 3)
        elif gp.ndim == 1:
            gp = gp.reshape(1, -1)
        if gp.shape[0] >= 4:
            start_ee_pos = gp[3, :3]
        else:
            start_ee_pos = np.array([0.0, 0.0, 0.0])
    else:
        start_ee_pos = np.array([0.0, 0.0, 0.0])
    
    ee_trajectory = extract_ee_trajectory_from_actions(predicted_actions, start_ee_pos)
    
    # Compute bounds
    bounds = _get_observation_bounds(obs, ee_trajectory=ee_trajectory)
    
    # Create figure with 3 views
    fig = plt.figure(figsize=(24, 8), dpi=100)
    fig.patch.set_facecolor("white")
    
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_scene_3d(ax1, obs, bounds, elev=20, azim=45, ee_trajectory=ee_trajectory)
    ax1.set_title(f"{title} | View 1", fontsize=10)
    
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_scene_3d(ax2, obs, bounds, elev=20, azim=225, ee_trajectory=ee_trajectory)
    ax2.set_title(f"{title} | View 2", fontsize=10)
    
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    _draw_scene_3d(ax3, obs, bounds, elev=20, azim=135, ee_trajectory=ee_trajectory)
    ax3.set_title(f"{title} | View 3", fontsize=10)
    
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf, mode="RGBA").convert("RGB")
    plt.close(fig)
    
    # Save as single image (not GIF for static)
    img.save(output_path.replace('.gif', '.png'))


def plot_refinement_initial_visualization(obs, predicted_actions, output_path, title="Refinement Initial"):
    """
    Visualize initial refinement (iteration 0) with observation and predicted EE trajectory.
    
    Parameters
    ----------
    obs : dict
        Observation dictionary (use last from n_obs=2 for part PC)
    predicted_actions : np.ndarray
        Shape (horizon, 10) - predicted actions
    output_path : str
        Path to save GIF
    title : str
        Title for visualization
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from PIL import Image
    
    # Extract last observation if obs has temporal dimension
    processed_obs = obs.copy()
    if "point_cloud" in processed_obs and processed_obs["point_cloud"].ndim == 3:
        # Shape (n_obs_steps, N, 3) - take last
        processed_obs["point_cloud"] = processed_obs["point_cloud"][-1]
    if "part_point_cloud" in processed_obs and processed_obs["part_point_cloud"].ndim == 3:
        processed_obs["part_point_cloud"] = processed_obs["part_point_cloud"][-1]
    if "gripper_pcd" in processed_obs and processed_obs["gripper_pcd"].ndim == 3:
        processed_obs["gripper_pcd"] = processed_obs["gripper_pcd"][-1]
    if "static_affordance" in processed_obs and processed_obs["static_affordance"].ndim == 2:
        processed_obs["static_affordance"] = processed_obs["static_affordance"][-1]
    
    # Extract EE trajectory
    if "gripper_pcd" in processed_obs:
        gp = processed_obs["gripper_pcd"]
        if gp.ndim == 3:
            gp = gp.reshape(-1, 3)
        elif gp.ndim == 1:
            gp = gp.reshape(1, -1)
        if gp.shape[0] >= 4:
            start_ee_pos = gp[3, :3]
        else:
            start_ee_pos = np.array([0.0, 0.0, 0.0])
    else:
        start_ee_pos = np.array([0.0, 0.0, 0.0])
    
    ee_trajectory = extract_ee_trajectory_from_actions(predicted_actions, start_ee_pos)
    
    # Compute bounds
    bounds = _get_observation_bounds(processed_obs, ee_trajectory=ee_trajectory)
    
    # Create figure with 3 views
    fig = plt.figure(figsize=(24, 8), dpi=100)
    fig.patch.set_facecolor("white")
    
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_scene_3d(ax1, processed_obs, bounds, elev=20, azim=45, ee_trajectory=ee_trajectory)
    ax1.set_title(f"{title} | View 1", fontsize=10)
    
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_scene_3d(ax2, processed_obs, bounds, elev=20, azim=225, ee_trajectory=ee_trajectory)
    ax2.set_title(f"{title} | View 2", fontsize=10)
    
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    _draw_scene_3d(ax3, processed_obs, bounds, elev=20, azim=135, ee_trajectory=ee_trajectory)
    ax3.set_title(f"{title} | View 3", fontsize=10)
    
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf, mode="RGBA").convert("RGB")
    plt.close(fig)
    
    img.save(output_path.replace('.gif', '.png'))


def plot_refinement_simulator_visualization(obs_sequence, actions_sequence, output_path, title="Refinement Simulator", initial_obs=None):
    """
    Visualize simulator execution showing how observations change and EE trajectory.
    
    Parameters
    ----------
    obs_sequence : list of dict
        List of observations AFTER simulator execution (obs[i] is result of action[i])
    actions_sequence : list of np.ndarray
        List of actions executed (each shape (1, 10) or (10,))
    output_path : str
        Path to save GIF
    title : str
        Title for visualization
    initial_obs : dict, optional
        Initial observation BEFORE any actions were executed. If provided, used to get
        the starting EE position for the first action.
    
    Note: obs_sequence[i] is the observation AFTER executing actions_sequence[i].
    To correctly show what trajectory led to each observation, we need the EE position
    BEFORE each action. This is:
    - For action 0: from initial_obs (if provided) or estimated by subtracting action delta
    - For action i>0: from obs_sequence[i-1]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from PIL import Image
    import tqdm
    
    if not obs_sequence:
        raise ValueError("No observations in sequence")
    
    # Extract EE trajectories for each step
    # Key insight: obs_sequence[i] is AFTER action[i], so to get start_ee_pos for action[i],
    # we need the gripper position BEFORE action[i] was executed:
    # - For i=0: use initial_obs if provided, otherwise estimate from obs[0] - action[0]
    # - For i>0: use obs_sequence[i-1] (the result after action[i-1])
    ee_trajectories = []
    for i, (obs, action) in enumerate(zip(obs_sequence, actions_sequence)):
        if len(action.shape) == 1:
            action = action.reshape(1, -1)
        
        # Get the EE position BEFORE this action was executed
        if i == 0:
            if initial_obs is not None and "gripper_pcd" in initial_obs:
                # Use initial observation
                gp = initial_obs["gripper_pcd"]
                if gp.ndim == 3:
                    gp = gp[-1]  # Take last obs step
                start_ee_pos = gp[3, :3] if gp.shape[0] >= 4 else np.array([0.0, 0.0, 0.0])
            elif "gripper_pcd" in obs:
                # Estimate: current position - action delta = position before action
                gp = obs["gripper_pcd"]
                if gp.ndim == 3:
                    gp = gp[-1]
                current_ee_pos = gp[3, :3] if gp.shape[0] >= 4 else np.array([0.0, 0.0, 0.0])
                # Subtract the action delta to get position before
                start_ee_pos = current_ee_pos - action[0, :3]
            else:
                start_ee_pos = np.array([0.0, 0.0, 0.0])
        else:
            # Use the previous observation (which is the state AFTER action[i-1], i.e., BEFORE action[i])
            prev_obs = obs_sequence[i - 1]
            if "gripper_pcd" in prev_obs:
                gp = prev_obs["gripper_pcd"]
                if gp.ndim == 3:
                    gp = gp[-1]
                start_ee_pos = gp[3, :3] if gp.shape[0] >= 4 else np.array([0.0, 0.0, 0.0])
            else:
                start_ee_pos = np.array([0.0, 0.0, 0.0])
        
        ee_traj = extract_ee_trajectory_from_actions(action, start_ee_pos)
        ee_trajectories.append(ee_traj)
    
    # Compute bounds from all observations
    all_bounds = []
    for obs, ee_traj in zip(obs_sequence, ee_trajectories):
        bounds = _get_observation_bounds(obs, ee_trajectory=ee_traj)
        all_bounds.append(bounds)
    
    # Use max bounds
    x_mins, x_maxs, y_mins, y_maxs, z_mins, z_maxs = zip(*all_bounds)
    bounds = (min(x_mins), max(x_maxs), min(y_mins), max(y_maxs), min(z_mins), max(z_maxs))
    
    # Render frames
    images = []
    for frame_idx in range(len(obs_sequence)):
        obs = obs_sequence[frame_idx]
        ee_traj = ee_trajectories[frame_idx]
        
        # Process obs to remove temporal dimension if present
        processed_obs = obs.copy()
        if "point_cloud" in processed_obs and processed_obs["point_cloud"].ndim == 3:
            processed_obs["point_cloud"] = processed_obs["point_cloud"][-1]
        if "part_point_cloud" in processed_obs and processed_obs["part_point_cloud"].ndim == 3:
            processed_obs["part_point_cloud"] = processed_obs["part_point_cloud"][-1]
        if "gripper_pcd" in processed_obs and processed_obs["gripper_pcd"].ndim == 3:
            processed_obs["gripper_pcd"] = processed_obs["gripper_pcd"][-1]
        if "static_affordance" in processed_obs and processed_obs["static_affordance"].ndim == 2:
            processed_obs["static_affordance"] = processed_obs["static_affordance"][-1]
        
        fig = plt.figure(figsize=(24, 8), dpi=100)
        fig.patch.set_facecolor("white")
        
        frame_title = f"{title} | Frame {frame_idx}/{len(obs_sequence)}"
        
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        _draw_scene_3d(ax1, processed_obs, bounds, elev=20, azim=45, ee_trajectory=ee_traj)
        ax1.set_title(f"{frame_title} | View 1", fontsize=10)
        
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        _draw_scene_3d(ax2, processed_obs, bounds, elev=20, azim=225, ee_trajectory=ee_traj)
        ax2.set_title(f"{frame_title} | View 2", fontsize=10)
        
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        _draw_scene_3d(ax3, processed_obs, bounds, elev=20, azim=135, ee_trajectory=ee_traj)
        ax3.set_title(f"{frame_title} | View 3", fontsize=10)
        
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(buf, mode="RGBA").convert("RGB")
        plt.close(fig)
        images.append(img)
    
    # Save GIF
    if not images:
        raise ValueError("No images to save!")
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / 5),  # 5 fps
        loop=0,
    )


def plot_refinement_input_visualization(obs, future_static_affordance, output_path, title="Refinement Input"):
    """
    Visualize refinement input showing future observations and current observation.
    
    Parameters
    ----------
    obs : dict
        Current observation
    future_static_affordance : np.ndarray
        Shape (n_future_steps, 3) - future static affordance trajectory
    output_path : str
        Path to save GIF
    title : str
        Title for visualization
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from PIL import Image
    
    # Process obs to remove temporal dimension if present
    processed_obs = obs.copy()
    if "point_cloud" in processed_obs and processed_obs["point_cloud"].ndim == 3:
        processed_obs["point_cloud"] = processed_obs["point_cloud"][-1]
    if "part_point_cloud" in processed_obs and processed_obs["part_point_cloud"].ndim == 3:
        processed_obs["part_point_cloud"] = processed_obs["part_point_cloud"][-1]
    if "gripper_pcd" in processed_obs and processed_obs["gripper_pcd"].ndim == 3:
        processed_obs["gripper_pcd"] = processed_obs["gripper_pcd"][-1]
    if "static_affordance" in processed_obs and processed_obs["static_affordance"].ndim == 2:
        processed_obs["static_affordance"] = processed_obs["static_affordance"][-1]
    
    # Ensure future_static_affordance is 2D
    if future_static_affordance.ndim == 3:
        # Shape (n_obs_steps, n_future_steps, 3) - take first obs step
        future_static_affordance = future_static_affordance[0]
    elif future_static_affordance.ndim == 1:
        future_static_affordance = future_static_affordance.reshape(1, -1)
    
    # Compute bounds
    bounds = _get_observation_bounds(processed_obs, future_affordance=future_static_affordance)
    
    # Create figure with 3 views
    fig = plt.figure(figsize=(24, 8), dpi=100)
    fig.patch.set_facecolor("white")
    
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_scene_3d(ax1, processed_obs, bounds, elev=20, azim=45, future_affordance=future_static_affordance)
    ax1.set_title(f"{title} | View 1", fontsize=10)
    
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_scene_3d(ax2, processed_obs, bounds, elev=20, azim=225, future_affordance=future_static_affordance)
    ax2.set_title(f"{title} | View 2", fontsize=10)
    
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    _draw_scene_3d(ax3, processed_obs, bounds, elev=20, azim=135, future_affordance=future_static_affordance)
    ax3.set_title(f"{title} | View 3", fontsize=10)
    
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf, mode="RGBA").convert("RGB")
    plt.close(fig)
    
    img.save(output_path.replace('.gif', '.png'))


def plot_refinement_prediction_visualization(obs, predicted_actions, future_static_affordance, output_path, title="Refinement Prediction", mean_diff=None):
    """
    Visualize refinement prediction showing predicted EE trajectory and future affordance.
    
    Parameters
    ----------
    obs : dict
        Current observation
    predicted_actions : np.ndarray
        Shape (horizon, 10) - predicted actions
    future_static_affordance : np.ndarray
        Shape (n_future_steps, 3) - future static affordance trajectory
    output_path : str
        Path to save GIF
    title : str
        Title for visualization
    mean_diff : float, optional
        Mean difference from previous iteration (will be added to title if provided)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from PIL import Image
    
    # Process obs to remove temporal dimension if present
    processed_obs = obs.copy()
    if "point_cloud" in processed_obs and processed_obs["point_cloud"].ndim == 3:
        processed_obs["point_cloud"] = processed_obs["point_cloud"][-1]
    if "part_point_cloud" in processed_obs and processed_obs["part_point_cloud"].ndim == 3:
        processed_obs["part_point_cloud"] = processed_obs["part_point_cloud"][-1]
    if "gripper_pcd" in processed_obs and processed_obs["gripper_pcd"].ndim == 3:
        processed_obs["gripper_pcd"] = processed_obs["gripper_pcd"][-1]
    if "static_affordance" in processed_obs and processed_obs["static_affordance"].ndim == 2:
        processed_obs["static_affordance"] = processed_obs["static_affordance"][-1]
    
    # Extract EE trajectory
    if "gripper_pcd" in processed_obs:
        gp = processed_obs["gripper_pcd"]
        if gp.ndim == 3:
            gp = gp.reshape(-1, 3)
        elif gp.ndim == 1:
            gp = gp.reshape(1, -1)
        if gp.shape[0] >= 4:
            start_ee_pos = gp[3, :3]
        else:
            start_ee_pos = np.array([0.0, 0.0, 0.0])
    else:
        start_ee_pos = np.array([0.0, 0.0, 0.0])
    
    ee_trajectory = extract_ee_trajectory_from_actions(predicted_actions, start_ee_pos)
    
    # Handle None or process future_static_affordance
    if future_static_affordance is not None:
        if future_static_affordance.ndim == 3:
            # Shape (n_obs_steps, n_future_steps, 3) - take first obs step
            future_static_affordance = future_static_affordance[0]
        elif future_static_affordance.ndim == 1:
            future_static_affordance = future_static_affordance.reshape(1, -1)
    
    # Compute bounds
    bounds = _get_observation_bounds(processed_obs, ee_trajectory=ee_trajectory, future_affordance=future_static_affordance)
    
    # Create figure with 3 views
    fig = plt.figure(figsize=(24, 8), dpi=100)
    fig.patch.set_facecolor("white")
    
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_scene_3d(ax1, processed_obs, bounds, elev=20, azim=45, ee_trajectory=ee_trajectory, future_affordance=future_static_affordance)
    ax1.set_title(f"{title} | View 1", fontsize=10)
    
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw_scene_3d(ax2, processed_obs, bounds, elev=20, azim=225, ee_trajectory=ee_trajectory, future_affordance=future_static_affordance)
    ax2.set_title(f"{title} | View 2", fontsize=10)
    
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    _draw_scene_3d(ax3, processed_obs, bounds, elev=20, azim=135, ee_trajectory=ee_trajectory, future_affordance=future_static_affordance)
    ax3.set_title(f"{title} | View 3", fontsize=10)
    
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf, mode="RGBA").convert("RGB")
    plt.close(fig)
    
    img.save(output_path.replace('.gif', '.png'))


def plot_all_refinement_trajectories(obs, all_predicted_trajectories, output_path, title="All Refinement Trajectories", mean_diffs=None, n_steps_executed=None, n_action_steps=None):
    """
    Visualize all refinement iteration trajectories overlaid on a single plot.

    Parameters
    ----------
    obs : dict
        Current observation (for context - point cloud, gripper, etc.)
    all_predicted_trajectories : list of np.ndarray
        List of EE trajectories, one for each refinement iteration
        Each trajectory has shape (horizon+1, 3)
    output_path : str
        Path to save image
    title : str
        Title for visualization (may already include diffs)
    mean_diffs : list of float or None, optional
        List of mean differences for each iteration (None for iter 0, float for others)
    n_steps_executed : int or None, optional
        Number of action steps actually executed. If provided, the executed portion
        of trajectories will be highlighted.
    n_action_steps : int or None, optional
        Total number of action steps predicted. Used with n_steps_executed to determine
        which portion of the trajectory was executed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from PIL import Image
    import matplotlib.cm as cm
    
    # Process obs to remove temporal dimension if present
    processed_obs = obs.copy()
    if "point_cloud" in processed_obs and processed_obs["point_cloud"].ndim == 3:
        processed_obs["point_cloud"] = processed_obs["point_cloud"][-1]
    if "part_point_cloud" in processed_obs and processed_obs["part_point_cloud"].ndim == 3:
        processed_obs["part_point_cloud"] = processed_obs["part_point_cloud"][-1]
    if "gripper_pcd" in processed_obs and processed_obs["gripper_pcd"].ndim == 3:
        processed_obs["gripper_pcd"] = processed_obs["gripper_pcd"][-1]
    if "static_affordance" in processed_obs and processed_obs["static_affordance"].ndim == 2:
        processed_obs["static_affordance"] = processed_obs["static_affordance"][-1]
    
    # Compute bounds including all trajectories
    all_traj_points = []
    for traj in all_predicted_trajectories:
        if traj.shape[0] > 0 and traj.shape[1] >= 3:
            all_traj_points.append(traj[:, :3])
    
    bounds = _get_observation_bounds(processed_obs)
    if all_traj_points:
        # Extend bounds to include all trajectories
        combined_traj_points = np.vstack(all_traj_points)
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        padding = 0.1
        x_min = min(x_min, combined_traj_points[:, 0].min() - padding)
        x_max = max(x_max, combined_traj_points[:, 0].max() + padding)
        y_min = min(y_min, combined_traj_points[:, 1].min() - padding)
        y_max = max(y_max, combined_traj_points[:, 1].max() + padding)
        z_min = min(z_min, combined_traj_points[:, 2].min() - padding)
        z_max = max(z_max, combined_traj_points[:, 2].max() + padding)
        bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
    
    # Create colormap for trajectories
    n_trajectories = len(all_predicted_trajectories)
    colors = cm.viridis(np.linspace(0, 1, n_trajectories))
    
    # Create figure with 3 views
    fig = plt.figure(figsize=(24, 8), dpi=100)
    fig.patch.set_facecolor("white")
    
    def draw_scene_with_all_trajectories(ax, elev, azim, subtitle):
        """Draw scene with all trajectories overlaid."""
        ax.set_facecolor("white")
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Draw point cloud
        if "part_point_cloud" in processed_obs:
            part_pc = processed_obs["part_point_cloud"]
            if part_pc.ndim == 3:
                part_pc = part_pc.reshape(-1, 3)
            elif part_pc.ndim == 1:
                part_pc = part_pc.reshape(1, -1)
            if part_pc.shape[0] > 0 and part_pc.shape[1] >= 3:
                ax.scatter(
                    part_pc[:, 0], part_pc[:, 1], part_pc[:, 2],
                    c="cyan", s=5, alpha=0.3, label="Part Point Cloud",
                    zorder=1, edgecolors="blue", linewidths=0.3
                )
        elif "point_cloud" in processed_obs:
            pc = processed_obs["point_cloud"]
            if pc.ndim == 3:
                pc = pc.reshape(-1, 3)
            elif pc.ndim == 1:
                pc = pc.reshape(1, -1)
            if pc.shape[0] > 0 and pc.shape[1] >= 3:
                ax.scatter(
                    pc[:, 0], pc[:, 1], pc[:, 2],
                    c="saddlebrown", s=1, alpha=0.2, label="Object Point Cloud",
                    zorder=1
                )
        
        # Draw gripper
        if "gripper_pcd" in processed_obs:
            gp = processed_obs["gripper_pcd"]
            if gp.ndim == 3:
                gp = gp.reshape(-1, 3)
            elif gp.ndim == 1:
                gp = gp.reshape(1, -1)
            if gp.shape[0] >= 4 and gp.shape[1] >= 3:
                # First 3 points are gripper fingers
                ax.scatter(
                    gp[0:3, 0], gp[0:3, 1], gp[0:3, 2],
                    c="green", s=50, alpha=1.0, marker="o",
                    label="Gripper", zorder=4, edgecolors="darkgreen", linewidth=1
                )
                # 4th point is end effector
                ax.scatter(
                    gp[3:4, 0], gp[3:4, 1], gp[3:4, 2],
                    c="darkgreen", s=100, alpha=1.0, marker="*",
                    label="End Effector", zorder=4, edgecolors="black", linewidth=1
                )
        
        # Draw static affordance
        if "static_affordance" in processed_obs:
            aff = processed_obs["static_affordance"]
            if aff.ndim == 1 and aff.shape[0] >= 3:
                ax.scatter(
                    aff[0], aff[1], aff[2],
                    c="red", s=200, alpha=1.0, marker="*",
                    label="Static Affordance", edgecolors="darkred",
                    linewidths=2, zorder=10000
                )
            elif aff.ndim == 2 and aff.shape[0] > 0 and aff.shape[1] >= 3:
                ax.scatter(
                    aff[0, 0], aff[0, 1], aff[0, 2],
                    c="red", s=200, alpha=1.0, marker="*",
                    label="Static Affordance", edgecolors="darkred",
                    linewidths=2, zorder=10000
                )
        
        # Draw all trajectories with different colors
        # If n_steps_executed is set, show executed portion highlighted
        for iter_idx, traj in enumerate(all_predicted_trajectories):
            if traj.shape[0] > 0:
                color = colors[iter_idx]
                label = f"Refinement {iter_idx}"
                if mean_diffs is not None and iter_idx < len(mean_diffs) and mean_diffs[iter_idx] is not None:
                    label += f" (diff={mean_diffs[iter_idx]:.6f})"

                # Determine executed portion
                # Trajectory has shape (horizon+1, 3) where first point is start position
                # So executed portion is indices 0 to n_steps_executed (inclusive)
                if n_steps_executed is not None and n_steps_executed < traj.shape[0] - 1:
                    exec_end_idx = n_steps_executed + 1  # +1 because first point is start

                    # Draw predicted (non-executed) portion - thin, dashed, semi-transparent
                    if exec_end_idx < traj.shape[0]:
                        ax.plot(
                            traj[exec_end_idx-1:, 0], traj[exec_end_idx-1:, 1], traj[exec_end_idx-1:, 2],
                            c=color, linewidth=1.0, alpha=0.4, linestyle='--',
                            zorder=5 + iter_idx
                        )

                    # Draw executed portion - thick, solid
                    ax.plot(
                        traj[:exec_end_idx, 0], traj[:exec_end_idx, 1], traj[:exec_end_idx, 2],
                        c=color, linewidth=2.5, alpha=0.9, label=label,
                        zorder=6 + iter_idx
                    )
                    # Add markers at executed waypoints
                    ax.scatter(
                        traj[1:exec_end_idx, 0], traj[1:exec_end_idx, 1], traj[1:exec_end_idx, 2],
                        c=[color], s=30, alpha=0.9, marker='o',
                        zorder=7 + iter_idx
                    )
                else:
                    # No n_steps_executed or all steps executed - draw normally
                    ax.plot(
                        traj[:, 0], traj[:, 1], traj[:, 2],
                        c=color, linewidth=1.5, alpha=0.7, label=label,
                        zorder=5 + iter_idx
                    )
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(subtitle, fontsize=10)
    
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    draw_scene_with_all_trajectories(ax1, elev=20, azim=45, subtitle=f"{title} | View 1")
    
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    draw_scene_with_all_trajectories(ax2, elev=20, azim=225, subtitle=f"{title} | View 2")
    
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    draw_scene_with_all_trajectories(ax3, elev=20, azim=135, subtitle=f"{title} | View 3")
    
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf, mode="RGBA").convert("RGB")
    plt.close(fig)
    
    img.save(output_path.replace('.gif', '.png'))
