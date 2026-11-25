# Third-party notices

This repository contains code and assets adapted from the projects below. Our own code is
released under the MIT licence (see `LICENSE`); the components below keep their original
licences.

## 3D Diffusion Policy (DP3)

- Source: https://github.com/YanjieZe/3D-Diffusion-Policy
- Licence: MIT, Copyright (c) 2024 Yanjie Ze
- Used in: `src/lwm/world_model/pointnet.py` (`PointNetEncoderXYZ`),
  `src/lwm/common/pytorch_util.py`, `src/lwm/planning/multistep_wrapper.py`.
  `src/lwm/planning/sim_utils/` is adapted from the evaluation utilities of the ArticuBot
  fork of DP3.

## Diffusion Policy

- Source: https://github.com/real-stanford/diffusion_policy
- Licence: MIT, Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
- Used in: DP3 derives `pytorch_util.py` and the multi-step environment wrapper from it.

## PointNet++ (PyTorch)

- Source: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- Licence: MIT, Copyright (c) 2019 benny
- Used in: `src/lwm/world_model/pointnet2.py` (set-abstraction layer, via ArticuBot's
  `weighted_displacement_model`).

## ArticuBot / RoboGen (simulation code)

- Sources: https://github.com/yufeiwang63/ArticuBot (Wang et al., RSS 2025), which builds on
  RoboGen (https://github.com/Genesis-Embodied-AI/RoboGen).
- Used in: `src/lwm/sim/` (simulator, robot, primitives, point-cloud wrapper, object
  metadata `data/partnet_mobility_dict.json`) and `src/lwm/planning/sim_utils/`.

## Franka Emika Panda description

- Source: franka_ros / franka_description (Franka Emika GmbH)
- Licence: Apache-2.0, see `src/lwm/sim/assets/panda_bullet/LICENSE.txt`
- Used in: `src/lwm/sim/assets/panda_bullet/`, `src/lwm/sim/assets/panda_ik/`.

## PyBullet ground plane

- Source: pybullet_data (Bullet Physics), zlib licence
- Used in: `src/lwm/sim/assets/plane/`.

## PartNet-Mobility / SAPIEN

- Not included in this repository. The simulator loads PartNet-Mobility objects from
  `PARTNET_DATASET_DIR`. Use of the dataset is subject to the SAPIEN / PartNet-Mobility terms
  of use (https://sapien.ucsd.edu/). Please cite Xiang et al., "SAPIEN: A SimulAted
  Part-based Interactive ENvironment" (CVPR 2020) and Mo et al., "PartNet" (CVPR 2019).

## Runtime dependencies

PyBullet (zlib), PyTorch (BSD-3-Clause) and the other
packages listed in `pyproject.toml` are installed from PyPI under their own licences.
