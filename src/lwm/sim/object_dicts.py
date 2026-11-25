"""PartNet-Mobility category index (from ArticuBot/RoboGen objaverse_utils/utils.py)."""

import json
import os
from collections import defaultdict


# Object metadata shipped with the package (lwm/sim/data/).
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if os.path.exists(f"{data_dir}/partnet_mobility_dict.json"):
    with open(f"{data_dir}/partnet_mobility_dict.json", "r") as f:
        partnet_mobility_dict = json.load(f)
else:
    partnet_mobility_dict = {}

if os.path.exists(f"{data_dir}/sapien_cannot_vhacd_part.json"):
    with open(f"{data_dir}/sapien_cannot_vhacd_part.json", "r") as f:
        sapaien_cannot_vhacd_part_dict = json.load(f)
else:
    sapaien_cannot_vhacd_part_dict = defaultdict(list)
