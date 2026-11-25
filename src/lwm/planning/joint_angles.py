"""
Canonical per-object joint angle constants from final_goal_based dataset.

- Doors: goal_040 (40% of object range).
- Drawers: goal_080 (80% of object range).

Use these for MPC success: fraction = (final - initial) / (OBJECT_JOINT_ANGLE_AT_GOAL[id] - initial).

To regenerate after dataset changes:
  python report_goal_joint_angles.py /path/to/final_goal_based -o joint_angle_constants.py
"""

# Nominal goal fraction (0.4 doors, 0.8 drawers) — for reference only
OBJECT_GOAL_JOINT_ANGLE = {
    "35059": 0.4, "45166": 0.4, "45249": 0.4, "45443": 0.4, "45526": 0.4,
    "46092": 0.4, "46417": 0.4, "47388": 0.4,
    "44817": 0.8, "44853": 0.8, "45092": 0.8, "45132": 0.8, "45135": 0.8,
    "45243": 0.8, "45290": 0.8, "45374": 0.8,
}

# Actual joint value (rad/m) at canonical goal — USE THIS FOR SUCCESS RATE
OBJECT_JOINT_ANGLE_AT_GOAL = {
    "35059": 1.1309733552923253, "45166": 0.6785840131753953,
    "45249": 1.2496557444279401, "45443": 0.6584778201924207,
    "45526": 0.6408849013323179, "46092": 0.6981317007977319,
    "46417": 0.6981317007977319, "47388": 0.6283185307179586,
    "44817": 0.23336194899428647, "44853": 0.35609129809859175,
    "45092": 0.31809827423459325, "45132": 0.3658141445890663,
    "45135": 0.2433225240235995, "45243": 0.23589725380428805,
    "45290": 0.31880955382734993, "45374": 0.24843825981407705,
}

# Object joint upper limit (rad doors, m drawers) — from opened_angle.txt line2
OBJECT_JOINT_ANGLE_MAX = {
    "35059": 2.8274333882308134, "45166": 1.6964600329384882,
    "45249": 3.12413936106985, "45443": 1.6461945504810518,
    "45526": 1.6022122533307945, "46092": 1.7453292519943295,
    "46417": 1.7453292519943295, "47388": 1.5707963267948966,
    "44817": 0.2917024362428581, "44853": 0.4451141226232397,
    "45092": 0.39762284279324156, "45132": 0.45726768073633284,
    "45135": 0.30415315502949936, "45243": 0.29487156725536007,
    "45290": 0.3985119422841874, "45374": 0.3105478247675963,
}
