"""Success metric of the paper's MPC table (Table 1), computed from an episode's ``metrics.json``.

Task progress is the fraction of the target joint displacement reached at the end of the
episode: ``(final - initial) / (goal - initial)``, where ``goal`` is the joint value at the
task-defined target (40% of the joint range for doors, 80% for drawers,
:data:`lwm.planning.joint_angles.OBJECT_JOINT_ANGLE_AT_GOAL`). SR@0.4 / SR@0.8 count episodes
with progress >= 0.4 / 0.8.
"""

from lwm.planning.joint_angles import OBJECT_JOINT_ANGLE_AT_GOAL

#: The eight training ("seen") object instances; every other PartNet-Mobility id is unseen.
SEEN_OBJECTS = frozenset({"45526", "35059", "46092", "45166", "44853", "44817", "45092", "45132"})


def fraction_of_goal(metrics):
    """Task progress in [0, 1] from a ``metrics.json`` dict written by ``lwm.planning.run_mpc``."""
    initial = metrics.get("initial_joint_angle", 0.0)
    final = metrics.get("final_joint_angle", initial)
    object_id = str(metrics.get("config", {}).get("object_id"))
    goal = OBJECT_JOINT_ANGLE_AT_GOAL.get(object_id, metrics.get("goal_joint_angle", 1.0))
    if goal <= initial:
        return 0.0
    return min(1.0, (final - initial) / (goal - initial))


def is_seen(object_id):
    return str(object_id) in SEEN_OBJECTS
