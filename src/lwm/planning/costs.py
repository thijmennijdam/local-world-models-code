"""Cost functions for MPC trajectory evaluation.

Two cost modes, selected via cost.cost_type:

  "dual" (default) — always-on EE→aff + aff→goal penalties.
  "progress"       — EE→aff penalty + proximity-gated progress bonus.
"""

import torch


def _clip_affordance_displacement(
    predicted_affordances: torch.Tensor,  # [N, H, 3]
    ee_positions: torch.Tensor,           # [N, H, 3]
) -> torch.Tensor:
    """Clip predicted affordance displacement to EE displacement per step.

    Physics prior: the handle can't move faster than the thing pushing it.
    """
    aff_deltas = predicted_affordances[:, 1:] - predicted_affordances[:, :-1]
    ee_deltas = ee_positions[:, 1:] - ee_positions[:, :-1]

    aff_delta_norms = torch.norm(aff_deltas, dim=-1, keepdim=True).clamp(min=1e-8)
    ee_delta_norms = torch.norm(ee_deltas, dim=-1, keepdim=True)

    scale = torch.clamp(ee_delta_norms / aff_delta_norms, max=1.0)
    clipped_deltas = aff_deltas * scale

    clipped = torch.zeros_like(predicted_affordances)
    clipped[:, 0] = predicted_affordances[:, 0]
    clipped[:, 1:] = clipped[:, 0:1] + torch.cumsum(clipped_deltas, dim=1)
    return clipped


# ── Original: always-on dual penalty ─────────────────────────────────────────

class DualTermCost:
    """Always-on dual cost: ee_aff * ||EE-aff|| + aff_goal * ||aff-goal||.

    Optional proximity-gated grip bonus: rewards closing the gripper (negative
    finger action) when EE is near the affordance.  The Gaussian gate ensures
    zero incentive when far away.
    """

    def __init__(
        self,
        ee_aff_weight: float = 1.0,
        aff_goal_weight: float = 1.0,
        terminal_weight: float = 5.0,
        clip_affordance: bool = True,
        grip_cost_weight: float = 0.0,
        proximity_scale: float = 0.04,
        **kwargs,
    ):
        self.ee_aff_weight = ee_aff_weight
        self.aff_goal_weight = aff_goal_weight
        self.terminal_weight = terminal_weight
        self.clip_affordance = clip_affordance
        self.grip_cost_weight = grip_cost_weight
        self.proximity_scale = proximity_scale

    def __call__(
        self,
        predicted_states: torch.Tensor,
        predicted_affordances: torch.Tensor,
        goal_affordance: torch.Tensor,
        action_sequences: torch.Tensor = None,
    ) -> torch.Tensor:
        ee_positions = predicted_states[:, :, :3]

        if self.clip_affordance:
            predicted_affordances = _clip_affordance_displacement(
                predicted_affordances, ee_positions
            )

        ee_to_aff = torch.norm(ee_positions - predicted_affordances, dim=-1)
        aff_to_goal = torch.norm(
            predicted_affordances - goal_affordance.unsqueeze(0).unsqueeze(0), dim=-1
        )

        step_costs = self.ee_aff_weight * ee_to_aff + self.aff_goal_weight * aff_to_goal
        total_costs = step_costs.sum(dim=1)

        if self.terminal_weight > 1.0:
            terminal = aff_to_goal[:, -1]
            total_costs = total_costs + (self.terminal_weight - 1.0) * self.aff_goal_weight * terminal

        # Proximity-gated grip bonus: responsive to distance AND rewards closing
        # gate = Gaussian(distance) → instant response to proximity changes
        # closing_fraction = finger delta normalized to [0,1] → requires actual closing
        grip_bonus_total = 0.0
        if self.grip_cost_weight > 0 and action_sequences is not None:
            gate = torch.exp(-ee_to_aff.pow(2) / (2.0 * self.proximity_scale ** 2))  # [N, H]
            finger_actions = action_sequences[:, :, -1]  # [N, H]
            GRIPPER_RANGE = 0.04
            closing_fraction = (-finger_actions / GRIPPER_RANGE).clamp(min=0, max=1)  # [N, H]
            grip_bonus = self.grip_cost_weight * gate * closing_fraction  # [N, H]
            total_costs = total_costs - grip_bonus.sum(dim=1)
            grip_bonus_total = float(grip_bonus.sum(dim=1).mean())

        # Diagnostics
        self.last_gate_mean = float(torch.exp(-ee_to_aff.pow(2) / (2.0 * self.proximity_scale ** 2)).mean()) if self.grip_cost_weight > 0 else 0.0
        self.last_progress_mean = 0.0
        self.last_bonus_mean = grip_bonus_total

        return total_costs


# ── New: proximity-gated progress bonus ──────────────────────────────────────

class ProximityProgressCost:
    """Hybrid cost: approach penalty + proximity-gated progress bonus.

    Cost = ee_aff_weight * ||EE - aff||                      (approach handle)
         + aff_goal_weight * ||aff - goal||                   (small tiebreaker)
         - gate * progress_weight * progress                  (bonus for moving handle)

    gate = exp(-||EE - aff||² / (2 * proximity_scale²))   — smooth Gaussian
    progress = clamp(||aff_goal[t=0]|| - ||aff_goal[t]||, min=0)

    No cost cliffs anywhere.
    """

    def __init__(
        self,
        ee_aff_weight: float = 5.0,
        aff_goal_weight: float = 0.5,
        progress_weight: float = 15.0,
        proximity_scale: float = 0.04,
        terminal_weight: float = 3.0,
        clip_affordance: bool = False,
        **kwargs,
    ):
        self.ee_aff_weight = ee_aff_weight
        self.aff_goal_weight = aff_goal_weight
        self.progress_weight = progress_weight
        self.proximity_scale = proximity_scale
        self.terminal_weight = terminal_weight
        self.clip_affordance = clip_affordance

    def __call__(
        self,
        predicted_states: torch.Tensor,
        predicted_affordances: torch.Tensor,
        goal_affordance: torch.Tensor,
        action_sequences: torch.Tensor = None,
    ) -> torch.Tensor:
        ee_positions = predicted_states[:, :, :3]

        if self.clip_affordance:
            predicted_affordances = _clip_affordance_displacement(
                predicted_affordances, ee_positions
            )

        ee_to_aff = torch.norm(ee_positions - predicted_affordances, dim=-1)  # [N, H]
        aff_to_goal = torch.norm(
            predicted_affordances - goal_affordance.unsqueeze(0).unsqueeze(0), dim=-1
        )  # [N, H]

        # Proximity gate: smooth Gaussian, ~1 when touching, ~0 when far
        gate = torch.exp(-ee_to_aff.pow(2) / (2.0 * self.proximity_scale ** 2))  # [N, H]

        # Progress: how much closer affordance got to goal vs start of horizon
        progress = (aff_to_goal[:, 0:1] - aff_to_goal).clamp(min=0)  # [N, H]

        # Penalties
        step_costs = self.ee_aff_weight * ee_to_aff + self.aff_goal_weight * aff_to_goal

        # Bonus
        step_bonus = gate * self.progress_weight * progress

        total_costs = step_costs.sum(dim=1) - step_bonus.sum(dim=1)

        # Terminal: extra gated progress reward at final step
        if self.terminal_weight > 0:
            terminal_progress = (aff_to_goal[:, 0] - aff_to_goal[:, -1]).clamp(min=0)
            terminal_gate = torch.exp(
                -ee_to_aff[:, -1].pow(2) / (2.0 * self.proximity_scale ** 2)
            )
            total_costs = total_costs - self.terminal_weight * terminal_gate * terminal_progress

        # Diagnostics
        self.last_gate_mean = float(gate.mean())
        self.last_progress_mean = float(progress.mean())
        self.last_bonus_mean = float(step_bonus.sum(dim=1).mean())

        return total_costs


# ── Factory ──────────────────────────────────────────────────────────────────

COST_TYPES = {
    "dual": DualTermCost,
    "progress": ProximityProgressCost,
}


def create_cost_function(cost_type: str = "dual", **kwargs):
    """Create cost function by name.

    Args:
        cost_type: "dual" (original) or "progress" (proximity-gated bonus)
        **kwargs: passed to the cost class constructor
    """
    cls = COST_TYPES.get(cost_type)
    if cls is None:
        raise ValueError(f"Unknown cost_type={cost_type!r}, choose from {list(COST_TYPES)}")
    return cls(**kwargs)
