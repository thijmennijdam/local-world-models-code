"""Sampling-based MPC planners: MPPI, CEM, and iCEM.

Provides a base class for sampling-based trajectory optimization and three
concrete implementations:
- MPPIPlanner: Model Predictive Path Integral (soft exponential weighting)
- CEMPlanner: Cross-Entropy Method (hard elite selection + Gaussian refit)
- iCEMPlanner: Improved CEM (elite history, kept elites, colored noise)

All share the same optimize() interface and warm-starting logic.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from termcolor import cprint


class SamplingPlanner(ABC):
    """Base class for sampling-based MPC planners.

    Shared functionality:
    - Action sampling with Gaussian noise
    - Batched trajectory rollout via world model
    - Cost evaluation
    - Action bounds enforcement
    - Warm-starting (shift + init last action)

    Subclasses implement _update_distribution() to define how the action
    distribution is updated given sample costs.
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        action_dim: int,
        noise_sigma: np.ndarray,
        action_bounds: Tuple[np.ndarray, np.ndarray],
        neutral_action: Optional[np.ndarray] = None,
        wm_batch_size: Optional[int] = None,
        device: str = "cuda",
    ):
        self.horizon = horizon
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.wm_batch_size = wm_batch_size
        self.device = device

        self.noise_sigma = torch.from_numpy(noise_sigma).float().to(device)
        self.action_bounds_low = torch.from_numpy(action_bounds[0]).float().to(device)
        self.action_bounds_high = torch.from_numpy(action_bounds[1]).float().to(device)

        # Neutral action: the "do nothing" action (e.g. zero pos/finger deltas
        # but identity rotation in 6D).  Falls back to all-zeros when not given.
        if neutral_action is not None:
            self.neutral_action = torch.from_numpy(neutral_action).float().to(device)
        else:
            self.neutral_action = torch.zeros(action_dim, device=device)

        # Current action distribution: mean and std
        self.mean_actions = self.neutral_action.unsqueeze(0).expand(horizon, -1).clone()
        self.std_actions = self.noise_sigma.unsqueeze(0).expand(horizon, -1).clone()

    def optimize(
        self,
        current_obs: Dict[str, torch.Tensor],
        goal_affordance: np.ndarray,
        cost_fn: Callable,
        world_model: "WorldModelPredictor",
        num_iterations: int = 3,
        initial_actions: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """Run optimization to find optimal action sequence.

        Args:
            current_obs: Dict with keys agent_state [1, D], affordance [1, 3],
                         point_cloud [1, N, 3]
            goal_affordance: Goal affordance position [3]
            cost_fn: Callable(predicted_states, predicted_affordances, goal) -> costs
            world_model: WorldModelPredictor for trajectory rollout
            num_iterations: Number of optimization iterations
            initial_actions: Optional initial action sequence [horizon, action_dim]
            verbose: Print iteration stats

        Returns:
            optimal_actions: [horizon, action_dim] numpy array
            info: Dict with costs, iteration stats
        """
        if initial_actions is not None:
            self.mean_actions = torch.from_numpy(initial_actions).float().to(self.device)

        goal_tensor = torch.from_numpy(goal_affordance).float().to(self.device)

        agent_state = current_obs["agent_state"]
        affordance = current_obs["affordance"]
        point_cloud = current_obs["point_cloud"]

        batch_agent_state = agent_state.expand(self.num_samples, -1)
        batch_affordance = affordance.expand(self.num_samples, -1)
        batch_point_cloud = point_cloud.expand(self.num_samples, -1, -1)

        iteration_costs = []
        iteration_best_costs = []

        for iteration in range(num_iterations):
            # 1. Sample action sequences
            action_samples = self._sample_actions()

            # 2. Predict trajectories (chunked if wm_batch_size set)
            predictions = world_model.predict_trajectory(
                batch_agent_state,
                batch_affordance,
                batch_point_cloud,
                action_samples,
                wm_batch_size=self.wm_batch_size,
            )
            predicted_states = predictions["predicted_states"]
            predicted_affordances = predictions["predicted_affordances"]

            # 3. Evaluate costs
            costs = cost_fn(
                predicted_states=predicted_states,
                predicted_affordances=predicted_affordances,
                goal_affordance=goal_tensor,
                action_sequences=action_samples,
            )

            # 4. Update distribution (planner-specific)
            self._update_distribution(action_samples, costs)

            # Clip mean to bounds
            self.mean_actions = torch.clamp(
                self.mean_actions,
                self.action_bounds_low,
                self.action_bounds_high,
            )

            mean_cost = costs.mean().item()
            min_cost = costs.min().item()
            iteration_costs.append(mean_cost)
            iteration_best_costs.append(min_cost)

            if verbose:
                cprint(
                    f"  Iter {iteration+1}/{num_iterations}: "
                    f"mean_cost={mean_cost:.4f}, min_cost={min_cost:.4f}",
                    "yellow",
                )

        optimal_actions = self.mean_actions.cpu().numpy()

        info = {
            "costs": costs.cpu().numpy(),
            "iteration_mean_costs": iteration_costs,
            "iteration_best_costs": iteration_best_costs,
            "final_mean_cost": iteration_costs[-1],
            "final_best_cost": iteration_best_costs[-1],
        }

        return optimal_actions, info

    def _sample_actions(self) -> torch.Tensor:
        """Sample action sequences from current distribution.

        Returns:
            [num_samples, horizon, action_dim] action samples
        """
        noise = torch.randn(
            self.num_samples, self.horizon, self.action_dim, device=self.device
        )
        samples = self.mean_actions.unsqueeze(0) + noise * self.std_actions.unsqueeze(0)
        samples = torch.clamp(
            samples,
            self.action_bounds_low.unsqueeze(0).unsqueeze(0),
            self.action_bounds_high.unsqueeze(0).unsqueeze(0),
        )
        return samples

    @abstractmethod
    def _update_distribution(
        self, action_samples: torch.Tensor, costs: torch.Tensor
    ):
        """Update action distribution given samples and their costs.

        Must update self.mean_actions (and optionally self.std_actions).

        Args:
            action_samples: [num_samples, horizon, action_dim]
            costs: [num_samples]
        """
        ...

    def shift_and_init_last_action(self, last_action: Optional[np.ndarray] = None):
        """Shift mean action sequence by one step for warm-starting.

        Args:
            last_action: Optional action for last timestep [action_dim].
                         If None, repeats the second-to-last action.
        """
        self.mean_actions[:-1] = self.mean_actions[1:].clone()
        if last_action is not None:
            self.mean_actions[-1] = torch.from_numpy(last_action).float().to(self.device)
        # Reset std to initial noise level
        self.std_actions = self.noise_sigma.unsqueeze(0).expand(self.horizon, -1).clone()

    def clear_history(self):
        """Clear any cross-call memory (no-op for base class)."""
        pass

    def reset(self):
        """Reset distribution to initial (neutral mean, default noise)."""
        self.mean_actions = self.neutral_action.unsqueeze(0).expand(
            self.horizon, -1
        ).clone()
        self.std_actions = self.noise_sigma.unsqueeze(0).expand(self.horizon, -1).clone()


class MPPIPlanner(SamplingPlanner):
    """Model Predictive Path Integral (MPPI) planner.

    Updates distribution via temperature-weighted averaging of samples:
        w_i = exp(-cost_i / temperature) / Z
        mean = sum(w_i * action_i)

    Lower temperature = greedier (exploits low-cost samples more).
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        action_dim: int,
        noise_sigma: np.ndarray,
        action_bounds: Tuple[np.ndarray, np.ndarray],
        temperature: float = 0.1,
        neutral_action: Optional[np.ndarray] = None,
        wm_batch_size: Optional[int] = None,
        device: str = "cuda",
    ):
        super().__init__(
            horizon=horizon,
            num_samples=num_samples,
            action_dim=action_dim,
            noise_sigma=noise_sigma,
            action_bounds=action_bounds,
            neutral_action=neutral_action,
            wm_batch_size=wm_batch_size,
            device=device,
        )
        self.temperature = temperature

        cprint(
            f"MPPI initialized: horizon={horizon}, samples={num_samples}, "
            f"temperature={temperature}",
            "cyan",
        )

    def _update_distribution(self, action_samples, costs):
        # Exponential weighting with numerical stability
        costs_normalized = costs - costs.min()
        weights = torch.exp(-costs_normalized / self.temperature)
        weights = weights / weights.sum()

        # Weighted mean
        self.mean_actions = (
            weights.unsqueeze(-1).unsqueeze(-1) * action_samples
        ).sum(dim=0)


class CEMPlanner(SamplingPlanner):
    """Cross-Entropy Method (CEM) planner.

    Updates distribution by refitting a Gaussian to the top-K elite samples:
        elites = top elite_fraction samples by cost
        mean = mean(elites)
        std  = std(elites)

    Alpha parameter controls smoothing between old and new distribution:
        mean = alpha * new_mean + (1 - alpha) * old_mean
        std  = alpha * new_std  + (1 - alpha) * old_std
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        action_dim: int,
        noise_sigma: np.ndarray,
        action_bounds: Tuple[np.ndarray, np.ndarray],
        elite_fraction: float = 0.1,
        alpha: float = 0.25,
        neutral_action: Optional[np.ndarray] = None,
        wm_batch_size: Optional[int] = None,
        device: str = "cuda",
    ):
        super().__init__(
            horizon=horizon,
            num_samples=num_samples,
            action_dim=action_dim,
            noise_sigma=noise_sigma,
            action_bounds=action_bounds,
            neutral_action=neutral_action,
            wm_batch_size=wm_batch_size,
            device=device,
        )
        self.elite_fraction = elite_fraction
        self.alpha = alpha
        self.num_elites = max(1, int(num_samples * elite_fraction))

        cprint(
            f"CEM initialized: horizon={horizon}, samples={num_samples}, "
            f"elites={self.num_elites}, alpha={alpha}",
            "cyan",
        )

    def _update_distribution(self, action_samples, costs):
        # Select elite samples (lowest cost)
        _, elite_indices = torch.topk(costs, self.num_elites, largest=False)
        elites = action_samples[elite_indices]  # [num_elites, horizon, action_dim]

        # Refit Gaussian to elites
        new_mean = elites.mean(dim=0)  # [horizon, action_dim]
        new_std = elites.std(dim=0).clamp(min=1e-6)  # [horizon, action_dim]

        # Smooth update
        self.mean_actions = self.alpha * new_mean + (1 - self.alpha) * self.mean_actions
        self.std_actions = self.alpha * new_std + (1 - self.alpha) * self.std_actions


class iCEMPlanner(SamplingPlanner):
    """Improved Cross-Entropy Method (iCEM) planner.

    Extends CEM with three key improvements:
    1. Elite history: keeps best solutions across plan calls (cross-call memory)
    2. Kept elites: reuses fraction of elite solutions between CEM iterations
    3. Colored noise: temporally correlated noise for smoother action sequences

    Also supports separate warmup (first call, more iterations) vs online
    (subsequent calls, fewer iterations) modes.

    Reference: Pinneri et al., "Sample-efficient Cross-Entropy Method for
    Real-time Planning" (2020).
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        action_dim: int,
        noise_sigma: np.ndarray,
        action_bounds: Tuple[np.ndarray, np.ndarray],
        elite_fraction: float = 0.1,
        alpha: float = 0.1,
        elites_keep_fraction: float = 0.5,
        noise_beta: float = 3.0,
        history_length: int = 3,
        history_elite_frac: float = 0.3,
        warmup_iters: int = 5,
        online_iters: int = 3,
        execute_best: bool = True,
        noise_correlate_time: bool = True,
        neutral_action: Optional[np.ndarray] = None,
        wm_batch_size: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        Args:
            elite_fraction: Fraction of samples selected as elites
            alpha: Distribution smoothing (0=keep old, 1=fully new)
            elites_keep_fraction: Fraction of elites to carry between iterations
            noise_beta: Colored noise exponent (0=white, 3=pink). Higher = smoother.
            history_length: Number of best solutions to keep across plan calls
            history_elite_frac: Fraction of population to fill with historical elites
            warmup_iters: Optimization iterations on first plan call
            online_iters: Optimization iterations on subsequent calls
            execute_best: Return the lowest-cost evaluated sequence of this call
                (iCEM); False returns the smoothed sampling mean, which is never
                evaluated by the world model.
            noise_correlate_time: Colored noise correlated over the horizon (True)
                or across action dims (False).
        """
        super().__init__(
            horizon=horizon,
            num_samples=num_samples,
            action_dim=action_dim,
            noise_sigma=noise_sigma,
            action_bounds=action_bounds,
            neutral_action=neutral_action,
            wm_batch_size=wm_batch_size,
            device=device,
        )
        self.elite_fraction = elite_fraction
        self.alpha = alpha
        self.num_elites = max(1, int(num_samples * elite_fraction))
        self.elites_keep_fraction = elites_keep_fraction
        self.num_kept = max(0, int(self.num_elites * elites_keep_fraction))
        self.noise_beta = noise_beta
        self.history_length = history_length
        self.history_elite_frac = history_elite_frac
        self.warmup_iters = warmup_iters
        self.online_iters = online_iters
        self.execute_best = execute_best
        self.noise_correlate_time = noise_correlate_time

        # Cross-call elite history: list of (actions [H, D], cost)
        self.elite_history: List[Tuple[torch.Tensor, float]] = []
        # Within-call kept elites from previous iteration
        self._kept_elites: Optional[torch.Tensor] = None
        # Track whether this is the first plan call
        self._is_first_call = True

        cprint(
            f"iCEM initialized: horizon={horizon}, samples={num_samples}, "
            f"elites={self.num_elites}, keep={self.num_kept}, "
            f"noise_beta={noise_beta}, history={history_length}, "
            f"warmup={warmup_iters}, online={online_iters}",
            "cyan",
        )

    def optimize(
        self,
        current_obs: Dict[str, torch.Tensor],
        goal_affordance: np.ndarray,
        cost_fn: Callable,
        world_model: "WorldModelPredictor",
        num_iterations: int = 3,
        initial_actions: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """Run iCEM optimization. Uses warmup_iters on first call, online_iters after."""
        # Override num_iterations based on warmup/online
        if self._is_first_call:
            num_iterations = self.warmup_iters
            self._is_first_call = False
        else:
            num_iterations = self.online_iters

        if initial_actions is not None:
            self.mean_actions = torch.from_numpy(initial_actions).float().to(self.device)

        goal_tensor = torch.from_numpy(goal_affordance).float().to(self.device)

        agent_state = current_obs["agent_state"]
        affordance = current_obs["affordance"]
        point_cloud = current_obs["point_cloud"]

        iteration_costs = []
        iteration_best_costs = []
        self._kept_elites = None
        call_best_actions, call_best_cost = None, float("inf")

        for iteration in range(num_iterations):
            # 1. Compose samples: new + kept elites + history elites
            action_samples = self._sample_with_elites_and_history()
            total_samples = action_samples.shape[0]

            # Expand obs to match total sample count
            batch_agent_state = agent_state.expand(total_samples, -1)
            batch_affordance = affordance.expand(total_samples, -1)
            batch_point_cloud = point_cloud.expand(total_samples, -1, -1)

            # 2. Predict trajectories (chunked if wm_batch_size set)
            predictions = world_model.predict_trajectory(
                batch_agent_state,
                batch_affordance,
                batch_point_cloud,
                action_samples,
                wm_batch_size=self.wm_batch_size,
            )

            # 3. Evaluate costs
            costs = cost_fn(
                predicted_states=predictions["predicted_states"],
                predicted_affordances=predictions["predicted_affordances"],
                goal_affordance=goal_tensor,
                action_sequences=action_samples,
            )

            # 4. Select elites
            _, elite_indices = torch.topk(costs, self.num_elites, largest=False)
            elites = action_samples[elite_indices]
            elite_costs = costs[elite_indices]

            # 5. Update distribution from elites
            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0).clamp(min=1e-6)
            self.mean_actions = self.alpha * new_mean + (1 - self.alpha) * self.mean_actions
            self.std_actions = self.alpha * new_std + (1 - self.alpha) * self.std_actions

            # Clip mean to bounds
            self.mean_actions = torch.clamp(
                self.mean_actions,
                self.action_bounds_low,
                self.action_bounds_high,
            )

            # 6. Keep fraction of elites for next iteration
            if self.num_kept > 0:
                self._kept_elites = elites[:self.num_kept].clone()

            # 7. Update cross-call history with best elite
            best_idx = elite_costs.argmin()
            best_actions = elites[best_idx].clone()
            best_cost = elite_costs[best_idx].item()
            self._update_elite_history(best_actions, best_cost)
            if best_cost < call_best_cost:
                call_best_actions, call_best_cost = best_actions, best_cost

            mean_cost = costs.mean().item()
            min_cost = costs.min().item()
            iteration_costs.append(mean_cost)
            iteration_best_costs.append(min_cost)

            if verbose:
                cprint(
                    f"  Iter {iteration+1}/{num_iterations}: "
                    f"mean_cost={mean_cost:.4f}, min_cost={min_cost:.4f}, "
                    f"samples={total_samples}",
                    "yellow",
                )

        if self.execute_best:
            optimal_actions = call_best_actions.cpu().numpy()
        else:
            optimal_actions = self.mean_actions.cpu().numpy()

        info = {
            "costs": costs.cpu().numpy(),
            "iteration_mean_costs": iteration_costs,
            "iteration_best_costs": iteration_best_costs,
            "final_mean_cost": iteration_costs[-1],
            "final_best_cost": iteration_best_costs[-1],
        }

        return optimal_actions, info

    def _sample_with_elites_and_history(self) -> torch.Tensor:
        """Compose action samples from new noise, kept elites, and history.

        Returns:
            [total_samples, horizon, action_dim] action samples
        """
        # How many slots for history elites
        num_history = 0
        history_samples = None
        if self.elite_history:
            num_history = min(
                int(self.num_samples * self.history_elite_frac),
                len(self.elite_history),
            )
            if num_history > 0:
                history_samples = torch.stack(
                    [h[0] for h in self.elite_history[:num_history]]
                ).to(self.device)

        # How many slots for kept elites
        num_kept = 0
        if self._kept_elites is not None:
            num_kept = self._kept_elites.shape[0]

        # Remaining slots for new samples
        num_new = self.num_samples - num_history - num_kept
        num_new = max(1, num_new)

        # Sample new action sequences (with colored noise)
        new_samples = self._sample_colored(num_new)

        # Concatenate
        parts = [new_samples]
        if self._kept_elites is not None and num_kept > 0:
            parts.append(self._kept_elites)
        if history_samples is not None and num_history > 0:
            parts.append(history_samples)

        return torch.cat(parts, dim=0)

    def _sample_colored(self, num_samples: int) -> torch.Tensor:
        """Sample action sequences with temporally correlated (colored) noise.

        Uses power-law spectral density: S(f) ~ 1/f^beta
        beta=0: white noise, beta=2: brown, beta=3: pink (smooth)
        """
        if self.noise_beta <= 0:
            return self._sample_actions_n(num_samples)

        import colorednoise as cn
        # colorednoise draws from a fresh OS-seeded generator unless given one, so seed a
        # planner-local generator once from the (seeded) global numpy RNG.
        if getattr(self, "_noise_rng", None) is None:
            self._noise_rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        if self.noise_correlate_time:
            # powerlaw_psd_gaussian correlates along the last axis, so generate
            # [num_samples, action_dim, horizon] and move time to dim 1.
            noise = cn.powerlaw_psd_gaussian(
                self.noise_beta, (num_samples, self.action_dim, self.horizon),
                random_state=self._noise_rng,
            ).transpose(0, 2, 1)
        else:
            # Correlated across action dims, white in time.
            noise = cn.powerlaw_psd_gaussian(
                self.noise_beta, (num_samples, self.horizon, self.action_dim),
                random_state=self._noise_rng,
            )
        noise = torch.from_numpy(np.ascontiguousarray(noise)).float().to(self.device)

        samples = self.mean_actions.unsqueeze(0) + noise * self.std_actions.unsqueeze(0)
        samples = torch.clamp(
            samples,
            self.action_bounds_low.unsqueeze(0).unsqueeze(0),
            self.action_bounds_high.unsqueeze(0).unsqueeze(0),
        )
        return samples

    def _sample_actions_n(self, num_samples: int) -> torch.Tensor:
        """Sample N action sequences with white noise (standard Gaussian)."""
        noise = torch.randn(
            num_samples, self.horizon, self.action_dim, device=self.device
        )
        samples = self.mean_actions.unsqueeze(0) + noise * self.std_actions.unsqueeze(0)
        samples = torch.clamp(
            samples,
            self.action_bounds_low.unsqueeze(0).unsqueeze(0),
            self.action_bounds_high.unsqueeze(0).unsqueeze(0),
        )
        return samples

    def _update_distribution(self, action_samples, costs):
        # Not used — iCEM overrides optimize() directly
        pass

    def _update_elite_history(self, actions: torch.Tensor, cost: float):
        """Add a solution to elite history, keeping sorted by cost."""
        self.elite_history.append((actions.cpu(), cost))
        self.elite_history.sort(key=lambda x: x[1])
        self.elite_history = self.elite_history[:self.history_length]

    def shift_and_init_last_action(self, last_action: Optional[np.ndarray] = None):
        """Shift distribution and history forward by one step."""
        # Shift mean
        self.mean_actions[:-1] = self.mean_actions[1:].clone()
        if last_action is not None:
            self.mean_actions[-1] = torch.from_numpy(last_action).float().to(self.device)

        # Reset std
        self.std_actions = self.noise_sigma.unsqueeze(0).expand(self.horizon, -1).clone()

        # Shift elite history sequences forward
        shifted_history = []
        for actions, cost in self.elite_history:
            shifted = torch.zeros_like(actions)
            shifted[:-1] = actions[1:]
            shifted[-1] = actions[-1]  # repeat last
            shifted_history.append((shifted, cost))
        self.elite_history = shifted_history

        # Clear kept elites (stale after shift)
        self._kept_elites = None

    def clear_history(self):
        """Clear elite history and trigger warmup on next call."""
        self.elite_history = []
        self._kept_elites = None
        self._is_first_call = True

    def reset(self):
        """Reset distribution, history, and state."""
        super().reset()
        self.clear_history()


def create_planner(planner_type: str, **kwargs) -> SamplingPlanner:
    """Factory function to create a planner by name.

    Args:
        planner_type: "mppi", "cem", or "icem"
        **kwargs: Planner-specific arguments

    Returns:
        SamplingPlanner instance
    """
    planners = {
        "mppi": MPPIPlanner,
        "cem": CEMPlanner,
        "icem": iCEMPlanner,
    }
    if planner_type not in planners:
        raise ValueError(f"Unknown planner type '{planner_type}'. Choose from: {list(planners.keys())}")
    return planners[planner_type](**kwargs)
