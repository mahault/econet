"""Phantom thermostat for sophisticated inference (Pitliya et al., 2025).

The phantom is a lightweight pymdp Agent mirroring the thermostat's
generative model.  The battery runs a T-step rollout through the phantom
to predict P(HVAC action) at each future planning step.  This prediction
is then used to build step-dependent B_effective matrices for the battery's
energy factor, enabling full sophisticated inference.

Key asymmetry: the thermostat's HVAC decisions are independent of battery
actions (thermostat doesn't observe battery state), so the phantom rollout
is pre-computed ONCE per simulation step and reused for all battery policies.
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from pymdp.agent import Agent

from .generative_model import (
    build_thermostat_model,
    build_battery_model,
    THERMO_NUM_ACTIONS,
    BATTERY_NUM_ACTIONS,
)
from .environment import (
    TARGET_TEMP_OCCUPIED, TARGET_TEMP_UNOCCUPIED,
    discretize_temp, discretize_soc, discretize_cost, discretize_ghg,
    discretize_energy,
    BATTERY_CAPACITY_KWH, BATTERY_STEP_FRAC,
    SOC_MIN_DISCHARGE, SOC_MAX_CHARGE,
    HVAC_KWH_PER_STEP,
)


# Thermostat obs keys in the order expected by the pymdp Agent
_THERMO_OBS_KEYS = ["room_temp", "outdoor_temp", "occupancy", "tou_high"]


def _marginalize_first_action(q_pi, policies):
    """Marginalise policy distribution to P(first action).

    Parameters
    ----------
    q_pi : jnp.ndarray, shape (batch=1, n_policies)
        Policy posterior.
    policies : jnp.ndarray, shape (n_policies, T, n_factors)
        Policy array from the Agent.

    Returns
    -------
    p_action : np.ndarray, shape (n_actions,)
        Marginal probability of each first action (cool/heat/off).
    """
    q_pi_np = np.asarray(q_pi).flatten()
    policies_np = np.asarray(policies)
    # First action of each policy for controlled factor 0
    first_actions = policies_np[:, 0, 0].astype(int)
    p_action = np.zeros(THERMO_NUM_ACTIONS)
    for pi_idx, a in enumerate(first_actions):
        p_action[a] += q_pi_np[pi_idx]
    # Normalise (should already sum to 1, but be safe)
    total = p_action.sum()
    if total > 0:
        p_action /= total
    return p_action


def estimate_thermo_obs(env_data: dict, step_idx: int) -> dict:
    """Estimate thermostat observations at a future step.

    The battery doesn't know the exact room temperature the thermostat
    will see, so it uses the comfort-target temperature as a best estimate
    (since an aligned thermostat maintains near-target).  Outdoor temp,
    occupancy, and TOU are known schedules.

    Parameters
    ----------
    env_data : dict
        Full simulation environment data.
    step_idx : int
        The absolute step index to estimate observations for.

    Returns
    -------
    obs_dict : dict
        Discretised observations: room_temp, outdoor_temp, occupancy, tou_high.
    """
    max_idx = len(env_data["outdoor_temp"]) - 1
    idx = min(step_idx, max_idx)

    occ = int(env_data["occupancy"][idx])
    target = TARGET_TEMP_OCCUPIED if occ else TARGET_TEMP_UNOCCUPIED

    return {
        "room_temp": discretize_temp(target),
        "outdoor_temp": discretize_temp(env_data["outdoor_temp"][idx]),
        "occupancy": occ,
        "tou_high": int(env_data["tou_high"][idx]),
    }


class PhantomThermostat:
    """Lightweight pymdp Agent mirroring the thermostat's generative model.

    Runs T-step rollout to predict HVAC action sequence P(a^o_tau) for
    tau = 0 .. T-1.  Used by the battery's sophisticated EFE computation.

    The phantom uses:
    - Same A, B, C, D as the real thermostat (no auditory, no learning)
    - Dynamic C alignment (occupancy + TOU-dependent comfort preference)
    - Deterministic action selection (matching aligned thermostat)
    """

    def __init__(self, env_data: dict, policy_len: int = 4,
                 gamma: float = 16.0):
        model = build_thermostat_model(env_data, policy_len=policy_len)

        self.agent = Agent(
            A=model["A"],
            B=model["B"],
            C=model["C"],
            D=model["D"],
            A_dependencies=model["A_dependencies"],
            B_dependencies=model["B_dependencies"],
            control_fac_idx=[0],
            policy_len=policy_len,
            gamma=gamma,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=False,
            action_selection="deterministic",
            inference_algo="fpi",
            num_iter=16,
            batch_size=1,
            learn_A=False,
            learn_B=False,
        )
        self._qs_prev = None
        self._action_prev = None
        self._empirical_prior = self.agent.D
        self._env_data = env_data
        self._rng_key = jr.PRNGKey(99)

    def _update_dynamic_C(self, obs_dict: dict):
        """Mirror the aligned thermostat's dynamic C vector."""
        import equinox as eqx
        from .environment import TEMP_LEVELS

        occupancy = obs_dict.get("occupancy", 1)
        tou_high = obs_dict.get("tou_high", 0)
        target = TARGET_TEMP_OCCUPIED if occupancy else TARGET_TEMP_UNOCCUPIED
        target_idx = discretize_temp(target)
        amplitude = -1.5 if tou_high else -4.0

        c_room = np.zeros(TEMP_LEVELS)
        for i in range(TEMP_LEVELS):
            dist = abs(i - target_idx)
            c_room[i] = amplitude * dist ** 2 / 4.0
        c_room -= c_room.max()

        new_C = list(self.agent.C)
        new_C[0] = jnp.array(c_room)[None, :]
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)

    def predict_action(self, obs_dict: dict) -> np.ndarray:
        """Single-step: infer states -> infer policies -> P(action).

        Updates internal beliefs for the next call (stateful).

        Parameters
        ----------
        obs_dict : dict
            Thermostat observations (room_temp, outdoor_temp, occupancy, tou_high).

        Returns
        -------
        p_action : np.ndarray, shape (3,)
            [P(cool), P(heat), P(off)]
        """
        # Dynamic C (mirror aligned thermostat)
        self._update_dynamic_C(obs_dict)

        obs = [jnp.array([[obs_dict[k]]]) for k in _THERMO_OBS_KEYS]

        qs = self.agent.infer_states(obs, self._empirical_prior)
        q_pi, _ = self.agent.infer_policies(qs)

        # Marginalise to P(first action)
        p_action = _marginalize_first_action(q_pi, self.agent.policies)

        # Select action for empirical prior update
        self._rng_key, subkey = jr.split(self._rng_key)
        action = self.agent.sample_action(q_pi)

        # Update empirical prior for next step
        pred = self.agent.update_empirical_prior(action, qs)
        self._empirical_prior = pred
        self._qs_prev = qs
        self._action_prev = action

        return p_action

    def rollout(self, start_step: int, horizon: int) -> list:
        """T-step lookahead: predict P(action) for tau = 0 .. horizon-1.

        Parameters
        ----------
        start_step : int
            Current simulation step (absolute index).
        horizon : int
            Number of steps to predict ahead.

        Returns
        -------
        sequence : list of np.ndarray
            [P(hvac)_0, ..., P(hvac)_{T-1}], each shape (3,).
        """
        sequence = []
        for tau in range(horizon):
            step = start_step + tau
            obs = estimate_thermo_obs(self._env_data, step)
            p_action = self.predict_action(obs)
            sequence.append(np.array(p_action))
        return sequence

    def reset(self):
        """Reset phantom to initial beliefs (for new simulation step).

        Called before each rollout so the phantom doesn't accumulate
        beliefs across simulation steps.
        """
        self._empirical_prior = self.agent.D
        self._qs_prev = None
        self._action_prev = None
        self._rng_key = jr.PRNGKey(99)


# Battery obs keys in the order expected by the pymdp Agent
_BATTERY_OBS_KEYS = ["soc", "cost", "ghg"]


def estimate_battery_obs(env_data: dict, step_idx: int, soc: float,
                         hvac_energy: float) -> dict:
    """Estimate battery observations at a future step.

    Parameters
    ----------
    env_data : dict
        Full simulation environment data.
    step_idx : int
        The absolute step index.
    soc : float
        Estimated SoC at this step.
    hvac_energy : float
        Estimated HVAC energy (kWh) at this step.

    Returns
    -------
    obs_dict : dict
        Discretised observations: soc, cost, ghg, tou_high.
    """
    max_idx = len(env_data["outdoor_temp"]) - 1
    idx = min(step_idx, max_idx)

    baseline = env_data["baseline_load"][idx]
    solar = env_data["solar_gen"][idx]
    total = baseline + hvac_energy - solar
    tou_val = env_data["tou_value"][idx]
    cost = max(0, total) * tou_val
    ghg = max(0, total) * env_data["ghg_rate"][idx]

    return {
        "soc": discretize_soc(soc),
        "cost": discretize_cost(cost),
        "ghg": discretize_ghg(ghg),
        "tou_high": int(env_data["tou_high"][idx]),
    }


class PhantomBattery:
    """Lightweight pymdp Agent mirroring the battery's generative model.

    Runs T-step rollout to predict P(battery action) for each planning
    step.  Used by the thermostat's sophisticated inference to estimate
    the cost impact of HVAC decisions.

    The phantom mirrors the aligned battery's dynamic C (TOU-flip).
    """

    C_SOC_PEAK = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    C_SOC_OFFPEAK = np.array([-1.5, -0.5, 0.0, 0.5, 1.0])

    def __init__(self, env_data: dict, policy_len: int = 4,
                 gamma: float = 16.0, initial_soc: float = 0.5):
        model = build_battery_model(env_data, initial_soc=initial_soc,
                                    policy_len=policy_len)

        self.agent = Agent(
            A=model["A"],
            B=model["B"],
            C=model["C"],
            D=model["D"],
            A_dependencies=model["A_dependencies"],
            B_dependencies=model["B_dependencies"],
            control_fac_idx=[0],
            policy_len=policy_len,
            gamma=gamma,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=False,
            action_selection="deterministic",
            inference_algo="fpi",
            num_iter=16,
            batch_size=1,
            learn_A=False,
            learn_B=False,
        )
        self._empirical_prior = self.agent.D
        self._rng_key = jr.PRNGKey(98)
        self._env_data = env_data
        self._soc = initial_soc  # track estimated SoC for rollout

    def _update_dynamic_C(self, tou_high: int):
        """Mirror the aligned battery's TOU-flip C vector."""
        import equinox as eqx

        if tou_high:
            new_c0 = jnp.array(self.C_SOC_PEAK)
        else:
            new_c0 = jnp.array(self.C_SOC_OFFPEAK)
        new_C = list(self.agent.C)
        new_C[0] = new_c0[None, :]
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)

    def predict_action(self, obs_dict: dict) -> np.ndarray:
        """Single-step: infer states -> policies -> P(action).

        Returns
        -------
        p_action : np.ndarray, shape (3,)
            [P(charge), P(discharge), P(off)]
        """
        self._update_dynamic_C(obs_dict.get("tou_high", 0))

        obs = [jnp.array([[obs_dict[k]]]) for k in _BATTERY_OBS_KEYS]

        qs = self.agent.infer_states(obs, self._empirical_prior)
        q_pi, _ = self.agent.infer_policies(qs)

        # Marginalise to P(first action)
        q_pi_np = np.asarray(q_pi).flatten()
        policies_np = np.asarray(self.agent.policies)
        first_actions = policies_np[:, 0, 0].astype(int)
        p_action = np.zeros(BATTERY_NUM_ACTIONS)
        for pi_idx, a in enumerate(first_actions):
            p_action[a] += q_pi_np[pi_idx]
        total = p_action.sum()
        if total > 0:
            p_action /= total

        # Select action for empirical prior update
        self._rng_key, _ = jr.split(self._rng_key)
        action = self.agent.sample_action(q_pi)
        pred = self.agent.update_empirical_prior(action, qs)
        self._empirical_prior = pred

        # Track estimated SoC for next step
        action_int = int(action[0, 0])
        if action_int == 0 and self._soc < SOC_MAX_CHARGE:
            self._soc += BATTERY_STEP_FRAC
        elif action_int == 1 and self._soc > SOC_MIN_DISCHARGE:
            self._soc -= BATTERY_STEP_FRAC
        self._soc = np.clip(self._soc, 0.0, 1.0)

        return p_action

    def rollout(self, start_step: int, horizon: int,
                hvac_energies: list = None) -> list:
        """T-step lookahead: predict P(battery action) for tau = 0..T-1.

        Parameters
        ----------
        start_step : int
            Current simulation step.
        horizon : int
            Number of steps to predict.
        hvac_energies : list of float, optional
            Estimated HVAC energy per step (from thermostat's policy).
            If None, assumes HVAC_KWH_PER_STEP for all steps.

        Returns
        -------
        sequence : list of np.ndarray
            [P(batt)_0, ..., P(batt)_{T-1}], each shape (3,).
        """
        sequence = []
        for tau in range(horizon):
            step = start_step + tau
            hvac_e = (hvac_energies[tau] if hvac_energies is not None
                      else HVAC_KWH_PER_STEP)
            obs = estimate_battery_obs(self._env_data, step,
                                       self._soc, hvac_e)
            p_action = self.predict_action(obs)
            sequence.append(np.array(p_action))
        return sequence

    def reset(self, initial_soc: float = 0.5):
        """Reset phantom beliefs and SoC estimate."""
        self._empirical_prior = self.agent.D
        self._rng_key = jr.PRNGKey(98)
        self._soc = initial_soc
