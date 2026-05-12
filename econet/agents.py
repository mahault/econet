"""Agent wrappers around JAX pymdp Agent."""

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx

from pymdp.agent import Agent

from .generative_model import (
    build_thermostat_model, build_battery_model,
    build_thermostat_model_tom, build_battery_model_tom,
    build_thermostat_model_cost_aware,
    THERMO_NUM_ACTIONS, BATTERY_NUM_ACTIONS,
    THERMO_NUM_STATES, BATTERY_NUM_STATES,
    THERMO_TOM_A_DEPS, BATTERY_TOM_A_DEPS,
    COMFORT_LEVELS, BATTERY_A_DEPS,
)
from .environment import (
    HVAC_KWH_PER_STEP, SOC_LEVELS, TEMP_LEVELS, TEMP_MIN, TEMP_STEP,
    TARGET_TEMP_OCCUPIED, TARGET_TEMP_UNOCCUPIED,
    ENERGY_LEVELS, BATTERY_CAPACITY_KWH, BATTERY_STEP_FRAC,
    BATTERY_EFFICIENCY,
    discretize_temp,
)


# ======================================================================
# pymdp API compatibility helpers
# ======================================================================

def _update_empirical_prior_compat(agent, action, qs):
    """Handle both pymdp API versions for update_empirical_prior."""
    result = agent.update_empirical_prior(action, qs)
    if isinstance(result, tuple):
        return result[0]  # (pred, qs) format
    return result  # single value format


def _infer_parameters_compat(agent, **kwargs):
    """Handle both pymdp API versions for infer_parameters."""
    try:
        return agent.infer_parameters(**kwargs)
    except TypeError:
        if 'outcomes' in kwargs:
            kwargs['observations'] = kwargs.pop('outcomes')
        return agent.infer_parameters(**kwargs)


# ======================================================================
# Predictive B matrix helpers (shared by standard + ToM agents)
# ======================================================================

def _build_thermo_predictive_B(env_data, step_idx):
    """Build 1-step-ahead B matrices for thermostat exogenous factors.

    Uses known schedules (TOU, occupancy) and weather forecast (outdoor temp)
    to build transition matrices that encode what the agent expects NEXT step.

    Returns (B1, B2, B3) for outdoor_temp, occupancy, tou_high.
    """
    total = len(env_data["outdoor_temp"])
    next_idx = min(step_idx + 1, total - 1)

    # B[1]: Outdoor temp (Gaussian around forecast, sigma=1.0 bin)
    next_temp = env_data["outdoor_temp"][next_idx]
    next_temp_idx = discretize_temp(next_temp)
    B1 = np.zeros((TEMP_LEVELS, TEMP_LEVELS, 1))
    for prev in range(TEMP_LEVELS):
        for delta in range(-3, 4):
            nxt = next_temp_idx + delta
            if 0 <= nxt < TEMP_LEVELS:
                w = np.exp(-0.5 * (delta / 1.0) ** 2)
                B1[nxt, prev, 0] += w
        col_sum = B1[:, prev, 0].sum()
        if col_sum > 0:
            B1[:, prev, 0] /= col_sum

    # B[2]: Occupancy (deterministic calendar)
    next_occ = int(env_data["occupancy"][next_idx])
    B2 = np.zeros((2, 2, 1))
    B2[next_occ, :, 0] = 1.0

    # B[3]: TOU (deterministic schedule)
    next_tou = int(env_data["tou_high"][next_idx])
    B3 = np.zeros((2, 2, 1))
    B3[next_tou, :, 0] = 1.0

    return B1, B2, B3


def _build_battery_predictive_tou(env_data, step_idx):
    """Build 1-step-ahead TOU B matrix for battery agent."""
    total = len(env_data["tou_high"])
    next_idx = min(step_idx + 1, total - 1)

    next_tou = int(env_data["tou_high"][next_idx])
    B1 = np.zeros((2, 2, 1))
    B1[next_tou, :, 0] = 1.0

    return B1


class ThermostatAgent:
    """Active inference thermostat agent.

    Controls HVAC: cool (0), heat (1), off (2).
    Uses JAX pymdp with B_dependencies for outdoor-temp-conditioned
    room temperature transitions.

    When aligned=True (default), dynamically adjusts C vector based on
    occupancy and TOU — enabling decentralized coordination with battery.
    When aligned=False, uses static C from the generative model.

    When learn_B=True, performs online Dirichlet updates to pB every step
    via agent.infer_parameters (requires 2-frame belief joint).
    """

    def __init__(self, env_data: dict, policy_len: int = 4,
                 gamma: float = 16.0, learn_B: bool = False,
                 aligned: bool = True, forecast_data: dict = None,
                 use_states_info_gain: bool = True,
                 comfort_scale: float = 1.0, soc_scale: float = 1.0):
        comfort_amplitude = -4.0 * comfort_scale
        model = build_thermostat_model(env_data, policy_len=policy_len,
                                       comfort_amplitude=comfort_amplitude)

        # pB for Dirichlet learning (only B[0] = room_temp factor)
        pB = None
        if learn_B:
            pB = [b + 1.0 for b in model["B"]]

        self.agent = Agent(
            A=model["A"],
            B=model["B"],
            C=model["C"],
            D=model["D"],
            pB=pB,
            A_dependencies=model["A_dependencies"],
            B_dependencies=model["B_dependencies"],
            control_fac_idx=[0],
            policy_len=policy_len,
            gamma=gamma,
            use_utility=True,
            use_states_info_gain=use_states_info_gain,
            use_param_info_gain=learn_B,
            action_selection="deterministic",
            inference_algo="fpi",
            num_iter=16,
            batch_size=1,
            learn_B=learn_B,
        )
        self.learn_B = learn_B
        self.aligned = aligned
        self._comfort_scale = comfort_scale
        self._env_data = env_data
        self._forecast_data = forecast_data if forecast_data is not None else env_data
        self._empirical_prior = self.agent.D
        self._rng_key = jr.PRNGKey(0)
        self.qs_history = []
        self.efe_history = []

        # Online B-learning state: store previous beliefs & action
        self._qs_prev = None
        self._action_prev = None

    def step(self, obs_dict: dict, step_idx: int = None) -> tuple:
        """Run one inference step.

        Parameters
        ----------
        obs_dict : dict
            Keys: room_temp, outdoor_temp, occupancy, tou_high (all int indices)
        step_idx : int, optional
            Current absolute simulation step. When provided, updates exogenous
            B matrices with 1-step-ahead forecasts (predictive B).

        Returns
        -------
        action : int
            HVAC action (0=cool, 1=heat, 2=off)
        hvac_energy : float
            Energy consumption in kWh
        info : dict
            Contains q_pi and neg_efe for analysis
        """
        # --- Predictive B matrices for exogenous factors ---
        if step_idx is not None:
            B1, B2, B3 = _build_thermo_predictive_B(self._forecast_data, step_idx)
            new_B = list(self.agent.B)
            new_B[1] = jnp.array(B1)[None, :]
            new_B[2] = jnp.array(B2)[None, :]
            new_B[3] = jnp.array(B3)[None, :]
            self.agent = eqx.tree_at(lambda a: a.B, self.agent, new_B)

        # --- Dynamic C: shift preference by occupancy + TOU every step ---
        # Thermostat relaxes comfort slightly during peak TOU, creating
        # space for battery to handle cost reduction.
        occupancy = obs_dict.get("occupancy", 1)
        tou_high = obs_dict.get("tou_high", 0)
        target = TARGET_TEMP_OCCUPIED if occupancy else TARGET_TEMP_UNOCCUPIED
        target_idx = discretize_temp(target)
        # Amplitude: strong off-peak (-4.0), slightly relaxed during peak (-3.0)
        # Scaled by comfort_scale to dominate info gain.
        amplitude = (-3.0 if tou_high else -4.0) * self._comfort_scale
        c_room = np.zeros(TEMP_LEVELS)
        for i in range(TEMP_LEVELS):
            dist = abs(i - target_idx)
            c_room[i] = amplitude * dist ** 2 / 4.0
        c_room -= c_room.max()
        new_c0 = jnp.array(c_room)[None, :]  # (1, TEMP_LEVELS)
        new_C = list(self.agent.C)
        new_C[0] = new_c0
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)

        # Shape: (batch=1, T=1) — JAX pymdp expects a time dimension
        obs = [
            jnp.array([[obs_dict["room_temp"]]]),
            jnp.array([[obs_dict["outdoor_temp"]]]),
            jnp.array([[obs_dict["occupancy"]]]),
            jnp.array([[obs_dict["tou_high"]]]),
        ]

        qs = self.agent.infer_states(obs, self._empirical_prior)
        q_pi, neg_efe = self.agent.infer_policies(qs)

        self._rng_key, subkey = jr.split(self._rng_key)
        action = self.agent.sample_action(q_pi)

        # action shape: (batch=1, num_factors) — extract controlled factor
        action_int = int(action[0, 0])

        # Online B-learning: update pB every step using 2-frame beliefs
        if self.learn_B and self._qs_prev is not None:
            beliefs_seq = jtu.tree_map(
                lambda prev, curr: jnp.concatenate([prev, curr], axis=1),
                self._qs_prev, qs
            )
            self.agent = _infer_parameters_compat(
                self.agent,
                beliefs_A=beliefs_seq,
                outcomes=obs,
                actions=self._action_prev,
                beliefs_B=beliefs_seq,
                lr_pB=1.0,
            )

        # Store for next step's B-learning
        if self.learn_B:
            self._qs_prev = qs
            self._action_prev = action

        # Update empirical prior for next step
        self._empirical_prior = _update_empirical_prior_compat(self.agent, action, qs)

        # Convert JAX arrays to numpy for storage
        q_pi_np = np.asarray(q_pi)
        neg_efe_np = np.asarray(neg_efe)

        self.qs_history.append(qs)
        self.efe_history.append(neg_efe_np)

        hvac_energy = HVAC_KWH_PER_STEP if action_int != 2 else 0.0

        return action_int, hvac_energy, {
            "q_pi": q_pi_np,
            "neg_efe": neg_efe_np,
        }

    def update_C(self, new_C: list):
        """Inject new C vectors from high-level strategy agent.

        Parameters
        ----------
        new_C : list of np.ndarray
            New C vectors (one per observation modality), without batch dim.
        """
        new_C_batched = [
            jnp.broadcast_to(jnp.array(c)[None], (1,) + jnp.array(c).shape)
            for c in new_C
        ]
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C_batched)


class BatteryAgent:
    """Active inference battery storage agent.

    Controls battery: charge (0), discharge (1), off (2).

    When aligned=True (default), uses dynamic C vector for TOU arbitrage:
    - High TOU (peak): prefer lower SoC -> discharge to offset grid pull
    - Low TOU (off-peak): prefer higher SoC -> charge for next peak
    This enables decentralized coordination with the thermostat.

    When aligned=False, uses static C from the generative model.
    """

    # SoC preference profiles (5 levels: 0.0, 0.2, 0.4, 0.6, 0.8)
    # Peak: prefer low SoC (discharge to offset grid pull)
    # Off-peak: prefer high SoC (charge for next peak) — moderate strength
    C_SOC_PEAK = np.array([2.0, 1.0, 0.0, -1.0, -2.0])      # prefer low SoC
    C_SOC_OFFPEAK = np.array([-1.5, -0.5, 0.0, 0.5, 1.0])   # prefer high SoC

    def __init__(self, env_data: dict, policy_len: int = 4,
                 gamma: float = 16.0, initial_soc: float = 0.5,
                 aligned: bool = True, forecast_data: dict = None,
                 use_states_info_gain: bool = True,
                 soc_scale: float = 1.0):
        model = build_battery_model(env_data, initial_soc=initial_soc,
                                    policy_len=policy_len,
                                    soc_scale=soc_scale)

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
            use_states_info_gain=use_states_info_gain,
            action_selection="deterministic",
            inference_algo="fpi",
            num_iter=16,
            batch_size=1,
        )
        self.aligned = aligned
        self._soc_scale = soc_scale
        # Instance-level SoC C vectors scaled by soc_scale
        self._c_soc_peak = np.array([v * soc_scale for v in [2.0, 1.0, 0.0, -1.0, -2.0]])
        self._c_soc_offpeak = np.array([v * soc_scale for v in [-1.5, -0.5, 0.0, 0.5, 1.0]])
        self._env_data = env_data
        self._forecast_data = forecast_data if forecast_data is not None else env_data
        self._empirical_prior = self.agent.D
        self._rng_key = jr.PRNGKey(1)
        self.qs_history = []
        self.efe_history = []

    def step(self, obs_dict: dict, step_idx: int = None) -> tuple:
        """Run one inference step.

        Parameters
        ----------
        obs_dict : dict
            Keys: soc, cost, ghg, tou_high (all int indices)
        step_idx : int, optional
            Current absolute simulation step. When provided, updates TOU
            B matrix with 1-step-ahead forecast (predictive B).

        Returns
        -------
        action : int
            Battery action (0=charge, 1=discharge, 2=off)
        info : dict
            Contains q_pi and neg_efe for analysis
        """
        # --- Predictive TOU B matrix ---
        if step_idx is not None:
            B1_new = _build_battery_predictive_tou(self._forecast_data, step_idx)
            new_B = list(self.agent.B)
            new_B[1] = jnp.array(B1_new)[None, :]
            self.agent = eqx.tree_at(lambda a: a.B, self.agent, new_B)

        # --- Dynamic C: flip SoC preference based on TOU every step ---
        # Battery discharges at peak (offsetting HVAC cost), charges off-peak.
        tou_high = obs_dict.get("tou_high", 0)
        if tou_high:
            new_c0 = jnp.array(self._c_soc_peak)
        else:
            new_c0 = jnp.array(self._c_soc_offpeak)

        new_c0_batched = new_c0[None, :]  # (1, 5)
        new_C = list(self.agent.C)
        new_C[0] = new_c0_batched
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)

        # Shape: (batch=1, T=1) — JAX pymdp expects a time dimension
        obs = [
            jnp.array([[obs_dict["soc"]]]),
            jnp.array([[obs_dict["cost"]]]),
            jnp.array([[obs_dict["ghg"]]]),
        ]

        qs = self.agent.infer_states(obs, self._empirical_prior)
        q_pi, neg_efe = self.agent.infer_policies(qs)

        self._rng_key, subkey = jr.split(self._rng_key)
        action = self.agent.sample_action(q_pi)

        action_int = int(action[0, 0])

        # Update empirical prior for next step
        self._empirical_prior = _update_empirical_prior_compat(self.agent, action, qs)

        # Convert JAX arrays to numpy for storage
        q_pi_np = np.asarray(q_pi)
        neg_efe_np = np.asarray(neg_efe)

        self.qs_history.append(qs)
        self.efe_history.append(neg_efe_np)

        return action_int, {
            "q_pi": q_pi_np,
            "neg_efe": neg_efe_np,
        }

    def update_C(self, new_C: list):
        """Inject new C vectors from high-level strategy agent.

        Parameters
        ----------
        new_C : list of np.ndarray
            New C vectors (one per observation modality), without batch dim.
        """
        new_C_batched = [
            jnp.broadcast_to(jnp.array(c)[None], (1,) + jnp.array(c).shape)
            for c in new_C
        ]
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C_batched)


# ======================================================================
# Sophisticated Inference Thermostat Agent (Pitliya et al., 2025)
# ======================================================================

class SophisticatedThermostatAgent:
    """Thermostat with principled cost awareness via extended generative model.

    Uses a 5-factor pymdp Agent where energy_demand is the 5th state factor
    and cost is the 5th observation modality.  Cost enters through the
    standard EFE (utility + ambiguity) — no custom loops or unit-mismatched
    penalties.

    At each step:
    1. Update B[4] (energy transition) from phantom battery prediction
    2. Update C[0] (comfort preference) from occupancy + TOU
    3. Feed 5 observations (including cost from previous step)
    4. Standard infer_states → infer_policies → sample_action

    The phantom battery is called ONCE per step to get P(charge/discharge/off),
    which builds B[4] as a proper mixture transition matrix.
    """

    def __init__(self, env_data: dict, policy_len: int = 4,
                 gamma: float = 16.0, learn_B: bool = False,
                 cost_scale: float = 3.0, initial_soc: float = 0.5,
                 forecast_data: dict = None,
                 use_states_info_gain: bool = True,
                 comfort_scale: float = 1.0):
        from itertools import product
        from .phantom import PhantomBattery

        self._comfort_scale = comfort_scale
        comfort_amplitude = -4.0 * comfort_scale
        model = build_thermostat_model_cost_aware(
            env_data, policy_len=policy_len, cost_scale=cost_scale,
            comfort_amplitude=comfort_amplitude)

        pB = None
        if learn_B:
            pB = [b + 1.0 for b in model["B"]]

        # Build constrained policies: a[4] == a[0] always
        # With 5 factors and control on [0, 4], unconstrained would be
        # 3^T × 3^T = 6561 policies (T=2). We constrain to 3^T = 81.
        base_seqs = list(product(range(THERMO_NUM_ACTIONS), repeat=policy_len))
        n_policies = len(base_seqs)
        policies_np = np.zeros((n_policies, policy_len, 5), dtype=int)
        for i, seq in enumerate(base_seqs):
            for t in range(policy_len):
                policies_np[i, t, 0] = seq[t]  # HVAC action
                policies_np[i, t, 4] = seq[t]  # energy mirrors HVAC (constrained)
                # factors 1-3: no-op (0)
        policies_jax = jnp.array(policies_np)

        self.agent = Agent(
            A=model["A"],
            B=model["B"],
            C=model["C"],
            D=model["D"],
            pB=pB,
            A_dependencies=model["A_dependencies"],
            B_dependencies=model["B_dependencies"],
            control_fac_idx=[0, 4],
            policies=policies_jax,
            gamma=gamma,
            use_utility=True,
            use_states_info_gain=use_states_info_gain,
            use_param_info_gain=learn_B,
            action_selection="deterministic",
            inference_algo="fpi",
            num_iter=16,
            batch_size=1,
            learn_B=learn_B,
        )
        self.learn_B = learn_B
        self.cost_scale = cost_scale
        self._empirical_prior = self.agent.D
        self._rng_key = jr.PRNGKey(4)
        self.qs_history = []
        self.efe_history = []
        self._qs_prev = None
        self._action_prev = None
        self._env_data = env_data
        self._forecast_data = forecast_data if forecast_data is not None else env_data

        # Phantom battery for B[4] construction
        self.phantom_battery = PhantomBattery(
            env_data, policy_len=policy_len, gamma=gamma,
            initial_soc=initial_soc)

        # Energy bin size for B[4] construction
        self._energy_bin_size = 9.0 / (ENERGY_LEVELS - 1)  # ~1.0 kWh

    def _build_predictive_B(self, step_idx: int) -> tuple:
        """Build predictive B matrices for exogenous factors.

        Uses known schedules (TOU, occupancy) and weather forecast
        (outdoor temp) to build one-step-ahead transition matrices.

        Parameters
        ----------
        step_idx : int
            Current absolute simulation step.

        Returns
        -------
        B1, B2, B3 : tuple of np.ndarray
            Transition matrices for outdoor_temp, occupancy, tou_high.
        """
        total = len(self._forecast_data["outdoor_temp"])
        next_idx = min(step_idx + 1, total - 1)

        # --- B[3]: TOU (deterministic schedule, published by grid) ---
        next_tou = int(self._forecast_data["tou_high"][next_idx])
        B3 = np.zeros((2, 2, 1))
        B3[next_tou, 0, 0] = 1.0
        B3[next_tou, 1, 0] = 1.0

        # --- B[2]: Occupancy (deterministic calendar) ---
        next_occ = int(self._forecast_data["occupancy"][next_idx])
        B2 = np.zeros((2, 2, 1))
        B2[next_occ, 0, 0] = 1.0
        B2[next_occ, 1, 0] = 1.0

        # --- B[1]: Outdoor temp (Gaussian around forecast, σ=1.0 bins) ---
        next_temp_cont = self._forecast_data["outdoor_temp"][next_idx]
        next_temp_idx = discretize_temp(next_temp_cont)
        B1 = np.zeros((TEMP_LEVELS, TEMP_LEVELS, 1))
        for prev in range(TEMP_LEVELS):
            for delta in range(-3, 4):
                nxt = next_temp_idx + delta
                if 0 <= nxt < TEMP_LEVELS:
                    w = np.exp(-0.5 * (delta / 1.0) ** 2)
                    B1[nxt, prev, 0] += w
            col_sum = B1[:, prev, 0].sum()
            if col_sum > 0:
                B1[:, prev, 0] /= col_sum

        return B1, B2, B3

    def _build_B4(self, phantom_probs: np.ndarray) -> np.ndarray:
        """Build energy transition matrix from phantom battery prediction.

        Parameters
        ----------
        phantom_probs : np.ndarray, shape (3,)
            [P(charge), P(discharge), P(off)] from phantom battery.

        Returns
        -------
        B4 : np.ndarray, shape (ENERGY_LEVELS, ENERGY_LEVELS, 3)
            P(energy' | energy, hvac_action).
        """
        B4 = np.zeros((ENERGY_LEVELS, ENERGY_LEVELS, THERMO_NUM_ACTIONS))
        batt_kwh_charge = BATTERY_STEP_FRAC * BATTERY_CAPACITY_KWH / BATTERY_EFFICIENCY
        batt_kwh_discharge = BATTERY_STEP_FRAC * BATTERY_CAPACITY_KWH * BATTERY_EFFICIENCY
        p_charge = float(phantom_probs[0])
        p_discharge = float(phantom_probs[1])
        p_off = float(phantom_probs[2])

        for action in range(THERMO_NUM_ACTIONS):
            hvac_kwh = HVAC_KWH_PER_STEP if action != 2 else 0.0
            for j in range(ENERGY_LEVELS):
                # Mixture over phantom battery outcomes
                for batt_shift, p_batt in [
                    (+batt_kwh_charge, p_charge),
                    (-batt_kwh_discharge, p_discharge),
                    (0.0, p_off),
                ]:
                    delta = round((hvac_kwh + batt_shift) / self._energy_bin_size)
                    j_next = int(np.clip(j + delta, 0, ENERGY_LEVELS - 1))
                    B4[j_next, j, action] += p_batt
        # Columns should already be normalized (p_batt sums to 1)
        return B4

    def step(self, obs_dict: dict, step_idx: int,
             cost_obs: int = 0) -> tuple:
        """Run one inference step with principled cost awareness.

        Parameters
        ----------
        obs_dict : dict
            Keys: room_temp, outdoor_temp, occupancy, tou_high
        step_idx : int
            Current absolute simulation step.
        cost_obs : int
            Discretized cost observation (from previous step or baseline).

        Returns
        -------
        action : int
            HVAC action (0=cool, 1=heat, 2=off)
        hvac_energy : float
            Energy consumption in kWh
        info : dict
        """
        # --- 1. Phantom battery rollout for B[4] ---
        self.phantom_battery.reset(initial_soc=0.5)
        phantom_batt_seq = self.phantom_battery.rollout(
            start_step=step_idx,
            horizon=1,  # only need first step for B[4]
        )
        phantom_p_batt = phantom_batt_seq[0] if phantom_batt_seq else np.array([0.33, 0.33, 0.34])

        # --- 2. Update all dynamic B matrices ---
        B4_new = self._build_B4(phantom_p_batt)
        B1_new, B2_new, B3_new = self._build_predictive_B(step_idx)
        new_B = list(self.agent.B)
        new_B[1] = jnp.array(B1_new)[None, :]  # outdoor temp forecast
        new_B[2] = jnp.array(B2_new)[None, :]  # occupancy schedule
        new_B[3] = jnp.array(B3_new)[None, :]  # TOU schedule
        new_B[4] = jnp.array(B4_new)[None, :]  # phantom battery
        self.agent = eqx.tree_at(lambda a: a.B, self.agent, new_B)

        # --- 3. Dynamic C[0] (comfort scaled by thermal stress) ---
        # Cost awareness flows through C[4], NOT through weakened comfort.
        # Comfort amplitude scales with outdoor-target gap: deviations are
        # more aversive when the environment is extreme (nonlinear health risk).
        occupancy = obs_dict.get("occupancy", 1)
        outdoor_idx = obs_dict.get("outdoor_temp", 0)
        target = TARGET_TEMP_OCCUPIED if occupancy else TARGET_TEMP_UNOCCUPIED
        target_idx = discretize_temp(target)
        outdoor_actual = TEMP_MIN + outdoor_idx * TEMP_STEP
        thermal_gap = abs(outdoor_actual - target)
        amplitude = -4.0 * self._comfort_scale * (1.0 + thermal_gap / 20.0)
        c_room = np.zeros(TEMP_LEVELS)
        for i in range(TEMP_LEVELS):
            dist = abs(i - target_idx)
            c_room[i] = amplitude * dist ** 2 / 4.0
        c_room -= c_room.max()
        new_C = list(self.agent.C)
        new_C[0] = jnp.array(c_room)[None, :]
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)

        # --- 4. Build observations (5 modalities) ---
        obs = [
            jnp.array([[obs_dict["room_temp"]]]),
            jnp.array([[obs_dict["outdoor_temp"]]]),
            jnp.array([[obs_dict["occupancy"]]]),
            jnp.array([[obs_dict["tou_high"]]]),
            jnp.array([[cost_obs]]),
        ]

        # --- 5. Standard inference ---
        qs = self.agent.infer_states(obs, self._empirical_prior)
        q_pi, neg_efe = self.agent.infer_policies(qs)

        self._rng_key, subkey = jr.split(self._rng_key)
        action = self.agent.sample_action(q_pi)

        # action shape: (batch=1, num_factors=5)
        action_int = int(action[0, 0])

        # --- B-learning ---
        if self.learn_B and self._qs_prev is not None:
            beliefs_seq = jtu.tree_map(
                lambda prev, curr: jnp.concatenate([prev, curr], axis=1),
                self._qs_prev, qs
            )
            self.agent = _infer_parameters_compat(
                self.agent,
                beliefs_A=beliefs_seq,
                outcomes=obs,
                actions=self._action_prev,
                beliefs_B=beliefs_seq,
                lr_pB=1.0,
            )

        # Store for next step
        if self.learn_B:
            self._qs_prev = qs
            self._action_prev = action

        # --- Update empirical prior ---
        self._empirical_prior = _update_empirical_prior_compat(self.agent, action, qs)

        q_pi_np = np.asarray(q_pi)
        neg_efe_np = np.asarray(neg_efe)

        self.qs_history.append(qs)
        self.efe_history.append(neg_efe_np)

        hvac_energy = HVAC_KWH_PER_STEP if action_int != 2 else 0.0

        return action_int, hvac_energy, {
            "q_pi": q_pi_np,
            "neg_efe": neg_efe_np,
            "phantom_p_batt": phantom_p_batt,
        }


# ======================================================================
# Sophisticated Inference Battery Agent (Pitliya et al., 2025)
# ======================================================================

class SophisticatedBatteryAgent:
    """Battery agent with full sophisticated inference.

    Maintains a phantom thermostat that predicts HVAC activity T steps
    ahead.  Custom EFE computation uses step-dependent B[energy] based
    on phantom predictions, allowing the battery to anticipate HVAC-driven
    energy costs across the planning horizon.

    Reference: Pitliya et al. (2025), "Theory of Mind Using Active Inference"
    (arXiv:2508.00401).
    """

    C_SOC_PEAK = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    C_SOC_OFFPEAK = np.array([-1.5, -0.5, 0.0, 0.5, 1.0])

    def __init__(self, env_data: dict, policy_len: int = 4,
                 gamma: float = 16.0, initial_soc: float = 0.5,
                 forecast_data: dict = None,
                 use_states_info_gain: bool = True,
                 soc_scale: float = 1.0):
        from .phantom import PhantomThermostat
        from .sophisticated import compute_sophisticated_efe

        model = build_battery_model(env_data, initial_soc=initial_soc,
                                    policy_len=policy_len,
                                    soc_scale=soc_scale)

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
            use_states_info_gain=use_states_info_gain,
            action_selection="deterministic",
            inference_algo="fpi",
            num_iter=16,
            batch_size=1,
        )
        self.gamma = gamma
        self.policy_len = policy_len
        self._soc_scale = soc_scale
        self._c_soc_peak = np.array([v * soc_scale for v in [2.0, 1.0, 0.0, -1.0, -2.0]])
        self._c_soc_offpeak = np.array([v * soc_scale for v in [-1.5, -0.5, 0.0, 0.5, 1.0]])
        self._empirical_prior = self.agent.D
        self._rng_key = jr.PRNGKey(2)
        self.qs_history = []
        self.efe_history = []
        self._env_data = env_data
        self._forecast_data = forecast_data if forecast_data is not None else env_data

        # Phantom thermostat for sophisticated inference (uses true env_data)
        self.phantom = PhantomThermostat(env_data, policy_len=policy_len,
                                         gamma=gamma)

        # HVAC delta in energy bins: HVAC_KWH_PER_STEP / bin_size
        # energy range: -3.0 to +6.0 kWh over 10 bins → bin_size = 1.0
        self._hvac_delta_bins = round(HVAC_KWH_PER_STEP / 1.0)  # = 2

    def step(self, obs_dict: dict, step_idx: int,
             current_hvac_action: int = None) -> tuple:
        """Run one inference step with sophisticated EFE.

        Parameters
        ----------
        obs_dict : dict
            Keys: soc, cost, ghg, tou_high (all int indices)
        step_idx : int
            Current absolute simulation step (needed for phantom rollout).
        current_hvac_action : int, optional
            Known HVAC action at current step (0=cool, 1=heat, 2=off).
            When provided, replaces phantom prediction for tau=0 with
            the actual observation, eliminating prediction error for the
            first planning step.

        Returns
        -------
        action : int
            Battery action (0=charge, 1=discharge, 2=off)
        info : dict
            Contains q_pi, neg_efe, phantom_p_hvac (first step prediction)
        """
        from .sophisticated import compute_sophisticated_efe

        # --- Predictive TOU B matrix (matches standard BatteryAgent) ---
        B1_new = _build_battery_predictive_tou(self._forecast_data, step_idx)
        new_B = list(self.agent.B)
        new_B[1] = jnp.array(B1_new)[None, :]
        self.agent = eqx.tree_at(lambda a: a.B, self.agent, new_B)

        # --- Build per-step TOU schedule for custom EFE ---
        total = len(self._forecast_data["tou_high"])
        tou_schedule = []
        for tau in range(self.policy_len):
            future_idx = min(step_idx + tau + 1, total - 1)
            tou_schedule.append(int(self._forecast_data["tou_high"][future_idx]))

        # --- Dynamic C (TOU arbitrage) ---
        tou_high = obs_dict.get("tou_high", 0)
        if tou_high:
            new_c0 = jnp.array(self._c_soc_peak)
        else:
            new_c0 = jnp.array(self._c_soc_offpeak)
        new_C = list(self.agent.C)
        new_C[0] = new_c0[None, :]
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)

        # --- Standard belief update ---
        obs = [
            jnp.array([[obs_dict["soc"]]]),
            jnp.array([[obs_dict["cost"]]]),
            jnp.array([[obs_dict["ghg"]]]),
        ]
        qs = self.agent.infer_states(obs, self._empirical_prior)

        # --- Phantom rollout: T-step HVAC prediction ---
        self.phantom.reset()
        phantom_seq = self.phantom.rollout(
            start_step=step_idx,
            horizon=self.policy_len,
        )

        # Override tau=0 with known HVAC action (exact observation)
        if current_hvac_action is not None:
            known = np.zeros(3)
            known[current_hvac_action] = 1.0
            phantom_seq[0] = known

        # --- Custom EFE with step-dependent B[energy] and TOU ---
        q_pi, neg_efe = compute_sophisticated_efe(
            A=self.agent.A,
            B=self.agent.B,
            C=self.agent.C,
            qs=qs,
            policies=self.agent.policies,
            phantom_sequence=phantom_seq,
            A_deps=BATTERY_A_DEPS,
            energy_factor_idx=2,
            hvac_delta_bins=self._hvac_delta_bins,
            gamma=self.gamma,
            tou_schedule=tou_schedule,
        )

        # --- Action selection ---
        # q_pi is over policies, not actions. Select best policy,
        # then extract the first action for the controlled factor.
        best_policy_idx = int(np.argmax(q_pi))
        policies_np = np.asarray(self.agent.policies)
        action_int = int(policies_np[best_policy_idx, 0, 0])  # first step, factor 0

        # --- Update empirical prior for next step ---
        # Build action array matching pymdp format: (batch=1, n_factors)
        action_arr = jnp.array([[action_int, 0, 0]])
        self._empirical_prior = _update_empirical_prior_compat(self.agent, action_arr, qs)

        self.qs_history.append(qs)
        self.efe_history.append(neg_efe)

        return action_int, {
            "q_pi": q_pi,
            "neg_efe": neg_efe,
            "phantom_p_hvac": phantom_seq[0] if phantom_seq else np.array([0, 0, 1.0]),
        }


class SophisticatedToMBatteryAgent:
    """Battery with sophisticated inference AND belief sharing.

    Combines:
    - Phantom thermostat for T-step HVAC prediction (sophisticated)
    - Received q(comfort) from thermostat to improve phantom obs estimate
    - ToM reliability filter on received beliefs

    When the battery receives q(comfort), it uses the belief to estimate
    the actual room temperature the thermostat observes, which improves
    the phantom's initial conditions for its rollout.
    """

    C_SOC_PEAK = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    C_SOC_OFFPEAK = np.array([-1.5, -0.5, 0.0, 0.5, 1.0])

    def __init__(self, env_data: dict, policy_len: int = 4,
                 gamma: float = 16.0, initial_soc: float = 0.5,
                 social_weight: float = 1.0,
                 forecast_data: dict = None,
                 use_states_info_gain: bool = True,
                 soc_scale: float = 1.0):
        from .phantom import PhantomThermostat
        from .tom import ThermostatToM

        model = build_battery_model(env_data, initial_soc=initial_soc,
                                    policy_len=policy_len,
                                    soc_scale=soc_scale)

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
            use_states_info_gain=use_states_info_gain,
            action_selection="deterministic",
            inference_algo="fpi",
            num_iter=16,
            batch_size=1,
        )
        self.gamma = gamma
        self.policy_len = policy_len
        self._soc_scale = soc_scale
        self._c_soc_peak = np.array([v * soc_scale for v in [2.0, 1.0, 0.0, -1.0, -2.0]])
        self._c_soc_offpeak = np.array([v * soc_scale for v in [-1.5, -0.5, 0.0, 0.5, 1.0]])
        self._empirical_prior = self.agent.D
        self._rng_key = jr.PRNGKey(3)
        self.qs_history = []
        self.efe_history = []
        self._env_data = env_data
        self._forecast_data = forecast_data if forecast_data is not None else env_data

        # Phantom thermostat for sophisticated inference (uses true env_data)
        self.phantom = PhantomThermostat(env_data, policy_len=policy_len,
                                         gamma=gamma)
        self._hvac_delta_bins = round(HVAC_KWH_PER_STEP / 1.0)

        # ToM filter for received comfort beliefs
        self.thermo_tom = ThermostatToM(n_states=COMFORT_LEVELS)
        self.social_weight = social_weight

        # Room temp estimate from received comfort beliefs
        self._received_room_temp_est = None

    def _comfort_to_room_temp_idx(self, q_comfort):
        """Estimate room temp index from received q(comfort).

        Comfort levels map to temp ranges relative to target:
          0: COLD    (target - 6)
          1: COOL    (target - 3)
          2: COMFY   (target)
          3: WARM    (target + 3)
          4: HOT     (target + 6)
        """
        from .tom import belief_to_obs
        comfort_idx = belief_to_obs(q_comfort)
        offsets = [-6, -3, 0, 3, 6]  # °C relative to target
        return offsets[comfort_idx]

    def step(self, obs_dict: dict, step_idx: int,
             received_q_comfort=None,
             current_hvac_action: int = None) -> tuple:
        """Run one inference step with sophisticated EFE + received beliefs.

        Parameters
        ----------
        obs_dict : dict
            Keys: soc, cost, ghg, tou_high (all int indices)
        step_idx : int
            Current absolute simulation step.
        received_q_comfort : np.ndarray, optional
            Thermostat's shared q(comfort), shape (5,).
        current_hvac_action : int, optional
            Known HVAC action at current step (0=cool, 1=heat, 2=off).

        Returns
        -------
        action : int
            Battery action (0=charge, 1=discharge, 2=off)
        info : dict
        """
        from .sophisticated import compute_sophisticated_efe
        from .phantom import estimate_thermo_obs
        from .tom import belief_to_obs

        # --- Process received comfort belief ---
        if received_q_comfort is not None:
            comfort_obs = belief_to_obs(received_q_comfort)
            self.thermo_tom.update(received_q_comfort, comfort_obs)
            # Estimate room temp offset from comfort
            self._received_room_temp_est = self._comfort_to_room_temp_idx(
                received_q_comfort)

        # --- Predictive TOU B matrix ---
        B1_new = _build_battery_predictive_tou(self._forecast_data, step_idx)
        new_B = list(self.agent.B)
        new_B[1] = jnp.array(B1_new)[None, :]
        self.agent = eqx.tree_at(lambda a: a.B, self.agent, new_B)

        # --- Build per-step TOU schedule for custom EFE ---
        total = len(self._forecast_data["tou_high"])
        tou_schedule = []
        for tau in range(self.policy_len):
            future_idx = min(step_idx + tau + 1, total - 1)
            tou_schedule.append(int(self._forecast_data["tou_high"][future_idx]))

        # --- Dynamic C (TOU arbitrage) ---
        tou_high = obs_dict.get("tou_high", 0)
        if tou_high:
            new_c0 = jnp.array(self._c_soc_peak)
        else:
            new_c0 = jnp.array(self._c_soc_offpeak)
        new_C = list(self.agent.C)
        new_C[0] = new_c0[None, :]
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)

        # --- Standard belief update ---
        obs = [
            jnp.array([[obs_dict["soc"]]]),
            jnp.array([[obs_dict["cost"]]]),
            jnp.array([[obs_dict["ghg"]]]),
        ]
        qs = self.agent.infer_states(obs, self._empirical_prior)

        # --- Phantom rollout with improved obs estimate ---
        self.phantom.reset()

        # If we have received comfort, override phantom's room temp estimate
        if self._received_room_temp_est is not None:
            # Patch the phantom's first obs with received info
            base_obs = estimate_thermo_obs(self._env_data, step_idx)
            occ = int(self._env_data["occupancy"][min(step_idx,
                       len(self._env_data["occupancy"]) - 1)])
            target = TARGET_TEMP_OCCUPIED if occ else TARGET_TEMP_UNOCCUPIED
            est_temp = target + self._received_room_temp_est
            base_obs["room_temp"] = discretize_temp(est_temp)
            # Run first step with improved obs, rest with standard estimate
            p0 = self.phantom.predict_action(base_obs)
            phantom_seq = [np.array(p0)]
            for tau in range(1, self.policy_len):
                obs_tau = estimate_thermo_obs(self._env_data, step_idx + tau)
                p_tau = self.phantom.predict_action(obs_tau)
                phantom_seq.append(np.array(p_tau))
        else:
            phantom_seq = self.phantom.rollout(
                start_step=step_idx, horizon=self.policy_len)

        # Override tau=0 with known HVAC action (exact observation)
        if current_hvac_action is not None:
            known = np.zeros(3)
            known[current_hvac_action] = 1.0
            phantom_seq[0] = known

        # --- Custom EFE with step-dependent B[energy] and TOU ---
        q_pi, neg_efe = compute_sophisticated_efe(
            A=self.agent.A,
            B=self.agent.B,
            C=self.agent.C,
            qs=qs,
            policies=self.agent.policies,
            phantom_sequence=phantom_seq,
            A_deps=BATTERY_A_DEPS,
            energy_factor_idx=2,
            hvac_delta_bins=self._hvac_delta_bins,
            gamma=self.gamma,
            tou_schedule=tou_schedule,
        )

        # --- Action selection ---
        best_policy_idx = int(np.argmax(q_pi))
        policies_np = np.asarray(self.agent.policies)
        action_int = int(policies_np[best_policy_idx, 0, 0])

        # --- Update empirical prior ---
        action_arr = jnp.array([[action_int, 0, 0]])
        self._empirical_prior = _update_empirical_prior_compat(self.agent, action_arr, qs)

        self.qs_history.append(qs)
        self.efe_history.append(neg_efe)

        # --- Extract q(SoC) for sharing ---
        q_soc = np.asarray(qs[0][0, 0, :])

        return action_int, {
            "q_pi": q_pi,
            "neg_efe": neg_efe,
            "phantom_p_hvac": phantom_seq[0] if phantom_seq else np.array([0, 0, 1.0]),
            "q_soc": q_soc.copy(),
            "tom_reliability": self.thermo_tom.reliability,
        }


# ======================================================================
# ToM + Belief Sharing Agents (Phase 3)
# ======================================================================

class ToMThermostatAgent:
    """Active inference thermostat with Theory of Mind + belief sharing.

    Extends ThermostatAgent with:
    - Receives battery's q(SoC) as auditory observation
    - Maintains BatteryToM filter for reliability tracking
    - Shares own q(comfort) derived from q(room_temp) posterior
    - Social C vector on auditory modality scales with ToM reliability

    Following Friston et al. (2023) "Federated inference and belief sharing".
    """

    def __init__(self, env_data: dict, policy_len: int = 4,
                 gamma: float = 16.0, learn_B: bool = False,
                 social_weight: float = 1.0,
                 auditory_mode: str = "full",
                 forecast_data: dict = None,
                 use_states_info_gain: bool = True,
                 comfort_scale: float = 1.0):
        """
        Parameters
        ----------
        auditory_mode : str
            "full"    — structured auditory A matrix (original behavior)
            "uniform" — auditory A is uniform (no info gain from auditory)
            "none"    — no auditory modality (standard 4-modality model)
        forecast_data : dict, optional
            Noisy forecast data for predictive B matrices. If None, uses env_data.
        comfort_scale : float
            Multiplier for comfort C amplitudes. Default 1.0.
        """
        from .tom import BatteryToM, belief_to_comfort, belief_to_obs

        self._auditory_mode = auditory_mode
        self._comfort_scale = comfort_scale
        comfort_amplitude = -4.0 * comfort_scale

        if auditory_mode == "none":
            # Use standard thermostat model (4 modalities, no auditory)
            model = build_thermostat_model(env_data, policy_len=policy_len,
                                           comfort_amplitude=comfort_amplitude)
            pA = None
            pB = None
            if learn_B:
                pB = [b + 1.0 for b in model["B"]]
        else:
            model = build_thermostat_model_tom(
                env_data, policy_len=policy_len, social_weight=social_weight,
                comfort_amplitude=comfort_amplitude,
            )
            if auditory_mode == "uniform":
                # Replace auditory A with uniform distribution → zero info gain
                n_obs = model["A"][4].shape[0]
                n_states = model["A"][4].shape[1]
                model["A"][4] = np.ones((n_obs, n_states)) / n_obs

            pB = None
            if learn_B:
                pB = [b + 1.0 for b in model["B"]]

            pA = [a * 1000.0 + 0.1 for a in model["A"][:4]] + [model["A"][4] + 1.0]

        self.agent = Agent(
            A=model["A"],
            B=model["B"],
            C=model["C"],
            D=model["D"],
            pA=pA,
            pB=pB,
            A_dependencies=model["A_dependencies"],
            B_dependencies=model["B_dependencies"],
            control_fac_idx=[0],
            policy_len=policy_len,
            gamma=gamma,
            use_utility=True,
            use_states_info_gain=use_states_info_gain,
            use_param_info_gain=False,
            action_selection="deterministic",
            inference_algo="fpi",
            num_iter=16,
            batch_size=1,
            learn_A=(auditory_mode != "none"),
            learn_B=learn_B,
        )
        self.learn_B = learn_B
        self.social_weight = social_weight
        self._env_data = env_data
        self._forecast_data = forecast_data if forecast_data is not None else env_data
        self._empirical_prior = self.agent.D
        self._rng_key = jr.PRNGKey(10)
        self.qs_history = []
        self.efe_history = []
        self._qs_prev = None
        self._action_prev = None

        # ToM filter: thermostat's model of battery
        self.battery_tom = BatteryToM(n_states=SOC_LEVELS)

        # Belief sharing output (updated each step)
        self._last_q_comfort = np.ones(COMFORT_LEVELS) / COMFORT_LEVELS
        if auditory_mode != "none":
            self._base_social_C = model["C"][4].copy()
        else:
            self._base_social_C = None

    def step(self, obs_dict: dict, received_q_soc: np.ndarray = None,
             step_idx: int = None) -> tuple:
        """Run one inference step with belief sharing.

        Parameters
        ----------
        obs_dict : dict
            Keys: room_temp, outdoor_temp, occupancy, tou_high (all int indices)
        received_q_soc : np.ndarray, optional
            Battery's shared q(SoC) posterior, shape (5,). None on first step.
        step_idx : int, optional
            Current absolute simulation step. When provided, updates exogenous
            B matrices with 1-step-ahead forecasts (predictive B).

        Returns
        -------
        action : int
            HVAC action (0=cool, 1=heat, 2=off)
        hvac_energy : float
            Energy consumption in kWh
        info : dict
            Contains q_pi, neg_efe, q_comfort (shared belief), tom_reliability
        """
        from .tom import belief_to_comfort, belief_to_obs

        # --- Predictive B matrices for exogenous factors ---
        if step_idx is not None:
            B1, B2, B3 = _build_thermo_predictive_B(self._forecast_data, step_idx)
            new_B = list(self.agent.B)
            new_B[1] = jnp.array(B1)[None, :]
            new_B[2] = jnp.array(B2)[None, :]
            new_B[3] = jnp.array(B3)[None, :]
            self.agent = eqx.tree_at(lambda a: a.B, self.agent, new_B)

        # --- Process received belief from battery ---
        if received_q_soc is not None:
            batt_obs = belief_to_obs(received_q_soc)
            self.battery_tom.update(received_q_soc, batt_obs)

            # Scale social C by ToM reliability (only if auditory modality exists)
            if self._base_social_C is not None:
                reliability = self.battery_tom.reliability
                scaled_social_C = self._base_social_C * reliability
                new_C = list(self.agent.C)
                new_C[4] = jnp.array(scaled_social_C)[None, :]
                self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)
        else:
            batt_obs = SOC_LEVELS // 2  # neutral prior: mid SoC

        # --- Dynamic C for room temp (aligned mode, always on for ToM agent) ---
        occupancy = obs_dict.get("occupancy", 1)
        tou_high = obs_dict.get("tou_high", 0)
        target = TARGET_TEMP_OCCUPIED if occupancy else TARGET_TEMP_UNOCCUPIED
        target_idx = discretize_temp(target)
        amplitude = (-3.0 if tou_high else -4.0) * self._comfort_scale
        c_room = np.zeros(TEMP_LEVELS)
        for i in range(TEMP_LEVELS):
            dist = abs(i - target_idx)
            c_room[i] = amplitude * dist ** 2 / 4.0
        c_room -= c_room.max()
        new_C = list(self.agent.C)
        new_C[0] = jnp.array(c_room)[None, :]
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)

        # --- Observations ---
        if self._auditory_mode == "none":
            obs = [
                jnp.array([[obs_dict["room_temp"]]]),
                jnp.array([[obs_dict["outdoor_temp"]]]),
                jnp.array([[obs_dict["occupancy"]]]),
                jnp.array([[obs_dict["tou_high"]]]),
            ]
        else:
            obs = [
                jnp.array([[obs_dict["room_temp"]]]),
                jnp.array([[obs_dict["outdoor_temp"]]]),
                jnp.array([[obs_dict["occupancy"]]]),
                jnp.array([[obs_dict["tou_high"]]]),
                jnp.array([[batt_obs]]),
            ]

        qs = self.agent.infer_states(obs, self._empirical_prior)
        q_pi, neg_efe = self.agent.infer_policies(qs)

        self._rng_key, subkey = jr.split(self._rng_key)
        action = self.agent.sample_action(q_pi)
        action_int = int(action[0, 0])

        # Online parameter learning (A and/or B)
        if self._qs_prev is not None:
            beliefs_seq = jtu.tree_map(
                lambda prev, curr: jnp.concatenate([prev, curr], axis=1),
                self._qs_prev, qs
            )
            self.agent = _infer_parameters_compat(
                self.agent,
                beliefs_A=qs, outcomes=obs,
                actions=self._action_prev, beliefs_B=beliefs_seq, lr_pB=1.0,
            )

        self._qs_prev = qs
        self._action_prev = action

        self._empirical_prior = _update_empirical_prior_compat(self.agent, action, qs)

        q_pi_np = np.asarray(q_pi)
        neg_efe_np = np.asarray(neg_efe)

        self.qs_history.append(qs)
        self.efe_history.append(neg_efe_np)

        # --- Extract q(room_temp) and project to q(comfort) for sharing ---
        # qs[0] is q(room_temp), shape (batch=1, T=1, TEMP_LEVELS)
        q_room = np.asarray(qs[0][0, 0, :])  # (TEMP_LEVELS,)
        self._last_q_comfort = belief_to_comfort(q_room, target_idx)

        hvac_energy = HVAC_KWH_PER_STEP if action_int != 2 else 0.0

        return action_int, hvac_energy, {
            "q_pi": q_pi_np,
            "neg_efe": neg_efe_np,
            "q_comfort": self._last_q_comfort.copy(),
            "tom_reliability": self.battery_tom.reliability,
        }

    @property
    def q_comfort(self) -> np.ndarray:
        """Current shared comfort belief."""
        return self._last_q_comfort.copy()


class ToMBatteryAgent:
    """Active inference battery with Theory of Mind + belief sharing.

    Extends BatteryAgent with:
    - Receives thermostat's q(comfort) as auditory observation
    - Maintains ThermostatToM filter for reliability tracking
    - Shares own q(SoC) directly (already 5 levels)
    - Social C vector on auditory modality scales with ToM reliability

    Following Friston et al. (2023) "Federated inference and belief sharing".
    """

    C_SOC_PEAK = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    C_SOC_OFFPEAK = np.array([-1.5, -0.5, 0.0, 0.5, 1.0])

    def __init__(self, env_data: dict, policy_len: int = 4,
                 gamma: float = 16.0, initial_soc: float = 0.5,
                 social_weight: float = 1.0, learn_B: bool = False,
                 auditory_mode: str = "full",
                 forecast_data: dict = None,
                 use_states_info_gain: bool = True,
                 soc_scale: float = 1.0):
        """
        Parameters
        ----------
        auditory_mode : str
            "full"    — structured auditory A matrix (original behavior)
            "uniform" — auditory A is uniform (no info gain from auditory)
            "none"    — no auditory modality (standard 3-modality model)
        forecast_data : dict, optional
            Noisy forecast data for predictive B matrices. If None, uses env_data.
        soc_scale : float
            Multiplier for SoC C preference values. Default 1.0.
        """
        from .tom import ThermostatToM

        self._auditory_mode = auditory_mode
        self._soc_scale = soc_scale
        self._c_soc_peak = np.array([v * soc_scale for v in [2.0, 1.0, 0.0, -1.0, -2.0]])
        self._c_soc_offpeak = np.array([v * soc_scale for v in [-1.5, -0.5, 0.0, 0.5, 1.0]])

        if auditory_mode == "none":
            # Use standard battery model (3 modalities, no auditory)
            model = build_battery_model(env_data, initial_soc=initial_soc,
                                        policy_len=policy_len,
                                        soc_scale=soc_scale)
            pA = None
            pB = None
            if learn_B:
                pB = [b + 1.0 for b in model["B"]]
        else:
            model = build_battery_model_tom(
                env_data, initial_soc=initial_soc,
                policy_len=policy_len, social_weight=social_weight,
                soc_scale=soc_scale,
            )
            if auditory_mode == "uniform":
                # Replace auditory A with uniform distribution → zero info gain
                n_obs = model["A"][3].shape[0]
                n_states = model["A"][3].shape[1]
                model["A"][3] = np.ones((n_obs, n_states)) / n_obs

            pB = None
            if learn_B:
                pB = [b + 1.0 for b in model["B"]]

            pA = [a * 1000.0 + 0.1 for a in model["A"][:3]] + [model["A"][3] + 1.0]

        self.agent = Agent(
            A=model["A"],
            B=model["B"],
            C=model["C"],
            D=model["D"],
            pA=pA,
            pB=pB,
            A_dependencies=model["A_dependencies"],
            B_dependencies=model["B_dependencies"],
            control_fac_idx=[0],
            policy_len=policy_len,
            gamma=gamma,
            use_utility=True,
            use_states_info_gain=use_states_info_gain,
            use_param_info_gain=False,
            action_selection="deterministic",
            inference_algo="fpi",
            num_iter=16,
            batch_size=1,
            learn_A=(auditory_mode != "none"),
            learn_B=learn_B,
        )
        self.learn_B = learn_B
        self.social_weight = social_weight
        self._env_data = env_data
        self._forecast_data = forecast_data if forecast_data is not None else env_data
        self._empirical_prior = self.agent.D
        self._rng_key = jr.PRNGKey(11)
        self.qs_history = []
        self.efe_history = []
        self._qs_prev = None
        self._action_prev = None

        # ToM filter: battery's model of thermostat
        self.thermo_tom = ThermostatToM(n_states=COMFORT_LEVELS)

        # Belief sharing output (updated each step)
        self._last_q_soc = np.ones(SOC_LEVELS) / SOC_LEVELS
        if auditory_mode != "none":
            self._base_social_C = model["C"][3].copy()
        else:
            self._base_social_C = None

    def step(self, obs_dict: dict, received_q_comfort: np.ndarray = None,
             step_idx: int = None) -> tuple:
        """Run one inference step with belief sharing.

        Parameters
        ----------
        obs_dict : dict
            Keys: soc, cost, ghg, tou_high (all int indices)
        received_q_comfort : np.ndarray, optional
            Thermostat's shared q(comfort) posterior, shape (5,). None on first step.
        step_idx : int, optional
            Current absolute simulation step. When provided, updates TOU
            B matrix with 1-step-ahead forecast (predictive B).

        Returns
        -------
        action : int
            Battery action (0=charge, 1=discharge, 2=off)
        info : dict
            Contains q_pi, neg_efe, q_soc (shared belief), tom_reliability
        """
        from .tom import belief_to_obs

        # --- Predictive TOU B matrix ---
        if step_idx is not None:
            B1_new = _build_battery_predictive_tou(self._forecast_data, step_idx)
            new_B = list(self.agent.B)
            new_B[1] = jnp.array(B1_new)[None, :]
            self.agent = eqx.tree_at(lambda a: a.B, self.agent, new_B)

        # --- Process received belief from thermostat ---
        if received_q_comfort is not None:
            comfort_obs = belief_to_obs(received_q_comfort)
            self.thermo_tom.update(received_q_comfort, comfort_obs)

            # Scale social C by ToM reliability (only if auditory modality exists)
            if self._base_social_C is not None:
                reliability = self.thermo_tom.reliability
                scaled_social_C = self._base_social_C * reliability
                new_C = list(self.agent.C)
                new_C[3] = jnp.array(scaled_social_C)[None, :]
                self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)
        else:
            comfort_obs = COMFORT_LEVELS // 2  # neutral: COMFY

        # --- Dynamic SoC C (TOU arbitrage, always on for ToM agent) ---
        tou_high = obs_dict.get("tou_high", 0)
        if tou_high:
            new_c0 = jnp.array(self._c_soc_peak)
        else:
            new_c0 = jnp.array(self._c_soc_offpeak)
        new_C = list(self.agent.C)
        new_C[0] = new_c0[None, :]
        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)

        # --- Observations ---
        if self._auditory_mode == "none":
            obs = [
                jnp.array([[obs_dict["soc"]]]),
                jnp.array([[obs_dict["cost"]]]),
                jnp.array([[obs_dict["ghg"]]]),
            ]
        else:
            obs = [
                jnp.array([[obs_dict["soc"]]]),
                jnp.array([[obs_dict["cost"]]]),
                jnp.array([[obs_dict["ghg"]]]),
                jnp.array([[comfort_obs]]),
            ]

        qs = self.agent.infer_states(obs, self._empirical_prior)
        q_pi, neg_efe = self.agent.infer_policies(qs)

        self._rng_key, subkey = jr.split(self._rng_key)
        action = self.agent.sample_action(q_pi)
        action_int = int(action[0, 0])

        # Online parameter learning (A and/or B) — only if there's something to learn
        if self._qs_prev is not None and (self._auditory_mode != "none" or self.learn_B):
            beliefs_seq = jtu.tree_map(
                lambda prev, curr: jnp.concatenate([prev, curr], axis=1),
                self._qs_prev, qs
            )
            self.agent = _infer_parameters_compat(
                self.agent,
                beliefs_A=qs, outcomes=obs,
                actions=self._action_prev, beliefs_B=beliefs_seq, lr_pB=1.0,
            )

        self._qs_prev = qs
        self._action_prev = action

        self._empirical_prior = _update_empirical_prior_compat(self.agent, action, qs)

        q_pi_np = np.asarray(q_pi)
        neg_efe_np = np.asarray(neg_efe)

        self.qs_history.append(qs)
        self.efe_history.append(neg_efe_np)

        # --- Extract q(SoC) for sharing ---
        # qs[0] is q(soc), shape (batch=1, T=1, 5)
        self._last_q_soc = np.asarray(qs[0][0, 0, :])  # (5,)

        return action_int, {
            "q_pi": q_pi_np,
            "neg_efe": neg_efe_np,
            "q_soc": self._last_q_soc.copy(),
            "tom_reliability": self.thermo_tom.reliability,
        }

    @property
    def q_soc(self) -> np.ndarray:
        """Current shared SoC belief."""
        return self._last_q_soc.copy()
