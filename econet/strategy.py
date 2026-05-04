"""Hierarchical strategy layer: high-level agent + observation aggregator.

The StrategyAgent operates on 12-hour epochs (6 low-level 2h steps).
It selects among 5 strategies that set C-vector profiles for the
low-level thermostat and battery agents.

Architecture:
    State factors: [energy_regime(4), demand_phase(3)]  = 12 states
    Observations:  [cost_trend(5), comfort_trend(4), soc_state(3)]
    Actions:       5 strategies
    Policy len:    2 (24h lookahead = 2 high-level steps)
"""

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx

from pymdp.agent import Agent

from .environment import (
    TARGET_TEMP_OCCUPIED, TEMP_LEVELS, TEMP_MIN,
    SOC_LEVELS, COST_LEVELS, GHG_LEVELS,
    discretize_temp,
)
from .generative_model import build_thermostat_C, build_battery_C

# ======================================================================
# Strategy definitions
# ======================================================================

STRATEGY_NAMES = [
    "MAXIMIZE_COMFORT",
    "BALANCED",
    "MINIMIZE_COST",
    "ECO",
    "PEAK_SHAVE",
]
NUM_STRATEGIES = len(STRATEGY_NAMES)

# --- High-level state/obs dimensions ---
NUM_ENERGY_REGIMES = 4   # COMFORT_FIRST, BALANCED, COST_SAVING, ECO
NUM_DEMAND_PHASES = 3    # PEAK, OFF_PEAK, SHOULDER
NUM_COST_TRENDS = 5      # very_low, low, medium, high, very_high
NUM_COMFORT_TRENDS = 4   # excellent, good, fair, poor
NUM_SOC_STATES = 3       # low, mid, high

STRATEGY_NUM_STATES = [NUM_ENERGY_REGIMES, NUM_DEMAND_PHASES]
STRATEGY_NUM_OBS = [NUM_COST_TRENDS, NUM_COMFORT_TRENDS, NUM_SOC_STATES]
STRATEGY_A_DEPS = [[0, 1], [0], [1]]
STRATEGY_B_DEPS = [[0], [1]]
STRATEGY_NUM_CONTROLS = [NUM_STRATEGIES, 1]  # only factor 0 controlled


# ======================================================================
# C-vector profiles: strategy -> low-level C vector overrides
# ======================================================================

def _build_thermo_C_profile(sigma: float, amplitude: float,
                             target_offset: float = 0.0) -> np.ndarray:
    """Build thermostat room-temp C vector with given width and amplitude.

    Parameters
    ----------
    sigma : float
        Width of Gaussian preference (larger = wider comfort band).
    amplitude : float
        Scaling of preference (more negative = stronger discomfort for deviation).
    target_offset : float
        Shift target temperature (e.g., -2 for precooling).
    """
    target = TARGET_TEMP_OCCUPIED + target_offset
    target_idx = discretize_temp(target)
    c = np.zeros(TEMP_LEVELS)
    for i in range(TEMP_LEVELS):
        dist = abs(i - target_idx)
        c[i] = amplitude * dist ** 2 / (sigma ** 2)
    c -= c.max()
    return c


# Pre-build all thermostat C profiles
# Amplitude must dominate info gain (~1-3 nats). -2.0 at sigma=2 gives
# -2.0 at 2°C dev, -4.5 at 3°C — strong enough to drive tracking.
THERMO_C_PROFILES = {
    "MAXIMIZE_COMFORT": {"sigma": 1.5, "amplitude": -3.0},   # very tight
    "BALANCED":         {"sigma": 2.0, "amplitude": -2.0},   # matches default
    "MINIMIZE_COST":    {"sigma": 4.0, "amplitude": -0.8},   # relaxed but not passive
    "ECO":              {"sigma": 3.0, "amplitude": -1.5},
    "PEAK_SHAVE":       {"sigma": 2.0, "amplitude": -2.0, "target_offset": -2},
}

BATTERY_C_PROFILES = {
    "MAXIMIZE_COMFORT": {"cost_weight": -3.0, "ghg_weight": -1.5},
    "BALANCED":         {"cost_weight": -6.0, "ghg_weight": -3.0},
    "MINIMIZE_COST":    {"cost_weight": -8.0, "ghg_weight": -2.0},
    "ECO":              {"cost_weight": -3.0, "ghg_weight": -7.0},
    "PEAK_SHAVE":       {"cost_weight": -6.0, "ghg_weight": -3.0,
                         "soc_peak": np.array([3.0, 2.0, 0.0, -1.5, -2.5])},
}


def build_thermo_C_for_strategy(strategy_name: str) -> list:
    """Build full thermostat C vector list for a given strategy."""
    params = THERMO_C_PROFILES[strategy_name]
    c_room = _build_thermo_C_profile(**params)
    # Other modalities: flat (no preference)
    return [c_room, np.zeros(TEMP_LEVELS), np.zeros(2), np.zeros(2)]


def build_battery_C_for_strategy(strategy_name: str) -> list:
    """Build full battery C vector list for a given strategy."""
    params = BATTERY_C_PROFILES[strategy_name]

    # SoC preference — default or custom for PEAK_SHAVE
    if "soc_peak" in params:
        c_soc = params["soc_peak"].copy()
    else:
        # Default SoC preference (prefer high SoC)
        c_soc = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])

    # Cost preference
    c_cost = np.zeros(COST_LEVELS)
    for i in range(COST_LEVELS):
        c_cost[i] = params["cost_weight"] * i / (COST_LEVELS - 1)

    # GHG preference
    c_ghg = np.zeros(GHG_LEVELS)
    for i in range(GHG_LEVELS):
        c_ghg[i] = params["ghg_weight"] * i / (GHG_LEVELS - 1)

    return [c_soc, c_cost, c_ghg]


def apply_strategy(strategy_idx: int, thermo_agent, battery_agent):
    """Apply a strategy's C vectors to both low-level agents.

    Parameters
    ----------
    strategy_idx : int
        Index into STRATEGY_NAMES (0-4).
    thermo_agent : ThermostatAgent
        Low-level thermostat agent.
    battery_agent : BatteryAgent
        Low-level battery agent.
    """
    name = STRATEGY_NAMES[strategy_idx]
    thermo_C = build_thermo_C_for_strategy(name)
    battery_C = build_battery_C_for_strategy(name)
    thermo_agent.update_C(thermo_C)
    battery_agent.update_C(battery_C)


# ======================================================================
# High-level generative model
# ======================================================================

def build_strategy_A() -> list:
    """Build A matrices for the strategy agent.

    A_dependencies = [[0, 1], [0], [1]]
        A[0]: (5, 4, 3)  -- P(o_cost_trend | energy_regime, demand_phase)
        A[1]: (4, 4)      -- P(o_comfort_trend | energy_regime)
        A[2]: (3, 3)      -- P(o_soc_state | demand_phase)
    """
    A = []

    # A[0]: cost_trend depends on energy_regime AND demand_phase
    # Shape: (NUM_COST_TRENDS, NUM_ENERGY_REGIMES, NUM_DEMAND_PHASES)
    A_cost = np.zeros((NUM_COST_TRENDS, NUM_ENERGY_REGIMES, NUM_DEMAND_PHASES))
    # COMFORT_FIRST(0): higher cost, especially during peak
    # BALANCED(1): medium cost
    # COST_SAVING(2): low cost
    # ECO(3): medium cost (GHG-focused, not cost-focused)
    cost_means = {
        # (regime, phase): expected cost_trend index
        (0, 0): 4.0, (0, 1): 2.5, (0, 2): 3.0,  # COMFORT_FIRST
        (1, 0): 3.0, (1, 1): 1.5, (1, 2): 2.0,  # BALANCED
        (2, 0): 2.0, (2, 1): 0.5, (2, 2): 1.0,  # COST_SAVING
        (3, 0): 3.0, (3, 1): 1.5, (3, 2): 2.0,  # ECO
    }
    for (regime, phase), mean in cost_means.items():
        for o in range(NUM_COST_TRENDS):
            A_cost[o, regime, phase] = np.exp(-0.5 * ((o - mean) / 1.0) ** 2)
    # Normalize over observation dimension
    A_cost /= A_cost.sum(axis=0, keepdims=True)
    A.append(A_cost)

    # A[1]: comfort_trend depends on energy_regime only
    # Shape: (NUM_COMFORT_TRENDS, NUM_ENERGY_REGIMES)
    A_comfort = np.zeros((NUM_COMFORT_TRENDS, NUM_ENERGY_REGIMES))
    # COMFORT_FIRST → excellent(0), BALANCED → good(1),
    # COST_SAVING → fair(2), ECO → good(1)
    comfort_means = {0: 0.3, 1: 1.2, 2: 2.5, 3: 1.5}
    for regime, mean in comfort_means.items():
        for o in range(NUM_COMFORT_TRENDS):
            A_comfort[o, regime] = np.exp(-0.5 * ((o - mean) / 0.8) ** 2)
    A_comfort /= A_comfort.sum(axis=0, keepdims=True)
    A.append(A_comfort)

    # A[2]: soc_state depends on demand_phase only
    # Shape: (NUM_SOC_STATES, NUM_DEMAND_PHASES)
    A_soc = np.zeros((NUM_SOC_STATES, NUM_DEMAND_PHASES))
    # PEAK(0): low SoC (discharged), OFF_PEAK(1): high SoC (charged),
    # SHOULDER(2): mid SoC
    soc_means = {0: 0.5, 1: 2.0, 2: 1.2}
    for phase, mean in soc_means.items():
        for o in range(NUM_SOC_STATES):
            A_soc[o, phase] = np.exp(-0.5 * ((o - mean) / 0.7) ** 2)
    A_soc /= A_soc.sum(axis=0, keepdims=True)
    A.append(A_soc)

    return A


def build_strategy_B() -> list:
    """Build B matrices for the strategy agent.

    B_dependencies = [[0], [1]]
        B[0]: (4, 4, 5)  -- P(regime' | regime, strategy_action)
        B[1]: (3, 3, 1)  -- P(phase' | phase) — exogenous, identity
    """
    B = []

    # B[0]: energy_regime transitions under strategy actions
    # Weak prior (near-uniform + small self-transition bias)
    # Agent learns actual transition dynamics via Dirichlet updates
    B_regime = np.ones((NUM_ENERGY_REGIMES, NUM_ENERGY_REGIMES, NUM_STRATEGIES)) * 0.1

    # Each strategy biases toward its corresponding regime
    # Strategy 0 (MAXIMIZE_COMFORT) → regime 0 (COMFORT_FIRST)
    # Strategy 1 (BALANCED) → regime 1 (BALANCED)
    # Strategy 2 (MINIMIZE_COST) → regime 2 (COST_SAVING)
    # Strategy 3 (ECO) → regime 3 (ECO)
    # Strategy 4 (PEAK_SHAVE) → regime 2 (COST_SAVING) with some regime 1
    strategy_to_regime = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2}
    for s_action, target_regime in strategy_to_regime.items():
        for prev_regime in range(NUM_ENERGY_REGIMES):
            B_regime[target_regime, prev_regime, s_action] += 0.5
            # Small self-transition bias
            B_regime[prev_regime, prev_regime, s_action] += 0.2

    # Normalize
    for prev_r in range(NUM_ENERGY_REGIMES):
        for a in range(NUM_STRATEGIES):
            col = B_regime[:, prev_r, a]
            B_regime[:, prev_r, a] = col / col.sum()
    B.append(B_regime)

    # B[1]: demand_phase — exogenous (identity with single no-op action)
    B.append(np.eye(NUM_DEMAND_PHASES).reshape(NUM_DEMAND_PHASES, NUM_DEMAND_PHASES, 1))

    return B


def build_strategy_C() -> list:
    """Build C vectors (meta-preferences) for strategy agent.

    c_cost_trend: prefer low aggregate cost
    c_comfort_trend: prefer excellent comfort
    c_soc_state: slight preference for mid/full SoC
    """
    # Cost trend: prefer low cost (index 0 = very_low)
    c_cost = np.array([0.0, -0.5, -1.5, -3.0, -5.0])

    # Comfort trend: prefer excellent (index 0 = excellent)
    c_comfort = np.array([0.0, -0.5, -2.0, -4.0])

    # SoC state: slight preference for mid/high
    c_soc = np.array([-0.5, 0.0, 0.3])

    return [c_cost, c_comfort, c_soc]


def build_strategy_D(initial_regime: int = 1,
                     initial_phase: int = 1) -> list:
    """Build D vectors for strategy agent.

    Defaults: BALANCED regime, OFF_PEAK phase.
    """
    d0 = np.zeros(NUM_ENERGY_REGIMES)
    d0[initial_regime] = 1.0

    d1 = np.zeros(NUM_DEMAND_PHASES)
    d1[initial_phase] = 1.0

    return [d0, d1]


# ======================================================================
# ObservationAggregator
# ======================================================================

class ObservationAggregator:
    """Summarizes low-level step records into high-level observations.

    Collects StepRecord objects from the low-level simulation and
    produces discretized observations for the StrategyAgent.
    """

    def __init__(self):
        self.records = []

    def add(self, record):
        """Add a StepRecord from the low-level simulation."""
        self.records.append(record)

    def reset(self):
        """Clear accumulated records."""
        self.records = []

    def summarize(self) -> dict:
        """Aggregate records into high-level observation dict.

        Returns
        -------
        dict with keys: cost_trend, comfort_trend, soc_state, demand_phase
        """
        if not self.records:
            return {
                "cost_trend": 2,      # medium
                "comfort_trend": 1,   # good
                "soc_state": 1,       # mid
                "demand_phase": 1,    # off_peak
            }

        total_cost = sum(r.cost for r in self.records)
        mean_comfort_dev = np.mean([
            abs(r.room_temp - r.target_temp) for r in self.records
        ])
        final_soc = self.records[-1].soc

        # Discretize cost trend: bins at [0.3, 0.6, 0.9, 1.2]
        cost_trend = int(np.clip(
            np.digitize(total_cost, [0.3, 0.6, 0.9, 1.2]),
            0, NUM_COST_TRENDS - 1
        ))

        # Discretize comfort trend: bins at [0.5, 1.5, 3.0]
        # 0=excellent (<0.5), 1=good, 2=fair, 3=poor (>3.0)
        comfort_trend = int(np.clip(
            np.digitize(mean_comfort_dev, [0.5, 1.5, 3.0]),
            0, NUM_COMFORT_TRENDS - 1
        ))

        # Discretize SoC: bins at [0.3, 0.6]
        # 0=low (<0.3), 1=mid, 2=high (>0.6)
        soc_state = int(np.clip(
            np.digitize(final_soc, [0.3, 0.6]),
            0, NUM_SOC_STATES - 1
        ))

        # Demand phase from TOU: majority vote
        tou_sum = sum(r.tou_high for r in self.records)
        tou_frac = tou_sum / len(self.records)
        if tou_frac > 0.6:
            demand_phase = 0   # PEAK
        elif tou_frac < 0.2:
            demand_phase = 1   # OFF_PEAK
        else:
            demand_phase = 2   # SHOULDER

        return {
            "cost_trend": cost_trend,
            "comfort_trend": comfort_trend,
            "soc_state": soc_state,
            "demand_phase": demand_phase,
        }


# ======================================================================
# StrategyAgent
# ======================================================================

class StrategyAgent:
    """High-level active inference agent selecting energy management strategies.

    Operates on 12-hour epochs. Observes aggregated cost/comfort/SoC trends
    and selects one of 5 strategies that set C vectors for low-level agents.
    """

    def __init__(self, policy_len: int = 2, gamma: float = 8.0,
                 learn_B: bool = True):
        A = build_strategy_A()
        B = build_strategy_B()
        C = build_strategy_C()
        D = build_strategy_D()

        pB = None
        if learn_B:
            pB = [b + 1.0 for b in B]

        self.agent = Agent(
            A=A,
            B=B,
            C=C,
            D=D,
            pB=pB,
            A_dependencies=STRATEGY_A_DEPS,
            B_dependencies=STRATEGY_B_DEPS,
            control_fac_idx=[0],
            policy_len=policy_len,
            gamma=gamma,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=learn_B,
            action_selection="deterministic",
            inference_algo="fpi",
            num_iter=16,
            batch_size=1,
            learn_B=learn_B,
        )
        self.learn_B = learn_B
        self._empirical_prior = self.agent.D
        self._rng_key = jr.PRNGKey(2)
        self.qs_history = []
        self.efe_history = []

        # Online B-learning state
        self._qs_prev = None
        self._action_prev = None

    def step(self, obs_dict: dict) -> tuple:
        """Run one high-level inference step.

        Parameters
        ----------
        obs_dict : dict
            Keys: cost_trend, comfort_trend, soc_state, demand_phase

        Returns
        -------
        strategy_idx : int
            Selected strategy (0-4).
        info : dict
            Contains q_pi and neg_efe.
        """
        # Update exogenous factor (demand_phase) in empirical prior
        demand_phase = obs_dict.get("demand_phase", 1)
        d_phase = np.zeros(NUM_DEMAND_PHASES)
        d_phase[demand_phase] = 1.0
        # Inject demand_phase into empirical prior's factor 1
        new_prior = list(self._empirical_prior)
        new_prior[1] = jnp.array(d_phase)[None, :]  # (1, 3)
        self._empirical_prior = new_prior

        obs = [
            jnp.array([[obs_dict["cost_trend"]]]),
            jnp.array([[obs_dict["comfort_trend"]]]),
            jnp.array([[obs_dict["soc_state"]]]),
        ]

        qs = self.agent.infer_states(obs, self._empirical_prior)
        q_pi, neg_efe = self.agent.infer_policies(qs)

        self._rng_key, subkey = jr.split(self._rng_key)
        action = self.agent.sample_action(q_pi)

        action_int = int(action[0, 0])

        # Online B-learning
        if self.learn_B and self._qs_prev is not None:
            beliefs_seq = jtu.tree_map(
                lambda prev, curr: jnp.concatenate([prev, curr], axis=1),
                self._qs_prev, qs
            )
            self.agent = self.agent.infer_parameters(
                beliefs_A=beliefs_seq,
                outcomes=obs,
                actions=self._action_prev,
                beliefs_B=beliefs_seq,
                lr_pB=1.0,
            )

        if self.learn_B:
            self._qs_prev = qs
            self._action_prev = action

        pred, _ = self.agent.update_empirical_prior(action, qs)
        self._empirical_prior = pred

        q_pi_np = np.asarray(q_pi)
        neg_efe_np = np.asarray(neg_efe)

        self.qs_history.append(qs)
        self.efe_history.append(neg_efe_np)

        return action_int, {
            "q_pi": q_pi_np,
            "neg_efe": neg_efe_np,
        }
