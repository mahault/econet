"""Main simulation loop with cascading updates between thermostat and battery agents."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .environment import Environment, StepRecord, generate_multi_day, STEPS_PER_DAY
from .agents import (
    ThermostatAgent, BatteryAgent, ToMThermostatAgent, ToMBatteryAgent,
    SophisticatedBatteryAgent, SophisticatedToMBatteryAgent,
    SophisticatedThermostatAgent,
)


@dataclass
class SimulationResult:
    """Container for simulation results."""
    history: list = field(default_factory=list)
    thermo_efe_history: list = field(default_factory=list)
    battery_efe_history: list = field(default_factory=list)
    thermo_qpi_history: list = field(default_factory=list)
    battery_qpi_history: list = field(default_factory=list)
    num_days: int = 0
    policy_len: int = 4
    learn_B: bool = False

    @property
    def total_steps(self):
        return len(self.history)

    @property
    def total_cost(self):
        return sum(r.cost for r in self.history)

    @property
    def total_ghg(self):
        return sum(r.ghg for r in self.history)

    @property
    def total_energy(self):
        return sum(r.total_energy for r in self.history)

    @property
    def cumulative_temp_deviation(self):
        return sum(abs(r.room_temp - r.target_temp) for r in self.history)

    @property
    def comfort_violation_hours(self):
        """Hours where temp deviates > 2C from target while occupied."""
        violations = 0
        for r in self.history:
            if r.occupancy and abs(r.room_temp - r.target_temp) > 2.0:
                violations += 2  # 2-hour time steps
        return violations

    @property
    def avg_daily_cost(self):
        if self.num_days > 0:
            return self.total_cost / self.num_days
        return 0.0

    def to_arrays(self) -> dict:
        """Convert history to numpy arrays for plotting."""
        if not self.history:
            return {}
        fields = [
            "step", "day", "time_of_day", "outdoor_temp", "room_temp",
            "target_temp", "occupancy", "tou_high", "tou_value", "solar_gen",
            "baseline_load", "hvac_action", "hvac_energy", "battery_action",
            "battery_energy", "soc", "total_energy", "cost", "ghg",
        ]
        return {f: np.array([getattr(r, f) for r in self.history])
                for f in fields}


def run_simulation(env_data: Optional[dict] = None,
                   num_days: int = 2,
                   policy_len: int = 4,
                   gamma: float = 16.0,
                   learn_B: bool = False,
                   aligned: bool = True,
                   initial_room_temp: float = 20.0,
                   initial_soc: float = 0.5,
                   verbose: bool = True,
                   seed: int = 42) -> SimulationResult:
    """Run multi-agent EcoNet simulation.

    Parameters
    ----------
    env_data : dict, optional
        Pre-generated environment data. If None, generates synthetic data.
    num_days : int
        Number of days to simulate.
    policy_len : int
        Planning horizon for agents.
    gamma : float
        Policy precision (inverse temperature).
    learn_B : bool
        Whether the thermostat agent learns B matrix parameters.
        When True, pB Dirichlet concentrations update online every step
        via agent.infer_parameters (2-frame belief joints).
    aligned : bool
        Whether agents use dynamic C vectors for decentralized coordination.
        When True (default), thermostat adapts to occupancy+TOU and battery
        flips SoC preference by TOU — enabling implicit coordination.
        When False, both use static C from the generative model.
    initial_room_temp : float
        Starting room temperature in C.
    initial_soc : float
        Starting battery state of charge.
    verbose : bool
        Print progress.
    seed : int
        Random seed.

    Returns
    -------
    SimulationResult
    """
    # Generate data if not provided
    if env_data is None:
        env_data = generate_multi_day(num_days=num_days, seed=seed)

    total_steps = len(env_data["time_of_day"])

    # Initialize environment
    env = Environment(env_data, initial_room_temp=initial_room_temp,
                      initial_soc=initial_soc)

    # Initialize agents
    thermo = ThermostatAgent(env_data, policy_len=policy_len,
                             gamma=gamma, learn_B=learn_B, aligned=aligned)
    battery = BatteryAgent(env_data, policy_len=policy_len,
                           gamma=gamma, initial_soc=initial_soc, aligned=aligned)

    result = SimulationResult(
        num_days=num_days, policy_len=policy_len, learn_B=learn_B,
    )

    for step in range(total_steps):
        if verbose and step % STEPS_PER_DAY == 0:
            day = step // STEPS_PER_DAY
            print(f"  Day {day + 1}/{num_days} ...", flush=True)

        # 1. Get thermostat observations
        thermo_obs = env.get_thermostat_obs(step)

        # 2. Thermostat acts first (cascading)
        #    B-learning happens inline in step() when learn_B=True
        hvac_action, hvac_energy, thermo_info = thermo.step(thermo_obs, step_idx=step)

        # 3. Apply thermostat action to environment
        actual_hvac_energy = env.apply_thermostat(hvac_action, step)

        # 4. Get battery observations (includes HVAC energy effect)
        battery_obs = env.get_battery_obs(step, actual_hvac_energy)

        # 5. Battery acts second
        battery_action, battery_info = battery.step(battery_obs, step_idx=step)

        # 6. Apply battery action and record
        record = env.apply_battery(battery_action, step, actual_hvac_energy)
        # Fix hvac_action in record (apply_battery sets it to battery_action)
        record.hvac_action = hvac_action

        result.history.append(record)
        result.thermo_efe_history.append(thermo_info["neg_efe"])
        result.battery_efe_history.append(battery_info["neg_efe"])
        result.thermo_qpi_history.append(thermo_info["q_pi"])
        result.battery_qpi_history.append(battery_info["q_pi"])

    if verbose:
        print(f"  Simulation complete: {total_steps} steps, "
              f"cost=${result.total_cost:.2f}, "
              f"GHG={result.total_ghg:.2f} kg CO2")

    return result


def run_deterministic(num_days: int = 2, policy_len: int = 4, **kwargs) -> SimulationResult:
    """Run Scenario 1: deterministic 2-day simulation."""
    return run_simulation(num_days=num_days, policy_len=policy_len,
                          learn_B=False, **kwargs)


def run_learning(num_days: int = 40, policy_len: int = 4, **kwargs) -> SimulationResult:
    """Run Scenario 2: parameter learning 40-day simulation."""
    return run_simulation(num_days=num_days, policy_len=policy_len,
                          learn_B=True, **kwargs)


def run_hierarchical_simulation(
    env_data: Optional[dict] = None,
    num_days: int = 7,
    policy_len: int = 4,
    gamma: float = 16.0,
    learn_B: bool = True,
    initial_room_temp: float = 20.0,
    initial_soc: float = 0.5,
    verbose: bool = True,
    seed: int = 42,
) -> SimulationResult:
    """Run hierarchical two-level EcoNet simulation.

    High-level StrategyAgent ticks every 6 low-level steps (12h epochs).
    It aggregates observations from the previous epoch, infers a strategy,
    and injects corresponding C vectors into the low-level agents.

    Parameters
    ----------
    env_data : dict, optional
        Pre-generated environment data. If None, generates synthetic data.
    num_days : int
        Number of days to simulate.
    policy_len : int
        Planning horizon for low-level agents.
    gamma : float
        Policy precision (inverse temperature).
    learn_B : bool
        Whether agents learn B matrix parameters online.
    initial_room_temp : float
        Starting room temperature in C.
    initial_soc : float
        Starting battery state of charge.
    verbose : bool
        Print progress.
    seed : int
        Random seed.

    Returns
    -------
    SimulationResult
    """
    from .strategy import StrategyAgent, ObservationAggregator, apply_strategy

    # Generate data if not provided
    if env_data is None:
        env_data = generate_multi_day(num_days=num_days, seed=seed)

    total_steps = len(env_data["time_of_day"])
    HIGH_LEVEL_PERIOD = 6  # steps per high-level epoch (12h)

    # Initialize environment
    env = Environment(env_data, initial_room_temp=initial_room_temp,
                      initial_soc=initial_soc)

    # Initialize low-level agents
    thermo = ThermostatAgent(env_data, policy_len=policy_len,
                             gamma=gamma, learn_B=learn_B)
    battery = BatteryAgent(env_data, policy_len=policy_len,
                           gamma=gamma, initial_soc=initial_soc)

    # Initialize high-level agent
    strategy = StrategyAgent(learn_B=learn_B)
    aggregator = ObservationAggregator()

    result = SimulationResult(
        num_days=num_days, policy_len=policy_len, learn_B=learn_B,
    )

    strategy_history = []

    for step in range(total_steps):
        if verbose and step % STEPS_PER_DAY == 0:
            day = step // STEPS_PER_DAY
            print(f"  Day {day + 1}/{num_days} ...", flush=True)

        # High-level tick: every HIGH_LEVEL_PERIOD steps
        if step % HIGH_LEVEL_PERIOD == 0 and step > 0:
            # Aggregate previous epoch observations
            high_obs = aggregator.summarize()
            aggregator.reset()

            # High-level agent selects strategy
            strategy_idx, strategy_info = strategy.step(high_obs)
            strategy_history.append(strategy_idx)

            if verbose:
                from .strategy import STRATEGY_NAMES
                print(f"    [Strategy] epoch {step // HIGH_LEVEL_PERIOD}: "
                      f"{STRATEGY_NAMES[strategy_idx]}")

            # Inject C vectors into low-level agents
            apply_strategy(strategy_idx, thermo, battery)

        # 1. Get thermostat observations
        thermo_obs = env.get_thermostat_obs(step)

        # 2. Thermostat acts first (cascading)
        hvac_action, hvac_energy, thermo_info = thermo.step(thermo_obs, step_idx=step)

        # 3. Apply thermostat action to environment
        actual_hvac_energy = env.apply_thermostat(hvac_action, step)

        # 4. Get battery observations (includes HVAC energy effect)
        battery_obs = env.get_battery_obs(step, actual_hvac_energy)

        # 5. Battery acts second
        battery_action, battery_info = battery.step(battery_obs, step_idx=step)

        # 6. Apply battery action and record
        record = env.apply_battery(battery_action, step, actual_hvac_energy)
        record.hvac_action = hvac_action

        # Feed record to aggregator for high-level obs
        aggregator.add(record)

        result.history.append(record)
        result.thermo_efe_history.append(thermo_info["neg_efe"])
        result.battery_efe_history.append(battery_info["neg_efe"])
        result.thermo_qpi_history.append(thermo_info["q_pi"])
        result.battery_qpi_history.append(battery_info["q_pi"])

    if verbose:
        print(f"  Hierarchical simulation complete: {total_steps} steps, "
              f"cost=${result.total_cost:.2f}, "
              f"GHG={result.total_ghg:.2f} kg CO2")
        if strategy_history:
            from .strategy import STRATEGY_NAMES
            from collections import Counter
            counts = Counter(strategy_history)
            print("  Strategy distribution:")
            for s_idx, name in enumerate(STRATEGY_NAMES):
                print(f"    {name}: {counts.get(s_idx, 0)} epochs")

    return result


def run_tom_simulation(
    env_data: Optional[dict] = None,
    num_days: int = 7,
    policy_len: int = 4,
    gamma: float = 16.0,
    learn_B: bool = True,
    initial_room_temp: float = 20.0,
    initial_soc: float = 0.5,
    social_weight: float = 1.0,
    auditory_mode: str = "full",
    verbose: bool = True,
    seed: int = 42,
) -> SimulationResult:
    """Run ToM + Belief Sharing simulation.

    Agents share posterior beliefs each step via auditory observation modality.
    Cascade: thermostat receives battery's q(SoC) from t-1, shares q(comfort)
    at t; battery receives thermostat's q(comfort) at t, shares q(SoC) for t+1.

    Parameters
    ----------
    env_data : dict, optional
        Pre-generated environment data. If None, generates synthetic data.
    num_days : int
        Number of days to simulate.
    policy_len : int
        Planning horizon for agents.
    gamma : float
        Policy precision (inverse temperature).
    learn_B : bool
        Whether thermostat learns B matrix parameters online.
    initial_room_temp : float
        Starting room temperature in C.
    initial_soc : float
        Starting battery state of charge.
    social_weight : float
        Amplitude scaling for social C vectors (auditory modality preferences).
    verbose : bool
        Print progress.
    seed : int
        Random seed.

    Returns
    -------
    SimulationResult
    """
    if env_data is None:
        env_data = generate_multi_day(num_days=num_days, seed=seed)

    total_steps = len(env_data["time_of_day"])

    env = Environment(env_data, initial_room_temp=initial_room_temp,
                      initial_soc=initial_soc)

    thermo = ToMThermostatAgent(
        env_data, policy_len=policy_len, gamma=gamma,
        learn_B=learn_B, social_weight=social_weight,
        auditory_mode=auditory_mode,
    )
    # Battery never learns B — its transitions (charge/discharge/idle) are
    # deterministic.  Only the thermostat benefits from B-learning (outdoor
    # temperature dynamics).  Passing learn_B=True here was inflating
    # transition uncertainty and making the battery ~25% less active.
    battery = ToMBatteryAgent(
        env_data, policy_len=policy_len, gamma=gamma,
        initial_soc=initial_soc, social_weight=social_weight,
        learn_B=False, auditory_mode=auditory_mode,
    )

    result = SimulationResult(
        num_days=num_days, policy_len=policy_len, learn_B=learn_B,
    )

    # Belief sharing state (1-step delay for battery->thermostat)
    prev_q_soc = None  # battery's q(SoC) from previous step

    # Track communication history
    belief_history = {
        "q_comfort": [],
        "q_soc": [],
        "thermo_tom_reliability": [],
        "battery_tom_reliability": [],
    }

    for step in range(total_steps):
        if verbose and step % STEPS_PER_DAY == 0:
            day = step // STEPS_PER_DAY
            print(f"  Day {day + 1}/{num_days} ...", flush=True)

        # 1. Thermostat receives battery's q(SoC) from step t-1
        thermo_obs = env.get_thermostat_obs(step)

        # 2. Thermostat infers + acts + produces q(comfort)
        hvac_action, hvac_energy, thermo_info = thermo.step(
            thermo_obs, received_q_soc=prev_q_soc, step_idx=step
        )

        # 3. Apply thermostat action
        actual_hvac_energy = env.apply_thermostat(hvac_action, step)

        # 4. Battery receives thermostat's q(comfort) from this step (cascade)
        battery_obs = env.get_battery_obs(step, actual_hvac_energy)
        q_comfort_shared = thermo_info["q_comfort"]

        # 5. Battery infers + acts + produces q(SoC)
        battery_action, battery_info = battery.step(
            battery_obs, received_q_comfort=q_comfort_shared, step_idx=step
        )

        # 6. Store battery's q(SoC) for thermostat at step t+1
        prev_q_soc = battery_info["q_soc"]

        # 7. Apply battery action and record
        record = env.apply_battery(battery_action, step, actual_hvac_energy)
        record.hvac_action = hvac_action

        result.history.append(record)
        result.thermo_efe_history.append(thermo_info["neg_efe"])
        result.battery_efe_history.append(battery_info["neg_efe"])
        result.thermo_qpi_history.append(thermo_info["q_pi"])
        result.battery_qpi_history.append(battery_info["q_pi"])

        # Track communication
        belief_history["q_comfort"].append(q_comfort_shared)
        belief_history["q_soc"].append(prev_q_soc)
        belief_history["thermo_tom_reliability"].append(
            thermo_info["tom_reliability"]
        )
        belief_history["battery_tom_reliability"].append(
            battery_info["tom_reliability"]
        )

    # Attach communication history to result for analysis
    result.belief_history = belief_history

    if verbose:
        final_thermo_rel = belief_history["thermo_tom_reliability"][-1]
        final_battery_rel = belief_history["battery_tom_reliability"][-1]
        print(f"  ToM simulation complete: {total_steps} steps, "
              f"cost=${result.total_cost:.2f}, "
              f"GHG={result.total_ghg:.2f} kg CO2")
        print(f"  ToM reliability: thermo={final_thermo_rel:.3f}, "
              f"battery={final_battery_rel:.3f}")

    return result


def run_sophisticated_simulation(
    env_data: Optional[dict] = None,
    num_days: int = 7,
    policy_len: int = 4,
    gamma: float = 16.0,
    learn_B: bool = True,
    initial_room_temp: float = 20.0,
    initial_soc: float = 0.5,
    verbose: bool = True,
    seed: int = 42,
) -> SimulationResult:
    """Run sophisticated inference simulation (Pitliya et al., 2025).

    Pairs a standard aligned ThermostatAgent with a SophisticatedBatteryAgent
    that maintains a phantom thermostat for T-step HVAC prediction.  The
    battery's EFE computation uses step-dependent B[energy] matrices based
    on phantom predictions, enabling it to anticipate HVAC-driven costs.

    Cascade: thermostat FIRST -> env update -> battery SECOND (same as
    standard simulation).  The battery additionally runs a phantom rollout
    each step to predict future HVAC activity.

    Parameters
    ----------
    env_data : dict, optional
        Pre-generated environment data. If None, generates synthetic data.
    num_days : int
        Number of days to simulate.
    policy_len : int
        Planning horizon for agents.
    gamma : float
        Policy precision (inverse temperature).
    learn_B : bool
        Whether the thermostat agent learns B matrix parameters online.
    initial_room_temp : float
        Starting room temperature in C.
    initial_soc : float
        Starting battery state of charge.
    verbose : bool
        Print progress.
    seed : int
        Random seed.

    Returns
    -------
    SimulationResult
    """
    if env_data is None:
        env_data = generate_multi_day(num_days=num_days, seed=seed)

    total_steps = len(env_data["time_of_day"])

    env = Environment(env_data, initial_room_temp=initial_room_temp,
                      initial_soc=initial_soc)

    # Standard aligned thermostat (with optional B-learning)
    thermo = ThermostatAgent(env_data, policy_len=policy_len,
                             gamma=gamma, learn_B=learn_B, aligned=True)

    # Sophisticated battery: phantom thermostat inside
    battery = SophisticatedBatteryAgent(
        env_data, policy_len=policy_len, gamma=gamma,
        initial_soc=initial_soc,
    )

    result = SimulationResult(
        num_days=num_days, policy_len=policy_len, learn_B=learn_B,
    )

    # Track phantom predictions for diagnostics
    phantom_history = []

    for step in range(total_steps):
        if verbose and step % STEPS_PER_DAY == 0:
            day = step // STEPS_PER_DAY
            print(f"  Day {day + 1}/{num_days} ...", flush=True)

        # 1. Thermostat observes and acts (standard)
        thermo_obs = env.get_thermostat_obs(step)
        hvac_action, hvac_energy, thermo_info = thermo.step(thermo_obs)

        # 2. Apply thermostat action to environment
        actual_hvac_energy = env.apply_thermostat(hvac_action, step)

        # 3. Battery observes (includes HVAC energy effect)
        battery_obs = env.get_battery_obs(step, actual_hvac_energy)

        # 4. Battery acts with sophisticated inference (passes step_idx)
        battery_action, battery_info = battery.step(battery_obs, step_idx=step)

        # 5. Apply battery action and record
        record = env.apply_battery(battery_action, step, actual_hvac_energy)
        record.hvac_action = hvac_action

        result.history.append(record)
        result.thermo_efe_history.append(thermo_info["neg_efe"])
        result.battery_efe_history.append(battery_info["neg_efe"])
        result.thermo_qpi_history.append(thermo_info["q_pi"])
        result.battery_qpi_history.append(battery_info["q_pi"])

        phantom_history.append(battery_info.get("phantom_p_hvac", None))

    # Attach phantom history for analysis
    result.phantom_history = phantom_history

    if verbose:
        print(f"  Sophisticated simulation complete: {total_steps} steps, "
              f"cost=${result.total_cost:.2f}, "
              f"GHG={result.total_ghg:.2f} kg CO2")

    return result


def run_sophisticated_tom_simulation(
    env_data: Optional[dict] = None,
    num_days: int = 7,
    policy_len: int = 4,
    gamma: float = 16.0,
    learn_B: bool = True,
    initial_room_temp: float = 20.0,
    initial_soc: float = 0.5,
    social_weight: float = 1.0,
    thermo_aligned: bool = True,
    verbose: bool = True,
    seed: int = 42,
) -> SimulationResult:
    """Run sophisticated inference + belief sharing simulation.

    Pairs a ToMThermostatAgent (which shares q(comfort)) with a
    SophisticatedToMBatteryAgent (which uses phantom + received beliefs).

    Cascade:
    1. Thermostat receives battery's q(SoC) from t-1
    2. Thermostat infers, acts, produces q(comfort)
    3. Battery receives q(comfort), uses it to improve phantom,
       runs sophisticated EFE, acts, produces q(SoC)
    4. Store q(SoC) for thermostat at t+1

    Parameters
    ----------
    thermo_aligned : bool
        If True, use ToMThermostatAgent (aligned + comm).
        If False, thermostat still shares beliefs but without aligned C.
    """
    if env_data is None:
        env_data = generate_multi_day(num_days=num_days, seed=seed)

    total_steps = len(env_data["time_of_day"])

    env = Environment(env_data, initial_room_temp=initial_room_temp,
                      initial_soc=initial_soc)

    # Thermostat with belief sharing
    thermo = ToMThermostatAgent(
        env_data, policy_len=policy_len, gamma=gamma,
        learn_B=learn_B, social_weight=social_weight,
        auditory_mode="full",
    )

    # Sophisticated battery with belief sharing
    battery = SophisticatedToMBatteryAgent(
        env_data, policy_len=policy_len, gamma=gamma,
        initial_soc=initial_soc, social_weight=social_weight,
    )

    result = SimulationResult(
        num_days=num_days, policy_len=policy_len, learn_B=learn_B,
    )

    prev_q_soc = None
    belief_history = {
        "q_comfort": [],
        "q_soc": [],
        "thermo_tom_reliability": [],
        "battery_tom_reliability": [],
        "phantom_p_hvac": [],
    }

    for step in range(total_steps):
        if verbose and step % STEPS_PER_DAY == 0:
            day = step // STEPS_PER_DAY
            print(f"  Day {day + 1}/{num_days} ...", flush=True)

        # 1. Thermostat receives battery's q(SoC) from t-1, infers + acts
        thermo_obs = env.get_thermostat_obs(step)
        hvac_action, hvac_energy, thermo_info = thermo.step(
            thermo_obs, received_q_soc=prev_q_soc
        )

        # 2. Apply thermostat action
        actual_hvac_energy = env.apply_thermostat(hvac_action, step)

        # 3. Battery receives q(comfort), runs sophisticated inference
        battery_obs = env.get_battery_obs(step, actual_hvac_energy)
        q_comfort_shared = thermo_info["q_comfort"]

        battery_action, battery_info = battery.step(
            battery_obs, step_idx=step,
            received_q_comfort=q_comfort_shared,
        )

        # 4. Store q(SoC) for thermostat at t+1
        prev_q_soc = battery_info.get("q_soc", None)

        # 5. Apply battery action and record
        record = env.apply_battery(battery_action, step, actual_hvac_energy)
        record.hvac_action = hvac_action

        result.history.append(record)
        result.thermo_efe_history.append(thermo_info["neg_efe"])
        result.battery_efe_history.append(battery_info["neg_efe"])
        result.thermo_qpi_history.append(thermo_info["q_pi"])
        result.battery_qpi_history.append(battery_info["q_pi"])

        belief_history["q_comfort"].append(q_comfort_shared)
        belief_history["q_soc"].append(prev_q_soc)
        belief_history["thermo_tom_reliability"].append(
            thermo_info["tom_reliability"])
        belief_history["battery_tom_reliability"].append(
            battery_info.get("tom_reliability", 0.0))
        belief_history["phantom_p_hvac"].append(
            battery_info.get("phantom_p_hvac", None))

    result.belief_history = belief_history

    if verbose:
        print(f"  Sophisticated+ToM simulation complete: {total_steps} steps, "
              f"cost=${result.total_cost:.2f}, "
              f"GHG={result.total_ghg:.2f} kg CO2")

    return result


def run_full_sophisticated_simulation(
    env_data: Optional[dict] = None,
    num_days: int = 7,
    policy_len: int = 6,
    gamma: float = 16.0,
    learn_B: bool = True,
    initial_room_temp: float = 20.0,
    initial_soc: float = 0.5,
    cost_scale: float = 3.0,
    cost_weight: float = None,
    verbose: bool = True,
    seed: int = 42,
) -> SimulationResult:
    """Run symmetric sophisticated inference simulation.

    BOTH agents maintain phantoms of each other:
    - SophisticatedThermostatAgent: 5-factor Agent with cost modality
      (energy_demand factor + cost observation, driven by phantom battery)
    - SophisticatedBatteryAgent: phantom thermostat for HVAC prediction

    Cost awareness is principled: cost enters through the standard EFE
    via A_cost likelihood and C_cost preference — no additive penalty.

    Parameters
    ----------
    env_data : dict, optional
        Pre-generated environment data. If None, generates synthetic data.
    num_days : int
        Number of days to simulate.
    policy_len : int
        Planning horizon for agents.
    gamma : float
        Policy precision (inverse temperature).
    learn_B : bool
        Whether the thermostat agent learns B matrix parameters online.
    initial_room_temp : float
        Starting room temperature in C.
    initial_soc : float
        Starting battery state of charge.
    cost_scale : float
        Amplitude of thermostat's cost preference C[4]. Higher = more
        cost-sensitive. Default 3.0 (comfort C peaks ~4.0 nats).
    cost_weight : float, optional
        Deprecated alias for cost_scale. If provided, overrides cost_scale.
    verbose : bool
        Print progress.
    seed : int
        Random seed.

    Returns
    -------
    SimulationResult
    """
    from .environment import discretize_cost

    # Backwards compatibility: cost_weight maps to cost_scale
    if cost_weight is not None:
        cost_scale = cost_weight * 3.0  # old weight=1.0 → new scale=3.0

    if env_data is None:
        env_data = generate_multi_day(num_days=num_days, seed=seed)

    total_steps = len(env_data["time_of_day"])

    env = Environment(env_data, initial_room_temp=initial_room_temp,
                      initial_soc=initial_soc)

    # Sophisticated thermostat: 5-factor Agent with cost modality
    thermo = SophisticatedThermostatAgent(
        env_data, policy_len=policy_len, gamma=gamma,
        learn_B=learn_B, cost_scale=cost_scale,
        initial_soc=initial_soc,
    )

    # Sophisticated battery: phantom thermostat for HVAC prediction
    battery = SophisticatedBatteryAgent(
        env_data, policy_len=policy_len, gamma=gamma,
        initial_soc=initial_soc,
    )

    result = SimulationResult(
        num_days=num_days, policy_len=policy_len, learn_B=learn_B,
    )

    phantom_history = {"p_hvac": [], "p_batt": [], "cost_obs": []}

    # Cost observation: thermostat acts FIRST, so at step 0 it uses
    # an estimated baseline cost. At step t>0, it uses actual cost
    # from the previous step's record.
    prev_cost_obs = discretize_cost(
        max(0, env_data["baseline_load"][0] - env_data["solar_gen"][0])
        * env_data["tou_value"][0]
    )

    for step in range(total_steps):
        if verbose and step % STEPS_PER_DAY == 0:
            day = step // STEPS_PER_DAY
            print(f"  Day {day + 1}/{num_days} ...", flush=True)

        # 1. Thermostat observes and acts (cost obs from previous step)
        thermo_obs = env.get_thermostat_obs(step)
        hvac_action, hvac_energy, thermo_info = thermo.step(
            thermo_obs, step_idx=step, cost_obs=prev_cost_obs)

        # 2. Apply thermostat action to environment
        actual_hvac_energy = env.apply_thermostat(hvac_action, step)

        # 3. Battery observes (includes HVAC energy effect)
        battery_obs = env.get_battery_obs(step, actual_hvac_energy)

        # 4. Battery acts with sophisticated inference (phantom thermostat)
        battery_action, battery_info = battery.step(
            battery_obs, step_idx=step)

        # 5. Apply battery action and record
        record = env.apply_battery(battery_action, step, actual_hvac_energy)
        record.hvac_action = hvac_action

        # 6. Update cost observation for next step from actual record
        prev_cost_obs = discretize_cost(record.cost)

        result.history.append(record)
        result.thermo_efe_history.append(thermo_info["neg_efe"])
        result.battery_efe_history.append(battery_info["neg_efe"])
        result.thermo_qpi_history.append(thermo_info["q_pi"])
        result.battery_qpi_history.append(battery_info["q_pi"])

        phantom_history["p_hvac"].append(
            battery_info.get("phantom_p_hvac", None))
        phantom_history["p_batt"].append(
            thermo_info.get("phantom_p_batt", None))
        phantom_history["cost_obs"].append(prev_cost_obs)

    result.phantom_history = phantom_history

    if verbose:
        print(f"  Symmetric sophisticated simulation complete: "
              f"{total_steps} steps, "
              f"cost=${result.total_cost:.2f}, "
              f"GHG={result.total_ghg:.2f} kg CO2")

    return result
