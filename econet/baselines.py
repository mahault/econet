"""Baseline strategies for comparison with EcoNet.

1. No-HEMS: HVAC off, battery off
2. Rule-based: simple thermostat rules + TOU-aware battery
3. Oracle: backward induction DP with perfect foresight
"""

import numpy as np

from .environment import (
    Environment, StepRecord, TEMP_MIN, TEMP_MAX, TEMP_LEVELS, SOC_LEVELS,
    THERMAL_LEAKAGE, HVAC_POWER, HVAC_KWH_PER_STEP,
    BATTERY_STEP_FRAC, BATTERY_CAPACITY_KWH,
    SOC_MIN_DISCHARGE, SOC_MAX_CHARGE,
    TARGET_TEMP_OCCUPIED, TARGET_TEMP_UNOCCUPIED,
    STEPS_PER_DAY,
    discretize_temp, continuous_temp,
    discretize_soc, continuous_soc,
)
from .simulation import SimulationResult


def run_no_hems(env_data: dict, initial_room_temp: float = 20.0,
                initial_soc: float = 0.5) -> SimulationResult:
    """Baseline 1: No HEMS -- HVAC always off, battery always off."""
    env = Environment(env_data, initial_room_temp=initial_room_temp,
                      initial_soc=initial_soc)
    total_steps = len(env_data["time_of_day"])
    result = SimulationResult(num_days=total_steps // STEPS_PER_DAY)

    for step in range(total_steps):
        hvac_action = 2  # off
        hvac_energy = env.apply_thermostat(hvac_action, step)
        battery_action = 2  # off
        record = env.apply_battery(battery_action, step, hvac_energy)
        record.hvac_action = hvac_action
        result.history.append(record)

    return result


def run_rule_based(env_data: dict, initial_room_temp: float = 20.0,
                   initial_soc: float = 0.5,
                   deadband: float = 1.0) -> SimulationResult:
    """Baseline 2: Rule-based HEMS.

    Thermostat rules:
    - Heat if room_temp < target - deadband
    - Cool if room_temp > target + deadband
    - Off otherwise

    Battery rules:
    - Charge during low-TOU + sufficient solar (solar > 1 kWh)
    - Discharge during high-TOU
    - Off otherwise
    """
    env = Environment(env_data, initial_room_temp=initial_room_temp,
                      initial_soc=initial_soc)
    total_steps = len(env_data["time_of_day"])
    result = SimulationResult(num_days=total_steps // STEPS_PER_DAY)

    for step in range(total_steps):
        # --- Thermostat rules ---
        target = (TARGET_TEMP_OCCUPIED if env_data["occupancy"][step]
                  else TARGET_TEMP_UNOCCUPIED)

        if env.room_temp < target - deadband:
            hvac_action = 1  # heat
        elif env.room_temp > target + deadband:
            hvac_action = 0  # cool
        else:
            hvac_action = 2  # off

        hvac_energy = env.apply_thermostat(hvac_action, step)

        # --- Battery rules ---
        tou_high = env_data["tou_high"][step]
        solar = env_data["solar_gen"][step]

        if not tou_high and solar > 1.0 and env.soc < SOC_MAX_CHARGE:
            battery_action = 0  # charge during low TOU + solar
        elif tou_high and env.soc > SOC_MIN_DISCHARGE:
            battery_action = 1  # discharge during high TOU
        else:
            battery_action = 2  # off

        record = env.apply_battery(battery_action, step, hvac_energy)
        record.hvac_action = hvac_action
        result.history.append(record)

    return result


def _simulate_step_physics(room_temp, soc, hvac_a, batt_a, step, env_data):
    """Simulate one step of physics without an Environment object.

    Returns (cost, comfort_penalty, new_room_temp, new_soc, hvac_energy, batt_energy).
    """
    outdoor = env_data["outdoor_temp"][step]

    # Thermostat physics
    hvac_effect = 0.0
    if hvac_a == 0:
        hvac_effect = -HVAC_POWER
    elif hvac_a == 1:
        hvac_effect = +HVAC_POWER

    T_new = room_temp + THERMAL_LEAKAGE * (outdoor - room_temp) + hvac_effect
    T_new = np.clip(T_new, TEMP_MIN, TEMP_MAX)

    hvac_energy = HVAC_KWH_PER_STEP if hvac_a != 2 else 0.0

    # Battery physics with constraints
    batt_delta = 0.0
    if batt_a == 0 and soc < SOC_MAX_CHARGE:
        batt_delta = BATTERY_STEP_FRAC
    elif batt_a == 1 and soc > SOC_MIN_DISCHARGE:
        batt_delta = -BATTERY_STEP_FRAC

    new_soc = np.clip(soc + batt_delta, 0.0, 1.0)
    batt_energy_kwh = batt_delta * BATTERY_CAPACITY_KWH

    # Energy/cost/GHG
    baseline = env_data["baseline_load"][step]
    solar = env_data["solar_gen"][step]
    total_energy = baseline + hvac_energy + batt_energy_kwh - solar
    tou_val = env_data["tou_value"][step]
    cost = max(0, total_energy) * tou_val

    # Comfort penalty
    target = (TARGET_TEMP_OCCUPIED if env_data["occupancy"][step]
              else TARGET_TEMP_UNOCCUPIED)
    comfort_pen = 0.5 * (T_new - target) ** 2

    return cost, comfort_pen, float(T_new), float(new_soc), hvac_energy, batt_energy_kwh


def run_oracle(env_data: dict, initial_room_temp: float = 20.0,
               initial_soc: float = 0.5,
               max_steps: int = 24,
               comfort_weight: float = 0.3) -> SimulationResult:
    """Baseline 3: Oracle with perfect foresight via backward induction DP.

    State space: room_temp (26 bins) x SoC (5 bins) = 130 states
    Action space: HVAC (3) x battery (3) = 9 joint actions
    Horizon: T steps

    Complexity: T * 130 * 9 = ~28K evaluations for T=24. Runs in <1s.

    Parameters
    ----------
    max_steps : int
        Maximum number of steps to optimize over.
    comfort_weight : float
        Weight on comfort penalty relative to cost. Default 0.3.
    """
    total_steps = min(len(env_data["time_of_day"]), max_steps)
    num_room = TEMP_LEVELS  # 26
    num_soc = SOC_LEVELS    # 5

    # V[t][r][s] = minimum future cost from (t, room_idx, soc_idx) to T
    V = np.zeros((total_steps + 1, num_room, num_soc))  # terminal cost = 0

    # policy[t][r][s] = (best_hvac, best_batt) at each state
    policy_hvac = np.zeros((total_steps, num_room, num_soc), dtype=int)
    policy_batt = np.zeros((total_steps, num_room, num_soc), dtype=int)

    # Backward pass
    for t in range(total_steps - 1, -1, -1):
        for r in range(num_room):
            T_room = continuous_temp(r)
            for s in range(num_soc):
                soc_val = continuous_soc(s)

                best_val = float("inf")
                best_h = 2
                best_b = 2

                for hvac_a in range(3):
                    for batt_a in range(3):
                        cost, comfort, T_new, new_soc, _, _ = \
                            _simulate_step_physics(
                                T_room, soc_val, hvac_a, batt_a, t, env_data)

                        r_new = discretize_temp(T_new)
                        s_new = discretize_soc(new_soc)

                        # Bellman equation
                        total = cost + comfort_weight * comfort + V[t + 1, r_new, s_new]

                        if total < best_val:
                            best_val = total
                            best_h = hvac_a
                            best_b = batt_a

                V[t, r, s] = best_val
                policy_hvac[t, r, s] = best_h
                policy_batt[t, r, s] = best_b

    # Forward pass: execute optimal policy on actual Environment
    env = Environment(env_data, initial_room_temp=initial_room_temp,
                      initial_soc=initial_soc)
    result = SimulationResult(num_days=total_steps // STEPS_PER_DAY)

    for t in range(total_steps):
        r = discretize_temp(env.room_temp)
        s = discretize_soc(env.soc)

        hvac_a = int(policy_hvac[t, r, s])
        batt_a = int(policy_batt[t, r, s])

        hvac_energy = env.apply_thermostat(hvac_a, t)
        record = env.apply_battery(batt_a, t, hvac_energy)
        record.hvac_action = hvac_a
        result.history.append(record)

    return result


def run_all_baselines(env_data: dict, **kwargs) -> dict:
    """Run all baseline strategies and return results dict.

    Returns
    -------
    dict[str, SimulationResult]
        Keys: 'no_hems', 'rule_based', 'oracle'
    """
    return {
        "no_hems": run_no_hems(env_data, **kwargs),
        "rule_based": run_rule_based(env_data, **kwargs),
        "oracle": run_oracle(env_data, **kwargs),
    }
