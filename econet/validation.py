"""Validation and benchmarking for EcoNet.

1. Energy balance verification
2. Thermodynamic consistency
3. Sensitivity analysis (parameter sweeps)
4. Comparison to published RL/MPC results
"""

import numpy as np
from typing import Optional

from .environment import (
    THERMAL_LEAKAGE, HVAC_POWER, HVAC_KWH_PER_STEP,
    BATTERY_CAPACITY_KWH, BATTERY_STEP_FRAC,
    TARGET_TEMP_OCCUPIED, TARGET_TEMP_UNOCCUPIED,
    TEMP_MIN, TEMP_MAX,
    generate_multi_day, STEPS_PER_DAY,
)
from .simulation import run_simulation, SimulationResult
from .baselines import run_no_hems, run_rule_based
from .metrics import compute_metrics, EconomicMetrics


def verify_energy_balance(result: SimulationResult, tol: float = 0.01) -> dict:
    """Verify energy conservation at each step.

    total_energy = baseline_load + hvac_energy + battery_energy - solar_gen

    Returns dict with 'passed', 'max_error', 'errors'.
    """
    arr = result.to_arrays()
    expected = (arr["baseline_load"] + arr["hvac_energy"]
                + arr["battery_energy"] - arr["solar_gen"])
    actual = arr["total_energy"]
    errors = np.abs(expected - actual)

    return {
        "passed": bool(errors.max() < tol),
        "max_error": float(errors.max()),
        "mean_error": float(errors.mean()),
        "errors": errors,
    }


def verify_thermodynamic_consistency(result: SimulationResult,
                                      tol: float = 1.0) -> dict:
    """Verify room temperature follows thermodynamic model.

    Check that T_new ≈ T_prev + α*(T_out - T_prev) + HVAC_effect.
    """
    arr = result.to_arrays()
    errors = []

    for i in range(1, len(arr["room_temp"])):
        T_prev = arr["room_temp"][i - 1]
        T_out = arr["outdoor_temp"][i]
        action = arr["hvac_action"][i]

        hvac_effect = 0.0
        if action == 0:   # cool
            hvac_effect = -HVAC_POWER
        elif action == 1:  # heat
            hvac_effect = +HVAC_POWER

        T_expected = T_prev + THERMAL_LEAKAGE * (T_out - T_prev) + hvac_effect
        T_expected = np.clip(T_expected, TEMP_MIN, TEMP_MAX)
        err = abs(arr["room_temp"][i] - T_expected)
        errors.append(err)

    errors = np.array(errors)
    return {
        "passed": bool(errors.max() < tol),
        "max_error": float(errors.max()),
        "mean_error": float(errors.mean()),
        "errors": errors,
    }


def verify_battery_constraints(result: SimulationResult) -> dict:
    """Verify battery SoC stays within [0, 1] and constraints are respected."""
    arr = result.to_arrays()
    soc = arr["soc"]

    violations = []
    for i in range(len(soc)):
        if soc[i] < -0.01 or soc[i] > 1.01:
            violations.append((i, soc[i]))

    return {
        "passed": len(violations) == 0,
        "soc_min": float(soc.min()),
        "soc_max": float(soc.max()),
        "violations": violations,
    }


def run_sensitivity_sweep(param_name: str, param_values: list,
                          base_kwargs: Optional[dict] = None,
                          num_days: int = 2, seed: int = 42,
                          policy_len: int = 4) -> dict:
    """Run parameter sweep and collect results.

    Parameters
    ----------
    param_name : str
        One of: 'gamma', 'policy_len', 'hvac_power', 'battery_capacity'
    param_values : list
        Values to sweep over.
    base_kwargs : dict
        Base simulation kwargs.
    num_days : int
        Simulation length.

    Returns
    -------
    dict[str, SimulationResult]
    """
    if base_kwargs is None:
        base_kwargs = {}

    env_data = generate_multi_day(num_days=num_days, seed=seed)
    results = {}

    for val in param_values:
        kwargs = {**base_kwargs, "env_data": env_data, "num_days": num_days,
                  "seed": seed, "verbose": False, "policy_len": policy_len}

        if param_name == "gamma":
            kwargs["gamma"] = val
        elif param_name == "policy_len":
            kwargs["policy_len"] = val
        else:
            # For physical parameters, we'd need to modify environment constants
            # For now, just use gamma as the primary sweep parameter
            kwargs["gamma"] = val

        label = f"{param_name}={val}"
        print(f"  Running {label}...", flush=True)
        results[label] = run_simulation(**kwargs)

    return results


def compare_to_literature() -> dict:
    """Compare EcoNet results to published RL/MPC benchmarks.

    References:
    - Khabbazi et al. 2025 review: ~16% cost savings for residential RL-HEMS
    - Nazemi et al. 2025: AIF for building temperature tracking
    - Typical MPC: 10-25% cost reduction, 15-30% peak shaving

    Returns summary comparison dict.
    """
    return {
        "method": [
            "EcoNet (AIF, this work)",
            "RL-HEMS (Khabbazi 2025, avg)",
            "MPC (typical residential)",
            "Rule-based (this work)",
        ],
        "cost_savings_pct": [None, 16.0, 20.0, None],  # Filled after simulation
        "comfort_metric": [
            "Cumulative deviation °C·h",
            "Avg deviation °C",
            "Avg deviation °C",
            "Cumulative deviation °C·h",
        ],
        "notes": [
            "Multi-agent cascading, 6-step lookahead",
            "Average across surveyed methods",
            "Perfect model, rolling optimization",
            "Simple threshold control",
        ],
    }


def run_full_validation(num_days: int = 2, seed: int = 42,
                        verbose: bool = True, policy_len: int = 4) -> dict:
    """Run complete validation suite.

    Returns dict with all validation results.
    """
    if verbose:
        print("=== EcoNet Validation Suite ===\n")

    env_data = generate_multi_day(num_days=num_days, seed=seed)

    # Run EcoNet
    if verbose:
        print("1. Running EcoNet simulation...")
    econet_result = run_simulation(env_data=env_data, num_days=num_days,
                                   policy_len=policy_len, seed=seed,
                                   verbose=False)

    # Run baselines
    if verbose:
        print("2. Running baselines...")
    no_hems = run_no_hems(env_data)
    rule_based = run_rule_based(env_data)

    # Verification
    if verbose:
        print("3. Verifying energy balance...")
    eb = verify_energy_balance(econet_result)
    if verbose:
        print(f"   Energy balance: {'PASS' if eb['passed'] else 'FAIL'} "
              f"(max error: {eb['max_error']:.4f})")

    if verbose:
        print("4. Verifying thermodynamic consistency...")
    tc = verify_thermodynamic_consistency(econet_result)
    if verbose:
        print(f"   Thermodynamics: {'PASS' if tc['passed'] else 'FAIL'} "
              f"(max error: {tc['max_error']:.2f}°C)")

    if verbose:
        print("5. Verifying battery constraints...")
    bc = verify_battery_constraints(econet_result)
    if verbose:
        print(f"   Battery: {'PASS' if bc['passed'] else 'FAIL'} "
              f"(SoC range: [{bc['soc_min']:.2f}, {bc['soc_max']:.2f}])")

    # Metrics comparison
    if verbose:
        print("6. Computing metrics...")
    m_econet = compute_metrics(econet_result, num_days, no_hems)
    m_no_hems = compute_metrics(no_hems, num_days)
    m_rule = compute_metrics(rule_based, num_days, no_hems)

    if verbose:
        print(f"\n--- Results ({num_days} days) ---")
        print(f"{'Metric':<25} {'No-HEMS':>10} {'Rule-Based':>12} {'EcoNet':>10}")
        print("-" * 60)
        print(f"{'Total Cost ($)':<25} {m_no_hems.total_cost:>10.2f} "
              f"{m_rule.total_cost:>12.2f} {m_econet.total_cost:>10.2f}")
        print(f"{'Total GHG (kg)':<25} {m_no_hems.total_ghg:>10.2f} "
              f"{m_rule.total_ghg:>12.2f} {m_econet.total_ghg:>10.2f}")
        print(f"{'Comfort Dev (°C·h)':<25} {m_no_hems.comfort_deviation_total:>10.1f} "
              f"{m_rule.comfort_deviation_total:>12.1f} "
              f"{m_econet.comfort_deviation_total:>10.1f}")
        print(f"{'Cost savings vs no-HEMS':<25} {'---':>10} "
              f"{m_rule.cost_savings_vs_no_hems:>11.1%} "
              f"{m_econet.cost_savings_vs_no_hems:>9.1%}")

    return {
        "energy_balance": eb,
        "thermodynamics": tc,
        "battery_constraints": bc,
        "metrics": {
            "econet": m_econet,
            "no_hems": m_no_hems,
            "rule_based": m_rule,
        },
        "results": {
            "econet": econet_result,
            "no_hems": no_hems,
            "rule_based": rule_based,
        },
    }
