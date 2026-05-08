"""Multi-seed experiment runner for EcoNet.

Runs 5 seeds x 9 conditions x 4 scenarios (synthetic climate).
Outputs mean +/- std for cost, comfort, GHG per condition.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time

from econet.climate import generate_climate_week
from econet.simulation import (
    run_simulation, run_tom_simulation, run_sophisticated_simulation,
)
from econet.baselines import run_no_hems, run_rule_based, run_oracle, run_mpc, run_rl
from econet.metrics import compute_metrics

SEEDS = [42, 137, 256, 789, 1024]
SCENARIOS = [
    ("london", "summer"),
    ("london", "winter"),
    ("montreal", "winter"),
    ("phoenix", "summer"),
]
NUM_DAYS = 7


def run_condition(name, env_data, seed):
    """Run a single condition on given env_data. Returns SimulationResult."""
    if name == "no_hems":
        return run_no_hems(env_data)
    elif name == "rule_based":
        return run_rule_based(env_data)
    elif name == "oracle":
        return run_oracle(env_data, max_steps=len(env_data["time_of_day"]))
    elif name == "mpc":
        return run_mpc(env_data, horizon=6)
    elif name == "rl":
        return run_rl(env_data, num_episodes=2000, seed=seed)
    elif name == "independent":
        return run_simulation(
            env_data=env_data, num_days=NUM_DAYS,
            policy_len=4, gamma=16.0,
            learn_B=False, aligned=False, seed=seed, verbose=False,
        )
    elif name == "aligned":
        return run_simulation(
            env_data=env_data, num_days=NUM_DAYS,
            policy_len=4, gamma=16.0,
            learn_B=False, aligned=True, seed=seed, verbose=False,
        )
    elif name == "federated":
        return run_tom_simulation(
            env_data=env_data, num_days=NUM_DAYS,
            policy_len=4, gamma=16.0,
            learn_B=False, social_weight=1.0, seed=seed, verbose=False,
        )
    elif name == "tom":
        return run_sophisticated_simulation(
            env_data=env_data, num_days=NUM_DAYS,
            policy_len=4, gamma=16.0,
            learn_B=False, seed=seed, verbose=False,
        )
    else:
        raise ValueError(f"Unknown condition: {name}")


CONDITIONS = [
    "no_hems", "rule_based", "oracle", "mpc", "rl",
    "independent", "aligned", "federated", "tom",
]


def main():
    print("=" * 80)
    print("EcoNet Multi-Seed Experiment")
    print(f"  Seeds: {SEEDS}")
    print(f"  Conditions ({len(CONDITIONS)}): {CONDITIONS}")
    print(f"  Scenarios: {SCENARIOS}")
    print(f"  Duration: {NUM_DAYS} days each")
    print(f"  Total runs: {len(SEEDS) * len(CONDITIONS) * len(SCENARIOS)}")
    print("=" * 80)

    # Results: scenario -> condition -> list of metrics
    all_results = {}
    t_start = time.time()

    for city, season in SCENARIOS:
        scenario_key = f"{city}_{season}"
        all_results[scenario_key] = {}

        for cond in CONDITIONS:
            all_results[scenario_key][cond] = []

        for seed in SEEDS:
            env_data = generate_climate_week(city, season, seed=seed)

            for cond in CONDITIONS:
                print(f"  {scenario_key} / {cond} / seed={seed}...", end=" ", flush=True)
                t0 = time.time()
                try:
                    result = run_condition(cond, env_data, seed)
                    m = compute_metrics(result, num_days=NUM_DAYS)
                    all_results[scenario_key][cond].append(m)
                    dt = time.time() - t0
                    print(f"cost=${m.total_cost:.2f}, comfort={m.comfort_deviation_total:.1f} "
                          f"({dt:.1f}s)")
                except Exception as e:
                    dt = time.time() - t0
                    print(f"FAILED ({dt:.1f}s): {e}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s")

    # Print summary table
    print("\n" + "=" * 120)
    print(f"{'Scenario':<20} {'Condition':<14} {'Cost ($)':<18} {'Comfort (C*h)':<18} "
          f"{'GHG (kg)':<18} {'Batt Util':<12}")
    print("-" * 120)

    for scenario_key in all_results:
        for cond in CONDITIONS:
            metrics_list = all_results[scenario_key][cond]
            if not metrics_list:
                print(f"{scenario_key:<20} {cond:<14} {'FAILED':<18} {'FAILED':<18} "
                      f"{'FAILED':<18} {'FAILED':<12}")
                continue
            costs = [m.total_cost for m in metrics_list]
            comforts = [m.comfort_deviation_total for m in metrics_list]
            ghgs = [m.total_ghg for m in metrics_list]
            batt_utils = [m.battery_utilization for m in metrics_list]

            cost_str = f"{np.mean(costs):.2f} +/- {np.std(costs):.2f}"
            comf_str = f"{np.mean(comforts):.1f} +/- {np.std(comforts):.1f}"
            ghg_str = f"{np.mean(ghgs):.2f} +/- {np.std(ghgs):.2f}"
            batt_str = f"{np.mean(batt_utils)*100:.1f}%"

            print(f"{scenario_key:<20} {cond:<14} {cost_str:<18} {comf_str:<18} "
                  f"{ghg_str:<18} {batt_str:<12}")
        print()

    print("=" * 120)

    # Grand aggregate across all scenarios
    print("\nGRAND AGGREGATE (mean across all scenarios and seeds):")
    for cond in CONDITIONS:
        all_costs = []
        all_comforts = []
        all_ghgs = []
        for scenario_key in all_results:
            for m in all_results[scenario_key][cond]:
                all_costs.append(m.total_cost)
                all_comforts.append(m.comfort_deviation_total)
                all_ghgs.append(m.total_ghg)
        if not all_costs:
            print(f"  {cond:<14}: FAILED (no successful runs)")
            continue
        print(f"  {cond:<14}: Cost=${np.mean(all_costs):.2f}+/-{np.std(all_costs):.2f}, "
              f"Comfort={np.mean(all_comforts):.1f}+/-{np.std(all_comforts):.1f}, "
              f"GHG={np.mean(all_ghgs):.2f}+/-{np.std(all_ghgs):.2f}")

    # Oracle-gap table for LaTeX (Table 5 format)
    DISPLAY_NAMES = {
        "no_hems": "No-HEMS", "rule_based": "Rule-based", "oracle": "Oracle",
        "mpc": "MPC", "rl": "RL", "independent": "Independent",
        "aligned": "Aligned", "federated": "Federated", "tom": "ToM",
    }
    print("\n" + "=" * 100)
    print("ORACLE GAP TABLE (for LaTeX Table 5)")
    print("=" * 100)
    print(f"{'Scenario':<20} {'Condition':<14} {'Cost':>8} {'Comfort':>10} "
          f"{'GHG':>8} {'Batt%':>8} {'dCost':>8} {'dGHG':>8}")
    print("-" * 100)

    for city, season in SCENARIOS:
        scenario_key = f"{city}_{season}"
        # Get oracle mean cost/GHG for gap calculation
        oracle_metrics = all_results[scenario_key].get("oracle", [])
        oracle_cost = np.mean([m.total_cost for m in oracle_metrics]) if oracle_metrics else 0
        oracle_ghg = np.mean([m.total_ghg for m in oracle_metrics]) if oracle_metrics else 0

        for cond in CONDITIONS:
            metrics_list = all_results[scenario_key][cond]
            if not metrics_list:
                continue
            cost = np.mean([m.total_cost for m in metrics_list])
            comfort = np.mean([m.comfort_deviation_total for m in metrics_list])
            ghg = np.mean([m.total_ghg for m in metrics_list])
            batt = np.mean([m.battery_utilization for m in metrics_list]) * 100

            if cond == "oracle":
                dc_str = "---"
                dg_str = "---"
            else:
                dc = cost - oracle_cost
                dg = ghg - oracle_ghg
                dc_str = f"{dc:+.2f}"
                dg_str = f"{dg:+.2f}"

            name = DISPLAY_NAMES.get(cond, cond)
            print(f"{scenario_key:<20} {name:<14} {cost:8.2f} {comfort:10.1f} "
                  f"{ghg:8.2f} {batt:8.1f} {dc_str:>8} {dg_str:>8}")
        print()


if __name__ == "__main__":
    main()
