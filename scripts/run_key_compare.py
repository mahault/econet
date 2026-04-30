"""Key comparison: Oracle vs best AIF conditions (fast version).

Runs only the conditions that matter for the Oracle gap analysis:
  1. No-HEMS (lower bound)
  2. Rule-based (industry baseline)
  3. Oracle (upper bound)
  4. Aligned (standard AIF)
  5. Aligned+Phantom (battery-side sophisticated)
  6. Symmetric (both agents sophisticated) — the gap-closer

Skips Selfish, Comm, Full (which need long JAX compilations for
ToM auditory models and add no value to the Oracle gap story).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from collections import OrderedDict

from econet.climate import generate_climate_week
from econet.environment import Environment, STEPS_PER_DAY
from econet.baselines import run_no_hems, run_rule_based, run_oracle
from econet.simulation import (
    SimulationResult, run_simulation,
    run_sophisticated_simulation, run_full_sophisticated_simulation,
)
from econet.metrics import compute_metrics

SCENARIOS = [
    ("london", "summer"),
    ("london", "winter"),
    ("montreal", "winter"),
    ("phoenix", "summer"),
]

NUM_DAYS = 7
POLICY_LEN = 6
GAMMA = 16.0
SEED = 42


def run_key_conditions(env_data):
    """Run 6 key conditions for one scenario."""
    results = OrderedDict()
    total_steps = len(env_data["time_of_day"])

    print("    [1/6] No-HEMS...", end=" ", flush=True)
    results["No-HEMS"] = run_no_hems(env_data)
    print("done")

    print("    [2/6] Rule-based...", end=" ", flush=True)
    results["Rule-based"] = run_rule_based(env_data)
    print("done")

    print("    [3/6] Oracle...", end=" ", flush=True)
    results["Oracle"] = run_oracle(env_data, max_steps=total_steps)
    print("done")

    print("    [4/6] Aligned...", end=" ", flush=True)
    results["Aligned"] = run_simulation(
        env_data=env_data, num_days=NUM_DAYS, policy_len=POLICY_LEN,
        gamma=GAMMA, learn_B=True, aligned=True, seed=SEED, verbose=False)
    print("done")

    print("    [5/6] Aligned+Phantom...", end=" ", flush=True)
    results["Aligned+Phantom"] = run_sophisticated_simulation(
        env_data=env_data, num_days=NUM_DAYS, policy_len=POLICY_LEN,
        gamma=GAMMA, learn_B=True, seed=SEED, verbose=False)
    print("done")

    print("    [6/6] Symmetric...", end=" ", flush=True)
    results["Symmetric"] = run_full_sophisticated_simulation(
        env_data=env_data, num_days=NUM_DAYS, policy_len=POLICY_LEN,
        gamma=GAMMA, learn_B=True, cost_scale=3.0,
        seed=SEED, verbose=False)
    print("done")

    return results


def main():
    print("=" * 100)
    print("KEY COMPARISON: Oracle Gap Analysis")
    print(f"  Scenarios: {len(SCENARIOS)}, Days: {NUM_DAYS}, "
          f"policy_len: {POLICY_LEN}, gamma: {GAMMA}")
    print("=" * 100)

    scenarios = {}
    for city, season in SCENARIOS:
        key = f"{city}_{season}"
        env_data = generate_climate_week(city, season, seed=SEED)
        scenarios[key] = env_data
        temps = env_data["outdoor_temp"]
        print(f"  {key}: T_out = [{temps.min():.1f}, {temps.max():.1f}]C")

    all_metrics = {}
    for key, env_data in scenarios.items():
        print(f"\n--- {key} ---")
        results = run_key_conditions(env_data)

        no_hems = results["No-HEMS"]
        all_metrics[key] = OrderedDict()
        for cond_name, result in results.items():
            all_metrics[key][cond_name] = compute_metrics(
                result, num_days=NUM_DAYS, no_hems_result=no_hems)

    # --- RESULTS ---
    conditions = list(next(iter(all_metrics.values())).keys())

    for scenario_key in scenarios:
        print(f"\n\n{'=' * 100}")
        print(f"  {scenario_key}")
        print(f"{'=' * 100}")

        header = f"{'Condition':<20}"
        for col in ["Cost($)", "Comfort", "Batt%", "GHG(kg)",
                     "CostSav%", "GHGSav%"]:
            header += f" {col:>10}"
        print(header)
        print("-" * 90)

        for cond in conditions:
            m = all_metrics[scenario_key][cond]
            cost_sav = m.cost_savings_vs_no_hems * 100
            ghg_sav = m.ghg_savings_vs_no_hems * 100
            row = f"{cond:<20}"
            row += f" {m.total_cost:>10.2f}"
            row += f" {m.comfort_deviation_total:>10.1f}"
            row += f" {m.battery_utilization*100:>10.1f}"
            row += f" {m.total_ghg:>10.2f}"
            row += f" {cost_sav:>10.1f}"
            row += f" {ghg_sav:>10.1f}"
            print(row)

    # --- CROSS-SCENARIO SUMMARY ---
    print(f"\n\n{'=' * 100}")
    print("CROSS-SCENARIO SUMMARY")
    print(f"{'=' * 100}")

    header = f"{'Condition':<20} {'Avg Cost':>10} {'Avg Comfort':>12} {'Avg Batt%':>10} {'Avg GHG':>10} {'CostSav%':>10} {'GHGSav%':>10}"
    print(header)
    print("-" * 92)

    for cond in conditions:
        costs = [all_metrics[s][cond].total_cost for s in scenarios]
        comforts = [all_metrics[s][cond].comfort_deviation_total for s in scenarios]
        batts = [all_metrics[s][cond].battery_utilization for s in scenarios]
        ghgs = [all_metrics[s][cond].total_ghg for s in scenarios]
        csavs = [all_metrics[s][cond].cost_savings_vs_no_hems for s in scenarios]
        gsavs = [all_metrics[s][cond].ghg_savings_vs_no_hems for s in scenarios]

        row = f"{cond:<20}"
        row += f" {np.mean(costs):>10.2f}"
        row += f" {np.mean(comforts):>12.1f}"
        row += f" {np.mean(batts)*100:>10.1f}"
        row += f" {np.mean(ghgs):>10.2f}"
        row += f" {np.mean(csavs)*100:>10.1f}"
        row += f" {np.mean(gsavs)*100:>10.1f}"
        print(row)

    # --- GAP TO ORACLE ---
    print(f"\n\n{'=' * 100}")
    print("GAP TO ORACLE (negative = AIF is cheaper/better)")
    print(f"{'=' * 100}")

    aif_conditions = [c for c in conditions
                      if c not in ("No-HEMS", "Rule-based", "Oracle")]

    for scenario_key in scenarios:
        oracle_m = all_metrics[scenario_key]["Oracle"]
        print(f"\n  {scenario_key}:")
        for cond in aif_conditions:
            m = all_metrics[scenario_key][cond]
            dc = m.total_cost - oracle_m.total_cost
            df = m.comfort_deviation_total - oracle_m.comfort_deviation_total
            dg = m.total_ghg - oracle_m.total_ghg
            print(f"    {cond:<20} cost:{dc:>+8.2f}  comfort:{df:>+8.1f}  ghg:{dg:>+8.2f}")

    # --- MEAN GAP ---
    print(f"\n  AVERAGE across scenarios:")
    for cond in aif_conditions:
        dcs = [all_metrics[s][cond].total_cost - all_metrics[s]["Oracle"].total_cost
               for s in scenarios]
        dfs = [all_metrics[s][cond].comfort_deviation_total -
               all_metrics[s]["Oracle"].comfort_deviation_total for s in scenarios]
        dgs = [all_metrics[s][cond].total_ghg - all_metrics[s]["Oracle"].total_ghg
               for s in scenarios]
        print(f"    {cond:<20} cost:{np.mean(dcs):>+8.2f}  "
              f"comfort:{np.mean(dfs):>+8.1f}  ghg:{np.mean(dgs):>+8.2f}")

    print(f"\n{'=' * 100}")
    print("Done.")


if __name__ == "__main__":
    main()
