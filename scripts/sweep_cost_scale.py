"""Sweep cost_scale for principled cost-aware thermostat.

Runs Symmetric simulation with varying cost_scale values (C[4] amplitude).
Goal: demonstrate smooth Pareto frontier between comfort and cost.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from econet.climate import generate_climate_week
from econet.baselines import run_no_hems, run_oracle
from econet.simulation import (
    run_simulation, run_full_sophisticated_simulation,
)
from econet.metrics import compute_metrics

SEED = 42
NUM_DAYS = 7
POLICY_LEN = 6
GAMMA = 16.0

# Cost scale values to sweep (C[4] amplitude)
SCALES = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]

# Run on 2 scenarios for speed (1 mild, 1 extreme)
SCENARIOS = [
    ("london", "summer"),
    ("phoenix", "summer"),
]


def main():
    print("=" * 100)
    print("COST SCALE SWEEP: Principled Cost-Aware Thermostat")
    print(f"  Scales: {SCALES}")
    print(f"  Scenarios: {[f'{c}_{s}' for c,s in SCENARIOS]}")
    print("=" * 100)

    for city, season in SCENARIOS:
        key = f"{city}_{season}"
        env_data = generate_climate_week(city, season, seed=SEED)
        total_steps = len(env_data["time_of_day"])

        print(f"\n--- {key} ---")

        # Baselines (once)
        no_hems = run_no_hems(env_data)
        oracle = run_oracle(env_data, max_steps=total_steps)
        oracle_m = compute_metrics(oracle, num_days=NUM_DAYS, no_hems_result=no_hems)

        aligned = run_simulation(
            env_data=env_data, num_days=NUM_DAYS, policy_len=POLICY_LEN,
            gamma=GAMMA, learn_B=True, aligned=True, seed=SEED, verbose=False)
        aligned_m = compute_metrics(aligned, num_days=NUM_DAYS, no_hems_result=no_hems)

        print(f"  Oracle:  cost=${oracle_m.total_cost:.2f}  "
              f"comfort={oracle_m.comfort_deviation_total:.1f}  "
              f"ghg={oracle_m.total_ghg:.2f}")
        print(f"  Aligned: cost=${aligned_m.total_cost:.2f}  "
              f"comfort={aligned_m.comfort_deviation_total:.1f}  "
              f"ghg={aligned_m.total_ghg:.2f}")
        print()

        header = (f"  {'Scale':>8} {'Cost($)':>10} {'Comfort':>10} "
                  f"{'Batt%':>8} {'GHG(kg)':>10} {'dCost':>8} {'dComfort':>10}")
        print(header)
        print("  " + "-" * 70)

        for s in SCALES:
            print(f"  s={s:.1f}...", end=" ", flush=True)

            if s == 0.0:
                # s=0 means no cost preference → equivalent to Aligned
                result = aligned
                m = aligned_m
            else:
                result = run_full_sophisticated_simulation(
                    env_data=env_data, num_days=NUM_DAYS,
                    policy_len=POLICY_LEN, gamma=GAMMA,
                    learn_B=True, cost_scale=s,
                    seed=SEED, verbose=False)
                m = compute_metrics(result, num_days=NUM_DAYS,
                                    no_hems_result=no_hems)

            dc = m.total_cost - oracle_m.total_cost
            df = m.comfort_deviation_total - oracle_m.comfort_deviation_total

            print(f"  {s:>8.1f} {m.total_cost:>10.2f} "
                  f"{m.comfort_deviation_total:>10.1f} "
                  f"{m.battery_utilization*100:>8.1f} "
                  f"{m.total_ghg:>10.2f} "
                  f"{dc:>+8.2f} {df:>+10.1f}")

        print()


if __name__ == "__main__":
    main()
