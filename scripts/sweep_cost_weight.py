"""Sweep cost_weight for SophisticatedThermostatAgent (DEPRECATED).

Use sweep_cost_scale.py instead — the new principled implementation uses
cost_scale (C[4] amplitude) rather than an additive cost_weight penalty.

This script still works via the backwards-compatibility shim in
run_full_sophisticated_simulation (cost_weight maps to cost_scale).
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
POLICY_LEN = 4
GAMMA = 16.0

# Cost weights to sweep
WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

# Run on 2 scenarios for speed (1 mild, 1 extreme)
SCENARIOS = [
    ("london", "summer"),
    ("phoenix", "summer"),
]


def main():
    print("=" * 100)
    print("COST WEIGHT SWEEP: Symmetric Sophisticated Inference")
    print(f"  Weights: {WEIGHTS}")
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

        header = f"  {'Weight':>8} {'Cost($)':>10} {'Comfort':>10} {'Batt%':>8} {'GHG(kg)':>10} {'dCost':>8} {'dComfort':>10}"
        print(header)
        print("  " + "-" * 70)

        for w in WEIGHTS:
            print(f"  w={w:.2f}...", end=" ", flush=True)

            if w == 0.0:
                # w=0 is equivalent to Aligned (no cost penalty)
                result = aligned
                m = aligned_m
            else:
                result = run_full_sophisticated_simulation(
                    env_data=env_data, num_days=NUM_DAYS,
                    policy_len=POLICY_LEN, gamma=GAMMA,
                    learn_B=True, cost_weight=w,
                    seed=SEED, verbose=False)
                m = compute_metrics(result, num_days=NUM_DAYS,
                                    no_hems_result=no_hems)

            dc = m.total_cost - oracle_m.total_cost
            df = m.comfort_deviation_total - oracle_m.comfort_deviation_total

            print(f"  {w:>8.2f} {m.total_cost:>10.2f} "
                  f"{m.comfort_deviation_total:>10.1f} "
                  f"{m.battery_utilization*100:>8.1f} "
                  f"{m.total_ghg:>10.2f} "
                  f"{dc:>+8.2f} {df:>+10.1f}")

        print()


if __name__ == "__main__":
    main()
