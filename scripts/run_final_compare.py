"""Final comparison after learn_B fix: Aligned vs ToM+Belief.

Battery agent now never learns B (matching standard BatteryAgent).
Tests auditory_mode full with social_weight sweep.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from econet.real_weather import build_real_weather_env
from econet.climate import generate_climate_week
from econet.simulation import run_simulation, run_tom_simulation
from econet.metrics import compute_metrics

SOCIAL_WEIGHTS = [0.0, 0.5, 1.0, 2.0]

SCENARIOS = [
    ("london", "summer"),
    ("london", "winter"),
    ("montreal", "winter"),
    ("phoenix", "summer"),
]

NUM_DAYS = 7


def main():
    print("=" * 110)
    print("Final Comparison: learn_B fix applied (battery never learns B)")
    print(f"  Social weights: {SOCIAL_WEIGHTS}")
    print("=" * 110)

    scenarios = {}
    for city, season in SCENARIOS:
        key = f"{city}_{season}_real"
        env_data = build_real_weather_env(city, season, use_cache=True)
        if env_data is None:
            env_data = generate_climate_week(city, season, seed=42)
        scenarios[key] = env_data
        temps = env_data["outdoor_temp"]
        print(f"  {key}: T_out = [{temps.min():.1f}, {temps.max():.1f}]C")

    keys = list(scenarios.keys())

    # Aligned baseline
    aligned_metrics = {}
    for key, env_data in scenarios.items():
        print(f"\n  Aligned: {key}...")
        result = run_simulation(
            env_data=env_data, num_days=NUM_DAYS,
            policy_len=4, gamma=16.0,
            learn_B=True, aligned=True, seed=42, verbose=False,
        )
        aligned_metrics[key] = compute_metrics(result, num_days=NUM_DAYS)
        ma = aligned_metrics[key]
        print(f"    Cost: ${ma.total_cost:.2f}, Comfort: {ma.comfort_deviation_total:.1f}, "
              f"Batt: {ma.battery_utilization*100:.1f}%")

    # ToM sweep
    tom_metrics = {}
    for w in SOCIAL_WEIGHTS:
        print(f"\n--- social_weight = {w} ---")
        tom_metrics[w] = {}
        for key, env_data in scenarios.items():
            print(f"  ToM(w={w}): {key}...")
            result = run_tom_simulation(
                env_data=env_data, num_days=NUM_DAYS,
                policy_len=4, gamma=16.0,
                learn_B=True, social_weight=w, seed=42, verbose=False,
            )
            tom_metrics[w][key] = compute_metrics(result, num_days=NUM_DAYS)
            mt = tom_metrics[w][key]
            print(f"    Cost: ${mt.total_cost:.2f}, Comfort: {mt.comfort_deviation_total:.1f}, "
                  f"Batt: {mt.battery_utilization*100:.1f}%")

    # Summary table
    print("\n\n")
    print("=" * 110)
    print(f"{'Scenario':<28} {'Metric':<14} {'Aligned':>10}", end="")
    for w in SOCIAL_WEIGHTS:
        print(f" {'w='+str(w):>10}", end="")
    print()
    print("-" * 110)

    for key in keys:
        ma = aligned_metrics[key]
        for metric_name, get_val in [
            ("Comfort", lambda m: f"{m.comfort_deviation_total:.1f}"),
            ("Cost ($)", lambda m: f"{m.total_cost:.2f}"),
            ("Batt util %", lambda m: f"{m.battery_utilization*100:.1f}"),
            ("GHG (kg)", lambda m: f"{m.total_ghg:.2f}"),
        ]:
            if metric_name == "Comfort":
                row = f"{key:<28} {metric_name:.<14} {get_val(ma):>10}"
            else:
                row = f"{'':<28} {metric_name:.<14} {get_val(ma):>10}"
            for w in SOCIAL_WEIGHTS:
                mt = tom_metrics[w][key]
                row += f" {get_val(mt):>10}"
            print(row)
        print()

    # Delta table
    print("=" * 110)
    print("DELTA vs ALIGNED (negative = ToM better)")
    print("-" * 110)
    for key in keys:
        ma = aligned_metrics[key]
        print(f"  {key}:")
        for w in SOCIAL_WEIGHTS:
            mt = tom_metrics[w][key]
            cd = mt.total_cost - ma.total_cost
            fd = mt.comfort_deviation_total - ma.comfort_deviation_total
            bd = mt.battery_utilization - ma.battery_utilization
            gd = mt.total_ghg - ma.total_ghg
            print(f"    w={w:<4}  cost: ${cd:+.2f}  comfort: {fd:+.1f}  batt: {bd*100:+.1f}%  ghg: {gd:+.2f}kg")
    print("=" * 110)


if __name__ == "__main__":
    main()
