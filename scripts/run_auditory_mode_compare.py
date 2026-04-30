"""Compare auditory_mode approaches: full vs uniform vs none.

Tests whether removing/neutralizing the auditory modality's ambiguity
contribution recovers battery utilization lost to the extra modality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from econet.real_weather import build_real_weather_env
from econet.climate import generate_climate_week
from econet.simulation import run_simulation, run_tom_simulation
from econet.metrics import compute_metrics

AUDITORY_MODES = ["full", "uniform", "none"]
SOCIAL_WEIGHTS = [0.0, 1.0]

# Use 2 scenarios: 1 temperate, 1 extreme
SCENARIOS = [
    ("london", "summer"),    # temperate — marginal decisions
    ("phoenix", "summer"),   # extreme heat — obvious policy
]

NUM_DAYS = 7


def main():
    print("=" * 110)
    print("Auditory Mode Comparison: full vs uniform vs none")
    print(f"  Modes: {AUDITORY_MODES}")
    print(f"  Social weights: {SOCIAL_WEIGHTS}")
    print("=" * 110)

    scenarios = {}
    for city, season in SCENARIOS:
        key = f"{city}_{season}_real"
        env_data = build_real_weather_env(city, season, use_cache=True)
        if env_data is None:
            print(f"  Falling back to synthetic for {city} {season}")
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

    # ToM sweep: auditory_mode x social_weight
    tom_metrics = {}
    for mode in AUDITORY_MODES:
        for w in SOCIAL_WEIGHTS:
            label = f"{mode}/w={w}"
            print(f"\n--- {label} ---")
            tom_metrics[label] = {}
            for key, env_data in scenarios.items():
                print(f"  ToM({label}): {key}...")
                result = run_tom_simulation(
                    env_data=env_data, num_days=NUM_DAYS,
                    policy_len=4, gamma=16.0,
                    learn_B=True, social_weight=w,
                    auditory_mode=mode,
                    seed=42, verbose=False,
                )
                tom_metrics[label][key] = compute_metrics(result, num_days=NUM_DAYS)
                mt = tom_metrics[label][key]
                print(f"    Cost: ${mt.total_cost:.2f}, Comfort: {mt.comfort_deviation_total:.1f}, "
                      f"Batt: {mt.battery_utilization*100:.1f}%")

    # Summary table
    labels = [f"{m}/w={w}" for m in AUDITORY_MODES for w in SOCIAL_WEIGHTS]
    print("\n\n")
    print("=" * 140)
    hdr = f"{'Scenario':<24} {'Metric':<14} {'Aligned':>10}"
    for label in labels:
        hdr += f" {label:>16}"
    print(hdr)
    print("-" * 140)

    for key in keys:
        ma = aligned_metrics[key]

        row = f"{key:<24} {'Comfort':.<14} {ma.comfort_deviation_total:>10.1f}"
        for label in labels:
            mt = tom_metrics[label][key]
            row += f" {mt.comfort_deviation_total:>16.1f}"
        print(row)

        row = f"{'':<24} {'Cost ($)':.<14} {ma.total_cost:>10.2f}"
        for label in labels:
            mt = tom_metrics[label][key]
            row += f" {mt.total_cost:>16.2f}"
        print(row)

        row = f"{'':<24} {'Batt util %':.<14} {ma.battery_utilization*100:>10.1f}"
        for label in labels:
            mt = tom_metrics[label][key]
            row += f" {mt.battery_utilization*100:>16.1f}"
        print(row)

        row = f"{'':<24} {'GHG (kg)':.<14} {ma.total_ghg:>10.2f}"
        for label in labels:
            mt = tom_metrics[label][key]
            row += f" {mt.total_ghg:>16.2f}"
        print(row)
        print()

    # Delta table
    print("=" * 140)
    print("DELTA vs ALIGNED (negative = ToM better)")
    print("-" * 140)
    for key in keys:
        ma = aligned_metrics[key]
        print(f"  {key}:")
        for label in labels:
            mt = tom_metrics[label][key]
            cd = mt.total_cost - ma.total_cost
            fd = mt.comfort_deviation_total - ma.comfort_deviation_total
            bd = mt.battery_utilization - ma.battery_utilization
            print(f"    {label:<20}  cost: ${cd:+.2f}  comfort: {fd:+.1f}  batt: {bd*100:+.1f}%")
    print("=" * 140)


if __name__ == "__main__":
    main()
