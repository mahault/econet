"""Sweep social_weight on temperate + extreme scenarios.

Hypothesis: ToM helps in ambiguous scenarios (London summer) where knowing
the other agent's state resolves genuine uncertainty, but not in extreme
scenarios (Montreal winter, Phoenix summer) where optimal policy is obvious.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from econet.real_weather import build_real_weather_env, STATIONS
from econet.climate import generate_climate_week
from econet.simulation import run_simulation, run_tom_simulation
from econet.metrics import compute_metrics

SOCIAL_WEIGHTS = [0.0, 0.5, 1.0, 2.0]

SWEEP_SCENARIOS = [
    ("london", "summer"),    # temperate — marginal decisions
    ("london", "winter"),    # mild cold — some ambiguity
    ("montreal", "winter"),  # extreme cold — obvious policy
    ("phoenix", "summer"),   # extreme heat — obvious policy
]


def main():
    print("=" * 100)
    print("Social Weight Sweep: Temperate vs Extreme scenarios")
    print(f"  Weights: {SOCIAL_WEIGHTS}")
    print("=" * 100)

    scenarios = {}
    for city, season in SWEEP_SCENARIOS:
        key = f"{city}_{season}_real"
        env_data = build_real_weather_env(city, season, use_cache=True)
        if env_data is None:
            print(f"  Falling back to synthetic for {city} {season}")
            env_data = generate_climate_week(city, season, seed=42)
        scenarios[key] = env_data
        # Show temp range for context
        temps = env_data["outdoor_temp"]
        print(f"  {key}: T_out = [{temps.min():.1f}, {temps.max():.1f}]°C")

    keys = list(scenarios.keys())

    # Aligned baseline
    aligned_metrics = {}
    for key, env_data in scenarios.items():
        print(f"\n  Aligned: {key}...")
        result = run_simulation(
            env_data=env_data, num_days=7,
            policy_len=4, gamma=16.0,
            learn_B=True, aligned=True, seed=42, verbose=False,
        )
        aligned_metrics[key] = compute_metrics(result, num_days=7)
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
                env_data=env_data, num_days=7,
                policy_len=4, gamma=16.0,
                learn_B=True, social_weight=w, seed=42, verbose=False,
            )
            tom_metrics[w][key] = compute_metrics(result, num_days=7)
            mt = tom_metrics[w][key]
            print(f"    Cost: ${mt.total_cost:.2f}, Comfort: {mt.comfort_deviation_total:.1f}, "
                  f"Batt: {mt.battery_utilization*100:.1f}%")

    # ── Summary table ──────────────────────────────────────────────────
    print("\n")
    print("=" * 110)
    print(f"{'Scenario':<28} {'Metric':<14} {'Aligned':>10}", end="")
    for w in SOCIAL_WEIGHTS:
        print(f" {'w='+str(w):>10}", end="")
    print()
    print("-" * 110)

    for key in keys:
        ma = aligned_metrics[key]

        row = f"{key:<28} {'Comfort':.<14} {ma.comfort_deviation_total:>10.1f}"
        for w in SOCIAL_WEIGHTS:
            mt = tom_metrics[w][key]
            row += f" {mt.comfort_deviation_total:>10.1f}"
        print(row)

        row = f"{'':<28} {'Cost ($)':.<14} {ma.total_cost:>10.2f}"
        for w in SOCIAL_WEIGHTS:
            mt = tom_metrics[w][key]
            row += f" {mt.total_cost:>10.2f}"
        print(row)

        row = f"{'':<28} {'Batt util %':.<14} {ma.battery_utilization*100:>10.1f}"
        for w in SOCIAL_WEIGHTS:
            mt = tom_metrics[w][key]
            row += f" {mt.battery_utilization*100:>10.1f}"
        print(row)

        row = f"{'':<28} {'GHG (kg)':.<14} {ma.total_ghg:>10.2f}"
        for w in SOCIAL_WEIGHTS:
            mt = tom_metrics[w][key]
            row += f" {mt.total_ghg:>10.2f}"
        print(row)
        print()

    # ── Delta vs Aligned per scenario ──────────────────────────────────
    print("=" * 110)
    print("DELTA vs ALIGNED per scenario (negative = ToM better)")
    print("-" * 110)
    for key in keys:
        ma = aligned_metrics[key]
        print(f"  {key}:")
        for w in SOCIAL_WEIGHTS:
            mt = tom_metrics[w][key]
            cd = mt.total_cost - ma.total_cost
            fd = mt.comfort_deviation_total - ma.comfort_deviation_total
            bd = mt.battery_utilization - ma.battery_utilization
            print(f"    w={w:<4}  cost: ${cd:+.2f}  comfort: {fd:+.1f}  batt: {bd*100:+.1f}%")
    print("=" * 110)


if __name__ == "__main__":
    main()
