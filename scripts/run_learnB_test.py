"""Test whether learn_B on battery agent is the root cause of the battery gap.

The standard BatteryAgent NEVER learns B (no pB, no learn_B kwarg).
ToMBatteryAgent learns B when learn_B=True is passed.
This test compares ToM with learn_B=True vs learn_B=False on the battery.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from econet.real_weather import build_real_weather_env
from econet.climate import generate_climate_week
from econet.simulation import run_simulation, run_tom_simulation
from econet.metrics import compute_metrics

SCENARIOS = [
    ("london", "summer"),
    ("phoenix", "summer"),
]

NUM_DAYS = 7


def main():
    print("=" * 100)
    print("learn_B Battery Test: is B-learning the confound?")
    print("=" * 100)

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

    configs = [
        ("Aligned (learn_B=True)",  "aligned", {"learn_B": True, "aligned": True}),
        ("Aligned (learn_B=False)", "aligned", {"learn_B": False, "aligned": True}),
        ("ToM none (learn_B=True)", "tom",     {"learn_B": True, "social_weight": 0.0, "auditory_mode": "none"}),
        ("ToM none (learn_B=False)","tom",     {"learn_B": False, "social_weight": 0.0, "auditory_mode": "none"}),
        ("ToM full (learn_B=True)", "tom",     {"learn_B": True, "social_weight": 1.0, "auditory_mode": "full"}),
        ("ToM full (learn_B=False)","tom",     {"learn_B": False, "social_weight": 1.0, "auditory_mode": "full"}),
    ]

    all_metrics = {}
    for label, sim_type, kwargs in configs:
        print(f"\n--- {label} ---")
        all_metrics[label] = {}
        for key, env_data in scenarios.items():
            print(f"  {label}: {key}...", end=" ", flush=True)
            if sim_type == "aligned":
                result = run_simulation(
                    env_data=env_data, num_days=NUM_DAYS,
                    policy_len=4, gamma=16.0, seed=42, verbose=False,
                    **kwargs,
                )
            else:
                result = run_tom_simulation(
                    env_data=env_data, num_days=NUM_DAYS,
                    policy_len=4, gamma=16.0, seed=42, verbose=False,
                    **kwargs,
                )
            m = compute_metrics(result, num_days=NUM_DAYS)
            all_metrics[label][key] = m
            print(f"Cost: ${m.total_cost:.2f}, Comfort: {m.comfort_deviation_total:.1f}, "
                  f"Batt: {m.battery_utilization*100:.1f}%")

    # Summary table
    print("\n\n")
    print("=" * 130)
    hdr = f"{'Scenario':<24} {'Metric':<14}"
    for label, _, _ in configs:
        hdr += f" {label:>18}"
    print(hdr)
    print("-" * 130)

    for key in keys:
        row = f"{key:<24} {'Cost ($)':.<14}"
        for label, _, _ in configs:
            m = all_metrics[label][key]
            row += f" {m.total_cost:>18.2f}"
        print(row)

        row = f"{'':<24} {'Batt util %':.<14}"
        for label, _, _ in configs:
            m = all_metrics[label][key]
            row += f" {m.battery_utilization*100:>18.1f}"
        print(row)

        row = f"{'':<24} {'Comfort':.<14}"
        for label, _, _ in configs:
            m = all_metrics[label][key]
            row += f" {m.comfort_deviation_total:>18.1f}"
        print(row)
        print()

    print("=" * 130)


if __name__ == "__main__":
    main()
