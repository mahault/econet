"""Experiment A: Uncertainty Robustness (AIF vs MPC under forecast noise).

Adds temperature forecast noise at sigma = {0, 1, 2, 3, 4} C.
MPC and AIF plan against noisy forecasts; environment evolves with true values.
Oracle retains perfect foresight.

Output: fig19_uncertainty_robustness.pdf
"""

import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from econet.environment import generate_forecasts
from econet.climate import generate_climate_week, get_scenario_label
from econet.simulation import run_simulation, run_tom_simulation
from econet.baselines import run_oracle, run_mpc

FIGURE_DIR = Path(__file__).parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

SEEDS = [42, 137, 256]
CLIMATES = ["london_summer", "london_winter", "montreal_winter", "phoenix_summer"]
TEMP_NOISE_SIGMAS = [0, 1, 2, 3, 4]  # C


def run_experiment():
    """Run uncertainty robustness experiment."""
    print("=" * 60)
    print("Experiment A: Uncertainty Robustness")
    print("=" * 60)

    # results[climate][method][sigma] = list of costs across seeds
    results = {}

    for climate_key in CLIMATES:
        city, season = climate_key.split("_")
        label = get_scenario_label(climate_key)
        print(f"\n{'='*50}")
        print(f"Climate: {label}")
        print(f"{'='*50}")

        results[climate_key] = {
            "Oracle": {s: [] for s in TEMP_NOISE_SIGMAS},
            "MPC": {s: [] for s in TEMP_NOISE_SIGMAS},
            "AIF Aligned": {s: [] for s in TEMP_NOISE_SIGMAS},
            "AIF ToM": {s: [] for s in TEMP_NOISE_SIGMAS},
        }

        for seed in SEEDS:
            env_data = generate_climate_week(city, season, seed=seed)
            num_days = 7
            total_steps = len(env_data["time_of_day"])

            for sigma in TEMP_NOISE_SIGMAS:
                # Proportional solar noise: sigma=4 C -> 40% solar noise
                solar_noise = sigma * 0.10

                print(f"  seed={seed}, sigma={sigma}C "
                      f"(solar_noise={solar_noise:.0%})...")

                # Generate noisy forecast
                forecast_seed = seed * 1000 + sigma
                if sigma == 0:
                    forecast = None  # perfect foresight
                else:
                    forecast = generate_forecasts(
                        env_data,
                        temp_noise_std=float(sigma),
                        solar_noise_std=solar_noise,
                        seed=forecast_seed,
                    )

                # Oracle: always uses true data (perfect foresight)
                oracle_result = run_oracle(
                    env_data, max_steps=total_steps)
                results[climate_key]["Oracle"][sigma].append(
                    oracle_result.total_cost)

                # MPC: plans against forecast, executes against real env
                mpc_result = run_mpc(
                    env_data, horizon=6,
                    forecast_data=forecast)
                results[climate_key]["MPC"][sigma].append(
                    mpc_result.total_cost)

                # AIF Aligned: predictive B uses forecast
                aif_result = run_simulation(
                    env_data=env_data, num_days=num_days, seed=seed,
                    aligned=True, verbose=False,
                    forecast_data=forecast)
                results[climate_key]["AIF Aligned"][sigma].append(
                    aif_result.total_cost)

                # AIF ToM: predictive B uses forecast
                tom_result = run_tom_simulation(
                    env_data=env_data, num_days=num_days, seed=seed,
                    verbose=False,
                    forecast_data=forecast)
                results[climate_key]["AIF ToM"][sigma].append(
                    tom_result.total_cost)

    return results


def plot_results(results):
    """Generate fig19_uncertainty_robustness.pdf."""
    n_climates = len(CLIMATES)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    method_colors = {
        "Oracle": "black",
        "MPC": "#e74c3c",
        "AIF Aligned": "#2ecc71",
        "AIF ToM": "#3498db",
    }
    method_markers = {
        "Oracle": "D",
        "MPC": "o",
        "AIF Aligned": "s",
        "AIF ToM": "^",
    }

    for ax, climate_key in zip(axes, CLIMATES):
        label = get_scenario_label(climate_key)
        data = results[climate_key]

        for method in ["Oracle", "MPC", "AIF Aligned", "AIF ToM"]:
            means = [np.mean(data[method][s]) for s in TEMP_NOISE_SIGMAS]
            stds = [np.std(data[method][s]) for s in TEMP_NOISE_SIGMAS]

            # Normalize to percentage increase from sigma=0
            base = means[0]
            pct_increase = [(m - base) / base * 100 for m in means]
            pct_std = [s / base * 100 for s in stds]

            ax.errorbar(
                TEMP_NOISE_SIGMAS, pct_increase, yerr=pct_std,
                color=method_colors[method],
                marker=method_markers[method],
                markersize=6, capsize=3, linewidth=1.5,
                label=method,
                linestyle="--" if method == "Oracle" else "-",
            )

        ax.set_xlabel("Forecast Noise $\\sigma$ ($\\degree$C)")
        ax.set_ylabel("Cost Increase (%)")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    fig.suptitle("Robustness Under Forecast Uncertainty", fontsize=14, y=1.01)
    fig.tight_layout()

    out_path = FIGURE_DIR / "fig19_uncertainty_robustness.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nFig 19 saved: {out_path}")

    # Also plot absolute costs
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    axes2 = axes2.flatten()

    for ax, climate_key in zip(axes2, CLIMATES):
        label = get_scenario_label(climate_key)
        data = results[climate_key]

        for method in ["Oracle", "MPC", "AIF Aligned", "AIF ToM"]:
            means = [np.mean(data[method][s]) for s in TEMP_NOISE_SIGMAS]
            stds = [np.std(data[method][s]) for s in TEMP_NOISE_SIGMAS]

            ax.errorbar(
                TEMP_NOISE_SIGMAS, means, yerr=stds,
                color=method_colors[method],
                marker=method_markers[method],
                markersize=6, capsize=3, linewidth=1.5,
                label=method,
                linestyle="--" if method == "Oracle" else "-",
            )

        ax.set_xlabel("Forecast Noise $\\sigma$ ($\\degree$C)")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Absolute Cost Under Forecast Uncertainty", fontsize=14, y=1.01)
    fig2.tight_layout()

    out_path2 = FIGURE_DIR / "fig19_uncertainty_absolute.pdf"
    fig2.savefig(out_path2, bbox_inches="tight")
    fig2.savefig(out_path2.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig2)
    print(f"  Absolute cost plot saved: {out_path2}")


def print_summary(results):
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("Uncertainty Robustness Summary")
    print("=" * 70)

    for climate_key in CLIMATES:
        label = get_scenario_label(climate_key)
        data = results[climate_key]
        print(f"\n--- {label} ---")
        print(f"{'Method':<15} {'sigma=0':>10} {'sigma=2':>10} {'sigma=4':>10} "
              f"{'Increase':>10}")
        print("-" * 58)

        for method in ["Oracle", "MPC", "AIF Aligned", "AIF ToM"]:
            c0 = np.mean(data[method][0])
            c2 = np.mean(data[method][2])
            c4 = np.mean(data[method][4])
            increase = (c4 - c0) / c0 * 100 if c0 > 0 else 0
            print(f"{method:<15} ${c0:>8.2f} ${c2:>8.2f} ${c4:>8.2f} "
                  f"{increase:>+8.1f}%")


def main():
    start = time.time()

    results = run_experiment()
    plot_results(results)
    print_summary(results)

    elapsed = time.time() - start
    print(f"\nExperiment A complete. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
