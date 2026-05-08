"""Experiment A: All-Variants Uncertainty Robustness.

Tests 8 methods (Oracle, MPC, 6 AIF variants) under forecast noise.
Adds temperature forecast noise at sigma = {0, 1, 2, 3, 4} C.
MPC and AIF plan against noisy forecasts; environment evolves with true values.
Oracle retains perfect foresight.

Output: fig19_uncertainty_robustness.pdf, fig19_uncertainty_absolute.pdf
"""

import sys
import time
import traceback
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from econet.environment import generate_forecasts
from econet.climate import generate_climate_week, get_scenario_label
from econet.simulation import (
    run_simulation,
    run_tom_simulation,
    run_sophisticated_simulation,
    run_sophisticated_tom_simulation,
    run_full_sophisticated_simulation,
)
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
CLIMATES = ["london_summer", "phoenix_summer"]
TEMP_NOISE_SIGMAS = [0, 1, 2, 3, 4]  # C

# All 8 methods to test
METHOD_NAMES = [
    "Oracle",
    "MPC",
    "AIF Independent",
    "AIF Aligned",
    "AIF Federated",
    "AIF Sophisticated",
    "AIF Soph+ToM",
    "AIF Full Soph",
]


TUNED = dict(gamma=64.0, comfort_scale=3.0, soc_scale=2.0)


def _run_method(method_name, env_data, forecast, seed, num_days):
    """Run a single method with error handling."""
    total_steps = len(env_data["time_of_day"])

    if method_name == "Oracle":
        result = run_oracle(env_data, max_steps=total_steps)
    elif method_name == "MPC":
        result = run_mpc(env_data, horizon=6, forecast_data=forecast)
    elif method_name == "AIF Independent":
        result = run_simulation(
            env_data=env_data, num_days=num_days, seed=seed,
            aligned=False, verbose=False, forecast_data=forecast,
            **TUNED)
    elif method_name == "AIF Aligned":
        result = run_simulation(
            env_data=env_data, num_days=num_days, seed=seed,
            aligned=True, verbose=False, forecast_data=forecast,
            **TUNED)
    elif method_name == "AIF Federated":
        result = run_tom_simulation(
            env_data=env_data, num_days=num_days, seed=seed,
            verbose=False, forecast_data=forecast,
            **TUNED)
    elif method_name == "AIF Sophisticated":
        result = run_sophisticated_simulation(
            env_data=env_data, num_days=num_days, seed=seed,
            verbose=False, forecast_data=forecast,
            **TUNED)
    elif method_name == "AIF Soph+ToM":
        result = run_sophisticated_tom_simulation(
            env_data=env_data, num_days=num_days, seed=seed,
            verbose=False, forecast_data=forecast,
            **TUNED)
    elif method_name == "AIF Full Soph":
        result = run_full_sophisticated_simulation(
            env_data=env_data, num_days=num_days, seed=seed,
            verbose=False, forecast_data=forecast,
            **TUNED)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return result.total_cost


def run_experiment():
    """Run uncertainty robustness experiment with all 8 methods."""
    print("=" * 60)
    print("Experiment A: All-Variants Uncertainty Robustness")
    print(f"  Methods: {len(METHOD_NAMES)}")
    print(f"  Climates: {CLIMATES}")
    print(f"  Noise levels: {TEMP_NOISE_SIGMAS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {len(METHOD_NAMES) * len(TEMP_NOISE_SIGMAS) * len(CLIMATES) * len(SEEDS)}")
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
            m: {s: [] for s in TEMP_NOISE_SIGMAS}
            for m in METHOD_NAMES
        }

        for seed in SEEDS:
            env_data = generate_climate_week(city, season, seed=seed)
            num_days = 7

            for sigma in TEMP_NOISE_SIGMAS:
                # Proportional solar noise: sigma=4 C -> 40% solar noise
                solar_noise = sigma * 0.10

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

                for method in METHOD_NAMES:
                    print(f"  {method} | seed={seed}, sigma={sigma}C...",
                          end=" ", flush=True)
                    try:
                        cost = _run_method(
                            method, env_data, forecast, seed, num_days)
                        results[climate_key][method][sigma].append(cost)
                        print(f"${cost:.2f}")
                    except Exception as e:
                        print(f"FAILED: {e}")
                        traceback.print_exc()
                        # Use NaN as fallback — will be filtered in plotting
                        results[climate_key][method][sigma].append(np.nan)

    return results


def plot_results(results):
    """Generate fig19_uncertainty_robustness.pdf (percentage cost increase)."""
    n_climates = len(CLIMATES)
    fig, axes = plt.subplots(1, n_climates, figsize=(6 * n_climates, 5),
                             sharey=False)
    if n_climates == 1:
        axes = [axes]

    method_colors = {
        "Oracle": "black",
        "MPC": "#e74c3c",
        "AIF Independent": "#95a5a6",
        "AIF Aligned": "#2ecc71",
        "AIF Federated": "#3498db",
        "AIF Sophisticated": "#9b59b6",
        "AIF Soph+ToM": "#e67e22",
        "AIF Full Soph": "#1abc9c",
    }
    method_markers = {
        "Oracle": "D",
        "MPC": "o",
        "AIF Independent": "v",
        "AIF Aligned": "s",
        "AIF Federated": "^",
        "AIF Sophisticated": "P",
        "AIF Soph+ToM": "*",
        "AIF Full Soph": "X",
    }
    method_linestyles = {
        "Oracle": "--",
        "MPC": "-",
        "AIF Independent": "-",
        "AIF Aligned": "-",
        "AIF Federated": "-",
        "AIF Sophisticated": "-",
        "AIF Soph+ToM": "-",
        "AIF Full Soph": "-",
    }

    for ax, climate_key in zip(axes, CLIMATES):
        label = get_scenario_label(climate_key)
        data = results[climate_key]

        for method in METHOD_NAMES:
            costs_by_sigma = []
            stds_by_sigma = []
            for s in TEMP_NOISE_SIGMAS:
                vals = [v for v in data[method][s] if not np.isnan(v)]
                if vals:
                    costs_by_sigma.append(np.mean(vals))
                    stds_by_sigma.append(np.std(vals))
                else:
                    costs_by_sigma.append(np.nan)
                    stds_by_sigma.append(0)

            # Normalize to percentage increase from sigma=0
            base = costs_by_sigma[0]
            if np.isnan(base) or base == 0:
                continue
            pct_increase = [(m - base) / base * 100 for m in costs_by_sigma]
            pct_std = [s / base * 100 for s in stds_by_sigma]

            ax.errorbar(
                TEMP_NOISE_SIGMAS, pct_increase, yerr=pct_std,
                color=method_colors[method],
                marker=method_markers[method],
                markersize=6, capsize=3, linewidth=1.5,
                label=method,
                linestyle=method_linestyles[method],
            )

        ax.set_xlabel("Forecast Noise $\\sigma$ ($\\degree$C)")
        ax.set_ylabel("Cost Increase (%)")
        ax.set_title(label)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    fig.suptitle("Robustness Under Forecast Uncertainty (All AIF Variants)",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    out_path = FIGURE_DIR / "fig19_uncertainty_robustness.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nFig 19a (percentage) saved: {out_path}")

    # Also plot absolute costs
    fig2, axes2 = plt.subplots(1, n_climates, figsize=(6 * n_climates, 5),
                               sharey=False)
    if n_climates == 1:
        axes2 = [axes2]

    for ax, climate_key in zip(axes2, CLIMATES):
        label = get_scenario_label(climate_key)
        data = results[climate_key]

        for method in METHOD_NAMES:
            means = []
            stds = []
            for s in TEMP_NOISE_SIGMAS:
                vals = [v for v in data[method][s] if not np.isnan(v)]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(np.nan)
                    stds.append(0)

            ax.errorbar(
                TEMP_NOISE_SIGMAS, means, yerr=stds,
                color=method_colors[method],
                marker=method_markers[method],
                markersize=6, capsize=3, linewidth=1.5,
                label=method,
                linestyle=method_linestyles[method],
            )

        ax.set_xlabel("Forecast Noise $\\sigma$ ($\\degree$C)")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title(label)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Absolute Cost Under Forecast Uncertainty (All AIF Variants)",
                  fontsize=14, y=1.01)
    fig2.tight_layout()

    out_path2 = FIGURE_DIR / "fig19_uncertainty_absolute.pdf"
    fig2.savefig(out_path2, bbox_inches="tight")
    fig2.savefig(out_path2.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig2)
    print(f"Fig 19b (absolute) saved: {out_path2}")


def print_summary(results):
    """Print summary table of results."""
    print("\n" + "=" * 90)
    print("Uncertainty Robustness Summary")
    print("=" * 90)

    for climate_key in CLIMATES:
        label = get_scenario_label(climate_key)
        data = results[climate_key]
        print(f"\n--- {label} ---")
        print(f"{'Method':<22} {'sigma=0':>10} {'sigma=2':>10} {'sigma=4':>10} "
              f"{'Increase':>10}")
        print("-" * 65)

        for method in METHOD_NAMES:
            vals0 = [v for v in data[method][0] if not np.isnan(v)]
            vals2 = [v for v in data[method][2] if not np.isnan(v)]
            vals4 = [v for v in data[method][4] if not np.isnan(v)]

            c0 = np.mean(vals0) if vals0 else np.nan
            c2 = np.mean(vals2) if vals2 else np.nan
            c4 = np.mean(vals4) if vals4 else np.nan
            increase = (c4 - c0) / c0 * 100 if (vals0 and vals4 and c0 > 0) else np.nan

            c0_s = f"${c0:.2f}" if not np.isnan(c0) else "N/A"
            c2_s = f"${c2:.2f}" if not np.isnan(c2) else "N/A"
            c4_s = f"${c4:.2f}" if not np.isnan(c4) else "N/A"
            inc_s = f"{increase:+.1f}%" if not np.isnan(increase) else "N/A"

            print(f"{method:<22} {c0_s:>10} {c2_s:>10} {c4_s:>10} {inc_s:>10}")


def main():
    start = time.time()

    results = run_experiment()
    plot_results(results)
    print_summary(results)

    elapsed = time.time() - start
    print(f"\nExperiment A complete. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
