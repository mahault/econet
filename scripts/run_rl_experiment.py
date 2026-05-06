"""Experiment B: Improved RL Baseline.

Part 1 — Convergence curve: RL at different episode counts with both
original (10 bins) and improved (34 bins) versions. Compare against
AIF Aligned and Oracle reference lines.

Part 2 — Cross-climate transfer: Train RL on London summer, evaluate
on Phoenix summer (no retraining). AIF works on any climate with zero
training.

Output: fig20_rl_convergence.pdf
"""

import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from econet.climate import generate_climate_week, get_scenario_label, CLIMATE_PROFILES
from econet.simulation import run_simulation
from econet.baselines import run_oracle, run_rl, run_rl_improved

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
CLIMATES = ["london_summer", "montreal_winter", "phoenix_summer"]
EPISODE_COUNTS = [500, 1000, 2000, 5000, 10000, 20000]


def run_convergence_experiment():
    """Part 1: RL convergence curves."""
    print("=" * 60)
    print("Experiment B, Part 1: RL Convergence Curves")
    print("=" * 60)

    results = {}

    for climate_key in CLIMATES:
        city, season = climate_key.split("_")
        label = get_scenario_label(climate_key)
        print(f"\n--- {label} ---")

        results[climate_key] = {
            "rl_original": {},
            "rl_improved": {},
            "aif_aligned": [],
            "oracle": [],
        }

        for seed in SEEDS:
            env_data = generate_climate_week(city, season, seed=seed)
            num_days = 7

            # Oracle reference
            print(f"  Oracle (seed={seed})...")
            oracle_result = run_oracle(env_data, max_steps=len(env_data["time_of_day"]))
            results[climate_key]["oracle"].append(oracle_result.total_cost)

            # AIF Aligned reference
            print(f"  AIF Aligned (seed={seed})...")
            aif_result = run_simulation(
                env_data=env_data, num_days=num_days, seed=seed,
                aligned=True, verbose=False)
            results[climate_key]["aif_aligned"].append(aif_result.total_cost)

            # RL original (10 bins) at each episode count
            for n_ep in EPISODE_COUNTS:
                key = (n_ep, seed)
                print(f"  RL original (10 bins, {n_ep} eps, seed={seed})...")
                rl_orig = run_rl(env_data, num_episodes=n_ep, seed=seed)
                results[climate_key]["rl_original"].setdefault(n_ep, []).append(
                    rl_orig.total_cost)

            # RL improved (34 bins) at each episode count
            for n_ep in EPISODE_COUNTS:
                print(f"  RL improved (34 bins, {n_ep} eps, seed={seed})...")
                rl_imp = run_rl_improved(env_data, num_episodes=n_ep, seed=seed)
                results[climate_key]["rl_improved"].setdefault(n_ep, []).append(
                    rl_imp.total_cost)

    return results


def run_transfer_experiment():
    """Part 2: Cross-climate transfer."""
    print("\n" + "=" * 60)
    print("Experiment B, Part 2: Cross-Climate Transfer")
    print("=" * 60)

    transfer_results = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # Train on London summer
        train_data = generate_climate_week("london", "summer", seed=seed)
        print("  Training RL on London summer (10,000 episodes)...")
        # We need the Q-table, so we'll train and keep it via run_rl_improved
        # For transfer test, train on London then evaluate on Phoenix
        # run_rl_improved trains and evaluates on same data; for transfer,
        # we need to separate train and eval. We'll use the internal structure.
        from econet.baselines import (
            _simulate_step_physics, _evaluate_greedy,
            TEMP_LEVELS, TEMP_MIN, TEMP_MAX, STEPS_PER_DAY,
            discretize_soc,
        )

        rng = np.random.RandomState(seed)
        total_steps = len(train_data["time_of_day"])
        num_temp_bins = TEMP_LEVELS

        def disc_temp(room_temp):
            frac = (room_temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
            return int(np.clip(int(frac * num_temp_bins), 0, num_temp_bins - 1))

        def time_block(step):
            hour = (step % STEPS_PER_DAY) * 2
            return min(int(hour // 6), 3)

        def get_state(room_temp, soc, tou_high, occ, step):
            return (disc_temp(room_temp), discretize_soc(soc),
                    int(tou_high), int(occ), time_block(step))

        Q = np.zeros((num_temp_bins, 5, 2, 2, 4, 9))
        alpha_init = 0.15
        gamma_rl = 0.99
        reward_mean, reward_var, reward_count = 0.0, 1.0, 0
        num_episodes = 10000
        comfort_weight = 0.3
        initial_room_temp = 20.0
        initial_soc = 0.5

        for ep in range(num_episodes):
            epsilon = max(0.02, 1.0 - ep / (num_episodes * 0.8))
            alpha = alpha_init * max(0.3, 1.0 - ep / num_episodes)
            room_temp = initial_room_temp + rng.uniform(-2, 2)
            soc = initial_soc

            for t in range(total_steps):
                s = get_state(room_temp, soc,
                              train_data["tou_high"][t],
                              train_data["occupancy"][t], t)
                if rng.random() < epsilon:
                    action = rng.randint(9)
                else:
                    action = int(np.argmax(Q[s]))

                hvac_a = action // 3
                batt_a = action % 3
                cost, comfort, T_new, new_soc, _, _ = _simulate_step_physics(
                    room_temp, soc, hvac_a, batt_a, t, train_data)

                raw_reward = -(cost + comfort_weight * comfort)
                reward_count += 1
                delta = raw_reward - reward_mean
                reward_mean += delta / reward_count
                delta2 = raw_reward - reward_mean
                reward_var += (delta * delta2 - reward_var) / reward_count
                std = max(np.sqrt(reward_var), 1e-6)
                reward = (raw_reward - reward_mean) / std

                t_next = min(t + 1, total_steps - 1)
                s_next = get_state(T_new, new_soc,
                                   train_data["tou_high"][t_next],
                                   train_data["occupancy"][t_next], t_next)
                Q[s][action] += alpha * (
                    reward + gamma_rl * np.max(Q[s_next]) - Q[s][action])
                room_temp = T_new
                soc = new_soc

        # Evaluate on London summer (in-distribution)
        london_eval = _evaluate_greedy(
            Q, train_data, initial_room_temp, initial_soc,
            get_state, num_temp_bins)

        # Evaluate on Phoenix summer (transfer — no retraining)
        phoenix_data = generate_climate_week("phoenix", "summer", seed=seed)
        phoenix_eval = _evaluate_greedy(
            Q, phoenix_data, initial_room_temp, initial_soc,
            get_state, num_temp_bins)

        # AIF on both (zero training)
        london_aif = run_simulation(
            env_data=train_data, num_days=7, seed=seed,
            aligned=True, verbose=False)
        phoenix_aif = run_simulation(
            env_data=phoenix_data, num_days=7, seed=seed,
            aligned=True, verbose=False)

        # Oracle on both
        london_oracle = run_oracle(train_data,
                                   max_steps=len(train_data["time_of_day"]))
        phoenix_oracle = run_oracle(phoenix_data,
                                    max_steps=len(phoenix_data["time_of_day"]))

        transfer_results.setdefault("london_rl", []).append(london_eval.total_cost)
        transfer_results.setdefault("phoenix_rl_transfer", []).append(phoenix_eval.total_cost)
        transfer_results.setdefault("london_aif", []).append(london_aif.total_cost)
        transfer_results.setdefault("phoenix_aif", []).append(phoenix_aif.total_cost)
        transfer_results.setdefault("london_oracle", []).append(london_oracle.total_cost)
        transfer_results.setdefault("phoenix_oracle", []).append(phoenix_oracle.total_cost)

        # Also train fresh RL on Phoenix for comparison
        print("  Training fresh RL on Phoenix summer...")
        phoenix_fresh = run_rl_improved(phoenix_data, num_episodes=10000, seed=seed)
        transfer_results.setdefault("phoenix_rl_fresh", []).append(
            phoenix_fresh.total_cost)

    return transfer_results


def plot_convergence(results):
    """Generate fig20_rl_convergence.pdf."""
    n_climates = len(CLIMATES)
    fig, axes = plt.subplots(1, n_climates, figsize=(5 * n_climates, 4.5),
                             sharey=False)
    if n_climates == 1:
        axes = [axes]

    for ax, climate_key in zip(axes, CLIMATES):
        label = get_scenario_label(climate_key)
        data = results[climate_key]

        # Oracle reference line
        oracle_mean = np.mean(data["oracle"])
        ax.axhline(oracle_mean, color="black", linestyle="--",
                    linewidth=1.5, label=f"Oracle (${oracle_mean:.2f})")

        # AIF reference line
        aif_mean = np.mean(data["aif_aligned"])
        ax.axhline(aif_mean, color="#2ecc71", linestyle="-.",
                    linewidth=1.5, label=f"AIF Aligned (${aif_mean:.2f})")

        # RL original (10 bins)
        means_orig = [np.mean(data["rl_original"][n]) for n in EPISODE_COUNTS]
        stds_orig = [np.std(data["rl_original"][n]) for n in EPISODE_COUNTS]
        ax.errorbar(EPISODE_COUNTS, means_orig, yerr=stds_orig,
                     color="#e74c3c", marker="o", markersize=5,
                     capsize=3, linewidth=1.5, label="RL (10 bins)")

        # RL improved (34 bins)
        means_imp = [np.mean(data["rl_improved"][n]) for n in EPISODE_COUNTS]
        stds_imp = [np.std(data["rl_improved"][n]) for n in EPISODE_COUNTS]
        ax.errorbar(EPISODE_COUNTS, means_imp, yerr=stds_imp,
                     color="#3498db", marker="s", markersize=5,
                     capsize=3, linewidth=1.5, label="RL (34 bins)")

        ax.set_xscale("log")
        ax.set_xlabel("Training Episodes")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("RL Convergence: Cost vs Training Episodes", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = FIGURE_DIR / "fig20_rl_convergence.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nFig 20 saved: {out_path}")


def plot_transfer(transfer_results):
    """Print transfer results table."""
    print("\n" + "=" * 60)
    print("Cross-Climate Transfer Results")
    print("=" * 60)

    def fmt(key):
        vals = transfer_results[key]
        return f"${np.mean(vals):.2f} +/- ${np.std(vals):.2f}"

    print(f"\n{'Method':<30} {'London Summer':>18} {'Phoenix Summer':>18}")
    print("-" * 68)
    print(f"{'Oracle':.<30} {fmt('london_oracle'):>18} {fmt('phoenix_oracle'):>18}")
    print(f"{'AIF Aligned (zero training)':.<30} {fmt('london_aif'):>18} {fmt('phoenix_aif'):>18}")
    print(f"{'RL (trained on London)':.<30} {fmt('london_rl'):>18} {fmt('phoenix_rl_transfer'):>18}")
    print(f"{'RL (trained on Phoenix)':.<30} {'N/A':>18} {fmt('phoenix_rl_fresh'):>18}")

    # Compute degradation
    london_rl = np.mean(transfer_results["london_rl"])
    phoenix_transfer = np.mean(transfer_results["phoenix_rl_transfer"])
    phoenix_fresh = np.mean(transfer_results["phoenix_rl_fresh"])
    degradation = (phoenix_transfer - phoenix_fresh) / phoenix_fresh * 100

    print(f"\nRL transfer degradation (London->Phoenix): {degradation:+.1f}%")
    print("AIF transfer degradation: 0% (model-based, no training needed)")


def main():
    start = time.time()

    # Part 1: Convergence curves
    convergence_results = run_convergence_experiment()
    plot_convergence(convergence_results)

    # Print summary
    print("\n--- Convergence Summary ---")
    for climate_key in CLIMATES:
        label = get_scenario_label(climate_key)
        data = convergence_results[climate_key]
        oracle = np.mean(data["oracle"])
        aif = np.mean(data["aif_aligned"])
        best_rl = np.mean(data["rl_improved"][20000])
        aif_gap = (aif - oracle) / oracle * 100
        rl_gap = (best_rl - oracle) / oracle * 100
        print(f"  {label}: Oracle=${oracle:.2f}, "
              f"AIF={aif_gap:+.1f}%, RL-20k={rl_gap:+.1f}%")

    # Part 2: Cross-climate transfer
    transfer_results = run_transfer_experiment()
    plot_transfer(transfer_results)

    elapsed = time.time() - start
    print(f"\nExperiment B complete. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
