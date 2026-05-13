"""Generate figures for coordination mode comparison.

Produces:
  fig15_coordination_modes.pdf — 10-condition bar chart comparison
  fig16_belief_dynamics.pdf    — Belief sharing dynamics over time (Federated mode)
  fig17_pareto_frontier.pdf    — Cost vs Comfort Pareto frontier
  fig18_tom_reliability.pdf    — ToM reliability convergence (Federated mode)

Conditions (10 total):
  Baselines:   No-HEMS, Rule-Based, Oracle, MPC, RL
  AIF modes:   Independent, Aligned, Hierarchical, Federated, ToM
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from econet.climate import generate_climate_week
from econet.environment import STEPS_PER_DAY
from econet.simulation import (
    run_simulation, run_tom_simulation,
    run_hierarchical_simulation, run_sophisticated_simulation,
)
from econet.baselines import run_no_hems, run_rule_based, run_oracle, run_mpc, run_rl
from econet.metrics import compute_metrics, compute_communication_metrics

FIGURE_DIR = Path(__file__).parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
})

# ── Run all simulations ──────────────────────────────────────────────
print("Running simulations...")
num_days = 7
env_data = generate_climate_week("london", "summer", seed=42)
total_steps = len(env_data["time_of_day"])

# Baselines
print("  [1/10] No-HEMS ...")
no_hems = run_no_hems(env_data)
print("  [2/10] Rule-Based ...")
rule = run_rule_based(env_data)
print("  [3/10] Oracle ...")
oracle = run_oracle(env_data, max_steps=total_steps)
print("  [4/10] MPC (horizon=4) ...")
mpc = run_mpc(env_data, horizon=4)
print("  [5/10] RL (Q-learning, 2000 episodes) ...")
rl = run_rl(env_data, num_episodes=2000)

# AIF modes
TUNED = dict(gamma=64.0, comfort_scale=3.0, soc_scale=2.0)

print("  [6/10] Independent (dynamic C) ...")
independent = run_simulation(env_data=env_data, num_days=num_days,
                             policy_len=4,
                             learn_B=False, aligned=False, verbose=False,
                             **TUNED)

print("  [7/10] Aligned (dynamic C) ...")
aligned = run_simulation(env_data=env_data, num_days=num_days,
                         policy_len=4,
                         learn_B=False, aligned=True, verbose=False,
                         **TUNED)

print("  [8/10] Hierarchical (meta-agent) ...")
hierarchical = run_hierarchical_simulation(
    env_data=env_data, num_days=num_days,
    policy_len=4,
    learn_B=False, verbose=False,
    **TUNED)

print("  [9/10] Federated (belief sharing) ...")
federated = run_tom_simulation(env_data=env_data, num_days=num_days,
                               policy_len=4,
                               learn_B=False, social_weight=1.0, verbose=False,
                               **TUNED)

print("  [10/10] ToM (sophisticated phantom inference) ...")
tom = run_sophisticated_simulation(
    env_data=env_data, num_days=num_days,
    policy_len=4,
    learn_B=False, verbose=False,
    **TUNED)

# Compute metrics
m_no = compute_metrics(no_hems, num_days)
m_rule = compute_metrics(rule, num_days, no_hems_result=no_hems)
m_oracle = compute_metrics(oracle, num_days, no_hems_result=no_hems)
m_mpc = compute_metrics(mpc, num_days, no_hems_result=no_hems)
m_rl = compute_metrics(rl, num_days, no_hems_result=no_hems)
m_indep = compute_metrics(independent, num_days, no_hems_result=no_hems)
m_ali = compute_metrics(aligned, num_days, no_hems_result=no_hems)
m_hier = compute_metrics(hierarchical, num_days, no_hems_result=no_hems)
m_fed = compute_metrics(federated, num_days, no_hems_result=no_hems)
m_tom = compute_metrics(tom, num_days, no_hems_result=no_hems)
cm = compute_communication_metrics(federated)

print("Simulations complete. Generating figures...")

# ═════════════════════════════════════════════════════════════════════
# Fig 15: 10-condition coordination comparison (grouped bar chart)
# ═════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

modes = ["No-\nHEMS", "Rule-\nBased", "Oracle", "MPC", "RL",
         "Indep.", "Aligned", "Hierar.", "Feder.", "ToM"]
metrics_list = [m_no, m_rule, m_oracle, m_mpc, m_rl,
                m_indep, m_ali, m_hier, m_fed, m_tom]
colors = ["#bdc3c7", "#95a5a6", "#f39c12", "#e67e22", "#d35400",
          "#3498db", "#2ecc71", "#9b59b6", "#e74c3c", "#c0392b"]
n_modes = len(modes)

# Cost
costs = [m.total_cost for m in metrics_list]
bars = axes[0].bar(range(n_modes), costs, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(range(n_modes))
axes[0].set_xticklabels(modes, fontsize=7)
axes[0].set_ylabel("Total Cost ($)")
axes[0].set_title("(a) Energy Cost")
axes[0].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, costs):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"${val:.1f}", ha="center", va="bottom", fontsize=6)

# Comfort deviation
comforts = [m.comfort_deviation_total for m in metrics_list]
bars = axes[1].bar(range(n_modes), comforts, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_xticks(range(n_modes))
axes[1].set_xticklabels(modes, fontsize=7)
axes[1].set_ylabel("Comfort Deviation ($^\\circ$C$\\cdot$h)")
axes[1].set_title("(b) Comfort Deviation")
axes[1].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, comforts):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=6)

# GHG
results_all = [no_hems, rule, oracle, mpc, rl,
               independent, aligned, hierarchical, federated, tom]
ghgs = []
for r in results_all:
    if hasattr(r, "total_ghg"):
        ghgs.append(float(r.total_ghg))
    elif hasattr(r, "to_arrays"):
        ghgs.append(float(r.to_arrays()["ghg"].sum()))
    else:
        ghgs.append(float(r["ghg"].sum()))
bars = axes[2].bar(range(n_modes), ghgs, color=colors, edgecolor="black", linewidth=0.5)
axes[2].set_xticks(range(n_modes))
axes[2].set_xticklabels(modes, fontsize=7)
axes[2].set_ylabel("GHG Emissions (kg CO$_2$)")
axes[2].set_title("(c) GHG Emissions")
axes[2].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, ghgs):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=6)

fig.suptitle("Coordination Mode Comparison (7-day London summer)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "fig15_coordination_modes.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "fig15_coordination_modes.png", bbox_inches="tight")
print("  fig15_coordination_modes.pdf saved")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Fig 16: Belief dynamics over time (Federated mode)
# ═════════════════════════════════════════════════════════════════════
bh = federated.belief_history
q_comforts = bh.get("q_comfort", [])
q_socs = bh.get("q_soc", [])
n_steps = len(q_comforts)
x = np.arange(n_steps)
day_bd = np.arange(0, n_steps + 1, STEPS_PER_DAY)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# q(comfort) heatmap
if q_comforts:
    q_c_matrix = np.array(q_comforts)  # (n_steps, 5)
    im0 = axes[0, 0].imshow(q_c_matrix.T, aspect="auto", cmap="YlOrRd",
                             origin="lower", vmin=0, vmax=1)
    axes[0, 0].set_yticks(range(5))
    axes[0, 0].set_yticklabels(["COLD", "COOL", "COMFY", "WARM", "HOT"])
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_title("(a) Shared $q$(comfort) — Thermostat $\\rightarrow$ Battery")
    plt.colorbar(im0, ax=axes[0, 0], label="Probability")
    for bd in day_bd:
        axes[0, 0].axvline(bd, color="white", ls=":", alpha=0.5)

# q(SoC) heatmap
if q_socs:
    q_s_matrix = np.array(q_socs)  # (n_steps, 5)
    im1 = axes[0, 1].imshow(q_s_matrix.T, aspect="auto", cmap="YlGnBu",
                             origin="lower", vmin=0, vmax=1)
    axes[0, 1].set_yticks(range(5))
    axes[0, 1].set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8"])
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_title("(b) Shared $q$(SoC) — Battery $\\rightarrow$ Thermostat")
    plt.colorbar(im1, ax=axes[0, 1], label="Probability")
    for bd in day_bd:
        axes[0, 1].axvline(bd, color="white", ls=":", alpha=0.5)

# Comfort entropy over time
if q_comforts:
    def _entropy(q):
        q = np.asarray(q)
        q = q[q > 0]
        return float(-np.sum(q * np.log(q + 1e-12)))

    c_entropies = [_entropy(q) for q in q_comforts]
    s_entropies = [_entropy(q) for q in q_socs] if q_socs else []

    axes[1, 0].plot(x, c_entropies, "r-", alpha=0.8, label="$q$(comfort)")
    if s_entropies:
        axes[1, 0].plot(x, s_entropies, "b-", alpha=0.8, label="$q$(SoC)")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Shannon Entropy (nats)")
    axes[1, 0].set_title("(c) Belief Entropy Over Time")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    for bd in day_bd:
        axes[1, 0].axvline(bd, color="gray", ls=":", alpha=0.3)

# Argmax trajectories
if q_comforts and q_socs:
    c_argmax = [np.argmax(q) for q in q_comforts]
    s_argmax = [np.argmax(q) for q in q_socs]
    axes[1, 1].step(x, c_argmax, "r-", alpha=0.8, where="mid",
                    label="comfort argmax")
    axes[1, 1].step(x, s_argmax, "b-", alpha=0.8, where="mid",
                    label="SoC argmax")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("MAP State Index")
    axes[1, 1].set_title("(d) Belief MAP Estimates")
    axes[1, 1].set_yticks(range(5))
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    for bd in day_bd:
        axes[1, 1].axvline(bd, color="gray", ls=":", alpha=0.3)

fig.suptitle("Belief Sharing Dynamics (Federated mode, 7-day simulation)",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "fig16_belief_dynamics.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "fig16_belief_dynamics.png", bbox_inches="tight")
print("  fig16_belief_dynamics.pdf saved")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Fig 17: Cost vs Comfort Pareto frontier
# ═════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))

pareto_data = [
    ("No-HEMS", m_no, "#bdc3c7", "s"),
    ("Rule-Based", m_rule, "#95a5a6", "D"),
    ("Oracle", m_oracle, "#f39c12", "P"),
    ("MPC", m_mpc, "#e67e22", "X"),
    ("RL", m_rl, "#d35400", "H"),
    ("Independent", m_indep, "#3498db", "v"),
    ("Aligned", m_ali, "#2ecc71", "^"),
    ("Hierarchical", m_hier, "#9b59b6", "d"),
    ("Federated", m_fed, "#e74c3c", "o"),
    ("ToM", m_tom, "#c0392b", "*"),
]

for name, m, color, marker in pareto_data:
    ax.scatter(m.total_cost, m.comfort_deviation_total,
              c=color, s=150, marker=marker, edgecolors="black",
              linewidth=0.8, zorder=5, label=name)

# Connect the AIF variants with a line to show the frontier
aif_modes = [(m_indep, "Independent"), (m_ali, "Aligned"),
             (m_hier, "Hierarchical"), (m_fed, "Federated"), (m_tom, "ToM")]
aif_costs = [m.total_cost for m, _ in aif_modes]
aif_comforts = [m.comfort_deviation_total for m, _ in aif_modes]
sorted_pairs = sorted(zip(aif_costs, aif_comforts))
ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
       "k--", alpha=0.4, lw=1.5, label="AIF frontier")

ax.set_xlabel("Total Energy Cost ($)", fontsize=12)
ax.set_ylabel("Comfort Deviation ($^\\circ$C$\\cdot$h)", fontsize=12)
ax.set_title("Cost vs. Comfort Pareto Frontier (7-day simulation)", fontsize=13)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Annotate direction
all_costs = [m.total_cost for _, m, _, _ in pareto_data]
all_comforts = [m.comfort_deviation_total for _, m, _, _ in pareto_data]
ax.annotate("", xy=(min(all_costs) - 1, min(all_comforts) - 15),
           xytext=(min(all_costs) + 2, min(all_comforts) + 15),
           arrowprops=dict(arrowstyle="->", color="green", lw=2))
ax.text(min(all_costs) - 0.5, min(all_comforts) - 5,
       "Better", fontsize=10, color="green", fontstyle="italic")

fig.tight_layout()
fig.savefig(FIGURE_DIR / "fig17_pareto_frontier.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "fig17_pareto_frontier.png", bbox_inches="tight")
print("  fig17_pareto_frontier.pdf saved")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Fig 18: ToM Reliability convergence (Federated mode)
# ═════════════════════════════════════════════════════════════════════
thermo_rels = bh.get("thermo_tom_reliability", [])
battery_rels = bh.get("battery_tom_reliability", [])

if thermo_rels and battery_rels:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(range(len(thermo_rels)), thermo_rels, "r-", alpha=0.8, lw=1.5,
           label=f"Thermostat reliability (final={thermo_rels[-1]:.3f})")
    ax.plot(range(len(battery_rels)), battery_rels, "b-", alpha=0.8, lw=1.5,
           label=f"Battery reliability (final={battery_rels[-1]:.3f})")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Initial prior (0.5)")

    for bd in day_bd:
        ax.axvline(bd, color="gray", ls=":", alpha=0.3)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reliability ($\\rho$)")
    ax.set_title("Belief Reliability Convergence (Federated mode)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig18_tom_reliability.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig18_tom_reliability.png", bbox_inches="tight")
    print("  fig18_tom_reliability.pdf saved")
    plt.close(fig)

# ── Print summary table for paper ────────────────────────────────────
print()
print("=" * 80)
print("RESULTS TABLE FOR PAPER (Single seed=42, London summer, 7-day)")
print("=" * 80)
print()
print(f"{'Strategy':<30} {'Cost ($)':>10} {'Comfort (C-h)':>15} {'GHG (kg)':>10} {'Batt%':>8}")
print("-" * 78)
for name, m, result in [
    ("No-HEMS", m_no, no_hems),
    ("Rule-based", m_rule, rule),
    ("Oracle", m_oracle, oracle),
    ("MPC", m_mpc, mpc),
    ("RL (Q-learning)", m_rl, rl),
    ("EcoNet: Independent", m_indep, independent),
    ("EcoNet: Aligned", m_ali, aligned),
    ("EcoNet: Hierarchical", m_hier, hierarchical),
    ("EcoNet: Federated", m_fed, federated),
    ("EcoNet: ToM", m_tom, tom),
]:
    if hasattr(result, "total_ghg"):
        ghg = result.total_ghg
    elif hasattr(result, "to_arrays"):
        ghg = float(result.to_arrays()["ghg"].sum())
    else:
        ghg = float(result["ghg"].sum())
    print(f"{name:<30} ${m.total_cost:>8.2f} {m.comfort_deviation_total:>13.1f} "
          f"{ghg:>10.1f} {m.battery_utilization:>7.1%}")

print()
print("Communication metrics (Federated mode):")
print(f"  Avg comfort entropy:       {cm.avg_comfort_entropy:.3f} nats")
print(f"  Avg SoC entropy:           {cm.avg_soc_entropy:.3f} nats")
print(f"  Comfort belief diversity:  {cm.comfort_belief_diversity:.1%}")
print(f"  SoC belief diversity:      {cm.soc_belief_diversity:.1%}")
print(f"  Thermo reliability:        {cm.final_thermo_tom_reliability:.3f}")
print(f"  Battery reliability:       {cm.final_battery_tom_reliability:.3f}")

print()
print("All figures saved to:", FIGURE_DIR)
