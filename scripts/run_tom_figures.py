"""Generate figures for Phase 3: Theory of Mind + Belief Sharing.

Produces:
  fig15_coordination_modes.pdf — 6-mode bar chart comparison
  fig16_belief_dynamics.pdf    — Belief sharing dynamics over time
  fig17_pareto_frontier.pdf    — Cost vs Comfort Pareto frontier
  fig18_tom_reliability.pdf    — ToM reliability convergence
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from econet.environment import generate_multi_day, STEPS_PER_DAY
from econet.simulation import run_simulation, run_hierarchical_simulation, run_tom_simulation
from econet.baselines import run_no_hems, run_rule_based
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
env_data = generate_multi_day(num_days=num_days, seed=42)

no_hems = run_no_hems(env_data)
rule = run_rule_based(env_data)

independent = run_simulation(env_data=env_data, num_days=num_days,
                             policy_len=4, gamma=16.0,
                             learn_B=True, aligned=False, verbose=False)

aligned = run_simulation(env_data=env_data, num_days=num_days,
                         policy_len=4, gamma=16.0,
                         learn_B=True, aligned=True, verbose=False)

hierarchical = run_hierarchical_simulation(env_data=env_data, num_days=num_days,
                                           policy_len=4, gamma=16.0,
                                           learn_B=True, verbose=False)

tom_belief = run_tom_simulation(env_data=env_data, num_days=num_days,
                                policy_len=4, gamma=16.0,
                                learn_B=True, social_weight=2.0, verbose=False)

# Compute metrics
m_no = compute_metrics(no_hems, num_days)
m_rule = compute_metrics(rule, num_days, no_hems_result=no_hems)
m_ind = compute_metrics(independent, num_days, no_hems_result=no_hems)
m_ali = compute_metrics(aligned, num_days, no_hems_result=no_hems)
m_hier = compute_metrics(hierarchical, num_days, no_hems_result=no_hems)
m_tom = compute_metrics(tom_belief, num_days, no_hems_result=no_hems)
cm = compute_communication_metrics(tom_belief)

print("Simulations complete. Generating figures...")

# ═════════════════════════════════════════════════════════════════════
# Fig 15: 6-mode coordination comparison (grouped bar chart)
# ═════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

modes = ["No-HEMS", "Rule-\nBased", "Indep.", "Aligned", "Hierarch.", "ToM+\nBelief"]
metrics_list = [m_no, m_rule, m_ind, m_ali, m_hier, m_tom]
colors = ["#bdc3c7", "#95a5a6", "#3498db", "#2ecc71", "#9b59b6", "#e74c3c"]

# Cost
costs = [m.total_cost for m in metrics_list]
bars = axes[0].bar(range(6), costs, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(range(6))
axes[0].set_xticklabels(modes, fontsize=9)
axes[0].set_ylabel("Total Cost ($)")
axes[0].set_title("(a) Energy Cost")
axes[0].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, costs):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"${val:.1f}", ha="center", va="bottom", fontsize=8)

# Comfort deviation
comforts = [m.comfort_deviation_total for m in metrics_list]
bars = axes[1].bar(range(6), comforts, color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_xticks(range(6))
axes[1].set_xticklabels(modes, fontsize=9)
axes[1].set_ylabel("Comfort Deviation ($^\\circ$C$\\cdot$h)")
axes[1].set_title("(b) Comfort Deviation")
axes[1].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, comforts):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)

# GHG
results_all = [no_hems, rule, independent, aligned, hierarchical, tom_belief]
ghgs = []
for r in results_all:
    if hasattr(r, "to_arrays"):
        ghgs.append(float(r.to_arrays()["ghg"].sum()))
    else:
        ghgs.append(float(r["ghg"].sum()))
bars = axes[2].bar(range(6), ghgs, color=colors, edgecolor="black", linewidth=0.5)
axes[2].set_xticks(range(6))
axes[2].set_xticklabels(modes, fontsize=9)
axes[2].set_ylabel("GHG Emissions (kg CO$_2$)")
axes[2].set_title("(c) GHG Emissions")
axes[2].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, ghgs):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Coordination Mode Comparison (7-day simulation)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "fig15_coordination_modes.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "fig15_coordination_modes.png", bbox_inches="tight")
print("  fig15_coordination_modes.pdf saved")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Fig 16: Belief dynamics over time
# ═════════════════════════════════════════════════════════════════════
bh = tom_belief.belief_history
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

fig.suptitle("Belief Sharing Dynamics (ToM+Belief mode, 7-day simulation)",
             fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "fig16_belief_dynamics.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "fig16_belief_dynamics.png", bbox_inches="tight")
print("  fig16_belief_dynamics.pdf saved")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Fig 17: Cost vs Comfort Pareto frontier
# ═════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

pareto_data = [
    ("No-HEMS", m_no, "#bdc3c7", "s"),
    ("Rule-Based", m_rule, "#95a5a6", "D"),
    ("Independent", m_ind, "#3498db", "o"),
    ("Aligned", m_ali, "#2ecc71", "^"),
    ("Hierarchical", m_hier, "#9b59b6", "v"),
    ("ToM+Belief", m_tom, "#e74c3c", "*"),
]

for name, m, color, marker in pareto_data:
    ax.scatter(m.total_cost, m.comfort_deviation_total,
              c=color, s=150, marker=marker, edgecolors="black",
              linewidth=0.8, zorder=5, label=name)

# Connect the EcoNet variants with a line to show the frontier
econet_costs = [m_ind.total_cost, m_tom.total_cost,
                m_ali.total_cost, m_hier.total_cost]
econet_comforts = [m_ind.comfort_deviation_total, m_tom.comfort_deviation_total,
                   m_ali.comfort_deviation_total, m_hier.comfort_deviation_total]
# Sort by cost for the frontier line
sorted_pairs = sorted(zip(econet_costs, econet_comforts))
ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
       "k--", alpha=0.4, lw=1.5, label="EcoNet frontier")

ax.set_xlabel("Total Energy Cost ($)", fontsize=12)
ax.set_ylabel("Comfort Deviation ($^\\circ$C$\\cdot$h)", fontsize=12)
ax.set_title("Cost vs. Comfort Pareto Frontier (7-day simulation)", fontsize=13)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)

# Annotate the arrow showing Pareto direction
ax.annotate("", xy=(min(econet_costs) - 1, min(econet_comforts) - 15),
           xytext=(min(econet_costs) + 2, min(econet_comforts) + 15),
           arrowprops=dict(arrowstyle="->", color="green", lw=2))
ax.text(min(econet_costs) - 0.5, min(econet_comforts) - 5,
       "Better", fontsize=10, color="green", fontstyle="italic")

fig.tight_layout()
fig.savefig(FIGURE_DIR / "fig17_pareto_frontier.pdf", bbox_inches="tight")
fig.savefig(FIGURE_DIR / "fig17_pareto_frontier.png", bbox_inches="tight")
print("  fig17_pareto_frontier.pdf saved")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# Fig 18: ToM Reliability convergence
# ═════════════════════════════════════════════════════════════════════
thermo_rels = bh.get("thermo_tom_reliability", [])
battery_rels = bh.get("battery_tom_reliability", [])

if thermo_rels and battery_rels:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(range(len(thermo_rels)), thermo_rels, "r-", alpha=0.8, lw=1.5,
           label=f"Thermostat ToM (final={thermo_rels[-1]:.3f})")
    ax.plot(range(len(battery_rels)), battery_rels, "b-", alpha=0.8, lw=1.5,
           label=f"Battery ToM (final={battery_rels[-1]:.3f})")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Initial prior (0.5)")

    for bd in day_bd:
        ax.axvline(bd, color="gray", ls=":", alpha=0.3)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("ToM Reliability ($\\rho$)")
    ax.set_title("Theory of Mind Reliability Convergence")
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
print("=" * 78)
print("RESULTS TABLE FOR PAPER")
print("=" * 78)
print()
print(f"{'Strategy':<26} {'Cost ($)':>10} {'Comfort (C-h)':>15} {'GHG (kg)':>10}")
print("-" * 68)
for name, m, result in [
    ("No-HEMS", m_no, no_hems),
    ("Rule-based", m_rule, rule),
    ("EcoNet: Independent", m_ind, independent),
    ("EcoNet: Aligned", m_ali, aligned),
    ("EcoNet: Hierarchical", m_hier, hierarchical),
    ("EcoNet: ToM+Belief", m_tom, tom_belief),
]:
    if hasattr(result, "total_ghg"):
        ghg = result.total_ghg
    elif hasattr(result, "to_arrays"):
        ghg = float(result.to_arrays()["ghg"].sum())
    else:
        ghg = float(result["ghg"].sum())
    print(f"{name:<26} ${m.total_cost:>8.2f} {m.comfort_deviation_total:>13.1f} "
          f"{ghg:>10.1f}")

print()
print("Communication metrics (ToM+Belief):")
print(f"  Avg comfort entropy:       {cm.avg_comfort_entropy:.3f} nats")
print(f"  Avg SoC entropy:           {cm.avg_soc_entropy:.3f} nats")
print(f"  Comfort belief diversity:  {cm.comfort_belief_diversity:.1%}")
print(f"  SoC belief diversity:      {cm.soc_belief_diversity:.1%}")
print(f"  Thermo ToM reliability:    {cm.final_thermo_tom_reliability:.3f}")
print(f"  Battery ToM reliability:   {cm.final_battery_tom_reliability:.3f}")

print()
print("All figures saved to:", FIGURE_DIR)
