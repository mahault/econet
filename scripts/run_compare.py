"""Compare coordination modes: Independent vs Aligned vs Hierarchical vs ToM+Belief.

Shows how sharing posterior beliefs (Friston 2023) enables superior coordination
vs implicit C-vector alignment alone.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from econet.environment import generate_multi_day
from econet.simulation import run_simulation, run_hierarchical_simulation, run_tom_simulation
from econet.baselines import run_no_hems, run_rule_based
from econet.metrics import compute_metrics, compute_communication_metrics

num_days = 7
env_data = generate_multi_day(num_days=num_days, seed=42)

print("=" * 78)
print("EcoNet: Coordination through Preference Alignment + Belief Sharing")
print("=" * 78)
print()
print("Modes:")
print("  Independent:   Static C vectors, no coordination signal")
print("  Aligned:       Dynamic C (TOU+occupancy aware) — decentralized coordination")
print("  Hierarchical:  Meta-agent sets C — centralized coordination")
print("  ToM+Belief:    Aligned + shared posteriors + ToM reliability gating")
print()

# Baselines
no_hems = run_no_hems(env_data)
rule = run_rule_based(env_data)

# Independent: static C, no TOU/occupancy adaptation
independent = run_simulation(env_data=env_data, num_days=num_days,
                             policy_len=4, gamma=16.0,
                             learn_B=True, aligned=False, verbose=False)

# Aligned (decentralized): dynamic C, agents independently respond to shared signals
aligned = run_simulation(env_data=env_data, num_days=num_days,
                         policy_len=4, gamma=16.0,
                         learn_B=True, aligned=True, verbose=False)

# Hierarchical: meta-agent explicitly coordinates both agents
hierarchical = run_hierarchical_simulation(env_data=env_data, num_days=num_days,
                                           policy_len=4, gamma=16.0,
                                           learn_B=True, verbose=False)

# ToM + Belief Sharing: aligned + shared posteriors + auditory A + ToM
tom_belief = run_tom_simulation(env_data=env_data, num_days=num_days,
                                policy_len=4, gamma=16.0,
                                learn_B=True, social_weight=2.0, verbose=False)

# Metrics
m_no = compute_metrics(no_hems, num_days)
m_rule = compute_metrics(rule, num_days, no_hems_result=no_hems)
m_ind = compute_metrics(independent, num_days, no_hems_result=no_hems)
m_ali = compute_metrics(aligned, num_days, no_hems_result=no_hems)
m_hier = compute_metrics(hierarchical, num_days, no_hems_result=no_hems)
m_tom = compute_metrics(tom_belief, num_days, no_hems_result=no_hems)

print(f"{'Strategy':<26} {'Cost ($)':>10} {'Comfort (C-h)':>15} {'GHG (kg)':>10}")
print("-" * 78)
for name, m, result in [
    ("No-HEMS (baseline)", m_no, no_hems),
    ("Rule-based", m_rule, rule),
    ("EcoNet: Independent", m_ind, independent),
    ("EcoNet: Aligned", m_ali, aligned),
    ("EcoNet: Hierarchical", m_hier, hierarchical),
    ("EcoNet: ToM+Belief", m_tom, tom_belief),
]:
    print(f"{name:<26} ${m.total_cost:>8.2f} {m.comfort_deviation_total:>13.1f} "
          f"{result.total_ghg:>10.1f}")

print()
print("=" * 78)
print("COORDINATION BENEFIT (lower comfort deviation = better, lower cost = better)")
print("=" * 78)
print()
print(f"Aligned vs Independent:")
print(f"  Comfort: {m_ind.comfort_deviation_total:.1f} -> {m_ali.comfort_deviation_total:.1f} "
      f"({m_ind.comfort_deviation_total - m_ali.comfort_deviation_total:+.1f} C-h)")
print(f"  Cost:    ${m_ind.total_cost:.2f} -> ${m_ali.total_cost:.2f} "
      f"(${m_ind.total_cost - m_ali.total_cost:+.2f})")
print()
print(f"Hierarchical vs Aligned:")
print(f"  Comfort: {m_ali.comfort_deviation_total:.1f} -> {m_hier.comfort_deviation_total:.1f} "
      f"({m_ali.comfort_deviation_total - m_hier.comfort_deviation_total:+.1f} C-h)")
print(f"  Cost:    ${m_ali.total_cost:.2f} -> ${m_hier.total_cost:.2f} "
      f"(${m_ali.total_cost - m_hier.total_cost:+.2f})")
print()
print(f"ToM+Belief vs Aligned:")
print(f"  Comfort: {m_ali.comfort_deviation_total:.1f} -> {m_tom.comfort_deviation_total:.1f} "
      f"({m_ali.comfort_deviation_total - m_tom.comfort_deviation_total:+.1f} C-h)")
print(f"  Cost:    ${m_ali.total_cost:.2f} -> ${m_tom.total_cost:.2f} "
      f"(${m_ali.total_cost - m_tom.total_cost:+.2f})")
print()
print(f"Aligned vs Rule-based:")
print(f"  Comfort: {m_rule.comfort_deviation_total:.1f} -> {m_ali.comfort_deviation_total:.1f} "
      f"({m_rule.comfort_deviation_total - m_ali.comfort_deviation_total:+.1f} C-h)")
print(f"  Cost:    ${m_rule.total_cost:.2f} -> ${m_ali.total_cost:.2f} "
      f"(${m_rule.total_cost - m_ali.total_cost:+.2f})")

# Communication metrics for ToM+Belief
print()
print("=" * 78)
print("BELIEF SHARING METRICS (ToM+Belief mode)")
print("=" * 78)
print()
cm = compute_communication_metrics(tom_belief)
print(f"  Avg comfort entropy (shared q(comfort)):  {cm.avg_comfort_entropy:.3f} nats")
print(f"  Avg SoC entropy (shared q(SoC)):          {cm.avg_soc_entropy:.3f} nats")
print(f"  Comfort belief diversity (argmax != COMFY): {cm.comfort_belief_diversity:.1%}")
print(f"  SoC belief diversity (argmax != mid):       {cm.soc_belief_diversity:.1%}")
print(f"  Thermo ToM reliability (final/avg):  {cm.final_thermo_tom_reliability:.3f} / "
      f"{cm.avg_thermo_tom_reliability:.3f}")
print(f"  Battery ToM reliability (final/avg): {cm.final_battery_tom_reliability:.3f} / "
      f"{cm.avg_battery_tom_reliability:.3f}")
