"""Smoke test: symmetric sophisticated simulation (1 scenario, 1 day)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from econet.climate import generate_climate_week
from econet.simulation import run_full_sophisticated_simulation

print("Generating london_summer data (1 day)...")
env_data = generate_climate_week("london", "summer", seed=42)
# Trim to 1 day (12 steps) for speed
for k in env_data:
    if hasattr(env_data[k], '__len__') and len(env_data[k]) > 12:
        env_data[k] = env_data[k][:12]

print("Running symmetric sophisticated simulation...")
result = run_full_sophisticated_simulation(
    env_data=env_data, num_days=1, policy_len=4,
    gamma=16.0, learn_B=True, cost_weight=1.0,
    seed=42, verbose=True,
)

print(f"\nResult: {result.total_steps} steps")
print(f"  Cost:    ${result.total_cost:.2f}")
print(f"  GHG:     {result.total_ghg:.2f} kg")
print(f"  Comfort: {result.cumulative_temp_deviation:.1f} C-steps")

# Check phantom histories
if hasattr(result, 'phantom_history'):
    ph = result.phantom_history
    print(f"  Phantom HVAC predictions: {len(ph['p_hvac'])} steps")
    print(f"  Phantom Batt predictions: {len(ph['p_batt'])} steps")
    if ph['p_hvac'][0] is not None:
        print(f"    Step 0 P(HVAC): {ph['p_hvac'][0]}")
    if ph['p_batt'][0] is not None:
        print(f"    Step 0 P(Batt): {ph['p_batt'][0]}")

print("\nSmoke test PASSED!")
