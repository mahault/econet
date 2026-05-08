# AIF Parameter Tuning: Closing the Gap with MPC

## Problem Statement

Initial sigma=0 (perfect foresight) results showed AIF underperforming MPC:

| Mode | Cost Gap vs MPC |
|------|----------------|
| AIF Independent | +3.8% |
| AIF Aligned | +10.1% |
| AIF Federated | +22.4% |

At sigma=0 both AIF and MPC have perfect forecasts, so the gap represents pure
decision-quality loss in AIF's policy selection.

## Root Cause Analysis

| Issue | Detail |
|-------|--------|
| **gamma too low** | `16.0` everywhere. Softmax over policies too flat -- exploration dominates exploitation |
| **Comfort C too weak** | Peak amplitude `-3.0` to `-4.0` nats. Info gain from state transitions can reach 2-10 nats, drowning out utility |
| **Battery SoC C mismatch** | `build_battery_C()` uses `0.3*(i-2)` giving [-0.6..0.6], but class constants `C_SOC_PEAK=[2,1,0,-1,-2]` are much stronger. The weak generative model version feeds into the Agent constructor |
| **use_states_info_gain hardcoded** | `True` in all 7 Agent() constructors -- not configurable. Epistemic drive competes with pragmatic value |
| **Aligned mode noise** | Dynamic C amplitudes oscillate between -3.0/-4.0 by TOU period, adding noise without strong enough signal |

### Why info gain dominates

In EFE = utility + info_gain, the utility term depends on C vector amplitudes
while info gain depends on entropy of state transitions. With 26 temperature
bins and stochastic B matrices, info gain easily reaches 2-10 nats. At
comfort_scale=1.0, the peak comfort penalty is only ~4 nats at dist=2 bins,
which is comparable to or weaker than epistemic drive.

## Tuning Strategy

Increase utility signal strength relative to info gain. Do NOT disable info gain
(it provides robustness under forecast uncertainty -- the paper's key claim).
Instead, make pragmatic value dominant so info gain acts as a secondary tiebreaker.

## Changes Made

### 1. Policy Precision: gamma 16 -> 64

**File:** `econet/simulation.py` (all 6 simulation functions)

Changed default gamma from 16.0 to 64.0. This makes policy selection ~4x more
deterministic. The agent more strongly commits to the highest-EFE policy rather
than sampling exploratory ones.

The agent-level constructors retain gamma=16.0 as a fallback default, but all
simulation functions now pass 64.0.

### 2. Comfort Scale (comfort_scale)

**Files:** `econet/agents.py`, `econet/generative_model.py`, `econet/simulation.py`

Added `comfort_scale: float = 1.0` parameter to all thermostat agents.

- `build_thermostat_C(comfort_amplitude)`: New parameter replaces hardcoded `-4.0`
- Agents compute `comfort_amplitude = -4.0 * comfort_scale`
- Dynamic C in aligned mode: `amplitude = (-3.0 if peak else -4.0) * comfort_scale`
- SophisticatedThermostatAgent: thermal-gap amplitude scaled by comfort_scale

With `comfort_scale=3.0`:
- Static C peak: -12.0 nats (was -4.0)
- Dynamic C peak: -9.0 (was -3.0), off-peak: -12.0 (was -4.0)
- Strongly dominates info gain (~2-10 nats)

### 3. SoC Scale (soc_scale)

**Files:** `econet/agents.py`, `econet/generative_model.py`, `econet/simulation.py`

Added `soc_scale: float = 1.0` parameter to all battery agents.

- `build_battery_C(soc_scale)`: Multiplies SoC preferences by scale factor
- Battery agent classes: `C_SOC_PEAK` and `C_SOC_OFFPEAK` class constants
  replaced by instance variables `_c_soc_peak` / `_c_soc_offpeak` scaled by soc_scale

With `soc_scale=2.0`:
- Generative model SoC C: [-1.2, -0.6, 0, 0.6, 1.2] (was [-0.6, -0.3, 0, 0.3, 0.6])
- Dynamic C peak: [4, 2, 0, -2, -4] (was [2, 1, 0, -1, -2])
- Dynamic C off-peak: [-3, -1, 0, 1, 2] (was [-1.5, -0.5, 0, 0.5, 1])

### 4. Configurable use_states_info_gain

**Files:** `econet/agents.py`, `econet/simulation.py`

Added `use_states_info_gain: bool = True` to all 7 agent classes and all 6
simulation functions. Passed through to `Agent()` constructor.

Default remains `True` so existing behavior is unchanged. Setting to `False`
disables epistemic drive entirely (pure exploitation).

### 5. Experiment Scripts

**Files:** `scripts/run_uncertainty_experiment.py`, `scripts/run_rl_experiment.py`

Both scripts define:
```python
TUNED = dict(gamma=64.0, comfort_scale=3.0, soc_scale=2.0)
```

Passed via `**TUNED` to all AIF method calls. Oracle and MPC are unaffected.

## Files Modified

| File | Changes |
|------|---------|
| `econet/generative_model.py` | `comfort_amplitude` param on `build_thermostat_C`; `soc_scale` param on `build_battery_C`; threaded through all model builders (standard, cost-aware, ToM) |
| `econet/agents.py` | `use_states_info_gain`, `comfort_scale`, `soc_scale` on all 7 agent classes; dynamic C scaling; instance-level SoC C vectors |
| `econet/simulation.py` | gamma default 16->64 in all 6 functions; `comfort_scale`, `soc_scale`, `use_states_info_gain` params threaded to agent constructors |
| `scripts/run_uncertainty_experiment.py` | `TUNED` dict passed to all AIF methods |
| `scripts/run_rl_experiment.py` | `TUNED` dict passed to all AIF reference runs |

## Agent Classes Modified

All 7 agent classes in `agents.py`:

1. `ThermostatAgent` -- comfort_scale, use_states_info_gain
2. `BatteryAgent` -- soc_scale, use_states_info_gain, instance _c_soc_peak/_c_soc_offpeak
3. `SophisticatedThermostatAgent` -- comfort_scale, use_states_info_gain
4. `SophisticatedBatteryAgent` -- soc_scale, use_states_info_gain, instance _c_soc_peak/_c_soc_offpeak
5. `SophisticatedToMBatteryAgent` -- soc_scale, use_states_info_gain, instance _c_soc_peak/_c_soc_offpeak
6. `ToMThermostatAgent` -- comfort_scale, use_states_info_gain
7. `ToMBatteryAgent` -- soc_scale, use_states_info_gain, instance _c_soc_peak/_c_soc_offpeak

## Expected Outcomes

- **gamma=64 + comfort_scale=3**: AIF Independent within ~3% of MPC at sigma=0
- **Under noise (sigma=2-4)**: AIF should degrade less than MPC because info gain
  still operates as an uncertainty buffer
- **Sophisticated variants**: May show additional advantage from phantom model lookahead
- **If AIF still underperforms at sigma=0 but degrades less**: The paper narrative
  becomes "AIF trades small baseline cost for superior robustness"

## Numerical Safety

Large C values can cause overflow in softmax. With comfort_scale=3.0, the
maximum C magnitude is ~12 nats at dist=2 bins. At gamma=64, the softmax
argument is gamma * EFE, where EFE sums utility (~12 nats) and info gain
(~2-10 nats). Maximum softmax argument ~64 * 22 = 1408. This is within
float32 range (max ~3.4e38) so no overflow risk.

## Backward Compatibility

All new parameters have defaults that preserve existing behavior:
- `gamma=64.0` in simulation functions (changed from 16.0)
- `comfort_scale=1.0` (no scaling)
- `soc_scale=1.0` (no scaling)
- `use_states_info_gain=True` (original behavior)

The only breaking change is the gamma default in simulation functions. Code that
explicitly passes `gamma=16.0` will continue to work as before.
