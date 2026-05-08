# EcoNet

**Multiagent Planning and Control of Household Energy Resources Using Active Inference**

John C. Boik, Kobus Esterhuysen, Jacqueline B. Hynes, Axel Constant, Ines Hipolito, Mahault Albarracin, Alex B. Kiefer, Karl Friston

## Overview

EcoNet is a Bayesian approach to household and neighborhood energy management based on active inference. It uses the free energy principle to unify perception, learning, and action under a single objective, enabling intelligent coordination of home energy devices (HVAC thermostat, battery storage, solar PV) under uncertainty.

Key features:

- **Active inference agents** — Thermostat and Battery agents modeled as partially observable Markov decision processes (POMDPs) using [pymdp](https://github.com/infer-actively/pymdp)
- **Predictive transition matrices** — Agents update B matrices at every step using weather forecasts, TOU tariff schedules, and occupancy calendars, enabling anticipatory planning
- **Cost-aware thermostat** — Five-factor generative model with energy demand state driven by a phantom Battery Agent for principled cost-comfort trade-offs
- **Five coordination modes** — Independent, Aligned, Hierarchical, Theory of Mind (federated belief sharing), and Sophisticated (mutual phantom inference)
- **Climate robustness** — Validated across four cities (London, Montreal, Phoenix, Miami) and two seasons with real weather data

## Results

- Up to 6% daily cost reduction vs. unmanaged operation; 52% vs. rule-based control
- ToM mode achieves 32% cost reduction and 11% comfort improvement over aligned mode
- Robust generalization across climate conditions without retraining

## Parameter Tuning

AIF agent performance is controlled by three key parameters that balance pragmatic value (utility) against epistemic value (information gain):

| Parameter | Default | Tuned | Effect |
|-----------|---------|-------|--------|
| `gamma` | 64.0 | 64.0 | Policy precision (inverse temperature). Higher = more deterministic policy selection, less exploration |
| `comfort_scale` | 1.0 | 3.0 | Multiplier for thermostat comfort C amplitudes. Scales both static and dynamic (TOU-dependent) preferences |
| `soc_scale` | 1.0 | 2.0 | Multiplier for battery SoC C preferences. Scales both generative model C and dynamic TOU arbitrage vectors |

These are passed to any simulation function:

```python
from econet.simulation import run_simulation

result = run_simulation(
    env_data=env_data,
    gamma=64.0,           # policy precision
    comfort_scale=3.0,    # 3x comfort amplitude
    soc_scale=2.0,        # 2x SoC preference strength
)
```

Additional configurable parameters:
- `use_states_info_gain` (bool, default `True`): Enable/disable epistemic drive. When `False`, agents rely purely on pragmatic value.

See [TUNING.md](TUNING.md) for full rationale, root cause analysis, and expected outcomes.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ with JAX.

## Usage

Run the key comparison (Oracle gap analysis):

```bash
python scripts/run_key_compare.py
```

Run the cost scale sweep (Pareto frontier):

```bash
python scripts/sweep_cost_scale.py
```

Run the uncertainty robustness experiment (8 methods x 5 noise levels x 2 climates x 3 seeds):

```bash
python scripts/run_uncertainty_experiment.py
```

Run the RL convergence and cross-climate transfer experiment:

```bash
python scripts/run_rl_experiment.py
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Project Structure

```
econet/
  agents.py           # Thermostat and Battery active inference agents (7 classes)
  baselines.py        # No-HEMS, rule-based, oracle, MPC, and RL baselines
  climate.py          # Synthetic and real weather data generation
  environment.py      # Energy environment simulation
  generative_model.py # POMDP generative model (A, B, C, D matrices)
  metrics.py          # Cost, comfort, GHG, and battery utilization metrics
  phantom.py          # Phantom agent models for sophisticated inference
  plotting.py         # Publication figures
  simulation.py       # Simulation runners for all 6 coordination modes
  sophisticated.py    # Sophisticated (mutual phantom) coordination
  strategy.py         # High-level strategy agent for hierarchical mode
  tom.py              # Theory of Mind / federated belief sharing
  validation.py       # Validation utilities
scripts/
  run_key_compare.py             # Oracle gap analysis across coordination modes
  run_uncertainty_experiment.py  # Experiment A: forecast noise robustness
  run_rl_experiment.py           # Experiment B: RL convergence + transfer
  sweep_cost_scale.py            # Cost-comfort Pareto frontier
tests/                # Test suite
figures/              # Generated figures
TUNING.md             # AIF parameter tuning documentation
mainV5.tex            # Paper (revised manuscript)
references.bib        # Bibliography
```

## Citation

If you use this code, please cite:

```bibtex
@article{boik2025econet,
  title={EcoNet: Multiagent Planning and Control of Household Energy Resources Using Active Inference},
  author={Boik, John C. and Esterhuysen, Kobus and Hynes, Jacqueline B. and Constant, Axel and Hipolito, Ines and Albarracin, Mahault and Kiefer, Alex B. and Friston, Karl},
  year={2025}
}
```

## License

See the repository for license details.
