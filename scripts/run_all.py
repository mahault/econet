"""Run all EcoNet scenarios and generate all figures.

Usage: python scripts/run_all.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    start = time.time()
    print("=" * 60)
    print("EcoNet: Running All Scenarios")
    print("=" * 60)

    # Scenario 1: Deterministic
    print("\n\n>>> SCENARIO 1: Deterministic 2-Day <<<")
    from scripts.run_deterministic import main as run_det
    run_det()

    # Scenario 3: Baselines (before learning, since it's fast)
    print("\n\n>>> SCENARIO 3: Baseline Comparisons <<<")
    from scripts.run_baselines import main as run_base
    run_base()

    # Scenario 4: Climate sensitivity
    print("\n\n>>> SCENARIO 4: Climate Sensitivity <<<")
    from scripts.run_climate import main as run_clim
    run_clim()

    # Validation
    print("\n\n>>> VALIDATION <<<")
    from scripts.run_validation import main as run_val
    run_val()

    # Scenario 2: Learning (slow — 40 days)
    print("\n\n>>> SCENARIO 2: Parameter Learning (40 days) <<<")
    from scripts.run_learning import main as run_learn
    run_learn()

    elapsed = time.time() - start
    print(f"\n\n{'=' * 60}")
    print(f"All scenarios complete. Total time: {elapsed:.1f}s")
    print(f"Figures saved in: {Path(__file__).parent.parent / 'figures'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
