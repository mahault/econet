#!/bin/bash
cd /tmp/econet
rm -rf econet/__pycache__

echo "=== Starting EcoNet experiments ==="
echo "$(date): Scenario 1 (deterministic 2-day)..."
python3.12 scripts/run_deterministic.py > /tmp/econet_scenario1.log 2>&1
echo "$(date): Scenario 1 done (exit=$?)"

echo "$(date): Scenario 3 (baselines)..."
python3.12 scripts/run_baselines.py > /tmp/econet_scenario3.log 2>&1
echo "$(date): Scenario 3 done (exit=$?)"

echo "$(date): Validation..."
python3.12 scripts/run_validation.py > /tmp/econet_validation.log 2>&1
echo "$(date): Validation done (exit=$?)"

echo "$(date): Scenario 4 (climate, 7 days each — this will take a while)..."
python3.12 scripts/run_climate.py > /tmp/econet_scenario4.log 2>&1
echo "$(date): Scenario 4 done (exit=$?)"

echo "=== All experiments complete ==="
echo "Figures:"
ls -la /tmp/econet/figures/
