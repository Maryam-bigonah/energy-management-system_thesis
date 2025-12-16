"""
Run the full forecasting + optimization pipeline, then validate.

This is the fastest way to test "all process all correct".

Usage:
  python3 src/validation/run_all.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    # Forecasting (already implemented in your repo)
    _run([sys.executable, "src/forecasting/thesis_forecasting_pipeline.py"])

    # Optimization + figures
    _run([sys.executable, "src/optimization/run_optimization_pipeline.py"])
    _run([sys.executable, "src/optimization/generate_optimization_figures.py"])

    # Validation (strict PASS/FAIL)
    _run([sys.executable, "src/validation/validate_pipeline.py"])

    print("\n[PASS] End-to-end run finished successfully.")


if __name__ == "__main__":
    main()


