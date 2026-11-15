"""Pareto figures from the Phase 2 sweep CSV (project-plan.md Phase 2 step 4).

Run:  .venv/bin/python experiments/plot_baselines.py [path/to/results.csv]
Writes results/phase2_<group>.png, one figure per network group.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import plot_pareto

CSV = ROOT / "results" / "phase2_baselines.csv"
OUT_DIR = ROOT / "results"


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV
    for path in plot_pareto(csv_path, OUT_DIR):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
