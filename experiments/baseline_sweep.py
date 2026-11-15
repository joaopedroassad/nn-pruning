"""Phase 2 baseline sweep (project-plan.md Phase 2 step 5): every baseline on all 27 final nets.

Run:  .venv/bin/python experiments/baseline_sweep.py

Needs nets/ with the final networks, the calibration batches and cbs_mnist.pt.
Appends one row per run to results/phase2_baselines.csv and skips runs already
there, so it can be stopped and restarted at any time.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import append_row, done_keys, load_net, read_rows, row_key, run_row
from prune import METHODS, prune_net
from train_nets import CBS_NAME, EPOCHS, NETS, NETS_DIR, accuracy, net_name

SMOKE = False
SPARSITIES = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
MODES = (("none", False), ("none", True), ("simple", False), ("simple", True), ("error_correcting", True))
RESULTS = ROOT / "results" / ("phase2_baselines_smoke.csv" if SMOKE else "phase2_baselines.csv")


def final_net_names():
    names = [net_name(dataset, hidden, activation, seed, EPOCHS)
             for _, dataset, hidden, activation, seeds in NETS for seed in seeds]
    return names + [CBS_NAME]


def main():
    names = final_net_names()[:1] if SMOKE else final_net_names()
    sparsities = (0.9, 0.98) if SMOKE else SPARSITIES
    missing = [n for n in names if not (NETS_DIR / f"{n}.pt").exists()]
    if missing:
        sys.exit(f"Missing networks in {NETS_DIR}: {missing}")

    done = done_keys(RESULTS)
    total = len(names) * len(sparsities) * len(METHODS) * len(MODES)
    finished = sum(row_key(n, m, s, q, f) in done
                   for n in names for s in sparsities for m in METHODS for q, f in MODES)
    print(f"{finished}/{total} runs already in {RESULTS.name}", flush=True)
    start = time.perf_counter()
    for name in names:
        net, calib = load_net(name)
        dense = accuracy(net, net["config"]["dataset"], train=False)
        t = time.perf_counter()
        for sparsity in sparsities:
            for method in METHODS:
                for sequential, fix in MODES:
                    if row_key(name, method, sparsity, sequential, fix) in done:
                        continue
                    pruned, info = prune_net(net, calib, method, sparsity, sequential, fix)
                    append_row(RESULTS, run_row(name, net, dense, pruned, info, method, sparsity, sequential, fix))
                    finished += 1
        print(f"{name}: {time.perf_counter() - t:.0f}s  ({finished}/{total} runs, "
              f"{(time.perf_counter() - start) / 60:.1f} min so far)", flush=True)

    beats = [r for r in read_rows(RESULTS) if r["beats_dense"] == "True"]
    print(f"\nDone. {len(beats)} runs beat their dense accuracy:")
    for r in beats:
        print(f"  {r['net']} {r['method']} {r['sparsity']} {r['sequential']} fix={r['fix']}: "
              f"{r['test_acc']} > {r['dense_acc']}")


if __name__ == "__main__":
    main()
