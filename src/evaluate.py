"""Evaluation harness: one CSV row per pruning run, and Pareto plots (project-plan.md Phase 2 step 4)."""
import csv
import platform
from collections import defaultdict

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_nets import CBS_NAME, NETS_DIR, accuracy

METHOD_STYLE = {"random": ("#2a78d6", "o"), "magnitude": ("#eb6834", "s"),
                "wanda": ("#1baf7a", "^"), "greedy": ("#eda100", "D")}
MODE_TITLES = {("none", "False"): "no fix", ("none", "True"): "fix",
               ("simple", "False"): "sequential", ("simple", "True"): "sequential + fix",
               ("error_correcting", "True"): "error-correcting + fix"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

MAX_LAYERS = 3
FIELDS = (["net", "split", "dataset", "hidden", "activation", "method", "sparsity", "real_sparsity",
           "sequential", "fix", "dense_acc", "test_acc", "beats_dense", "objective_total", "output_error_total"]
          + [f"objective_l{i + 1}" for i in range(MAX_LAYERS)]
          + [f"output_error_l{i + 1}" for i in range(MAX_LAYERS)]
          + ["time_setup", "time_select", "time_fix", "torch", "platform"])
KEY = ("net", "method", "sparsity", "sequential", "fix")


def load_net(name):
    """A saved network and the calibration batch of its dataset."""
    net = torch.load(NETS_DIR / f"{name}.pt", weights_only=True)
    calib = torch.load(NETS_DIR / f"calib_{net['config']['dataset']}.pt", weights_only=True)["x"]
    return net, calib


def run_row(name, net, dense_acc, pruned, info, method, sparsity, sequential, fix):
    c = net["config"]
    acc = accuracy(pruned, c["dataset"], train=False)
    row = {"net": name, "split": c["split"], "dataset": c["dataset"],
           "hidden": "-".join(map(str, c["hidden"])), "activation": c["activation"],
           "method": method, "sparsity": sparsity,
           "real_sparsity": round(1 - info["kept"] / info["total"], 6),
           "sequential": sequential, "fix": fix,
           "dense_acc": round(dense_acc, 4), "test_acc": round(acc, 4), "beats_dense": acc > dense_acc,
           "objective_total": sum(info["objective"]), "output_error_total": sum(info["output_error"]),
           "time_setup": round(info["time_setup"], 4), "time_select": round(info["time_select"], 4),
           "time_fix": round(info["time_fix"], 4),
           "torch": torch.__version__, "platform": f"{platform.system()}-{platform.machine()}"}
    for i in range(MAX_LAYERS):
        has = i < len(info["objective"])
        row[f"objective_l{i + 1}"] = info["objective"][i] if has else ""
        row[f"output_error_l{i + 1}"] = info["output_error"][i] if has else ""
    return row


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def done_keys(path):
    return {tuple(r[k] for k in KEY) for r in read_rows(path)}


def row_key(name, method, sparsity, sequential, fix):
    return tuple(str(v) for v in (name, method, sparsity, sequential, fix))


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def plot_groups(rows):
    """Nets averaged together in one figure: the main architecture per dataset, plus CBS and the tiny net."""
    groups = defaultdict(list)
    for r in rows:
        if r["net"] == CBS_NAME:
            groups["cbs_mnist"].append(r)
        elif r["dataset"] == "mnist14":
            groups["tiny_mnist14"].append(r)
        elif r["hidden"] == "128-64" and r["activation"] == "relu":
            groups[f"{r['dataset']}_relu128"].append(r)
    return groups


def mean_std(values):
    t = torch.tensor(values, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)


def label_line_ends(ax, ends, min_gap_px=12):
    """Direct labels at the right end of each line, pushed apart vertically so they never overlap."""
    placed = sorted((ax.transData.transform((x, y))[1], x, y, text) for x, y, text in ends)
    last = -float("inf")
    for y_px, x, y, text in placed:
        target = max(y_px, last + min_gap_px)
        ax.annotate(text, (x, y), xytext=(8, target - y_px), textcoords="offset pixels",
                    va="center", fontsize=8, color=TEXT)
        last = target


def plot_pareto(csv_path, out_dir):
    """Per group: test accuracy and output error against sparsity, one panel per pipeline mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for group, rows in plot_groups(read_rows(csv_path)).items():
        modes = [m for m in MODE_TITLES if any((r["sequential"], r["fix"]) == m for r in rows)]
        n_nets = len({r["net"] for r in rows})
        fig, axes = plt.subplots(2, len(modes), figsize=(3.3 * len(modes) + 0.8, 6.2),
                                 sharex=True, sharey="row", squeeze=False)
        for col, mode in enumerate(modes):
            ax_acc, ax_err = axes[0, col], axes[1, col]
            mode_rows = [r for r in rows if (r["sequential"], r["fix"]) == mode]
            dense = mean_std([float(r["dense_acc"]) for r in mode_rows])[0]
            ax_acc.axhline(dense, color=MUTED, lw=1, ls="--")
            ends = {ax_acc: [], ax_err: []}
            for method, (color, marker) in METHOD_STYLE.items():
                by_s = defaultdict(lambda: ([], []))
                for r in mode_rows:
                    if r["method"] == method:
                        by_s[float(r["sparsity"])][0].append(float(r["test_acc"]))
                        by_s[float(r["sparsity"])][1].append(float(r["output_error_total"]))
                if not by_s:
                    continue
                xs = sorted(by_s)
                for ax, j in ((ax_acc, 0), (ax_err, 1)):
                    stats = [mean_std(by_s[x][j]) for x in xs]
                    mean = [m for m, _ in stats]
                    pct = [100 * x for x in xs]
                    ax.plot(pct, mean, color=color, marker=marker, lw=2, ms=6, label=method,
                            markeredgecolor="white", markeredgewidth=1)
                    ax.fill_between(pct, [m - s for m, s in stats], [m + s for m, s in stats],
                                    color=color, alpha=0.15, lw=0)
                    ends[ax].append((pct[-1], mean[-1], method))
            ax_acc.set_title(MODE_TITLES[mode], fontsize=10, color=TEXT)
            ax_err.set_yscale("log")
            if col == len(modes) - 1:
                for ax, ax_ends in ends.items():
                    label_line_ends(ax, ax_ends)
            ax_err.set_xlabel("sparsity (%)", color=MUTED)
            for ax in (ax_acc, ax_err):
                ax.grid(True, color=GRID, lw=0.8)
                ax.set_axisbelow(True)
                ax.tick_params(colors=MUTED, labelsize=8)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    ax.spines[side].set_color(GRID)
        axes[0, 0].set_ylabel("test accuracy (%)", color=MUTED)
        axes[1, 0].set_ylabel("output error vs dense\n(sum over layers)", color=MUTED)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color=MUTED, lw=1, ls="--"))
        labels.append("dense")
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 0.99), fontsize=9)
        fig.suptitle(f"{group}: baselines (mean ± std over {n_nets} net{'s' if n_nets > 1 else ''})",
                     y=1.04, fontsize=11, color=TEXT)
        fig.tight_layout()
        path = out_dir / f"phase2_{group}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths
