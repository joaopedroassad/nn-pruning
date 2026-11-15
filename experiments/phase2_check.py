"""Phase 2 check (project-plan.md Phase 2), on MNIST ReLU-128 seed 0 and the CBS reference net.

Run:  .venv/bin/python experiments/phase2_check.py
"""
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

import baselines
import scores
from evaluate import load_net
from prune import METHODS, prune_net, select
from train_nets import CBS_NAME, accuracy, layer_inputs

NET_NAMES = ("mnist_relu128_s0_ep20", CBS_NAME)
SPARSITIES = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
MODES = (("none", False), ("none", True), ("simple", False), ("simple", True), ("error_correcting", True))
N_TOY, TOY_N, TOY_K = 200, 12, 6
CBS_PAPER_MAGNITUDE = {0.9: 90.30, 0.98: 32.25}
CBS_TOL = 0.5
GREEDY_MIN_OPTIMAL = 0.8


def load(name):
    net, calib = load_net(name)
    return net, calib, [X.double() for X in layer_inputs(net, calib)[0]]


def check_score_identity(nets):
    print("1. S = dᵀCd matches the measured mean ‖X·d‖², and layer-1 objective = output error")
    worst = 0.0
    for name, (net, calib, inputs) in nets.items():
        for i, ((W, _), X) in enumerate(zip(net["layers"], inputs)):
            C = scores.covariance(X)
            M = baselines.random_mask(W, baselines.keep_count(W.shape[1], 0.9), seed=i)
            S = scores.damage(W, W.double() * M, C)
            direct = (X @ (W.double() * ~M).T).pow(2).mean(0)
            worst = max(worst, ((S - direct).abs() / direct.clamp(min=1e-300)).max().item())
        _, info = prune_net(net, calib, "greedy", 0.9, "none", fix=True)
        o, e = info["objective"][0], info["output_error"][0]
        worst = max(worst, abs(o - e) / e)
    ok = worst < 1e-10
    print(f"   worst relative error {worst:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def check_exact_k(nets):
    print("2. every mask keeps exactly k weights per row")
    ok = True
    for name, (net, _, inputs) in nets.items():
        for (W, _), X in zip(net["layers"], inputs):
            C = scores.covariance(X)
            for method, s in itertools.product(METHODS, (0.0,) + SPARSITIES):
                k = baselines.keep_count(W.shape[1], s)
                ok &= bool((select(method, W, C, k, seed=0).sum(1) == k).all())
    print(f"   {'PASS' if ok else 'FAIL'}")
    return ok


def check_zero_sparsity(nets):
    print("3. at 0% sparsity every method and mode reproduces the dense accuracy")
    ok = True
    for name, (net, calib, _) in nets.items():
        dense = accuracy(net, net["config"]["dataset"], train=False)
        for method, (sequential, fix) in itertools.product(METHODS, MODES):
            pruned, _ = prune_net(net, calib, method, 0.0, sequential, fix)
            acc = accuracy(pruned, net["config"]["dataset"], train=False)
            if acc != dense:
                print(f"   FAIL {name} {method} {sequential} fix={fix}: {acc:.2f} vs {dense:.2f}")
                ok = False
    print(f"   {'PASS' if ok else 'FAIL'}")
    return ok


def check_fix_never_increases(nets):
    print("4. the fix never increases S")
    worst = -float("inf")
    for name, (net, _, inputs) in nets.items():
        for (W, _), X in zip(net["layers"], inputs):
            C = scores.covariance(X)
            for method, s in itertools.product(METHODS, SPARSITIES):
                M = select(method, W, C, baselines.keep_count(W.shape[1], s), seed=0)
                before = scores.damage(W, W.double() * M, C)
                after = scores.damage(W, scores.fix(W, M, C), C)
                worst = max(worst, ((after - before) / before.clamp(min=1e-300)).max().item())
    ok = worst <= 1e-9
    print(f"   largest relative increase {worst:.2e} (negative = always decreased)  {'PASS' if ok else 'FAIL'}")
    return ok


def check_wanda_vs_magnitude(nets):
    print("5. Wanda's S ≤ magnitude's S on most rows at high sparsity")
    ok = True
    for name, (net, _, inputs) in nets.items():
        for i, ((W, _), X) in enumerate(zip(net["layers"], inputs)):
            C = scores.covariance(X)
            fractions = []
            for s in (0.9, 0.95, 0.98):
                k = baselines.keep_count(W.shape[1], s)
                S_wanda = scores.damage(W, W.double() * baselines.wanda_mask(W, C, k), C)
                S_mag = scores.damage(W, W.double() * baselines.magnitude_mask(W, k), C)
                fractions.append((S_wanda <= S_mag).double().mean().item())
            ok &= all(f > 0.5 for f in fractions)
            print(f"   {name} layer {i + 1}: " + ", ".join(f"{s:.0%}: {f:.0%}" for s, f in zip((0.9, 0.95, 0.98), fractions)))
    print(f"   {'PASS' if ok else 'FAIL'}")
    return ok


def check_toy_rows(nets):
    print(f"6. brute force on {N_TOY} toy rows (n = {TOY_N}, k = {TOY_K}, all 924 masks)")
    g = torch.Generator().manual_seed(0)
    masks = torch.zeros(924, TOY_N, dtype=torch.bool)
    for r, keep in enumerate(itertools.combinations(range(TOY_N), TOY_K)):
        masks[r, list(keep)] = True
    layers = [(W, X) for net, _, inputs in nets.values() for (W, _), X in zip(net["layers"], inputs)
              if W.shape[1] >= 2 * TOY_N]
    ok, optimal, gaps = True, {m: 0 for m in METHODS}, {m: [] for m in METHODS}
    for _ in range(N_TOY):
        W, X = layers[torch.randint(len(layers), (1,), generator=g).item()]
        live = (X.pow(2).mean(0) > 0).nonzero().squeeze(1)
        idx = live[torch.randperm(len(live), generator=g)[:TOY_N]]
        row = torch.randint(W.shape[0], (1,), generator=g).item()
        w = W[row:row + 1, idx]
        C = scores.covariance(X[:, idx])
        best = scores.damage(w.expand(924, -1), w.double() * masks, C).min().item()
        for method in METHODS:
            S = scores.damage(w, w.double() * select(method, w, C, TOY_K, seed=0), C).item()
            ok &= S >= best * (1 - 1e-9)
            optimal[method] += S <= best * (1 + 1e-9)
            gaps[method].append(S / best - 1)
    for method in METHODS:
        print(f"   {method:9}: optimal on {optimal[method] / N_TOY:4.0%} of rows, mean gap {sum(gaps[method]) / N_TOY:7.2%}")
    rate = optimal["greedy"] / N_TOY
    if rate < GREEDY_MIN_OPTIMAL:
        print(f"   WARNING greedy optimal on only {rate:.0%} (expected ≥ {GREEDY_MIN_OPTIMAL:.0%})")
    print(f"   no method below the brute-force optimum: {'PASS' if ok else 'FAIL'}")
    return ok


def check_cbs_anchor(nets):
    print("7. global magnitude on the CBS net reproduces the paper's magnitude numbers")
    net = nets[CBS_NAME][0]
    ok = True
    for s, paper in CBS_PAPER_MAGNITUDE.items():
        masks = baselines.global_magnitude_masks(net["layers"], s)
        pruned = {**net, "layers": [(W * M, b) for (W, b), M in zip(net["layers"], masks)]}
        acc = accuracy(pruned, "mnist", train=False)
        passed = abs(acc - paper) <= CBS_TOL
        ok &= passed
        print(f"   {s:.0%} sparsity: {acc:.2f}% (paper {paper:.2f}%)  {'PASS' if passed else 'FAIL'}")
    return ok


def check_beats_dense(nets):
    print("8. pruned runs that beat their dense accuracy (warning only)")
    flagged = []
    for name, (net, calib, _) in nets.items():
        dense = accuracy(net, net["config"]["dataset"], train=False)
        for method, s, fix in itertools.product(METHODS, SPARSITIES, (False, True)):
            pruned, _ = prune_net(net, calib, method, s, "none", fix)
            acc = accuracy(pruned, net["config"]["dataset"], train=False)
            if acc > dense:
                flagged.append(f"{name} {method} {s:.0%} fix={fix}: {acc:.2f} > {dense:.2f}")
    for line in flagged:
        print(f"   WARNING {line}")
    print(f"   {len(flagged)} flagged")
    return True


def main():
    nets = {name: load(name) for name in NET_NAMES}
    checks = [check_score_identity, check_exact_k, check_zero_sparsity, check_fix_never_increases,
              check_wanda_vs_magnitude, check_toy_rows, check_cbs_anchor, check_beats_dense]
    results = [check(nets) for check in checks]
    print(f"\nPhase 2 check: {'PASS' if all(results) else 'FAIL'}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
