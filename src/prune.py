"""Layer-by-layer pruning pipeline (project-plan.md §3.2 and Phase 2 step 3).

sequential="none":             every layer's C comes from the dense network's inputs.
sequential="simple":           layer ℓ's C comes from the inputs produced by the already
                               pruned (and fixed) layers, for both selection and fix.
sequential="error_correcting": selection as in "simple"; the fix targets the dense
                               pre-activation X·w from the pruned inputs X̂. Without the
                               fix it is identical to "simple".
"""
import time

import torch

import baselines
import scores
from train_nets import ACTIVATIONS, layer_inputs

METHODS = ("random", "magnitude", "wanda", "greedy")


def select(method, W, C, k, seed):
    if method == "random":
        return baselines.random_mask(W, k, seed)
    if method == "magnitude":
        return baselines.magnitude_mask(W, k)
    if method == "wanda":
        return baselines.wanda_mask(W, C, k)
    if method == "greedy":
        return baselines.greedy_mask(W, C, k)
    raise ValueError(method)


def pre_activation(h, W, b):
    z = h @ W.T
    return z if b is None else z + b.double()


def prune_net(net, calib_x, method, sparsity, sequential="none", fix=False, seed=0):
    """Prune every layer to the same per-row sparsity.

    Returns the pruned net and an info dict with, per layer:
    objective     S summed over rows, under the C the selector saw;
    output_error  mean squared difference between the pruned and dense network's
                  pre-activations at that layer on the calibration batch, summed over rows;
    plus kept/total weight counts and setup/select/fix times.
    """
    t = time.perf_counter()
    dense_inputs, _ = layer_inputs(net, calib_x)
    act = ACTIVATIONS[net["activation"]]
    h = dense_inputs[0].double()
    info = {"objective": [], "output_error": [], "kept": 0, "total": 0,
            "time_setup": time.perf_counter() - t, "time_select": 0.0, "time_fix": 0.0}
    layers = []
    for i, (W, b) in enumerate(net["layers"]):
        t = time.perf_counter()
        X = dense_inputs[i].double()
        X_hat = X if sequential == "none" else h
        C = scores.covariance(X_hat)
        k = baselines.keep_count(W.shape[1], sparsity)
        info["time_setup"] += time.perf_counter() - t

        t = time.perf_counter()
        M = select(method, W, C, k, seed + i)
        info["time_select"] += time.perf_counter() - t

        t = time.perf_counter()
        W_new = W.double() * M
        if fix:
            C_cross = scores.covariance(X_hat, X) if sequential == "error_correcting" else None
            W_new = scores.fix(W, M, C, C_cross)
        info["time_fix"] += time.perf_counter() - t

        z_dense = pre_activation(X, W.double(), b)
        z_pruned = pre_activation(h, W_new, b)
        info["objective"].append(scores.damage(W, W_new, C).sum().item())
        info["output_error"].append((z_pruned - z_dense).pow(2).mean(0).sum().item())
        info["kept"] += int(M.sum())
        info["total"] += M.numel()
        layers.append((W_new.float(), b))
        h = act(z_pruned) if i < len(net["layers"]) - 1 else z_pruned
    return {**net, "layers": layers}, info
