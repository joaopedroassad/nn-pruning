"""Exact-k baseline masks (project-plan.md Phase 2 step 2).

Every function returns a boolean keep-mask M of shape (rows, n) with exactly k
True entries per row, for a layer's weights W (rows, n) and input covariance C.
"""
import torch


def keep_count(n, sparsity):
    """Weights kept per row. At least 1, so small rows sit slightly below the target sparsity."""
    return max(1, round((1 - sparsity) * n))


def top_k_mask(scores, k):
    idx = scores.topk(k, dim=1).indices
    return torch.zeros(scores.shape, dtype=torch.bool).scatter_(1, idx, True)


def random_mask(W, k, seed):
    return top_k_mask(torch.rand(W.shape, generator=torch.Generator().manual_seed(seed)), k)


def magnitude_mask(W, k):
    return top_k_mask(W.abs().double(), k)


def wanda_mask(W, C, k):
    """Wanda score |w|·‖Xⱼ‖, which ranks the same as |w|·√Cⱼⱼ."""
    return top_k_mask(W.abs().double() * C.diagonal().clamp(min=0).sqrt(), k)


def greedy_mask(W, C, k):
    """Remove, one at a time and in every row at once, the weight that increases S least.

    With d the removed weights so far, removing j next costs 2wⱼ(Cd)ⱼ + wⱼ²Cⱼⱼ.
    """
    W = W.double()
    rows, n = W.shape
    removed = torch.zeros(rows, n, dtype=torch.bool)
    Cd = torch.zeros(rows, n, dtype=torch.float64)
    diag = C.diagonal()
    arange = torch.arange(rows)
    for _ in range(n - k):
        cost = 2 * W * Cd + W ** 2 * diag
        cost[removed] = float("inf")
        j = cost.argmin(dim=1)
        removed[arange, j] = True
        Cd += W[arange, j][:, None] * C[j]
    return ~removed


def global_magnitude_masks(layers, sparsity):
    """One |w| threshold across all layers, as in CBS's magnitude baseline (for the CBS anchor check)."""
    magnitudes = torch.cat([W.abs().flatten() for W, _ in layers])
    keep = int(len(magnitudes) * (1 - sparsity))
    threshold = magnitudes.topk(keep).values[-1]
    return [W.abs() >= threshold for W, _ in layers]
