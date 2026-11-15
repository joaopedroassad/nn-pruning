"""Per-row damage score, input covariance and the least-squares fix (project-plan.md §3.1).

All maths runs in float64. W, W_new, M have shape (rows, n); C is (n, n).
"""
import torch

DAMPING = 1e-3
FIX_CHUNK = 32


def covariance(X, Y=None):
    """C = XᵀX / B, or the cross term XᵀY / B when Y is given."""
    X = X.double()
    Y = X if Y is None else Y.double()
    return X.T @ Y / X.shape[0]


def damage(W, W_new, C):
    """Mean squared change in each row's pre-activation: (w − w_new)ᵀ C (w − w_new).

    For a plain mask, W_new = W ⊙ M and this is the plan's S(m) = dᵀCd.
    """
    D = W.double() - W_new.double()
    return torch.einsum("rn,nm,rm->r", D, C, D)


def fix(W, M, C, C_cross=None):
    """Least-squares fix of every row's kept weights.

    Minimises (w − x)ᵀC(w − x) + δ‖x − w‖² over x supported on the row's kept set,
    i.e. (C_KK + δI) x = C_K,: w + δ w_K. Damping towards w, not towards 0, means
    keeping the old weights is always feasible, so the fix can never increase S.
    With C_cross = X̂ᵀX / B the target is the dense pre-activation X·w instead
    (error-correcting sequential pruning). Removed weights come out exactly 0.
    """
    W = W.double()
    M = M.double()
    n = W.shape[1]
    delta = DAMPING * C.diagonal().mean()
    B = W @ (C if C_cross is None else C_cross.T)
    eye = torch.eye(n, dtype=torch.float64)
    out = []
    for start in range(0, W.shape[0], FIX_CHUNK):
        m = M[start:start + FIX_CHUNK]
        A = m[:, :, None] * C * m[:, None, :] + eye * (delta * m + (1 - m))[:, None, :]
        rhs = m * (B[start:start + FIX_CHUNK] + delta * W[start:start + FIX_CHUNK])
        out.append(torch.linalg.solve(A, rhs))
    return torch.cat(out)
