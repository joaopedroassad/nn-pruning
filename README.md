# nn-pruning

Self-supervised, iterative neural-network pruning with a ConsFormer-style learned solver.

## Goal

Pruning a trained network means choosing exactly *k* weights to keep. The damage from removing weights is not additive, because weights interact: two can do the same job, or cancel each other out. That makes pruning a combinatorial problem.

This project trains a Transformer, adapted from [ConsFormer](https://github.com/khalil-research/ConsFormer) (Xu et al., ICML 2025), to iteratively repair pruning masks. It is trained without labels and always enforces exact-k sparsity. For each output row of a linear layer, it minimises the exact reconstruction damage

```
S(m) = dᵀ C d,   d = w ⊙ (1 − m),   C = XᵀX / B
```

where `w` is the row's weights, `m` the keep-mask and `X` the layer's calibration inputs.

It is compared on the same objective, and on final accuracy, against magnitude pruning, Wanda, the Combinatorial Brain Surgeon's swap local search, and an exact solver (Gurobi). The focus is high sparsity (80–98%), and the pruner is tested on networks and datasets it has never seen.

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy scipy matplotlib pandas tqdm gurobipy
```

The exact-solver baseline needs a Gurobi licence. A free academic licence is available at [gurobi.com/academia](https://www.gurobi.com/academia/).

## Status

Early setup. Code and results to follow.
