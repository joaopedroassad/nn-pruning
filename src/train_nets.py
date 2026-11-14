"""Phase 1: train the network population (project-plan.md, Phase 1).

Run:  .venv/bin/python src/train_nets.py

Trains every network in NETS that isn't saved yet, imports the CBS reference
if its checkpoint has been downloaded into nets/, then runs the Phase 1 check.
Re-running resumes after a crash, or only re-runs the check.

A saved network is a plain dict (no model class), so later phases can prune
by editing the W tensors directly:
    layers      [(W, b), ...]   W: (n_out, n_in), b: (n_out,) or None
    activation  "relu" | "gelu" | "tanh"
    input_mean, input_std       applied to [0, 1] pixels before layer 1
    config      dataset, split, seed, epoch, training settings, accuracies
"""
import csv
import functools
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
NETS_DIR = ROOT / "nets"

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 128
CHECKPOINT_EPOCHS = (1, 5, 20)
N_CALIB = 1024
CALIB_SEED = 0

NETS = [
    ("train", "mnist",   (128, 64), "relu", [0, 1, 2, 3, 4]),
    ("train", "fashion", (128, 64), "relu", [0, 1, 2, 3, 4]),
    ("train", "fashion", (64, 32),  "relu", [0]),
    ("train", "fashion", (64, 32),  "gelu", [0]),
    ("train", "fashion", (128, 64), "gelu", [0]),
    ("val",   "mnist",   (128, 64), "relu", [5]),
    ("val",   "fashion", (128, 64), "relu", [5]),
    ("test",  "fashion", (128, 64), "relu", [6, 7]),
    ("test",  "kmnist",  (128, 64), "relu", [0, 1, 2]),
    ("test",  "fashion", (64, 32),  "tanh", [0]),
    ("test",  "fashion", (128, 64), "tanh", [0]),
    ("test",  "fashion", (256, 128), "tanh", [0]),
    ("test",  "fashion", (256, 128), "relu", [0]),
    ("test",  "fashion", (256, 128), "gelu", [0]),
    ("tiny",  "mnist14", (32,),     "relu", [0]),
]

MIN_TEST_ACC = {"mnist": 97.5, "fashion": 87.5, "kmnist": 89.0, "mnist14": 95.0}

CBS_CKPT = NETS_DIR / "mnist_25_epoch_93.97.ckpt"
CBS_NAME = "cbs_mnist"
CBS_ACC = 93.97
CBS_TOL = 0.05

TORCHVISION = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST, "kmnist": datasets.KMNIST}
ACTIVATIONS = {"relu": torch.relu, "gelu": F.gelu, "tanh": torch.tanh}


@functools.lru_cache(maxsize=None)
def load_dataset(name, train):
    """Images as (N, n_in) floats in [0, 1], labels as (N,) ints."""
    base = "mnist" if name == "mnist14" else name
    ds = TORCHVISION[base](DATA_DIR, train=train, download=True)
    x = ds.data.float().div(255)
    if name == "mnist14":
        x = F.avg_pool2d(x[:, None], 2)[:, 0]
    return x.flatten(1), ds.targets


def layer_inputs(net, x):
    """Input to every linear layer [(B, n_in) per layer], plus the logits (B, 10)."""
    act = ACTIVATIONS[net["activation"]]
    h = (x - net["input_mean"]) / net["input_std"]
    inputs = []
    for i, (W, b) in enumerate(net["layers"]):
        inputs.append(h)
        h = h @ W.T
        if b is not None:
            h = h + b
        if i < len(net["layers"]) - 1:
            h = act(h)
    return inputs, h


def forward(net, x):
    return layer_inputs(net, x)[1]


@torch.no_grad()
def accuracy(net, dataset, train):
    x, y = load_dataset(dataset, train)
    return 100 * (forward(net, x).argmax(1) == y).float().mean().item()


def init_net(n_in, hidden, activation):
    sizes = [n_in, *hidden, 10]
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        lin = torch.nn.Linear(a, b)
        layers.append((lin.weight.detach().clone().requires_grad_(),
                       lin.bias.detach().clone().requires_grad_()))
    return {"layers": layers, "activation": activation, "input_mean": 0.0, "input_std": 1.0}


def snapshot(net):
    return {**net, "layers": [(W.detach().clone(), b.detach().clone()) for W, b in net["layers"]]}


def train(dataset, hidden, activation, seed, save_epochs):
    """Train one net on CPU (deterministic) and return {epoch: net} at save_epochs."""
    torch.manual_seed(seed)
    x, y = load_dataset(dataset, train=True)
    net = init_net(x.shape[1], hidden, activation)
    opt = torch.optim.Adam([p for W, b in net["layers"] for p in (W, b)], lr=LR)
    steps_per_epoch = -(-len(x) // BATCH_SIZE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    snapshots = {}
    for epoch in range(1, max(save_epochs) + 1):
        for idx in torch.randperm(len(x)).split(BATCH_SIZE):
            loss = F.cross_entropy(forward(net, x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if epoch in save_epochs:
            snapshots[epoch] = snapshot(net)
    return snapshots


def net_name(dataset, hidden, activation, seed, epoch):
    return f"{dataset}_{activation}{hidden[0]}_s{seed}_ep{epoch}"


def make_calibration(dataset):
    """One fixed set of training images per dataset, shared by every net on it."""
    path = NETS_DIR / f"calib_{dataset}.pt"
    if path.exists():
        return
    x, _ = load_dataset(dataset, train=True)
    g = torch.Generator().manual_seed(CALIB_SEED)
    indices = torch.randperm(len(x), generator=g)[:N_CALIB]
    torch.save({"x": x[indices], "indices": indices}, path)


def import_cbs():
    """Convert their checkpoint to our format: 784→40→20→10, ReLU, no biases."""
    out = NETS_DIR / f"{CBS_NAME}.pt"
    if out.exists() or not CBS_CKPT.exists():
        return
    try:
        with torch.serialization.safe_globals([torch.optim.SGD, torch.optim.lr_scheduler.ExponentialLR]):
            ckpt = torch.load(CBS_CKPT, map_location="cpu", weights_only=True)
    except Exception as e:
        sys.exit(f"Safe load of {CBS_CKPT.name} failed; inspect the file before loading it fully.\n{e}")
    state = ckpt["model_state_dict"]
    expected = {f"module.fc{i}.{p}" for i in (1, 2, 3) for p in ("_layer.weight", "_weight_mask")}
    if set(state) != expected:
        sys.exit(f"Unexpected keys in {CBS_CKPT.name}: {list(state)}")
    net = {
        "layers": [((state[f"module.fc{i}._layer.weight"] * state[f"module.fc{i}._weight_mask"]).float(), None)
                   for i in (1, 2, 3)],
        "activation": "relu",
        "input_mean": 0.1307,
        "input_std": 0.3081,
    }
    net["config"] = {"split": "reference", "dataset": "mnist", "hidden": (40, 20),
                     "activation": "relu", "seed": None, "epoch": 25, "source": CBS_CKPT.name,
                     "train_acc": accuracy(net, "mnist", True),
                     "test_acc": accuracy(net, "mnist", False)}
    torch.save(net, out)
    print(f"imported {CBS_NAME}")


def write_index():
    rows = []
    for path in sorted(NETS_DIR.glob("*.pt")):
        if path.name.startswith("calib_"):
            continue
        c = torch.load(path, weights_only=True)["config"]
        rows.append({"name": path.stem, "split": c["split"], "dataset": c["dataset"],
                     "hidden": "-".join(map(str, c["hidden"])), "activation": c["activation"],
                     "seed": c["seed"], "epoch": c["epoch"],
                     "train_acc": f"{c['train_acc']:.2f}", "test_acc": f"{c['test_acc']:.2f}"})
    with open(NETS_DIR / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def check():
    """Phase 1 check: sensible accuracies, no calibration/test overlap, reproducibility."""
    ok = True
    rows = write_index()
    print(f"\n{'name':30} {'split':9} {'hidden':8} {'train':>6} {'test':>6}")
    for r in rows:
        print(f"{r['name']:30} {r['split']:9} {r['hidden']:8} {r['train_acc']:>6} {r['test_acc']:>6}")

    print("\nAccuracy (final checkpoints):")
    for r in rows:
        acc = float(r["test_acc"])
        if r["name"] == CBS_NAME:
            passed = abs(acc - CBS_ACC) <= CBS_TOL
            target = f"{CBS_ACC} ± {CBS_TOL}"
        elif int(r["epoch"]) == EPOCHS:
            passed = acc >= MIN_TEST_ACC[r["dataset"]]
            target = f">= {MIN_TEST_ACC[r['dataset']]}"
        else:
            continue
        if not passed:
            print(f"  FAIL {r['name']}: {acc:.2f} (expected {target})")
            ok = False
    n_final = sum(int(r["epoch"]) == EPOCHS or r["name"] == CBS_NAME for r in rows)
    print(f"  {n_final} final nets checked" + ("" if ok else ", with failures above"))
    if not (NETS_DIR / f"{CBS_NAME}.pt").exists():
        print(f"  (CBS reference not imported yet: download {CBS_CKPT.name} into nets/)")

    print("\nCalibration/test overlap (exact duplicate images):")
    for dataset in MIN_TEST_ACC:
        calib = torch.load(NETS_DIR / f"calib_{dataset}.pt", weights_only=True)["x"]
        test_x, _ = load_dataset(dataset, train=False)
        test_hashes = {hashlib.sha1(row.numpy().tobytes()).hexdigest() for row in test_x}
        shared = sum(hashlib.sha1(row.numpy().tobytes()).hexdigest() in test_hashes for row in calib)
        print(f"  {dataset}: {shared} of {len(calib)} calibration images also in the test set")
        ok &= shared == 0

    print("\nReproducibility (retrain mnist relu128 seed 0 for 1 epoch, compare with saved):")
    saved = torch.load(NETS_DIR / f"{net_name('mnist', (128, 64), 'relu', 0, 1)}.pt", weights_only=True)
    fresh = train("mnist", (128, 64), "relu", 0, save_epochs=(1,))[1]
    same = all(torch.equal(W1, W2) and torch.equal(b1, b2)
               for (W1, b1), (W2, b2) in zip(saved["layers"], fresh["layers"]))
    print(f"  bit-for-bit identical: {same}")
    ok &= same

    print(f"\nPhase 1 check: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    NETS_DIR.mkdir(exist_ok=True)
    for dataset in MIN_TEST_ACC:
        make_calibration(dataset)

    for split, dataset, hidden, activation, seeds in NETS:
        save_epochs = CHECKPOINT_EPOCHS if split == "train" else (EPOCHS,)
        for seed in seeds:
            paths = {e: NETS_DIR / f"{net_name(dataset, hidden, activation, seed, e)}.pt" for e in save_epochs}
            if all(p.exists() for p in paths.values()):
                continue
            print(f"training {paths[EPOCHS].stem}", flush=True)
            for epoch, net in train(dataset, hidden, activation, seed, save_epochs).items():
                net["config"] = {"split": split, "dataset": dataset, "hidden": hidden,
                                 "activation": activation, "seed": seed, "epoch": epoch,
                                 "lr": LR, "lr_schedule": "cosine", "epochs": EPOCHS,
                                 "batch_size": BATCH_SIZE, "optimizer": "Adam",
                                 "train_acc": accuracy(net, dataset, True),
                                 "test_acc": accuracy(net, dataset, False)}
                torch.save(net, paths[epoch])

    import_cbs()
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
