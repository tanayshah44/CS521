"""Record, per image, the worst-case margin lower bound min_j lb[j] of a compiled certifier,
plus the time per image, so two certifiers can be compared beyond the verified percentage."""
import contextlib, io, sys, time
import numpy as np
import torch
from torchvision import datasets, transforms
from constraintflow.lib.globals import set_device

build, tag = sys.argv[1], sys.argv[2]
eps_list = [float(e) for e in sys.argv[3].split(",")]
n = int(sys.argv[4])
set_device("cpu")
sys.path.insert(0, build)
from main import run

test = datasets.MNIST("./data", train=False, download=False, transform=transforms.ToTensor())
X = torch.stack([test[i][0] for i in range(n)])
y = torch.tensor([test[i][1] for i in range(n)])
out = {}
for eps in eps_list:
    lbs, t0 = [], time.perf_counter()
    for i in range(n):
        with contextlib.redirect_stdout(io.StringIO()):
            lb, _ = run("mnist_relu_3_100.onnx", 1, eps, X[i:i+1], y[i:i+1], dataset="mnist",
                        train=False, print_intermediate_results=False, no_sparsity=False)
        lbs.append(lb.reshape(-1).min().item())
    dt = (time.perf_counter() - t0) / n
    out[f"eps{eps}"] = np.array(lbs)
    print(f"{tag} eps={eps} time/img={dt:.4f}s verified={np.mean(np.array(lbs) >= 0):.2%}", flush=True)
np.savez(f"diag_{tag}.npz", **out)
