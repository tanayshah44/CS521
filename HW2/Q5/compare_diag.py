"""Compare per-image margin lower bounds of two certifiers (outputs of diag.py)."""
import sys
import numpy as np

TOL = 1e-5  # float32 noise: smaller differences count as ties
a, b = np.load(f"diag_{sys.argv[1]}.npz"), np.load(f"diag_{sys.argv[2]}.npz")
for k in a.files:
    d = b[k] - a[k]
    unv = a[k] < 0
    print(f"{k}: tighter={int((d > TOL).sum())} looser={int((d < -TOL).sum())} "
          f"max_gain={d.max():.4f} max_loss={max(0.0, -d.min()):.4f} mean_change={d.mean():+.4f}")
    print(f"   closest unverified: {sys.argv[1]}={a[k][unv].max():.4f} {sys.argv[2]}={b[k][b[k] < 0].max():.4f}  "
          f"verified: {int((a[k] >= 0).sum())} / {int((b[k] >= 0).sum())}")
