# Week 6 code outline: optimizers and Lagrange multipliers

Skeletons only. Syntax reminders: [numpy-refresher.md](../../docs/numpy-refresher.md) §9 (contours).

## Packages

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm               # log-scaled contour colors
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess   # answer keys

from sklearn.datasets import make_blobs, make_moons
from sklearn.svm import SVC

import sys; sys.path.insert(0, "../..")
from mlutils.gradcheck import numerical_gradient, numerical_jacobian, relative_error
```

## Suggested layout

```
weeks/week06-optimization/
  functions.py      # rosenbrock (f, grad, hess), quadratic, saddle example
  optimizers.py     # gd, momentum, nesterov, rmsprop, adam, newton  -> could also live in mlutils/
  svm_dual.py
  week06.ipynb
```

---

## Step 1: Test functions

```python
def rosenbrock(p: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """p = [x, y]. Minimum at (a, a²), f = 0."""
    x, y = p                                       # tuple-unpack a length-2 array
    ...

def rosenbrock_grad(p, a=1.0, b=100.0) -> np.ndarray: ...   # shape (2,)
def rosenbrock_hess(p, a=1.0, b=100.0) -> np.ndarray: ...   # shape (2, 2)

def make_quadratic(kappa: float):
    """Return (f, grad, hess) for f(x) = ½ xᵀ A x with A = diag(1, kappa)."""
    A = np.diag([1.0, kappa])
    ...
```

Check against SciPy: `np.allclose(rosenbrock_grad(p), rosen_der(p))`, same for the Hessian.

---

## Step 2: Optimizers

A common interface makes the comparison loop trivial:

```python
def gd(grad_f, x0, lr, n_iters) -> np.ndarray:
    """Return path, shape (n_iters + 1, 2), including x0."""
    x = x0.astype(float).copy()
    path = [x.copy()]
    for _ in range(n_iters):
        ...                                         # x = x - lr * grad_f(x)
        path.append(x.copy())                       # copy! otherwise every entry is the same array
    return np.array(path)
```

Write the others with the same signature (plus their hyperparameters):

```python
def momentum(grad_f, x0, lr, n_iters, beta=0.9):
    """Heavy ball: v = beta * v - lr * g;  x = x + v"""
    ...

def nesterov(grad_f, x0, lr, n_iters, beta=0.9):
    """Evaluate the gradient at the look-ahead point x + beta * v."""
    ...

def rmsprop(grad_f, x0, lr, n_iters, beta=0.9, eps=1e-8):
    """s = beta * s + (1 - beta) * g**2;  x -= lr * g / (sqrt(s) + eps)    (elementwise)"""
    ...

def adam(grad_f, x0, lr, n_iters, beta1=0.9, beta2=0.999, eps=1e-8):
    """m, v moving averages of g and g**2; bias-correct with 1 - beta**t (t starts at 1)."""
    ...

def newton(grad_f, hess_f, x0, n_iters, damping=1.0):
    """x -= damping * solve(H, g). Stop early if ||g|| < 1e-12 (pad the path or just return it shorter)."""
    ...
```

Divergence guard (useful for learning-rate sweeps): inside the loop, `if not np.all(np.isfinite(x)): break`.

---

## Step 3: Plots

Log-scaled contour of Rosenbrock with paths:

```python
xs, ys = np.linspace(-2, 2, 400), np.linspace(-1, 3, 400)
XX, YY = np.meshgrid(xs, ys)
ZZ = rosenbrock(np.stack([XX, YY]))          # if your function is written with x, y = p, it vectorizes over grids
fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(XX, YY, ZZ, levels=np.logspace(-1, 3.5, 20), norm=LogNorm(), cmap="viridis")
for name, path in paths.items():
    ax.plot(path[:, 0], path[:, 1], ".-", ms=2, lw=1, label=name)
ax.plot(1, 1, "r*", ms=15); ax.legend()
```

Convergence: `ax.semilogy([rosenbrock(p) for p in path])`, one line per optimizer.

**Empirical rate on the quadratic:** with `f*=0`, look at the ratio `f(x_{t+1}) / f(x_t)` (or `‖x_{t+1}‖/‖x_t‖`) for large `t`, and compare it to your derived formula in `κ`.

---

## Step 4: SVM dual (B1)

```python
X, y = make_blobs(n_samples=60, centers=2, cluster_std=0.8, random_state=4)
y = 2 * y - 1                                         # ±1 labels

def svm_dual_fit(X, y, C=None):
    """Return (alpha, w, b, support_mask).

    Q = (y[:, None] * y[None, :]) * (X @ X.T)          # (n, n), the outer product of labels times the Gram matrix
    Minimize the NEGATED dual (SciPy minimizes):  f(α) = ½ αᵀQα − Σα,  grad = Qα − 1
    bounds = [(0, C)] * n          (C=None means no upper bound: hard margin)
    constraints = [{"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y.astype(float)}]
    res = minimize(f, x0=np.zeros(n), jac=grad, bounds=bounds, constraints=constraints, method="SLSQP")

    Then: w = Σ αᵢ yᵢ xᵢ;  support vectors: α > 1e-6;  b = mean over SVs of (yᵢ − wᵀxᵢ)
    """
    ...
```

Answer key:

```python
svc = SVC(kernel="linear", C=1e6).fit(X, y)          # huge C ≈ hard margin
svc.coef_, svc.intercept_, svc.support_               # support_ = indices of support vectors
```

Plot margins with `ax.contour(XX, YY, grid @ w + b, levels=[-1, 0, 1], linestyles=["--", "-", "--"])` and circle SVs with `ax.scatter(..., s=150, facecolors="none", edgecolors="k")`.

## Step 5: PCA via Lagrange (B2)

```python
def top_component_slsqp(S, w0):
    """maximize wᵀSw s.t. wᵀw = 1  ⇔  minimize −wᵀSw.  Equality constraint: w @ w - 1."""
    ...

def top_component_projected(S, lr=0.1, n_iters=500, seed=0):
    """w ← w + lr * 2 S w;  w ← w / ||w||.  (Compare to power iteration from week 3. What's the difference?)"""
    ...
```

Lagrange picture in 2-D: plot `ax.contour` of `wᵀSw` over a grid, the unit circle (`np.cos(t), np.sin(t)`), and `ax.quiver` arrows for `∇f = 2Sw*` and `∇g = 2w*` at the optimum `w*`.

---

## Syntax reminders for this week

- `x, y = p` unpacks along the first axis, which works for both a `(2,)` point and a `(2, H, W)` grid. Handy for vectorized contour plots.
- `np.stack([XX, YY])` → shape `(2, H, W)`.
- Lambda functions for SciPy constraints: `lambda a: a @ y`.
- `[(0, None)] * n` builds a list of `n` identical bound tuples.
- Dict of results: `paths = {"GD": gd(...), "Adam": adam(...)}` then loop with `.items()`.
- `np.logspace(-1, 3.5, 20)` gives log-spaced contour levels (essential for Rosenbrock's banana valley).

---

## PyTorch corner

PyTorch's optimizers work on any tensor with `requires_grad`, not just neural networks:

```python
import torch

def rosen_t(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2

p = torch.tensor([-1.5, 2.0], dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([p], lr=0.05)                 # note: a LIST of tensors
path = [p.detach().clone().numpy()]
for _ in range(2000):
    opt.zero_grad()
    rosen_t(p).backward()
    opt.step()
    path.append(p.detach().clone().numpy())         # clone! .numpy() shares memory with p
```

Compare this path with your `adam(...)` using the same hyperparameters. They should be **identical** to ~1e-12.

LBFGS (quasi-Newton) needs a *closure* because it evaluates the function several times per step:

```python
opt = torch.optim.LBFGS([p], lr=1, line_search_fn="strong_wolfe")
def closure():
    opt.zero_grad(); loss = rosen_t(p); loss.backward(); return loss
for _ in range(20):
    opt.step(closure)
```

Hessian for Newton: `torch.autograd.functional.hessian(rosen_t, p.detach())`, an answer key for `rosenbrock_hess`.

**Distinctions to watch:**
- **Momentum formula:** PyTorch's SGD uses `v = μ·v + g; p -= lr·v`, while the "heavy ball" form above is `v = μ·v − lr·g; p += v`. They're equivalent for a constant `lr` but differ if `lr` changes during training (schedulers). PyTorch also has a `dampening` argument (default 0).
- **Nesterov:** `torch.optim.SGD(..., momentum=0.9, nesterov=True)` uses a reformulated (but equivalent) update, so paths may differ slightly in the first steps.
- **Adam's `eps`** is added *after* the square root, as in the paper. Implementations that put it inside the sqrt behave slightly differently near zero gradients.
- **`.detach().clone()`** when recording parameters. Without `clone()`, every saved entry points at the same, final memory.
- `weight_decay` in `Adam` is plain L2 added to the gradient; `AdamW` decouples it. They are *not* the same for adaptive methods.
