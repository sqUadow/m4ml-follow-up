# Week 1 code outline: linear regression three ways

Skeletons only: you write the bodies. Syntax reminders link to [numpy-refresher.md](../../docs/numpy-refresher.md).

## Packages

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats                                # t-distribution quantiles for CIs

# Answer keys only (don't call these inside your implementations)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import statsmodels.api as sm
```

## Suggested layout

```
weeks/week01-linear-regression/
  linreg.py          # your functions (importable)
  week01.ipynb       # experiments, plots, results table
mlutils/             # repo root: small helpers you'll reuse in later weeks
  __init__.py
  gradcheck.py       # numerical_gradient() from this week
```

To import `mlutils` from a notebook inside `weeks/week01-.../`:

```python
import sys; sys.path.insert(0, "../..")      # repo root
from mlutils.gradcheck import numerical_gradient
```

In Jupyter, add `%load_ext autoreload` and `%autoreload 2` at the top so edits to `linreg.py` are picked up without restarting.

---

## Step 0: Data, split, standardize

```python
data = fetch_california_housing(as_frame=True)
X_raw, y = data.data.to_numpy(), data.target.to_numpy()      # (20640, 8), (20640,)
feature_names = list(data.feature_names)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=0)
```

```python
def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) per feature, each shape (d,). Fit on TRAIN only."""
    ...

def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(X - mean) / std, broadcasting over rows."""
    ...

def add_intercept(X: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, d+1) with a column of ones FIRST."""
    # hint: np.column_stack([np.ones(len(X)), X])
    ...
```

> **Why train-only statistics?** Using the test set's mean/std leaks information about the test set into the model.

**Synthetic warm-up** (do this first; you know the right answer):

```python
rng = np.random.default_rng(0)
n = 200
X_syn = rng.normal(size=(n, 2))
y_syn = 2 + 3 * X_syn[:, 0] - 1 * X_syn[:, 1] + 0.1 * rng.normal(size=n)
# every fit_* function below should return approximately [2, 3, -1]
```

---

## Step 1: Normal equation

```python
def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve (XᵀX) w = Xᵀy.  X: (n, p) incl. intercept column, y: (n,) -> w: (p,)."""
    # use np.linalg.solve(A, b), not np.linalg.inv
    ...
```

## Step 2: SVD / pseudoinverse

```python
def fit_svd(X: np.ndarray, y: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Minimum-norm least-squares solution via thin SVD.

    1. U, s, Vt = np.linalg.svd(X, full_matrices=False)   # U (n,p), s (p,), Vt (p,p)
    2. invert only the singular values above rcond * s.max(); set the rest to 0
    3. w = Vt.T @ (s_inv * (U.T @ y))                    # think about why this order is cheap
    """
    ...
```

Syntax notes:
- `s` is 1-D. Multiplying `s_inv * vector` scales elementwise, which is the same as `diag(s_inv) @ vector` without building the matrix.
- `np.where(s > tol, 1 / s, 0.0)` will warn about division by zero for tiny `s`. Use `s_inv = np.zeros_like(s); mask = s > tol; s_inv[mask] = 1 / s[mask]` instead.

## Step 3: Gradient descent

```python
def mse_loss(X, y, w) -> float:
    """(1/n) * ||Xw - y||²"""
    ...

def mse_grad(X, y, w) -> np.ndarray:
    """Your hand-derived gradient of mse_loss w.r.t. w. Shape (p,)."""
    ...

def fit_gd(X, y, lr: float = 0.1, n_iters: int = 1000, tol: float = 1e-10,
           w0: np.ndarray | None = None) -> tuple[np.ndarray, list[float]]:
    """Plain batch gradient descent. Return (w, loss_history).

    Loop: compute grad, w = w - lr * grad, record loss.
    Stop early if np.linalg.norm(grad) < tol.
    Guard: if not np.isfinite(loss), break (diverged) and report it.
    """
    ...
```

<details>
<summary>Check your gradient derivation (peek after deriving)</summary>

`∇_w (1/n)‖Xw − y‖² = (2/n) Xᵀ(Xw − y)`, shape `(p,)`. Setting it to zero gives `XᵀX w = Xᵀy`.

</details>

## Step 4: Gradient checking (reused in weeks 2, 5, 6) → `mlutils/gradcheck.py`

```python
def numerical_gradient(f, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences: g[i] ≈ (f(w + eps·eᵢ) − f(w − eps·eᵢ)) / (2·eps).

    f: callable taking w (same shape as w) and returning a float.
    Loop over np.ndindex(w.shape) so it also works for matrices later (week 5).
    Remember to copy w before perturbing it, or restore it after.
    """
    ...

def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / max(||a|| + ||b||, 1e-12). Below ~1e-7 is great in float64."""
    ...
```

Usage: `relative_error(mse_grad(X, y, w), numerical_gradient(lambda w_: mse_loss(X, y, w_), w))` at a **random** `w` (not zeros).

## Step 5: Evaluation

```python
def rmse(y, y_hat) -> float: ...
def r2(y, y_hat) -> float:
    """1 - SS_res / SS_tot."""
    ...
```

Results table idea:

```python
results = {}
for name, fit in [("normal", fit_normal_equation), ("svd", fit_svd)]:
    t0 = time.perf_counter(); w = fit(Xtr, y_train); dt = time.perf_counter() - t0
    results[name] = dict(w=w, train_rmse=rmse(y_train, Xtr @ w), test_rmse=rmse(y_test, Xte @ w), sec=dt)
# then add GD, and print each row with f-strings
```

## Step 6: Answer keys

```python
w_lstsq, *_ = np.linalg.lstsq(Xtr, y_train, rcond=None)     # *_ discards the other 3 return values

sk = LinearRegression(fit_intercept=False).fit(Xtr, y_train)  # Xtr already has an intercept column
w_sk = sk.coef_

np.testing.assert_allclose(w_mine, w_lstsq, rtol=1e-6, atol=1e-8)
```

## Step 7: Break it with collinearity

```python
X_dup = np.column_stack([Xtr, Xtr[:, 1]])                                # exact duplicate of a feature
X_near = np.column_stack([Xtr, Xtr[:, 1] + 1e-6 * rng.normal(size=len(Xtr))])

for M in (X_dup, X_near):
    print(np.linalg.matrix_rank(M), np.linalg.cond(M.T @ M))
```

Things to observe and explain:
- `np.linalg.solve` on the exact duplicate may raise `LinAlgError: Singular matrix`, **or** return huge, meaningless weights (floating-point rounding can make a singular matrix look barely invertible). Wrap it in `try/except np.linalg.LinAlgError`.
- On the exact duplicate, `fit_svd` gives the **minimum-norm** solution: the weight for the duplicated feature is split evenly between the two copies.
- On the *near* duplicate, the smallest singular value is tiny but not zero (print `s`). With `rcond=1e-10` it is still inverted, and `1/s_min` is huge, so you get weights like `+500` and `−500` that cancel. That's the true least-squares answer, fitting noise in the 1e-6 column. Only truncating (`rcond=1e-4`) discards that direction. **The pseudoinverse helps only if you drop small singular values.** Ridge (stretch) is the smooth version of this.
- Predictions `X @ w` may still be fine for all methods. Collinearity hurts *interpretability* of `w` more than *fit*.

## Step 8: Confidence intervals (12.3.5–12.3.6)

```python
def ols_confidence_intervals(X, y, w, alpha: float = 0.05):
    """Return (se, lower, upper), each shape (p,).

    n, p = X.shape
    residuals -> RSS -> sigma2_hat = RSS / (n - p)       # unbiased; note n - p, not n
    cov_w = sigma2_hat * inv(XᵀX)                         # here inv is fine: you need the matrix
    se = sqrt(diag(cov_w))
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - p)
    """
    ...
```

Answer key:

```python
ols = sm.OLS(y_train, Xtr).fit()      # Xtr already has the intercept column (or use sm.add_constant)
print(ols.summary())                  # look for the coef, std err, [0.025 0.975] columns
ols.bse, ols.conf_int(alpha=0.05)     # standard errors, (p, 2) array of intervals
```

## Step 9: Polynomial regression

```python
def poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    """1-D x (n,) -> (n, degree+1) with columns [1, x, x², ..., x^degree].
    hint: np.vander(x, degree + 1, increasing=True)
    Standardize x first, otherwise x^10 overflows the conditioning."""
    ...
```

Plot `train_rmse` and `test_rmse` vs. degree on one axis, and `np.linalg.cond(X_poly)` on a log-scale second axis (`ax.twinx()`).

## Stretch: ridge

```python
def fit_ridge(X, y, lam: float, penalize_intercept: bool = False) -> np.ndarray:
    """Solve (XᵀX + λD) w = Xᵀy, where D = I but with D[0,0] = 0 if the intercept is unpenalized."""
    ...
```

Check against sklearn's `Ridge`, whose objective is `‖y − Xw‖² + α‖w‖²` (a sum, not a mean), so `alpha` = your `λ` if your loss is also a sum. Two equivalent comparisons:
- `Ridge(alpha=lam, fit_intercept=False).fit(Xtr, y)` penalizes *every* column, including your ones column, so compare with `penalize_intercept=True`.
- `Ridge(alpha=lam).fit(Xtr[:, 1:], y)` fits an unpenalized intercept itself (`.intercept_`), so compare with `penalize_intercept=False`.

---

## Syntax reminders for this week

- `X.T @ X` is `(p, n) @ (n, p) -> (p, p)`. Write shapes in comments.
- `y - X @ w`: make sure `y` is `(n,)` and not `(n, 1)` (see refresher §1).
- `np.diag(M)` extracts a diagonal; `np.diag(v)` builds a diagonal matrix.
- Unpack and ignore: `w, *_ = np.linalg.lstsq(...)`.
- `plt.semilogy(losses)` for loss curves.

---

## PyTorch corner

This week PyTorch is an **autograd answer key** for your gradient, plus a preview of the same model in PyTorch style. See [pytorch-primer.md](../../docs/pytorch-primer.md) §2–3.

```python
import torch

Xt = torch.from_numpy(Xtr)                    # stays float64, so precision matches NumPy
yt = torch.from_numpy(y_train)
w = torch.randn(Xt.shape[1], dtype=torch.float64, requires_grad=True)

loss = ((Xt @ w - yt) ** 2).mean()
loss.backward()
grad_torch = w.grad.numpy()
# compare to mse_grad(Xtr, y_train, w.detach().numpy())
```

Closed form in PyTorch: `torch.linalg.lstsq(Xt, yt).solution`.

Gradient descent "the PyTorch way" (you'll use this pattern all the time from week 5):

```python
model = torch.nn.Linear(d, 1)                 # includes its own bias, so DON'T add an intercept column
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()                  # mean of squared errors, no 1/2
Xf, yf = torch.from_numpy(X_std).float(), torch.from_numpy(y).float()

for t in range(1000):
    pred = model(Xf).squeeze(1)               # (n, 1) -> (n,). Without squeeze, MSELoss broadcasts (n,1) vs (n,) -> (n,n)!
    loss = loss_fn(pred, yf)
    opt.zero_grad(); loss.backward(); opt.step()
```

**Distinctions to watch:**
- `nn.Linear` handles the bias separately: no column of ones.
- `nn.MSELoss` averages, so its gradient is `(2/n)Xᵀ(Xw − y)`, the same as yours *if* your loss is a mean. If yours is `½‖·‖²` or a sum, the effective learning rate differs.
- The `(n,1)` vs `(n,)` shape bug is even sneakier in PyTorch: `MSELoss` only *warns*, then broadcasts.
