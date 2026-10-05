# Week 2 code outline: logistic regression, MLE, Newton

Skeletons only. Syntax reminders: [numpy-refresher.md](../../docs/numpy-refresher.md) (§7 numerical safety is important this week).

## Packages

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize          # generic optimizer for the MLE warm-up
from scipy.special import expit              # stable sigmoid, a good answer key for yours

# Answer keys / data
from sklearn.datasets import load_breast_cancer, make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm

import sys; sys.path.insert(0, "../..")
from mlutils.gradcheck import numerical_gradient, relative_error      # from week 1
```

## Suggested layout

```
weeks/week02-logistic-regression/
  logreg.py          # sigmoid, nll, grad, hessian, fit_gd, fit_newton
  week02.ipynb
mlutils/gradcheck.py # add numerical_jacobian() this week
```

---

## Step 1: MLE warm-up

```python
rng = np.random.default_rng(0)
coin = rng.binomial(n=1, p=0.3, size=200)              # Bernoulli = binomial with n=1
heights = rng.normal(loc=170, scale=8, size=500)

def bernoulli_nll(p: float, data: np.ndarray) -> float:
    """-log L(p) = -Σ [yᵢ log p + (1-yᵢ) log(1-p)]"""
    ...

def gaussian_nll(params: np.ndarray, data: np.ndarray) -> float:
    """params = [mu, log_sigma]. Optimizing log σ keeps σ > 0 without constraints.
    -log L = n·log σ + (n/2)·log 2π + Σ (xᵢ - μ)² / (2σ²)"""
    ...
```

Fitting with SciPy (answer key for your closed-form MLEs):

```python
res = minimize(gaussian_nll, x0=np.array([150.0, np.log(5.0)]), args=(heights,))
mu_hat, sigma_hat = res.x[0], np.exp(res.x[1])
# compare to heights.mean() and heights.std(ddof=0)   <- ddof=0 is the MLE (divide by n)
```

- `minimize(f, x0, args=(...))` calls `f(x, *args)`. Check `res.success` and `res.message`.
- For a bounded scalar like `p ∈ (0,1)`, use `minimize_scalar(f, bounds=(1e-9, 1-1e-9), method="bounded", args=(coin,))`.

Plot `ℓ(p)` over `np.linspace(0.01, 0.99, 200)` and add `plt.axvline(coin.mean())`.

---

## Step 2: Model pieces

Use a design matrix with an intercept column (reuse `add_intercept` from week 1). Labels `y ∈ {0, 1}`, shape `(n,)`.

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    """σ(z) = 1/(1+e^{-z}), without overflow for large |z|.
    Compare to scipy.special.expit. A naive version warns on z = -1000."""
    ...

def nll(w, X, y, lam: float = 0.0) -> float:
    """Sum (not mean) negative log-likelihood + (lam/2)·||w[1:]||²  (intercept unpenalized).
    Avoid computing log(sigmoid(z)) directly; see the hint below."""
    ...

def grad(w, X, y, lam: float = 0.0) -> np.ndarray:
    """Shape (p,). Your hand-derived gradient (+ penalty term on w[1:])."""
    ...

def hessian(w, X, y, lam: float = 0.0) -> np.ndarray:
    """Shape (p, p). Your hand-derived Hessian (+ penalty term on the diagonal, except [0,0])."""
    # building diag(s) as an (n, n) matrix is wasteful: X.T @ (s[:, None] * X) does the same thing
    ...
```

<details>
<summary>Hint: stable log-likelihood</summary>

With `z = Xw`: `−[y log σ(z) + (1−y) log(1−σ(z))] = log(1 + e^{z}) − y·z`.
`np.logaddexp(0, z)` computes `log(e⁰ + e^z) = log(1 + e^z)` without overflow.

</details>

<details>
<summary>Check your gradient and Hessian (peek after deriving)</summary>

With `p = σ(Xw)` (shape `(n,)`) and `S = diag(p ⊙ (1 − p))`:

- `∇NLL = Xᵀ(p − y)`
- `∇²NLL = Xᵀ S X`

Plus `λ·w` (except the intercept) and `λ·I` (except `[0,0]`) for the penalty.

</details>

## Step 3: Gradient and Hessian checks → add to `mlutils/gradcheck.py`

```python
def numerical_jacobian(F, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """F: R^p -> R^m. Returns (m, p) with column j ≈ (F(w + eps·eⱼ) − F(w − eps·eⱼ)) / (2 eps).
    The Hessian is the Jacobian of the gradient: numerical_jacobian(lambda v: grad(v, X, y), w)."""
    ...
```

```python
w_test = rng.normal(size=X.shape[1])          # random point, NOT zeros (symmetry can hide bugs)
print(relative_error(grad(w_test, X, y), numerical_gradient(lambda v: nll(v, X, y), w_test)))
print(relative_error(hessian(w_test, X, y), numerical_jacobian(lambda v: grad(v, X, y), w_test)))
```

## Step 4: Optimizers

```python
def fit_gd(X, y, lr, n_iters, lam=0.0, w0=None, tol=1e-10):
    """Return (w, history) where history is a dict of lists: 'nll', 'grad_norm', 'w_norm'.
    Tip: with a SUM loss, a good lr is ~1/n-ish; with a MEAN loss, ~0.1–1. Pick one convention."""
    ...

def fit_newton(X, y, n_iters=50, lam=0.0, w0=None, tol=1e-10):
    """w ← w − H⁻¹ g. Compute the step with np.linalg.solve(H, g), never inv(H).
    Stop when ||g|| < tol. Return (w, history)."""
    ...
```

Convergence plot:

```python
nll_star = min(min(h_gd["nll"]), min(h_newton["nll"]))
plt.semilogy(np.array(h_gd["nll"]) - nll_star + 1e-16, label="GD")
plt.semilogy(np.array(h_newton["nll"]) - nll_star + 1e-16, "o-", label="Newton")
```

## Step 5: Decision boundary (2-D toy data)

The boundary is where `w₀ + w₁x₁ + w₂x₂ = 0`, a line. Either solve for `x₂` and `plt.plot` it, or use the meshgrid/contour skeleton in the refresher §9 with `levels=[0.5]` on `sigmoid(grid @ w)`.

## Step 6: Answer keys

```python
# Unregularized (sklearn fits the intercept itself, so pass X WITHOUT your ones column)
sk = LogisticRegression(C=np.inf, max_iter=10_000).fit(X_no_intercept, y)
w_sk = np.r_[sk.intercept_, sk.coef_.ravel()]          # np.r_ concatenates 1-D pieces

# L2-regularized: sklearn minimizes  C·Σ NLLᵢ + ½||w||²  ⇔ your sum-NLL + (λ/2)||w||² with C = 1/λ
sk = LogisticRegression(C=1 / lam, max_iter=10_000, tol=1e-10).fit(X_no_intercept, y)

# Standard errors (stretch)
logit = sm.Logit(y, X_with_intercept).fit()
logit.params, logit.bse
```

Notes:
- In recent scikit-learn, the old `penalty="none"` / `penalty=None` argument is deprecated. `C=np.inf` means "no regularization."
- On perfectly separable data (breast cancer!), `C=np.inf` has no finite answer either. sklearn will stop at `max_iter` with large coefficients and a `ConvergenceWarning`. That's the point of the exercise.

## Step 7: Breast cancer

```python
data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.2,
                                          random_state=0, stratify=data.target)
# standardize with train stats (week 1 functions), add intercept, then fit
```

Track and plot `history["w_norm"]` for λ = 0 vs. λ = 1. Then:

```python
y_pred = (sigmoid(X_te @ w) >= 0.5).astype(int)
print(confusion_matrix(y_te, y_pred)); print(classification_report(y_te, y_pred))
```

## Stretch: softmax regression

```python
def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    """(n,) ints -> (n, k). hint: np.eye(k)[y]"""
    ...

def softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise. Subtract Z.max(axis=1, keepdims=True) first for stability."""
    ...

def ce_loss_and_grad(W, X, Y):
    """W (p, k), X (n, p), Y one-hot (n, k). Return (loss, dW) with dW shape (p, k)."""
    ...
```

---

## Syntax reminders for this week

- `np.r_[a, b]` concatenates scalars and 1-D arrays. Handy for `[intercept, *coefs]`.
- `X.T @ (s[:, None] * X)` scales row `i` of `X` by `s[i]`. Same as `X.T @ np.diag(s) @ X` without the `(n, n)` matrix.
- `(p >= 0.5).astype(int)` converts booleans to 0/1.
- `stratify=y` in `train_test_split` keeps class proportions equal in both splits.
- `np.errstate(over="ignore")` is a context manager to silence expected overflow warnings locally (but prefer stable formulas).

---

## PyTorch corner

PyTorch is your **answer key for both derivatives** this week. Use float64 so tolerances match NumPy.

```python
import torch
torch.set_default_dtype(torch.float64)

Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(float))

def nll_t(w):
    z = Xt @ w
    return torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="sum")

w = torch.from_numpy(w_test).requires_grad_(True)
nll_t(w).backward()
g_torch = w.grad.numpy()                                          # vs your grad()
H_torch = torch.autograd.functional.hessian(nll_t, w.detach()).numpy()   # vs your hessian()
```

The equivalent model, PyTorch style:

```python
model = torch.nn.Linear(d, 1)                       # bias = intercept
loss_fn = torch.nn.BCEWithLogitsLoss()              # mean over batch; takes LOGITS
opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=lam)
# loop: logits = model(Xf).squeeze(1); loss = loss_fn(logits, yf); opt.zero_grad(); loss.backward(); opt.step()
```

**Distinctions to watch:**
- `BCEWithLogitsLoss` takes **logits** (`Xw`), not probabilities, and targets must be **float** (`0.`/`1.`). `BCELoss(sigmoid(z), y)` exists but is less stable. Prefer the logits version (it uses the same `logaddexp` trick as you).
- Default `reduction="mean"` divides by `n`, so its gradient is yours divided by `n`.
- `weight_decay` in `torch.optim.SGD` penalizes **all** parameters, including the bias, unlike your unpenalized intercept. To exclude the bias, pass parameter groups: `SGD([{"params": model.weight, "weight_decay": lam}, {"params": model.bias, "weight_decay": 0}], lr=0.1)`.
- For multiclass (stretch), `nn.CrossEntropyLoss` takes raw logits `(n, k)` and integer labels `(n,)`, so no one-hot and no softmax.
