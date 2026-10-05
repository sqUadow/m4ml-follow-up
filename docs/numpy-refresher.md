# Python + NumPy refresher

Syntax reminders for the things these projects use constantly. Each week's code outline points back here. Official references: [NumPy for absolute beginners](https://numpy.org/doc/stable/user/absolute_beginners.html), [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), [linear algebra (`numpy.linalg`)](https://numpy.org/doc/stable/reference/routines.linalg.html).

## 1. Shapes are everything

Write the shape next to every variable, in a comment, until it's automatic.

```python
import numpy as np

X = np.random.default_rng(0).normal(size=(100, 3))   # (n, d): rows = samples, cols = features
w = np.zeros(3)                                       # (d,)   1-D vector, NOT (d, 1)
y_hat = X @ w                                         # (n,)
X.shape, X.ndim, X.dtype                              # ((100, 3), 2, dtype('float64'))
```

**1-D vs column vectors.** NumPy's `(d,)` is neither a row nor a column. `X @ w` works for `(n,d) @ (d,) -> (n,)`. Mixing `(n,)` and `(n,1)` silently broadcasts to `(n,n)`:

```python
a = np.ones(5)          # (5,)
b = np.ones((5, 1))     # (5, 1)
(a - b).shape           # (5, 5)  <- almost always a bug (e.g. residuals y - X @ w)
b.ravel()               # (5,)    flatten back to 1-D
a[:, None]              # (5, 1)  add an axis (same as a.reshape(-1, 1) or a[:, np.newaxis])
```

Guard against it with `assert y.shape == y_hat.shape`.

## 2. Creating arrays

```python
np.zeros((n, d)); np.ones(d); np.eye(d)           # zeros, ones, identity
np.full((2, 3), 7.0)                              # constant array
np.arange(0, 10, 2)                               # [0 2 4 6 8] (end excluded)
np.linspace(-2, 2, 100)                           # 100 points, end included (for plotting)
np.column_stack([np.ones(n), X])                  # prepend an intercept column -> (n, d+1)
np.hstack([A, B]); np.vstack([A, B])              # concatenate horizontally / vertically
np.diag(v)                                        # vector -> diagonal matrix; np.diag(M) -> diagonal of M
```

**Random numbers.** Use the Generator API (not `np.random.seed`):

```python
rng = np.random.default_rng(42)
rng.normal(loc=0, scale=1, size=(n, d))
rng.uniform(0, 1, size=n)
rng.integers(0, 10, size=5)
rng.permutation(n)                                # shuffled indices 0..n-1
rng.multivariate_normal(mean, cov, size=n)        # (n, d) samples
rng.choice(k, size=n, p=weights)                  # categorical draws
```

## 3. Indexing, slicing, masks

```python
X[0]            # first row, shape (d,)
X[:, 0]         # first column, shape (n,)
X[:, [0]]       # first column, shape (n, 1)  (list index keeps the axis)
X[:10, 1:3]     # rows 0-9, cols 1-2
X[-1]           # last row
mask = y == 1   # boolean array (n,)
X[mask]         # rows where y == 1
X[mask].mean(axis=0)                      # per-feature mean of class 1
idx = rng.permutation(n); X, y = X[idx], y[idx]   # shuffle X and y together
np.where(p > 0.5, 1, 0)                   # vectorized if/else
np.argmax(scores, axis=1)                 # index of max per row (class prediction)
```

Slices are **views** (modifying them modifies the original). Use `.copy()` when you need an independent array, e.g. `w = w0.copy()` before an optimization loop.

## 4. Axis arguments

`axis=0` collapses rows (one result per column); `axis=1` collapses columns (one result per row).

```python
X.mean(axis=0)            # (d,) feature means
X.sum(axis=1)             # (n,) row sums
X.max(axis=1, keepdims=True)   # (n, 1), keeps the dimension so it broadcasts back against X
```

## 5. Broadcasting

Shapes are compared right-to-left; dimensions must be equal or 1.

```python
X_centered = X - X.mean(axis=0)            # (n,d) - (d,)  -> (n,d)
X_std = X_centered / X.std(axis=0)          # standardize each feature
P = E / E.sum(axis=1, keepdims=True)        # (n,k) / (n,1) -> normalize each row (softmax)
D2 = ((A[:, None, :] - B[None, :, :])**2).sum(-1)   # (n,1,d)-(1,m,d) -> (n,m) pairwise sq. distances
```

## 6. Linear algebra

```python
A @ B                     # matrix product (prefer over np.dot)
A.T                       # transpose
np.outer(u, v)            # (n,) x (m,) -> (n, m)
u @ v                     # dot product of two 1-D vectors -> scalar
np.linalg.norm(v)         # Euclidean norm; norm(M, 'fro') for Frobenius
np.linalg.solve(A, b)     # solve Ax = b. Prefer this to inv(A) @ b (faster, more accurate)
np.linalg.inv(A)          # only when you truly need the inverse (e.g. a covariance of estimates)
np.linalg.lstsq(X, y, rcond=None)     # least squares; returns (w, residuals, rank, singular values)
np.linalg.pinv(X)         # Moore-Penrose pseudoinverse (via SVD)
U, s, Vt = np.linalg.svd(X, full_matrices=False)   # "thin" SVD; s is 1-D, sorted descending; Vt is V transposed
evals, evecs = np.linalg.eigh(S)     # symmetric matrices: real, ASCENDING order, columns are eigenvectors
evals, evecs = np.linalg.eig(A)      # general matrices (may be complex)
np.linalg.det(A); np.linalg.slogdet(A)   # slogdet -> (sign, log|det|), use for log-likelihoods
np.linalg.matrix_rank(X); np.linalg.cond(X)
np.trace(A)
np.einsum('ij,ij->i', A, B)          # row-wise dot products; einsum is worth learning for week 4/7
```

## 7. Numerical safety

```python
np.allclose(a, b, rtol=1e-5, atol=1e-8)   # compare floats; never use == on floats
np.isfinite(x).all()                      # catch nan/inf early
np.log1p(x); np.expm1(x)                  # accurate log(1+x), exp(x)-1 for small x
np.logaddexp(0, z)                        # log(1 + e^z) without overflow
from scipy.special import logsumexp, expit, softmax   # stable log-sum-exp, sigmoid, softmax
np.clip(p, 1e-12, 1 - 1e-12)              # keep probabilities away from 0/1 before log
np.seterr(all='raise')                    # turn silent overflow/nan warnings into errors while debugging
```

## 8. Python reminders

```python
# f-strings with formatting
print(f"iter {t:4d}  loss {loss:.6f}  |grad| {g_norm:.2e}")

# enumerate / zip
for i, (xi, yi) in enumerate(zip(X, y)):
    ...

# list comprehension -> array
losses = []
for t in range(T):
    losses.append(loss)
losses = np.array(losses)

# dicts for results tables
results = {"normal_eq": w1, "svd": w2, "gd": w3}
for name, w in results.items():
    print(f"{name:10s} {np.round(w, 4)}")

# small classes (week 5 autograd, week 7/8 models)
class Model:
    def __init__(self, k):
        self.k = k
    def fit(self, X):
        ...
        return self          # returning self allows Model(3).fit(X).predict(X)
    def __repr__(self):
        return f"Model(k={self.k})"

# type hints + docstrings (used in all the outlines)
def f(X: np.ndarray, w: np.ndarray) -> float:
    """One-line summary. Shapes: X (n, d), w (d,)."""
    ...

# timing
import time
t0 = time.perf_counter(); ...; print(time.perf_counter() - t0)
```

## 9. Plotting skeleton (matplotlib)

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(losses, label="GD")
axes[0].set_yscale("log"); axes[0].set_xlabel("iteration"); axes[0].legend()
axes[1].scatter(X[:, 0], X[:, 1], c=y, s=10, cmap="coolwarm")
fig.tight_layout(); plt.show()

# contour / decision boundary grid
xx, yy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])     # (40000, 2)
zz = model_predict(grid).reshape(xx.shape)
plt.contourf(xx, yy, zz, alpha=0.3); plt.contour(xx, yy, zz, levels=[0.5])

# images
plt.imshow(img.reshape(64, 64), cmap="gray"); plt.axis("off")
```

## 10. Testing your own code

```python
assert X.shape == (n, d), X.shape
np.testing.assert_allclose(mine, reference, rtol=1e-6)   # raises with a readable diff
```

Always test on a **tiny synthetic problem with a known answer** first (e.g. `y = 3*x + 2 + small noise` should give back `[2, 3]`).
