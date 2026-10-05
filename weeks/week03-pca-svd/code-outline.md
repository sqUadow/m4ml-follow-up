# Week 3 code outline: PCA, SVD, eigenfaces, compression

Skeletons only. Syntax reminders: [numpy-refresher.md](../../docs/numpy-refresher.md) §6 (linear algebra) and §9 (images).

## Packages

```python
import numpy as np
import matplotlib.pyplot as plt

# Data + answer keys
from sklearn.datasets import fetch_olivetti_faces, load_digits, load_sample_image
from sklearn.decomposition import PCA
```

## Suggested layout

```
weeks/week03-pca-svd/
  pca.py            # center, pca_eig, pca_svd, project, reconstruct, low_rank
  week03.ipynb
```

---

## Step 1: Covariance and PCA via eigendecomposition

```python
def center(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Xc, mean). mean has shape (d,)."""
    ...

def sample_cov(Xc: np.ndarray) -> np.ndarray:
    """(d, d) sample covariance of CENTERED data, dividing by n-1.
    Answer key: np.cov(X, rowvar=False). np.cov treats ROWS as variables by default!"""
    ...

def pca_eig(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (components (k, d), eigenvalues (k,), mean (d,)), sorted by eigenvalue DESCENDING.

    np.linalg.eigh(S) -> (evals ascending, evecs as COLUMNS).
    Sorting: order = np.argsort(evals)[::-1]; evals[order]; evecs[:, order]
    Return components as ROWS (evecs[:, order[:k]].T) to match sklearn's components_.
    """
    ...
```

- `[::-1]` reverses an array.
- `eigh` (for symmetric matrices) not `eig`: guaranteed real output and orthonormal eigenvectors.

## Step 2: PCA via SVD

```python
def pca_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same return signature as pca_eig.
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)   # s already descending
    components = rows of Vt;  eigenvalues = s**2 / (n - 1)
    """
    ...
```

**Comparing up to sign:** each component may be flipped. Two options:

```python
# 1) compare absolute values of the inner products: should be ~1 for each pair
np.abs(np.sum(C_eig * C_svd, axis=1))         # (k,), all ≈ 1.0

# 2) or fix a sign convention, e.g. make the largest-|entry| of each component positive
def fix_signs(C):
    idx = np.argmax(np.abs(C), axis=1)
    signs = np.sign(C[np.arange(len(C)), idx])    # "fancy indexing": one element per row
    return C * signs[:, None]
```

Answer key:

```python
sk = PCA(n_components=k).fit(X)
sk.components_, sk.explained_variance_, sk.explained_variance_ratio_, sk.mean_
```

## Step 3: Project and reconstruct

```python
def project(X, components, mean) -> np.ndarray:
    """(n, d) -> (n, k) scores: (X - mean) @ components.T"""
    ...

def reconstruct(Z, components, mean) -> np.ndarray:
    """(n, k) -> (n, d): Z @ components + mean"""
    ...

def reconstruction_mse(X, X_hat) -> float: ...
```

Check: the **total** squared error `np.sum((X - X_hat)**2)` should equal `(n − 1) · sum(all_eigenvalues[k:])`, which is the same as `sum(s[k:]**2)` from the SVD. Work out why on paper. If you're off by a factor of `n` or `n − 1`, that's a convention slip (mean vs. sum, `n` vs. `n−1`), not a bug in PCA.

## Step 4: Eigenfaces

```python
faces = fetch_olivetti_faces()
X = faces.data                                 # (400, 4096)

def show_grid(rows: np.ndarray, shape=(64, 64), ncols=8, titles=None):
    """Plot each row of `rows` as an image in a grid."""
    n = len(rows); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.8 * nrows))
    for ax, r in zip(axes.ravel(), rows):
        ax.imshow(r.reshape(shape), cmap="gray"); ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")                         # hide unused panels
    fig.tight_layout()
```

(This is plotting boilerplate, not the learning goal, so it's given in full.)

- Eigenfaces have positive and negative values. `imshow` auto-scales, which is fine for viewing.
- `pca_eig` on a 4096×4096 covariance is slow (seconds to a minute). `pca_svd` on 400×4096 is fast. Time both with `time.perf_counter()`.

## Step 5: Low-rank image compression

```python
img = load_sample_image("china.jpg")           # (427, 640, 3) uint8, needs Pillow installed
A = img.mean(axis=2) / 255.0                   # grayscale float

def low_rank(A: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation U[:, :k] @ diag(s[:k]) @ Vt[:k].
    Faster without diag: (U[:, :k] * s[:k]) @ Vt[:k]   (broadcast s across columns)"""
    ...
```

Eckart–Young check:

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
for k in [5, 20, 50, 100]:
    err2 = np.linalg.norm(A - low_rank(A, k), "fro") ** 2
    print(k, err2, np.sum(s[k:] ** 2))         # should match to ~1e-10 relative
```

Display: `plt.imshow(np.clip(A_k, 0, 1), cmap="gray")`. Low-rank approximations can step slightly outside [0, 1].

## Stretch: n ≪ d trick, power iteration

```python
def pca_small_gram(X, k):
    """Eigendecompose G = Xc @ Xc.T (n x n). If G u = λ u then Xc.T @ u is an eigenvector of Xc.T @ Xc
    with the same λ. Normalize it to unit length."""
    ...

def power_iteration(S, n_iters=1000, tol=1e-10, seed=0):
    """Return (eigval, eigvec) for the largest eigenvalue of symmetric S.
    Rayleigh quotient: lam = v @ S @ v."""
    ...
```

1-NN classifier with broadcasting (refresher §5):

```python
def nn1_predict(Z_train, y_train, Z_test):
    D2 = ((Z_test[:, None, :] - Z_train[None, :, :]) ** 2).sum(axis=-1)   # (n_test, n_train)
    ...
```

For splitting faces so each person appears in train and test: `train_test_split(..., stratify=faces.target)`.

---

## Syntax reminders for this week

- `np.linalg.svd` returns `Vt` (V transposed), so components are **rows** of `Vt`.
- `np.linalg.eigh` returns eigenvectors as **columns**, ascending.
- `np.cumsum(ratios)` for cumulative explained variance; `np.searchsorted(cum, 0.95) + 1` finds the number of components for 95%.
- Images: `.reshape(64, 64)` to display a row, `.ravel()` / `.reshape(-1)` to flatten back.
- `ax.arrow(x, y, dx, dy, width=...)` or `ax.quiver` to draw principal axes in the 2-D warm-up; use `ax.set_aspect("equal")` so perpendicular axes look perpendicular.

---

## PyTorch corner

No autograd this week, but the linear algebra API is almost identical:

```python
import torch
Xt = torch.from_numpy(X)
U, S, Vh = torch.linalg.svd(Xt - Xt.mean(dim=0), full_matrices=False)   # Vh ≙ NumPy's Vt
evals, evecs = torch.linalg.eigh(torch.cov(Xt.T))                         # torch.cov wants VARIABLES AS ROWS -> pass X.T
U, S, V = torch.pca_lowrank(Xt, q=50)                                      # randomized PCA: returns V, NOT Vh!
```

**Distinctions to watch:**
- `torch.linalg.svd` returns `Vh` (like NumPy's `Vt`). The older `torch.svd` (deprecated) returned `V`, and `torch.pca_lowrank` also returns `V`. Check the docs whenever you switch functions.
- `torch.cov` has no `rowvar` argument; it always treats rows as variables.
- Randomized methods (`torch.pca_lowrank`, `sklearn PCA(svd_solver="randomized")`) are approximate: compare with a tolerance, and set seeds.
- Sign ambiguity is the same in every library, and can differ *between* libraries on the same data.
