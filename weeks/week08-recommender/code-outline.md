# Week 8 code outline: matrix-factorization recommender

Skeletons only. Syntax reminders: [numpy-refresher.md](../../docs/numpy-refresher.md) §3 (indexing), §6 (linear algebra).

## Packages

```python
import zipfile
import urllib.request
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, "../..")
# from mlutils... reuse your week 3 PCA for the item-factor plot
```

No library answer key does exactly this model, so your checks are **baselines**, **monotone training loss**, and **gradient checks**.

## Suggested layout

```
weeks/week08-recommender/
  data.py           # download + load MovieLens, id mapping, validation split
  baselines.py      # global mean, biases, SVD baseline
  mf.py             # MF-SGD and MF-ALS
  week08.ipynb
```

---

## Step 1: Data

Download boilerplate (given in full):

```python
DATA = Path("../../data"); DATA.mkdir(exist_ok=True)
zip_path = DATA / "ml-100k.zip"
if not zip_path.exists():
    urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-100k.zip", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA)                      # creates data/ml-100k/

cols = ["user", "item", "rating", "ts"]
train_df = pd.read_csv(DATA / "ml-100k" / "u1.base", sep="\t", names=cols)
test_df  = pd.read_csv(DATA / "ml-100k" / "u1.test", sep="\t", names=cols)
items = pd.read_csv(DATA / "ml-100k" / "u.item", sep="|", header=None, encoding="latin-1",
                    usecols=[0, 1], names=["item", "title"])
```

Convert to NumPy arrays of 0-based indices. Represent ratings as **triplets**, not a dense matrix:

```python
n_users, n_items = 943, 1682
u_tr = train_df["user"].to_numpy() - 1          # (N,) int
i_tr = train_df["item"].to_numpy() - 1
r_tr = train_df["rating"].to_numpy().astype(float)
```

```python
def train_val_split(u, i, r, frac_val=0.1, seed=0):
    """Random 90/10 split of the triplets via rng.permutation."""
    ...

def rmse(r_true, r_pred) -> float: ...
```

Exploration: `np.bincount(u_tr, minlength=n_users)` gives ratings per user. Sparsity = `len(r_tr) / (n_users * n_items)`.

---

## Step 2: Baselines

```python
def fit_biases(u, i, r, n_users, n_items, lam=10.0, n_iters=10):
    """r̂ = μ + b_u + b_i. Alternate closed-form updates:
       b_u = Σ_{i ∈ I(u)} (r − μ − b_i) / (λ + |I(u)|)     for all users at once
       b_i = Σ_{u ∈ U(i)} (r − μ − b_u) / (λ + |U(i)|)
    Vectorized group sums: np.bincount(u, weights=residual, minlength=n_users).
    Return (mu, b_user (n_users,), b_item (n_items,))."""
    ...
```

`np.bincount(idx, weights=w, minlength=m)` is the key trick of this week: it sums `w` into `m` bins by index in a single call, with no Python loop over users.

```python
def svd_baseline(u, i, r, n_users, n_items, k):
    """1. Dense R (n_users, n_items) filled with each user's mean rating (or μ + b_u + b_i).
       2. Subtract that fill matrix F, so observed entries hold residuals and missing ones hold 0.
       3. U, s, Vt = np.linalg.svd(R - F, full_matrices=False); keep rank k.
       4. Prediction matrix = F + U_k diag(s_k) Vt_k.
    Return the (n_users, n_items) prediction matrix. 943×1682 dense is small enough."""
    ...
```

Predict for triplets with fancy indexing: `pred_matrix[u_val, i_val]`.

---

## Step 3: MF with SGD

```python
class MFSGD:
    def __init__(self, n_users, n_items, k=20, lr=0.01, lam=0.05, n_epochs=30, seed=0):
        """P (n_users, k), Q (n_items, k) ~ N(0, 0.1²); biases zero; mu = global mean (set in fit)."""
        ...

    def predict(self, u, i) -> np.ndarray:
        """Vectorized: mu + bu[u] + bi[i] + (P[u] * Q[i]).sum(axis=1)"""
        ...

    def fit(self, u, i, r, u_val=None, i_val=None, r_val=None):
        """Each epoch: shuffle the order (rng.permutation), then loop over ratings ONE AT A TIME:
             e = r - predict(single)
             update bu, bi, P[u], Q[i] with your derived rules
        Careful: compute the update for P[u] using the OLD Q[i] (copy P[u] first).
        Record train/val RMSE per epoch in self.history_."""
        ...
        return self
```

Performance notes:
- A pure-Python loop over 72k ratings × 30 epochs takes roughly tens of seconds to a few minutes. That's fine. Pull arrays into local variables (`P = self.P`) inside the loop; attribute lookups are slow.
- Optional speedup: `pip install numba` and decorate a plain function (not a method) with `@numba.njit`.

<details>
<summary>Check your SGD updates (peek after deriving)</summary>

With `e = r − r̂` and learning rate `η` (the factor 2 absorbed into `η`):

```
b_u ← b_u + η (e − λ b_u)
b_i ← b_i + η (e − λ b_i)
p_u ← p_u + η (e q_i − λ p_u)
q_i ← q_i + η (e p_u − λ q_i)      (using the OLD p_u)
```

</details>

---

## Step 4: MF with ALS

```python
def build_index(u, i, r, n_users, n_items):
    """Lists of arrays: for each user, the item indices and ratings they have; and vice versa.
    Sort-based grouping: order = np.argsort(u); split points from np.cumsum(np.bincount(u, minlength=n_users)).
    np.split(i[order], splits[:-1]) gives a list of per-user item arrays."""
    ...

class MFALS:
    def __init__(self, n_users, n_items, k=20, lam=0.1, n_iters=15, seed=0): ...

    def _solve_side(self, fixed: np.ndarray, groups_idx, groups_r, lam) -> np.ndarray:
        """For each row (user or item): gather the fixed factors of what it rated, F (m, k), and its
        ratings y (m,), then solve (FᵀF + λ·m·I) x = Fᵀ y with np.linalg.solve.
        (Scaling λ by m, the count, is the common 'weighted-λ' ALS variant; try both.)
        Rows with m = 0 keep zeros."""
        ...

    def fit(self, u, i, r, ...):
        """Center ratings by μ (or by the bias baseline) first, then alternate:
             P = _solve_side(Q, items_of_user, ratings_of_user)
             Q = _solve_side(P, users_of_item, ratings_of_item)
        Record train/val RMSE per iteration. The regularized objective J should never increase
        (each half-step minimizes J exactly over one block), so assert that with the unweighted λ.
        Training RMSE alone usually decreases too, but it isn't guaranteed because it ignores the penalty."""
        ...
```

Each `_solve_side` call is `n_users` (or `n_items`) small `k×k` ridge regressions, which is week 1's `fit_ridge` in a loop.

---

## Step 5: Gradient check (a cheap insurance policy)

Write the full objective `J(P, Q, bu, bi)` over all training triplets and its full-batch gradient. Check the gradient with `numerical_gradient` (week 1) on a **tiny** random problem (5 users, 4 items, k = 2). If the full-batch gradient is right, your per-rating SGD updates (which are the same terms, one rating at a time) almost certainly are too.

---

## Step 6: Interpretation

```python
def most_similar(Q, item_idx, titles, top=10):
    """Cosine similarity: normalize rows of Q, then sims = Qn @ Qn[item_idx]. argsort descending, skip itself."""
    ...

title_to_idx = {t: k for k, t in enumerate(items["title"])}     # u.item rows are in item-id order
most_similar(model.Q, title_to_idx["Star Wars (1977)"], items["title"].to_numpy())
```

Movies with very few ratings have noisy factors. Filter to items with ≥ 50 ratings for nicer lists.

2-D map: `Z = project(Q[popular], *pca_svd(Q[popular], 2))` with your week 3 functions, then `ax.annotate(title, Z[j])` for a handful of famous titles.

---

## Syntax reminders for this week

- `pd.read_csv(path, sep="\t", names=[...])` for header-less files; `encoding="latin-1"` for `u.item`.
- `df["col"].to_numpy()` to leave pandas for NumPy.
- `np.bincount(idx, weights=w, minlength=m)` for vectorized group sums.
- `np.argsort` + `np.split` for grouping indices.
- `P[u] * Q[i]` with index arrays `u, i` of length N gives `(N, k)` rowwise products; `.sum(axis=1)` gives the N dot products.
- `pred.clip(1, 5)` before computing RMSE (ratings live in [1, 5]). Usually a small free gain.

---

## PyTorch corner

`nn.Embedding(n, k)` is just a learnable `(n, k)` matrix indexed by integer ids, exactly your `P` and `Q`:

```python
import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, n_users, n_items, k, mu):
        super().__init__()
        self.P, self.Q = nn.Embedding(n_users, k), nn.Embedding(n_items, k)
        self.bu, self.bi = nn.Embedding(n_users, 1), nn.Embedding(n_items, 1)
        for e in (self.P, self.Q):
            nn.init.normal_(e.weight, std=0.1)
        for e in (self.bu, self.bi):
            nn.init.zeros_(e.weight)
        self.mu = mu
    def forward(self, u, i):                       # u, i: LongTensors of shape (batch,)
        dot = (self.P(u) * self.Q(i)).sum(dim=1)
        return self.mu + self.bu(u).squeeze(1) + self.bi(i).squeeze(1) + dot

model = MF(n_users, n_items, k=20, mu=float(r_tr.mean()))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
# loop over shuffled mini-batches (DataLoader over TensorDataset(u, i, r)):
#   pred = model(ub, ib); loss = ((pred - rb) ** 2).mean() + lam * (model.P(ub).pow(2).sum(1) + model.Q(ib).pow(2).sum(1)).mean()
#   opt.zero_grad(); loss.backward(); opt.step()
```

**Distinctions to watch:**
- **Indices must be `torch.long`** (`torch.from_numpy(u_tr)` is int64 if `u_tr` is int64; check `.dtype`).
- **Regularization:** `weight_decay` in the optimizer shrinks *every* embedding row at *every* step, even for users not in the batch. Your NumPy SGD only regularizes the rows touched by each rating. Penalizing only the batch's rows (as in the loop above) matches your objective more closely.
- **Mini-batches vs. one rating at a time:** your NumPy SGD is batch size 1. With batches, the learning rate needs retuning. Adam usually makes this much less painful.
- **Sparse gradients:** `nn.Embedding(..., sparse=True)` + `torch.optim.SparseAdam` only update touched rows, which matters for big catalogs but not here.
- **float32:** `r_tr` must be `.float()` to match the model's parameters.
