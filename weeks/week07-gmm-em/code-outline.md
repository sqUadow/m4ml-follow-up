# Week 7 code outline: GMM with EM, k-means, BIC, simulations

Skeletons only. Syntax reminders: [numpy-refresher.md](../../docs/numpy-refresher.md) §2 (random), §5 (broadcasting), §7 (logsumexp).

## Packages

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.special import logsumexp
from scipy import stats                      # chi2, t, expon, norm for the simulation section

# Data + answer keys
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Reuse from week 4 (copy into mlutils/ if you haven't yet)
import sys; sys.path.insert(0, "../..")
# from mlutils.gaussians import mvn_logpdf, plot_cov_ellipse
```

## Suggested layout

```
weeks/week07-gmm-em/
  gmm.py            # sample_gmm, GMM class (EM), bic
  kmeans.py         # kmeans_pp_init, kmeans
  simulations.py    # optional CLT / CI coverage experiments
  week07.ipynb
mlutils/gaussians.py  # mvn_logpdf + plot_cov_ellipse moved here from week 4
```

---

## Step 1: Sampling from a GMM (simulation, 10.2.6)

```python
def sample_gmm(n: int, weights, means, covs, rng) -> tuple[np.ndarray, np.ndarray]:
    """Return (X (n, d), z (n,) true component labels).
    1. z = rng.choice(K, size=n, p=weights)
    2. for each k: X[z == k] = rng.multivariate_normal(means[k], covs[k], size=(z == k).sum())
    """
    ...

true_w = np.array([0.5, 0.3, 0.2])
true_mu = np.array([[0, 0], [4, 4], [-3, 5]], dtype=float)
true_cov = np.array([[[1, 0.6], [0.6, 1]], [[1.5, -0.5], [-0.5, 0.5]], [[0.3, 0], [0, 2]]])
```

## Step 2: The GMM class

```python
class GMM:
    def __init__(self, K: int, max_iter: int = 200, tol: float = 1e-6,
                 reg_covar: float = 1e-6, seed: int = 0):
        ...

    def _init_params(self, X):
        """weights_ = uniform (K,); means_ = K random distinct rows of X (rng.choice(n, K, replace=False));
        covs_ = K copies of the overall data covariance (np.tile(cov, (K, 1, 1)))."""
        ...

    def _log_joint(self, X) -> np.ndarray:
        """(n, K) with entry [i, k] = log π_k + log N(x_i; μ_k, Σ_k).
        Build it column by column with mvn_logpdf, then np.column_stack."""
        ...

    def _e_step(self, X) -> tuple[np.ndarray, float]:
        """Return (log_resp (n, K), total log-likelihood).
        log_px = logsumexp(log_joint, axis=1)                  (n,)
        log_resp = log_joint - log_px[:, None]
        loglik = log_px.sum()
        """
        ...

    def _m_step(self, X, resp):
        """resp = exp(log_resp), shape (n, K).
        Nk = resp.sum(axis=0)                                   (K,)
        weights_ = Nk / n
        means_   = (resp.T @ X) / Nk[:, None]                   (K, d)
        covs_[k] = weighted covariance around means_[k] + reg_covar * I
        """
        ...

    def fit(self, X):
        """Loop: E-step, record loglik, check monotonicity, M-step; stop when the improvement < tol.
        Store self.loglik_history_ (a list) and self.converged_."""
        ...
        return self

    def predict_proba(self, X): ...          # responsibilities
    def predict(self, X): ...                # argmax responsibility
    def score(self, X) -> float: ...         # mean log-likelihood per sample, same as sklearn's .score
```

<details>
<summary>Check your M-step covariance (peek after deriving)</summary>

`Σₖ = (1/Nₖ) Σᵢ γᵢₖ (xᵢ − μₖ)(xᵢ − μₖ)ᵀ`.
Vectorized: `D = X - means_[k]` (n, d), then `(resp[:, k, None] * D).T @ D / Nk[k]`.

</details>

**Monotonicity assertion** (put it in `fit`):

```python
if len(hist) > 1:
    assert hist[-1] >= hist[-2] - 1e-9, f"log-likelihood decreased at iter {it}: {hist[-2]} -> {hist[-1]}"
```

**Matching components to the truth** (labels can come out permuted): for K=3, try all `itertools.permutations(range(3))` and pick the one minimizing `‖means_[perm] − true_mu‖`. Or just sort components by their first mean coordinate.

## Step 3: Answer key

```python
sk = GaussianMixture(n_components=2, covariance_type="full", reg_covar=1e-6,
                     random_state=0, tol=1e-6, max_iter=500).fit(X)
sk.weights_, sk.means_, sk.covariances_, sk.score(X), sk.bic(X), sk.converged_, sk.n_iter_
```

sklearn initializes with k-means by default (`init_params="kmeans"`), so its path differs from yours. The **final** parameters should match if both find the same optimum. Try `n_init=10` if they don't.

## Step 4: k-means

```python
def kmeans_pp_init(X, K, rng) -> np.ndarray:
    """First center uniform at random; each next center sampled with prob ∝ D(x)², the squared distance
    to the nearest chosen center. rng.choice(n, p=d2 / d2.sum())."""
    ...

def kmeans(X, K, max_iter=100, seed=0) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (centers (K, d), labels (n,), inertia).
    Assign: pairwise squared distances via broadcasting (n, K) -> argmin.
    Update: centers[k] = X[labels == k].mean(axis=0) (handle empty clusters!).
    Stop when labels stop changing."""
    ...
```

Answer key: `KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)` → `.cluster_centers_`, `.labels_`, `.inertia_`.

An anisotropic dataset where k-means struggles: `X_aniso = X_blobs @ np.array([[0.6, -0.6], [-0.4, 0.8]])`.

## Step 5: BIC

```python
def n_params_full_gmm(K: int, d: int) -> int:
    """(K − 1) weights + K·d means + K·d(d+1)/2 covariance entries."""
    ...

def bic(loglik_total: float, n: int, p: int) -> float:
    return -2 * loglik_total + p * np.log(n)
```

Check your value against `GaussianMixture(...).bic(X)` (it uses the same formula; small differences come from slightly different optima).

## Step 6: Plots

Soft cluster colors for K=2: `plt.scatter(X[:,0], X[:,1], c=resp[:, 1], cmap="coolwarm")`.

Ellipse sized by χ² (95% region):

```python
r = np.sqrt(stats.chi2.ppf(0.95, df=2))      # ≈ 2.45. Use this instead of "2σ" in 2-D
plot_cov_ellipse(ax, mu_k, Sigma_k, n_std=r)  # from week 4
```

Iteration snapshots: run EM manually for `[0, 1, 5, 20]` iterations (or store params each iteration in `fit`), and plot in `plt.subplots(1, 4)`.

## Step 7 (optional): simulations

```python
rng = np.random.default_rng(0)
true_mean = 2.0                                     # exponential with scale 2 has mean 2

def ci_coverage(sampler, true_mean, n, n_reps=10_000, alpha=0.05, rng=rng) -> float:
    """Draw n_reps samples of size n at once: sampler(size=(n_reps, n)).
    Means/stds along axis=1 (ddof=1!), t critical value from stats.t.ppf(1 - alpha/2, df=n-1).
    Return the fraction of intervals containing true_mean."""
    ...

ci_coverage(lambda size: rng.exponential(scale=2.0, size=size), 2.0, n=5)
```

CLT: `means = rng.exponential(2.0, size=(10_000, n)).mean(axis=1)`, then `plt.hist(means, bins=60, density=True)` and overlay `stats.norm(loc=2, scale=2/np.sqrt(n)).pdf(xs)`.

---

## Syntax reminders for this week

- `rng.choice(n, size=K, replace=False)` gives K distinct indices.
- `np.tile(A, (K, 1, 1))` stacks K copies of a matrix into `(K, d, d)`.
- `resp[:, k, None]` is the same as `resp[:, k][:, None]` → `(n, 1)` for broadcasting.
- `itertools.permutations(range(K))` for label matching.
- `assert cond, message` for invariants like monotone log-likelihood.
- Vectorized simulation: generate a 2-D array `(n_reps, n)` and reduce along `axis=1` instead of looping 10,000 times.

---

## PyTorch corner

EM doesn't need gradients (each M-step is closed form), so PyTorch is just an answer key this week. But the same model *can* be fit by gradient ascent, which is a useful contrast.

```python
import torch
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily

gmm_t = MixtureSameFamily(
    mixture_distribution=Categorical(probs=torch.from_numpy(weights)),
    component_distribution=MultivariateNormal(loc=torch.from_numpy(means),
                                              covariance_matrix=torch.from_numpy(covs)),  # batched (K, d, d)
)
gmm_t.log_prob(torch.from_numpy(X)).sum()     # compare to your final total log-likelihood
gmm_t.sample((500,))                          # an answer key for your sample_gmm
```

**Gradient-ascent GMM (optional experiment):** parameterize `logits` (for π, via softmax), `means`, and a lower-triangular `L` with positive diagonal (for `Σ = LLᵀ`, passed as `scale_tril=`), then maximize `log_prob(X).sum()` with Adam.

**Distinctions to watch:**
- Constraints: EM keeps `π` on the simplex and `Σ` positive definite *automatically*. Gradient methods need reparameterizations (softmax, Cholesky factor with `exp` on the diagonal), which is a big reason EM is preferred for GMMs.
- `MultivariateNormal` accepts `covariance_matrix`, `precision_matrix`, or `scale_tril` (Cholesky factor). `scale_tril` is the cheapest and most stable.
- Batched distributions: a `(K, d)` loc with a `(K, d, d)` covariance gives K independent components in one object (`batch_shape = (K,)`), the same broadcasting idea you used in your E-step.
