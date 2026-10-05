# Week 4 code outline: Naive Bayes and Gaussian discriminant analysis

Skeletons only. Syntax reminders: [numpy-refresher.md](../../docs/numpy-refresher.md) §3 (masks), §5 (broadcasting), §7 (log-space math).

## Packages

```python
import re                                  # regular expressions for tokenizing
import zipfile
import urllib.request
from collections import Counter            # word counting
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse     # covariance ellipses
from scipy.special import logsumexp        # normalize log-probabilities
from scipy.stats import multivariate_normal  # answer key for your log-density

# Data + answer keys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
```

## Suggested layout

```
weeks/week04-probabilistic-classifiers/
  naive_bayes.py     # tokenize, build_vocab, vectorize, MultinomialNaiveBayes
  gda.py             # mvn_logpdf, GDA(shared_cov=True/False)
  week04.ipynb
```

---

## Step 1: Get the SMS data

Download boilerplate (given in full; not the learning goal):

```python
DATA = Path("../../data"); DATA.mkdir(exist_ok=True)
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = DATA / "sms_spam.zip"
if not zip_path.exists():
    urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(DATA / "sms_spam")

labels, texts = [], []
with open(DATA / "sms_spam" / "SMSSpamCollection", encoding="utf-8") as f:
    for line in f:
        label, text = line.rstrip("\n").split("\t", 1)     # split on the FIRST tab only
        labels.append(label); texts.append(text)
y = np.array([1 if l == "spam" else 0 for l in labels])
```

If the URL ever moves, download the zip manually from the [dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) into `data/`.

Split with `train_test_split(texts, y, test_size=0.2, random_state=0, stratify=y)` (it works on Python lists too).

## Step 2: Tokenize, vocabulary, count matrix

```python
def tokenize(text: str) -> list[str]:
    """Lowercase, then extract word-like tokens.
    hint: re.findall(r"[a-z0-9']+", text.lower())"""
    ...

def build_vocab(texts: list[str], min_count: int = 1) -> dict[str, int]:
    """Map word -> column index, from TRAINING texts only.
    hint: Counter().update(tokens) in a loop; then {w: i for i, w in enumerate(sorted(words))}"""
    ...

def vectorize(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    """(n_docs, V) counts. Ignore words not in vocab (unseen test words).
    A dense array is fine here (~4.5k docs × ~8k words). Real text pipelines use scipy.sparse."""
    ...
```

Python reminders: `dict.get(key)` returns `None` if missing; `if w in vocab:` is O(1); `Counter(tokens).most_common(10)`.

## Step 3: Multinomial Naive Bayes

```python
class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Set:
          self.classes_          (K,)
          self.log_prior_        (K,)   log π_c = log(count_c / n)
          self.log_theta_        (K, V) log of smoothed word probabilities per class
        Use a boolean mask X[y == c] to get each class's rows.
        """
        ...
        return self

    def joint_log_likelihood(self, X) -> np.ndarray:
        """(n, K): log π_c + Σ_j x_j log θ_{c,j}. One matrix multiply plus broadcasting."""
        ...

    def predict_proba(self, X) -> np.ndarray:
        """Normalize in log space: jll - logsumexp(jll, axis=1, keepdims=True), then exp."""
        ...

    def predict(self, X) -> np.ndarray:
        ...
```

Answer key (use the **same** vocabulary so columns line up):

```python
cv = CountVectorizer(vocabulary=vocab, token_pattern=r"[a-z0-9']+", lowercase=True)
X_train_sk = cv.transform(train_texts)                  # sparse matrix; .toarray() to compare with yours
sk = MultinomialNB(alpha=1.0).fit(X_train_sk, y_train)
np.allclose(sk.feature_log_prob_, my_nb.log_theta_)    # should be True
```

Spammiest words: `np.argsort(log_theta[1] - log_theta[0])[::-1][:15]`, then map indices back to words with an inverted vocab `{i: w for w, i in vocab.items()}`.

## Step 4: Multivariate normal log-density

```python
def mvn_logpdf(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """log N(x; μ, Σ) for every row of X. Shapes: X (n, d), mu (d,), Sigma (d, d) -> (n,).

    Pieces:
      diff = X - mu                                    (n, d)
      Mahalanobis term: diff_i ᵀ Σ⁻¹ diff_i for each row.
        Avoid inv: sol = np.linalg.solve(Sigma, diff.T).T   (n, d)
        then a row-wise dot product: np.einsum("ij,ij->i", diff, sol)   or (diff * sol).sum(axis=1)
      log-determinant: sign, logdet = np.linalg.slogdet(Sigma)   (det() under/overflows in high d)
    """
    ...
```

Check: `np.allclose(mvn_logpdf(X, mu, S), multivariate_normal(mean=mu, cov=S).logpdf(X))`.

**einsum crash course:** `"ij,ij->i"` means "for each `i`, sum over `j` of `A[i,j]*B[i,j]`". Letters that disappear from the output are summed over. `"ij,jk->ik"` is matrix multiplication.

## Step 5: GDA (LDA and QDA in one class)

```python
class GDA:
    def __init__(self, shared_cov: bool = False, reg: float = 1e-9):
        """shared_cov=True -> LDA, False -> QDA. reg is added to the covariance diagonal for stability."""
        ...

    def fit(self, X, y):
        """Per class c: prior π_c, mean μ_c, covariance Σ_c (MLE divides by n_c. Older sklearn versions' QDA divided by n_c − 1, so check `qda.covariance_` against both).
        LDA: pooled Σ = Σ_c (n_c · Σ_c) / n  (the within-class scatter, MLE version)."""
        ...
        return self

    def joint_log_likelihood(self, X) -> np.ndarray:
        """(n, K): log π_c + mvn_logpdf(X, μ_c, Σ_c), stacking one column per class with np.column_stack."""
        ...

    def predict_proba(self, X): ...
    def predict(self, X): ...
```

Answer keys:

```python
lda = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)    # compare lda.means_, lda.covariance_, predict_proba
qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)   # qda.covariance_ is a list of (d,d)
```

Small numeric differences in probabilities can come from `n` vs. `n−1` conventions. Labels should agree almost everywhere; check where they disagree (probably points right on the boundary).

## Step 6: Plots

Decision regions: use the meshgrid skeleton from the refresher §9 with `model.predict(grid)`.

Covariance ellipse (eigen-decomposition again!):

```python
def plot_cov_ellipse(ax, mu, Sigma, n_std=2.0, **kwargs):
    """Axes of the ellipse = eigenvectors of Σ; half-lengths = n_std * sqrt(eigenvalues)."""
    evals, evecs = np.linalg.eigh(Sigma)
    angle = np.degrees(np.arctan2(evecs[1, -1], evecs[0, -1]))     # direction of the largest eigenvector
    width, height = 2 * n_std * np.sqrt(evals[::-1])                # full lengths, largest first
    ax.add_patch(Ellipse(xy=mu, width=width, height=height, angle=angle, fill=False, **kwargs))
```

(Given because the matplotlib `Ellipse` API is fiddly. Make sure you understand why eigenvectors give the axes.)

Side-by-side: `fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)`.

---

## Syntax reminders for this week

- `line.split("\t", 1)` splits at most once, so tabs inside the message stay.
- `np.unique(y, return_counts=True)` for class balance.
- `np.column_stack([f(c) for c in classes])` builds an `(n, K)` matrix from K length-`n` vectors.
- `logsumexp(a, axis=1, keepdims=True)` is the stable `log Σ exp`. Never `np.log(np.exp(a).sum())`.
- Inverting a dict: `{v: k for k, v in d.items()}`.

---

## PyTorch corner

`torch.distributions` is another answer key, and it shows the same log-space thinking:

```python
import torch
from torch.distributions import MultivariateNormal, Categorical

mvn = MultivariateNormal(loc=torch.from_numpy(mu), covariance_matrix=torch.from_numpy(Sigma))
mvn.log_prob(torch.from_numpy(X))          # (n,), compare to your mvn_logpdf

# a whole generative classifier = prior + class-conditionals
log_joint = torch.stack([torch.log(prior[c]) + dists[c].log_prob(Xt) for c in range(K)], dim=1)   # (n, K)
post = torch.softmax(log_joint, dim=1)      # same as exp(log_joint - logsumexp)
```

**Distinctions to watch:**
- `MultivariateNormal` checks that the covariance is positive definite and raises an error if not. Add a small `reg * I` (as you do in your GDA).
- `torch.stack(list, dim=1)` ≈ `np.column_stack(list)` for 1-D pieces; `torch.cat` joins along an *existing* dim.
- Text in PyTorch normally means token **ids** + `nn.Embedding`, not count matrices. Your count-matrix NB is the "bag of words" ancestor of that.
