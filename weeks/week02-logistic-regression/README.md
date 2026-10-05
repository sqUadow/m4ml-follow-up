# Week 2: Logistic regression, maximum likelihood, and Newton's method

**Goal:** Derive logistic regression as a maximum-likelihood problem, fit it with gradient descent *and* Newton's method (using the Hessian), and verify every derivative numerically.

**Time:** ~6–10 hours. **Code outline:** [code-outline.md](code-outline.md)

## Math to review

Full lesson list: [docs/week2-math-review.md](../../docs/week2-math-review.md). The key ideas:

| Idea | Math4ML lesson | You'll use it to... |
|---|---|---|
| Product notation, log turns ∏ into Σ | 12.2.1–12.2.2 | write the likelihood and log-likelihood |
| Likelihood / log-likelihood (discrete & continuous) | 12.2.3–12.2.6 | Bernoulli likelihood for labels; Gaussian likelihood warm-up |
| Maximum likelihood estimation | 12.2.7 | turn "fit the model" into "maximize ℓ(w)" |
| Gradient vector, multivariable chain rule | 9.3.3, 9.3.6 | derive `∇ℓ(w)` through the sigmoid |
| Jacobian, second derivative (Hessian) | 9.4.1, 9.4.5 | derive the Hessian for Newton's method |
| Second-degree Taylor polynomial | 9.4.6 | understand *why* Newton's step is `−H⁻¹g` |

**Derive before coding:**

1. For coin flips `y₁..yₙ ~ Bernoulli(p)`, write `L(p)`, `ℓ(p)`, and show `p̂ = ȳ`. For `xᵢ ~ N(μ, σ²)`, show `μ̂ = x̄` and `σ̂² = (1/n)Σ(xᵢ − x̄)²` (biased!).
2. Show `σ'(z) = σ(z)(1 − σ(z))` for `σ(z) = 1/(1 + e^{−z})`.
3. With `pᵢ = σ(xᵢᵀw)`, write the negative log-likelihood `NLL(w)` and derive its gradient (a vector, shape `(p,)`) and Hessian (a matrix, shape `(p, p)`). Write each as a single matrix expression in `X`, `y`, `p`.
4. Minimize the quadratic Taylor approximation `NLL(w + Δ) ≈ NLL(w) + gᵀΔ + ½ΔᵀHΔ` over `Δ`. That's Newton's step.
5. Is `H` positive semidefinite? What does that say about local vs. global minima?

## Learn / read (pick 1–2)

- ★ **CS229 notes, Ch. 2 "Classification and logistic regression"**, including the section on Newton's method: <https://cs229.stanford.edu/main_notes.pdf>
- ★ **Bishop PRML, §4.3.2–4.3.3** (logistic regression; iteratively reweighted least squares = Newton): <https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/>
- **MML book, §8.3** (parameter estimation, MLE) and **§5.7–5.8** (higher-order derivatives, multivariate Taylor): <https://mml-book.github.io/>
- **ISLP, §4.3 "Logistic Regression"**: <https://www.statlearning.com/>
- **d2l, Maximum Likelihood** appendix: <https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.html>
- **CS231n notes, "Linear classification"** (softmax/cross-entropy, for the stretch goal): <https://cs231n.github.io/linear-classify/>
- Gentle option: Andrew Ng's ML Specialization, Course 1, Week 3.

## Datasets

**Breast Cancer Wisconsin (Diagnostic)**: 569 tumors, 30 numeric features, binary label (malignant/benign). Bundled with scikit-learn, no download.

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target            # (569, 30), (569,)  target: 0 = malignant, 1 = benign
```

**2-D toy data** for plotting decision boundaries:

```python
from sklearn.datasets import make_classification
X2, y2 = make_classification(n_samples=300, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, class_sep=1.0, random_state=0)
```

Stretch: **Digits** (`sklearn.datasets.load_digits()`, 1,797 8×8 images, 10 classes) for softmax regression.

## Tasks

### Core
- [ ] **MLE warm-up:** simulate Bernoulli and Gaussian samples (`rng.binomial`, `rng.normal`), then (a) compute the closed-form MLEs, and (b) numerically minimize the NLL with `scipy.optimize.minimize` and confirm they agree. Plot the log-likelihood curve `ℓ(p)` and mark its maximum.
- [ ] Implement a **numerically stable** sigmoid and NLL (no `log(0)`, no overflow for large `|z|`).
- [ ] Implement `nll`, `grad`, `hessian` from your derivations. **Gradient-check both** the gradient (vs. finite differences of `nll`) and the Hessian (vs. finite differences of `grad`), using `mlutils/gradcheck.py` from week 1.
- [ ] **Gradient descent** on the 2-D toy data. Plot the decision boundary over the data.
- [ ] **Newton's method** on the same data. Plot `NLL − NLL*` vs. iteration on a log scale for GD and Newton on the *same* axes. Count iterations to reach 1e-8.
- [ ] Compare your weights with `sklearn.linear_model.LogisticRegression(C=np.inf)` (unregularized).

### Breast cancer: when MLE doesn't exist
- [ ] Standardize, split 80/20, and fit **unregularized** logistic regression with GD for 10,000 iterations. Track `‖w‖` over iterations. What happens, and why? (Hint: training accuracy hits 100%. If the classes are perfectly separable, what does the likelihood do as `‖w‖ → ∞`?)
- [ ] Add an L2 penalty `(λ/2)‖w‖²` (don't penalize the intercept). Update your gradient and Hessian, re-check them numerically, and refit with Newton.
- [ ] Match sklearn: `LogisticRegression(C=1/λ)` minimizes `C·Σ NLLᵢ + ½‖w‖²`, so use a **sum** NLL. Compare coefficients.
- [ ] Report test accuracy, a confusion matrix, and precision/recall (`sklearn.metrics` is fine for metrics).

### Stretch
- [ ] **Softmax (multinomial) regression** on Digits: derive the cross-entropy gradient `Xᵀ(P − Y)`, gradient-check it, and fit with mini-batch GD.
- [ ] **Confidence intervals from the Hessian:** at the MLE, `Cov(ŵ) ≈ H⁻¹`. Compute standard errors on the 2-D toy data and compare to `statsmodels.api.Logit(y, X).fit().bse`.
- [ ] Damped Newton / line search: start Newton far from the optimum (`w0 = 10·ones`) and see if pure Newton misbehaves.

## Done when

- Gradient and Hessian pass numerical checks (relative error < 1e-6).
- One plot shows GD vs. Newton convergence (log scale); one plot shows the 2-D decision boundary.
- You can explain why `‖w‖` blew up on breast cancer and how λ fixes it.

## Reflection questions

1. Newton converged in a handful of iterations. Why don't we use it for neural networks with millions of parameters?
2. Logistic regression's Hessian is `XᵀSX` and linear regression's is `XᵀX`. What's the role of `S`, and why is Newton for logistic regression called *iteratively reweighted least squares*?
3. The MLE for the Gaussian variance divides by `n`. `np.var` defaults to `ddof=0`, but `np.cov` and pandas `.var()` default to dividing by `n−1`. Which is "right," and for what purpose?

## What I learned

<!-- Your write-up goes here. -->
