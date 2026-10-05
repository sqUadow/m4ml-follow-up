# Week 7: Gaussian mixture models with EM

**Goal:** Implement the Expectation–Maximization algorithm for a Gaussian mixture model, including a stable multivariate-normal log-likelihood. Cluster real data, compare with your own k-means and with scikit-learn, and choose the number of clusters with BIC. Optional: simulation experiments for confidence intervals and the CLT.

**Time:** ~6–10 hours. **Code outline:** [code-outline.md](code-outline.md)

## Math to review

Full lesson list: [docs/week7-math-review.md](../../docs/week7-math-review.md). The key ideas:

| Idea | Math4ML lesson | You'll use it to... |
|---|---|---|
| Likelihood, log-likelihood, MLE | 12.2.1–12.2.7 | the GMM objective `ℓ = Σᵢ log Σₖ πₖ N(xᵢ; μₖ, Σₖ)` and the M-step updates |
| Bivariate / multivariate normal | 11.5.2–11.5.5 | component densities; you already wrote `mvn_logpdf` in week 4 |
| Chi-square distribution | 10.6.5 | sizing confidence ellipses: `(x−μ)ᵀΣ⁻¹(x−μ) ~ χ²_d` |
| Simulating random observations | 10.2.6 | sampling from a known GMM to test parameter recovery |
| Student's t, exponential, uniform distributions | 10.6.1–10.6.7 | optional simulation experiments |
| Central limit theorem | 12.1.6 | optional CLT experiment |
| Bayes' theorem (week 4) | 10.1.2 | the E-step *is* Bayes' rule for "which component generated xᵢ?" |

**Derive before coding:**

1. **Generative story:** pick `zᵢ ~ Categorical(π)`, then `xᵢ | zᵢ = k ~ N(μₖ, Σₖ)`. Write `p(xᵢ)` by marginalizing over `zᵢ` (law of total probability).
2. **E-step:** with Bayes' rule, write the responsibility `γᵢₖ = P(zᵢ = k | xᵢ)`.
3. **M-step:** treating `γ` as fixed weights, maximize `Σᵢ Σₖ γᵢₖ [log πₖ + log N(xᵢ; μₖ, Σₖ)]`. Show that `μₖ` is a weighted mean, `Σₖ` a weighted covariance, and (with a Lagrange multiplier for `Σπₖ = 1`, from week 6!) `πₖ = Nₖ / n` where `Nₖ = Σᵢ γᵢₖ`.
4. Why can't you just set `∂ℓ/∂μₖ = 0` and solve directly? (Look at where the log is.)
5. Show k-means is the limit of EM with shared spherical covariance `σ²I` as `σ → 0` (responsibilities become hard 0/1 assignments).

## Learn / read (pick 1–2)

- ★ **MML book, Ch. 11 "Density Estimation with Gaussian Mixture Models"** (§11.2 parameter learning via MLE, §11.3 EM algorithm, §11.4 latent-variable perspective). Companion notebook: [`tutorial_gmm.ipynb`](https://github.com/mml-book/mml-book.github.io/tree/master/tutorials).
- ★ **Bishop PRML, Ch. 9 "Mixture Models and EM"** (§9.1 k-means, §9.2 GMMs, uses Old Faithful as its running example!): <https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/>
- **CS229 notes, "EM algorithms"** chapter (mixture of Gaussians, Jensen's inequality, general EM): <https://cs229.stanford.edu/main_notes.pdf>
- **ISLP, §12.4 "Clustering Methods"** (k-means, hierarchical): <https://www.statlearning.com/>
- Gentle option: Andrew Ng's ML Specialization, Course 3, Week 1 (k-means, Gaussian anomaly detection).

## Datasets

**Old Faithful geyser**: 272 eruptions, 2 features (eruption `duration` in minutes, `waiting` time to next eruption in minutes). Two obvious clusters, which makes it the classic GMM example.

```python
import pandas as pd
geyser = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv")
# or: import seaborn as sns; geyser = sns.load_dataset("geyser")
X = geyser[["duration", "waiting"]].to_numpy()      # (272, 2)
# geyser["kind"] has "short"/"long" labels you can compare against afterwards
```

The two features have very different scales (minutes ~2–5 vs. ~40–100). Standardize before clustering, or see what happens if you don't.

**Iris** (`load_iris()`, 4-D, 3 species): cluster without labels, then compare clusters to species with `adjusted_rand_score`.

**Synthetic GMM**: sample from a GMM with known `π, μ, Σ` (that's simulation, lesson 10.2.6) and check that EM recovers them.

## Tasks

### Core: EM for GMMs
- [ ] Write a sampler for a GMM with known parameters (3 components in 2-D). Plot the samples colored by true component.
- [ ] Implement EM: initialization, E-step (in **log space** with `logsumexp`), M-step, and log-likelihood tracking. Reuse your week 4 `mvn_logpdf`.
- [ ] **Assert the log-likelihood never decreases** (allow ~1e-9 slack). This is the best bug detector for EM.
- [ ] On synthetic data, check that the recovered `π, μ, Σ` match the truth (up to a permutation of component labels).
- [ ] Old Faithful: fit `K = 2`; plot data colored by responsibility (soft colors), plus 1σ and 2σ (or 95% via χ²) ellipses for each component. Animate or show a grid of ellipses at iterations 0, 1, 5, 20.
- [ ] Compare to `sklearn.mixture.GaussianMixture(n_components=2, covariance_type="full")`: means, covariances, weights, and the final average log-likelihood (`.score(X)`).

### k-means and model selection
- [ ] Implement k-means (Lloyd's algorithm) with k-means++ initialization. Compare its clusters to GMM's on Old Faithful and on a stretched (anisotropic) synthetic dataset where k-means fails.
- [ ] Implement **BIC** = `−2ℓ + p·log n` (count the free parameters `p` for a full-covariance GMM yourself). Plot BIC vs. K = 1..6 for Old Faithful and iris; compare to `GaussianMixture(...).bic(X)`.
- [ ] Iris: fit K = 3, compare clusters to species with `sklearn.metrics.adjusted_rand_score`.

### Optional: statistics by simulation (10.2.6, 10.6, 12.1.6)
- [ ] **CLT:** draw 10,000 sample means of size n = 1, 2, 5, 30 from an exponential distribution. Plot histograms with the CLT normal density on top.
- [ ] **CI coverage:** draw 10,000 samples of size n = 5 and 30 from an exponential (skewed!), build 95% t-intervals for the mean, and estimate how often they contain the true mean. Is it 95%? Repeat with normal data.
- [ ] **χ² check:** sample from a 2-D normal and verify that the squared Mahalanobis distance follows `χ²₂` (histogram vs. `scipy.stats.chi2.pdf`). This justifies your ellipse sizes.

### Stretch
- [ ] Multiple random restarts; keep the best log-likelihood. How often does a single run get stuck in a worse local optimum?
- [ ] Implement `covariance_type="diag"` and `"spherical"` variants and compare BIC.
- [ ] Use your GMM as a density estimator for **anomaly detection**: flag points with the lowest `log p(x)`.

## Done when

- The log-likelihood curve is monotone and EM recovers synthetic parameters.
- An Old Faithful figure shows soft assignments and fitted ellipses that match sklearn's.
- A BIC-vs-K plot for at least one real dataset, plus your k-means comparison.

## Reflection questions

1. EM always increases the likelihood, but can converge to a bad local optimum. What does a "degenerate" solution look like (hint: a component collapsing onto one point), and what does `reg_covar` protect against?
2. When would you prefer k-means over a GMM, and vice versa?
3. The M-step formulas look like weighted versions of week 4's GDA estimates. What's the relationship between supervised GDA and unsupervised GMM?

## What I learned

<!-- Your write-up goes here. -->
