# Week 1: Least squares and linear regression, three ways

**Goal:** Solve one regression problem three ways (normal equation, SVD pseudoinverse, gradient descent), see exactly where each breaks, and put confidence intervals on the answer.

**Time:** ~6–10 hours. **Code outline:** [code-outline.md](code-outline.md)

## Math to review

Full lesson list: [docs/week1-math-review.md](../../docs/week1-math-review.md). The key ideas:

| Idea | Math4ML lesson | You'll use it to... |
|---|---|---|
| Least-squares solution `XᵀX w = Xᵀy` | 8.1.1 | solve the normal equation |
| Collinearity, non-unique solutions | 8.1.2 | understand why `XᵀX` becomes singular |
| Regression with matrices (design matrix, polynomial features) | 8.2.1–8.2.3 | build `X` with an intercept column and polynomial columns |
| Vector / matrix gradients | 9.5.3–9.5.5 | derive `∇_w ‖Xw − y‖²` by hand |
| Pseudoinverse via SVD | 7.2.6 (week 3 preview) | get the minimum-norm solution when `XᵀX` is singular |
| Confidence intervals for slope and intercept | 12.3.5–12.3.6 | compute standard errors and t-intervals |

**Derive before coding** (paper, with shapes):

1. `L(w) = (1/n)‖Xw − y‖²`. Expand it, take the gradient with respect to `w`, set it to zero, and get the normal equation.
2. With `X = U Σ Vᵀ` (thin SVD), show `w = V Σ⁻¹ Uᵀ y` solves the normal equation when `X` has full column rank.
3. Why does an exact duplicate column make `XᵀX` singular? What is its rank?

## Learn / read (pick 1–2)

- ★ **MML book, Ch. 9 "Linear Regression"** (§9.1–9.2 parameter estimation, §9.4 MLE as orthogonal projection) and **§5.2–5.5** for the gradients. Companion notebook: [`tutorial_linear_regression.ipynb`](https://github.com/mml-book/mml-book.github.io/tree/master/tutorials).
- ★ **MIT 18.065, Lecture 9: "Four Ways to Solve Least Squares Problems"** (Strang) on [OCW](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/). This is exactly this week's project.
- **VMLS (Boyd & Vandenberghe), Ch. 12–13** (least squares, least-squares data fitting) and **Ch. 15** (regularized LS for the stretch goal): <https://web.stanford.edu/~boyd/vmls/>
- **ISLP, Ch. 3 "Linear Regression"**, especially §3.1.2 on standard errors and confidence intervals: <https://www.statlearning.com/>
- **CS229 notes, Ch. 1** (LMS / gradient descent and the normal equations): <https://cs229.stanford.edu/main_notes.pdf>
- **d2l §3.1 "Linear Regression"** for the PyTorch view: <https://d2l.ai/chapter_linear-regression/linear-regression.html>
- Gentle option: Andrew Ng's ML Specialization, Course 1, Weeks 1–2.

## Dataset

**California housing**: 20,640 districts, 8 numeric features (median income, house age, rooms, ...), target = median house value in units of $100k.

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)   # downloads once to ~/scikit_learn_data
X, y = data.data.to_numpy(), data.target.to_numpy()
print(data.feature_names); print(data.DESCR[:1500])
```

Docs: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html>

## Tasks

### Core
- [ ] **Warm-up on synthetic data**: generate `y = 2 + 3x₁ − x₂ + noise` and confirm all methods below recover `[2, 3, −1]`.
- [ ] Load California housing, do a train/test split (80/20, fixed seed), and **standardize** features using *training* means/stds only.
- [ ] Build the design matrix with an intercept column.
- [ ] **Method 1, normal equation:** solve `XᵀX w = Xᵀy` with `np.linalg.solve`.
- [ ] **Method 2, SVD pseudoinverse:** compute `w = V Σ⁺ Uᵀ y` yourself from `np.linalg.svd` (then compare to `np.linalg.pinv`).
- [ ] **Method 3, gradient descent:** use your hand-derived gradient. Plot loss vs. iteration for 3 learning rates (one too small, one good, one that diverges).
- [ ] Confirm all three agree with each other and with `np.linalg.lstsq` and `sklearn.linear_model.LinearRegression` (to ~1e-6 for the closed forms).
- [ ] Report train/test RMSE and R².
- [ ] **Gradient check:** compare your analytic gradient to a finite-difference gradient (you'll reuse this checker in weeks 2, 5, 6).

### Break it on purpose
- [ ] Add a column that is an exact copy of `MedInc` (or `2·MedInc`). Look at `np.linalg.matrix_rank`, `np.linalg.cond(XᵀX)`, and what `solve` does.
- [ ] Add a *nearly* collinear column (`MedInc + 1e-6·noise`). Compare the weights from the normal equation, the pseudoinverse with a tiny cutoff (`rcond=1e-10`), and a **truncated** pseudoinverse (`rcond=1e-4`). Print the singular values. Why do the first two give huge, opposite-signed weights while the truncated one doesn't?
- [ ] Re-run gradient descent on **unstandardized** features. How many more iterations does it need? (Relate to the condition number; you'll see why in week 6.)

### Confidence intervals
- [ ] Compute `σ̂² = RSS/(n − p)`, the covariance `σ̂²(XᵀX)⁻¹`, standard errors, and 95% t-intervals for every coefficient.
- [ ] Compare with `statsmodels` `OLS(...).fit().conf_int()` (they should match to many decimals).
- [ ] Which features have intervals that contain 0?

### Polynomial regression (8.2.2)
- [ ] Using only `MedInc`, fit polynomials of degree 1–10 (Vandermonde matrix). Plot train vs. test RMSE against degree. Note what happens to `cond(X)` as the degree grows.

### Stretch
- [ ] **Ridge regression** in closed form, `(XᵀX + λI)⁻¹Xᵀy`, and via SVD shrinkage factors `σᵢ/(σᵢ² + λ)`. Plot coefficient paths vs. `log λ`. Compare to `sklearn.linear_model.Ridge`.
- [ ] Stochastic / mini-batch gradient descent: compare loss curves vs. full-batch.

## Done when

- A results table: method × (weights match? train RMSE, test RMSE, time).
- A plot of GD loss curves, a plot of polynomial train/test error, and a CI table that matches statsmodels.
- You can explain in two sentences why the (truncated) pseudoinverse survives collinearity and the normal equation doesn't.

## Reflection questions

1. Why is `np.linalg.solve(X.T @ X, X.T @ y)` better than `np.linalg.inv(X.T @ X) @ X.T @ y`, and why is `lstsq`/SVD better still? (Hint: `cond(XᵀX) = cond(X)²`.)
2. What does standardizing do to the *shape* of the loss surface?
3. The CI for a coefficient assumes something about the residuals. What, and is it plausible here? (Plot a residual histogram.)

## What I learned

<!-- Your write-up goes here: results table, plots, surprises. -->
