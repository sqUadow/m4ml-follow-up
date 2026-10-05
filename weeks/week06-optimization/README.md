# Week 6: Optimization (GD, momentum, Adam, Newton, and Lagrange multipliers)

**Goal:** (A) Implement the optimizers behind all of ML and race them on hard 2-D functions, with paths drawn on level-curve plots. (B) Use Lagrange multipliers for constrained problems: a hard-margin SVM via its dual, and PCA as constrained variance maximization.

**Time:** ~6–10 hours. **Code outline:** [code-outline.md](code-outline.md)

## Math to review

Full lesson list: [docs/week6-math-review.md](../../docs/week6-math-review.md). The key ideas:

| Idea | Math4ML lesson | You'll use it to... |
|---|---|---|
| Level curves | 9.2.2 | contour plots of `f(x, y)` with optimizer paths on top |
| Critical points, local vs. global extrema | 9.8.1, 9.8.3 | find and classify where `∇f = 0` |
| Second partial derivatives test | 9.8.2 | Hessian eigenvalues → min / max / saddle |
| Lagrange multipliers (one and multiple constraints) | 9.8.4–9.8.6 | SVM dual; PCA derivation |
| Constrained optimization of quadratic forms | 7.1.5–7.1.6 | max `wᵀSw` subject to `‖w‖ = 1` → top eigenvector |
| Second-order Taylor (from week 2) | 9.4.6 | why Newton's method takes `−H⁻¹∇f` |

**Derive before coding:**

1. Rosenbrock `f(x, y) = (1 − x)² + 100(y − x²)²`: compute `∇f` and the Hessian by hand. Find all critical points and classify them with the second-partials test.
2. For `f(x) = ½xᵀAx` with `A = diag(1, κ)`: one GD step is `x ← (I − ηA)x`. For which `η` does it converge? Which `η` gives the fastest worst-case rate, and what is that rate in terms of `κ`? (This is why standardizing features in week 1 helped.)
3. **PCA via Lagrange:** maximize `wᵀSw` subject to `wᵀw = 1`. Write the Lagrangian, set its gradient to zero, and show `Sw = λw`. Which eigenvector maximizes, and what is the maximum value?
4. **Hard-margin SVM:** primal is `min ½‖w‖²` s.t. `yᵢ(wᵀxᵢ + b) ≥ 1`. With multipliers `αᵢ ≥ 0`, derive `w = Σαᵢyᵢxᵢ`, `Σαᵢyᵢ = 0`, and the dual `max Σαᵢ − ½ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ`. (MML §12.3 walks through it.)

## Learn / read (pick 1–2)

- ★ **MML book, Ch. 7 "Continuous Optimization"** (§7.1 GD + momentum, §7.2 Lagrange multipliers, §7.3 convexity/duality) and **Ch. 12 "Classification with SVMs"** (§12.2 primal, §12.3 dual): <https://mml-book.github.io/>
- ★ **"Why Momentum Really Works"** (Goh, Distill). Interactive, and it is derivation 2 done beautifully: <https://distill.pub/2017/momentum/>
- **d2l, Ch. 12 "Optimization Algorithms"**: [GD](https://d2l.ai/chapter_optimization/gd.html), [momentum](https://d2l.ai/chapter_optimization/momentum.html), [Adam](https://d2l.ai/chapter_optimization/adam.html)
- **Ruder, "An overview of gradient descent optimization algorithms"**: <https://www.ruder.io/optimizing-gradient-descent/>
- **CS229 notes, SVM chapter** (Lagrange duality, KKT, the dual): <https://cs229.stanford.edu/main_notes.pdf>
- **MIT 18.065, Lectures 22–23** (gradient descent; accelerating with momentum) on [OCW](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)
- **Adam paper** (short and readable, Algorithm 1 is all you need): <https://arxiv.org/abs/1412.6980>

## Datasets

Mostly **synthetic functions**: Rosenbrock, an ill-conditioned quadratic, and a function with a saddle point.

For the SVM, **linearly separable 2-D data**:

```python
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=60, centers=2, cluster_std=0.8, random_state=4)
y = 2 * y - 1                         # labels {0,1} -> {-1,+1}  (the SVM math assumes ±1)
```

(Or iris setosa vs. versicolor on petal length/width, which is also linearly separable.) For PCA via Lagrange, reuse week 3's data (digits or a 2-D Gaussian).

## Tasks

### Part A: Unconstrained optimization race
- [ ] **Critical points warm-up:** for `f(x, y) = x³ − 3x + y²`, find critical points by hand, classify them with the Hessian, and confirm visually on a contour plot. Start GD *exactly* at the saddle and then slightly off it. What happens?
- [ ] Implement **GD, momentum (heavy ball), Nesterov momentum, RMSProp, Adam, and Newton's method**, each as a function that takes `grad_f` (and `hess_f` for Newton) and returns the full path of iterates.
- [ ] **Ill-conditioned quadratic** (`κ = 10, 100`): plot GD paths for several step sizes (including the optimal one and one that diverges). Verify the empirical convergence rate matches derivation 2.
- [ ] **Rosenbrock** from `(−1.5, 2)`: plot all optimizer paths on one log-scaled contour plot and `f(xₜ) − f*` vs. iteration on a semilog plot. Tune each one's learning rate briefly (a small grid), and report the iterations each needs to reach `f < 1e-6`.
- [ ] Use the gradient checker from week 1 on your Rosenbrock gradient, and a numerical Jacobian of the gradient (week 2) to check your Hessian.
- [ ] Compare to `scipy.optimize.minimize(rosen, x0, jac=rosen_der, method="BFGS")` and `method="Newton-CG"` (SciPy ships `rosen`, `rosen_der`, `rosen_hess` as answer keys).

### Part B: Lagrange multipliers (do at least one; both if you can)
- [ ] **B1. Hard-margin SVM via the dual.** Build the matrix `Q[i, j] = yᵢyⱼxᵢᵀxⱼ`, solve the dual with `scipy.optimize.minimize(method="SLSQP")` using bounds `αᵢ ≥ 0` and the equality constraint `Σαᵢyᵢ = 0`. Recover `w`, `b`, and the support vectors (`αᵢ > 1e-6`). Plot the data, the decision line, the margins `wᵀx + b = ±1`, and circle the support vectors. Compare to `sklearn.svm.SVC(kernel="linear", C=1e6)`.
- [ ] Verify the KKT conditions numerically: `αᵢ > 0` only where `yᵢ(wᵀxᵢ + b) ≈ 1`.
- [ ] **B2. PCA as constrained optimization.** Solve `max wᵀSw s.t. wᵀw = 1` with SLSQP and with projected gradient ascent (step, then renormalize). Confirm you get week 3's first principal component and that `wᵀSw` equals the top eigenvalue. Then add the constraint `wᵀw₁ = 0` to get the second component.
- [ ] **Visualize the Lagrange condition** in 2-D: draw level curves of `wᵀSw`, the unit circle constraint, and the gradients `∇f` and `∇g` at the optimum (they're parallel).

### Stretch
- [ ] **Soft-margin SVM:** add the upper bound `αᵢ ≤ C` and use overlapping blobs. How do the support vectors change with `C`?
- [ ] **Kernel SVM:** replace `xᵢᵀxⱼ` with an RBF kernel and classify `make_moons`.
- [ ] **Line search:** add backtracking (Armijo) to GD and damped Newton; start Newton far away on Rosenbrock.
- [ ] Re-train your week 5 MNIST MLP with your own momentum and Adam implementations and compare training curves.

## Done when

- One contour figure with all optimizer paths on Rosenbrock and one convergence plot, plus a table of iterations-to-tolerance.
- The empirical GD rate on the quadratic matches the theory.
- SVM (dual) matches sklearn's `coef_`, `intercept_`, and support vectors, **or** Lagrange-PCA matches week 3's components.

## Reflection questions

1. Newton converged in a handful of iterations on Rosenbrock. Why is Adam (and not Newton) the default in deep learning?
2. Momentum and Adam both "remember" past gradients, but for different purposes. What does each one fix?
3. In the SVM dual, the data only appears through inner products `xᵢᵀxⱼ`. Why does that matter (think kernels)?

## What I learned

<!-- Your write-up goes here. -->
