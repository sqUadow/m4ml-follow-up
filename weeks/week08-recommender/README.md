# Week 8 (capstone): matrix-factorization recommender

**Goal:** Predict movie ratings on MovieLens by approximating a mostly-missing ratings matrix `R ≈ P Qᵀ` with low-rank factors. Start with baselines and a plain SVD, then train regularized matrix factorization with **SGD** and with **alternating least squares (ALS)**. Report test RMSE and write up the whole 8-week arc.

**Time:** ~8–12 hours. **Code outline:** [code-outline.md](code-outline.md)

## Math to review

Full lesson list: [docs/week8-math-review.md](../../docs/week8-math-review.md). The key ideas:

| Idea | Math4ML lesson | You'll use it to... |
|---|---|---|
| SVD, low-rank approximation | 7.2.1–7.2.6 (+ week 3) | an SVD baseline on a filled-in ratings matrix |
| Least squares | 8.1.1–8.1.2 (+ week 1) | each ALS half-step is a ridge regression per user / per item |
| Critical points, optimization | 9.8.1–9.8.3 (+ week 6) | the MF objective is non-convex jointly but convex in `P` *or* `Q` alone |
| Lagrange multipliers (perspective) | 9.8.4–9.8.6 | regularization as a soft version of a norm constraint |

**Derive before coding:**

1. Model: `r̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤᵀqᵢ`. Objective over *observed* ratings `Ω`:
   `J = Σ_{(u,i)∈Ω} (rᵤᵢ − r̂ᵤᵢ)² + λ(‖pᵤ‖² + ‖qᵢ‖² + bᵤ² + bᵢ²)`.
   Derive the SGD updates for `bᵤ, bᵢ, pᵤ, qᵢ` from a single rating's error `eᵤᵢ = rᵤᵢ − r̂ᵤᵢ`.
2. **ALS:** fix `Q`. Show that the optimal `pᵤ` solves `(Q_uᵀQ_u + λI) pᵤ = Q_uᵀ r_u`, where `Q_u` holds only the items user `u` rated. Recognize this as week 1's ridge regression.
3. Why can't you just run `np.linalg.svd` on the ratings matrix? (What do you put in the ~94% of missing entries, and what does that do to the fit?)
4. Is `J` convex in `(P, Q)` jointly? (Hint: `p·q` for scalars, `f(p, q) = (1 − pq)²`. Check the Hessian at `(0, 0)` with the second-partials test.)

## Learn / read (pick 1–2)

- ★ **Simon Funk, "Netflix Update: Try This at Home"** (2006), the blog post that popularized SGD matrix factorization: <https://sifter.org/~simon/journal/20061211.html>
- ★ **Koren, Bell & Volinsky, "Matrix Factorization Techniques for Recommender Systems"** (IEEE Computer, 2009). Short and readable, with the biased MF model above. Search the title for the PDF.
- **d2l, "Recommender Systems" chapter**: [MovieLens](https://d2l.ai/chapter_recommender-systems/movielens.html) and [Matrix Factorization](https://d2l.ai/chapter_recommender-systems/mf.html) sections (MF in PyTorch-style code).
- **ISLP, §12.3 "Missing Values and Matrix Completion"**, the SVD-with-missing-values view (iterative soft-impute): <https://www.statlearning.com/>
- **VMLS, Ch. 15 "Multi-objective least squares"** (regularized LS, the ALS building block): <https://web.stanford.edu/~boyd/vmls/>
- **MIT 18.065**, low-rank approximation lectures (6–7) on [OCW](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/)
- Gentle option: Andrew Ng's ML Specialization, Course 3, Week 2 (collaborative filtering).

## Dataset

**MovieLens 100K**: 100,000 ratings (1–5 stars) from 943 users on 1,682 movies. Comes with ready-made 80/20 train/test splits (`u1.base` / `u1.test`, ..., `u5`).

- Page: <https://grouplens.org/datasets/movielens/100k/>
- Zip: <https://files.grouplens.org/datasets/movielens/ml-100k.zip> (download snippet in the [code outline](code-outline.md#step-1-data))
- `u.data` / `u1.base` / `u1.test`: tab-separated `user_id  item_id  rating  timestamp` (IDs start at **1**)
- `u.item`: pipe-separated (`|`), **latin-1** encoded; field 0 = item id, field 1 = title

Larger alternative once things work: `ml-latest-small` (100k ratings, 9k movies, CSV) from the same page.

## Tasks

### Core
- [ ] Load `u1.base` / `u1.test`, convert to 0-based indices, and explore: rating histogram, ratings per user and per movie (long tail!), sparsity of the matrix.
- [ ] Hold out 10% of `u1.base` as a **validation** set for tuning. Touch `u1.test` only for final numbers.
- [ ] **Baselines:** (a) global mean `μ`, (b) `μ + bᵤ + bᵢ` with regularized biases fit by a few rounds of alternating closed-form updates. Report validation RMSE.
- [ ] **SVD baseline:** fill missing entries with (user or item) means, subtract them, compute a truncated SVD for rank k, add the means back, and predict. Plot validation RMSE vs. k = 1..50. Why does a larger k eventually get *worse*?
- [ ] **MF with SGD** (biased model): implement the updates from derivation 1; shuffle ratings each epoch. Plot train/validation RMSE per epoch.
- [ ] **MF with ALS:** alternate solving the per-user and per-item ridge systems with `np.linalg.solve`. Plot train/validation RMSE per iteration. Compare convergence speed to SGD.
- [ ] Tune `k` (e.g. 10, 20, 50) and `λ` on validation; report the best model's **test RMSE** on `u1.test`, alongside all baselines, in one table.

Rough ballpark on `u1.test` for sanity (yours will vary): global mean ≈ 1.15, bias baseline ≈ 0.94–0.96, tuned MF ≈ 0.90–0.93.

### Make it interpretable
- [ ] **Similar movies:** for a few well-known titles (e.g. *Star Wars (1977)*, *Toy Story (1995)*), list the 10 nearest movies by cosine similarity of item factors `qᵢ`.
- [ ] Project item factors to 2-D with your **week 3 PCA** and plot, labeling a few famous movies. Are there interpretable directions?
- [ ] Top-10 recommendations for one user: highest predicted ratings among movies they *haven't* rated.

### Write-up (the capstone part)
- [ ] Fill in the "What I learned" section below: results table, 2–3 plots, and what each previous week contributed (least squares → ALS, SVD → baseline, optimization → SGD/Adam, MLE → why squared error ≈ Gaussian likelihood).
- [ ] Update the top-level README with links to your favorite results from each week.

### Stretch
- [ ] Repeat on all five splits `u1`–`u5` and report mean ± std of test RMSE.
- [ ] **Soft-impute** (ISLP §12.3): iterate "fill missing with current low-rank prediction → SVD → soft-threshold singular values."
- [ ] Train the MF model in PyTorch with `nn.Embedding` and Adam (see the PyTorch corner) and compare with your NumPy SGD.
- [ ] Try `ml-latest-small` or `ml-1m`, and watch how your dense-matrix approaches scale.

### Alternative capstone
If you'd rather do something else, enter a beginner Kaggle competition **using only your own implementations**: [Titanic](https://www.kaggle.com/competitions/titanic) (your week 2 logistic regression + feature engineering) or [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (your week 1 linear/ridge regression on log prices). Keep the same write-up requirements.

## Done when

- One table: global mean, bias baseline, best SVD-k, MF-SGD, MF-ALS, as validation and test RMSE.
- Learning-curve plots for SGD and ALS, plus the similar-movies lists.
- The write-up below is complete.

## Reflection questions

1. ALS solves an exact least-squares problem at each half-step, so why can it still end at a worse solution than SGD (or vice versa)?
2. Squared error on ratings corresponds to which likelihood? (Connect to week 2's MLE.) Is that a sensible model for 1–5 star integers?
3. What does the model do for a brand-new user with no ratings (the "cold start" problem)? What information could help?

## What I learned

<!-- Capstone write-up: results table, plots, and the thread from week 1 to week 8. -->
