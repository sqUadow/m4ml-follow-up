# Week 3: PCA and SVD (eigenfaces and image compression)

**Goal:** Implement PCA two ways (eigendecomposition of the covariance matrix, and SVD of the centered data), show they agree, and use them for eigenfaces and low-rank image compression.

**Time:** ~6–10 hours. **Code outline:** [code-outline.md](code-outline.md)

> This week overlaps with your current linear algebra course (eigenvalues, orthogonality, SVD). It's a good one to revisit after you finish those chapters.

## Math to review

Full lesson list: [docs/week3-math-review.md](../../docs/week3-math-review.md). The key ideas:

| Idea | Math4ML lesson | You'll use it to... |
|---|---|---|
| What PCA is; computing principal components | 8.3.1–8.3.2 | center data, eigendecompose the covariance, sort by eigenvalue |
| PCA ↔ SVD connection | 8.3.3 | show `Vᵀ` rows = principal directions, `s²/(n−1)` = eigenvalues |
| Feature extraction with PCA | 8.3.4 | project onto the top-k components, then reconstruct |
| Singular values; SVD of larger matrices | 7.2.1–7.2.5 | compute and interpret `U`, `Σ`, `Vᵀ` |
| SVD and the pseudoinverse | 7.2.6 | connect back to week 1 |
| Covariance matrix (prereq) | 11.4.4, 12.1.9 | build the sample covariance `XcᵀXc / (n−1)` |

**Derive before coding:**

1. With centered data `Xc = U Σ Vᵀ`, show `XcᵀXc = V Σ² Vᵀ`. So what are the eigenvectors and eigenvalues of the sample covariance?
2. The projection of `x` onto the top-k directions `V_k` is `V_k V_kᵀ x`. Show the reconstruction error of the whole dataset equals the sum of the discarded eigenvalues (times `n−1`).
3. **Eckart–Young:** the best rank-k approximation of `A` in Frobenius norm is `A_k = U_k Σ_k V_kᵀ`, with error² `= Σ_{i>k} σᵢ²`. You'll verify this numerically.

## Learn / read (pick 1–2)

- ★ **MML book, Ch. 10 "Dimensionality Reduction with PCA"** (§10.2 maximum-variance view, §10.3 projection view, §10.4 eigenvectors and low-rank approximation, §10.5 PCA in high dimensions) and **§4.5–4.6** (SVD, matrix approximation). Companion notebook: [`tutorial_pca.ipynb`](https://github.com/mml-book/mml-book.github.io/tree/master/tutorials).
- ★ **MIT 18.065, Lecture 6 "Singular Value Decomposition"** and **Lecture 7 "Eckart–Young: The Closest Rank k Matrix to A"** (Strang) on [OCW](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/).
- **ISLP, §12.2 "Principal Components Analysis"**: <https://www.statlearning.com/>
- **CS229 notes, "Principal components analysis" chapter**: <https://cs229.stanford.edu/main_notes.pdf>
- **fast.ai Computational Linear Algebra**, notebooks 2–3 (SVD for topic modeling, robust PCA for background removal): <https://github.com/fastai/numerical-linear-algebra>
- Visual intuition: 3Blue1Brown's *Essence of linear algebra*, the eigenvectors episode: <https://www.3blue1brown.com/topics/linear-algebra>
- scikit-learn's eigenfaces example (answer key for the stretch): <https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html>

## Datasets

**Olivetti faces**: 400 grayscale 64×64 face images (40 people × 10 images), already scaled to [0, 1].

```python
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()               # downloads once, then cached
X = faces.data                               # (400, 4096): each row is a flattened 64x64 image
images, person = faces.images, faces.target  # (400, 64, 64), (400,)
```

**A photo for compression**, bundled with scikit-learn:

```python
from sklearn.datasets import load_sample_image
img = load_sample_image("china.jpg")         # (427, 640, 3) uint8
gray = img.mean(axis=2) / 255.0              # (427, 640) float in [0, 1]
```

**Digits** (`load_digits()`, 64-D): good for a 2-D PCA scatter plot colored by digit.

## Tasks

### Core: PCA two ways
- [ ] Warm-up: 2-D correlated Gaussian data (`rng.multivariate_normal`). Compute PCA and draw the principal axes as arrows scaled by `√eigenvalue` over the scatter plot.
- [ ] Implement `pca_eig` (covariance → `np.linalg.eigh` → sort descending) and `pca_svd` (center → `np.linalg.svd`).
- [ ] Show they agree: eigenvalues vs. `s²/(n−1)`, and components equal **up to sign** (explain why sign is arbitrary).
- [ ] Compare to `sklearn.decomposition.PCA` (`components_`, `explained_variance_`).
- [ ] Digits: project to 2-D and scatter-plot colored by label. Plot the cumulative explained-variance ratio. How many components for 90%? 95%?

### Eigenfaces
- [ ] Compute the mean face and the top components on Olivetti; display the mean face and the first 16 eigenfaces as 64×64 images.
- [ ] Reconstruct a few faces using k = 5, 20, 50, 150 components; show originals vs. reconstructions in a grid.
- [ ] Plot reconstruction MSE vs. k and verify it matches the sum of discarded eigenvalues.
- [ ] Note the shape: `n = 400 < d = 4096`. How many non-zero eigenvalues can the covariance have? Why is SVD much cheaper than `eigh` on a 4096×4096 matrix here?

### Image compression with low-rank SVD
- [ ] Compute `A_k` for k = 5, 20, 50, 100 on the grayscale photo and display them.
- [ ] Verify Eckart–Young numerically: `‖A − A_k‖_F² == Σ_{i>k} σᵢ²`.
- [ ] Plot the singular-value spectrum (log scale) and the compression ratio `k(m + n + 1) / (m·n)` vs. relative error.
- [ ] Optional: compress each RGB channel separately and recombine.

### Stretch
- [ ] **High-dimensional trick (MML §10.5):** compute eigenvectors of the small `n×n` matrix `XcXcᵀ` and map them back to `d`-space. Confirm they match.
- [ ] **Face recognition:** project faces to k-D PCA space, classify with 1-nearest-neighbor (your own, via broadcasting distances), and plot accuracy vs. k.
- [ ] **Power iteration:** find the top eigenvector by repeated `v ← Sv/‖Sv‖`; then deflate to get the second. Compare to `eigh`.
- [ ] **Whitening:** transform data so its covariance is the identity; check with `np.cov`.

## Done when

- `pca_eig` and `pca_svd` agree with each other and with sklearn (up to sign).
- You have an eigenface grid, a reconstruction grid, a reconstruction-error-vs-k plot, and compressed images with the Eckart–Young check passing.

## Reflection questions

1. Why must you center the data before PCA? What goes wrong if you don't? (Try it.)
2. Should you standardize features before PCA? When would that change the answer (think about California housing from week 1 vs. pixels)?
3. How is the PCA reconstruction `V_k V_kᵀ x` related to the orthogonal projections from your linear algebra course?

## What I learned

<!-- Your write-up goes here. -->
