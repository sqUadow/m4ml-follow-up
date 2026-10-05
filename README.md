# m4ml follow up

Practice applying the math from the Math Academy **Mathematics for Machine Learning** course by building core ML algorithms from scratch.

The rule for every week: **implement it in NumPy first, then check your results against a library** (scikit-learn, statsmodels, SciPy, or PyTorch). The library is the answer key, not the solution.

## How this repo is organized

```
README.md                 <- you are here: schedule + workflow
requirements.txt          <- packages for all 8 weeks
docs/
  math-review.md          <- full Math4ML lesson index (Math Academy links)
  weekN-math-review.md    <- the lessons to re-read before week N
  resources.md            <- every course / book / video referenced, in one place
  numpy-refresher.md      <- Python + NumPy syntax reminders used throughout
  pytorch-primer.md       <- NumPy -> PyTorch translation and the gotchas
weeks/
  weekNN-topic/
    README.md             <- the project: goals, readings, dataset, task checklist
    code-outline.md       <- packages, function skeletons, syntax hints, PyTorch corner
data/                     <- downloaded datasets (git-ignored, created by you)
```

Each week's `code-outline.md` gives you **function signatures, shapes, and hints, not implementations**. Bodies are left as `...` / `# TODO`. Where a derivation result is useful for checking your own work, it is hidden in a collapsible `<details>` block, so derive it first and peek after.

## The 8-week sequence

| Week | Project | Core math (Math4ML sections) | Main dataset | Math review |
|---|---|---|---|---|
| 1 | [Least squares & linear regression, three ways](weeks/week01-linear-regression/README.md) | 8.1, 8.2, 9.5, 12.3.5–6 | California housing | [week 1](docs/week1-math-review.md) |
| 2 | [Logistic regression, MLE & Newton's method](weeks/week02-logistic-regression/README.md) | 12.2, 9.3, 9.4 | Breast cancer (Wisconsin) | [week 2](docs/week2-math-review.md) |
| 3 | [PCA, SVD, eigenfaces & image compression](weeks/week03-pca-svd/README.md) | 8.3, 7.2 | Olivetti faces, sample image | [week 3](docs/week3-math-review.md) |
| 4 | [Probabilistic classifiers: Naive Bayes & Gaussian discriminant analysis](weeks/week04-probabilistic-classifiers/README.md) | 10.1, 11.4, 11.5 | SMS Spam, Iris | [week 4](docs/week4-math-review.md) |
| 5 | [Autograd engine & a NumPy neural network](weeks/week05-autograd-mlp/README.md) | 9.2.13, 9.4, 9.5 | MNIST | [week 5](docs/week5-math-review.md) |
| 6 | [Optimization: GD, momentum, Adam, Newton & Lagrange multipliers](weeks/week06-optimization/README.md) | 9.8, 7.1.5–6, 9.2.2 | Rosenbrock, separable blobs | [week 6](docs/week6-math-review.md) |
| 7 | [Gaussian mixture models with EM](weeks/week07-gmm-em/README.md) | 12.2, 11.5, 10.6, 12.1.6 | Old Faithful geyser, Iris | [week 7](docs/week7-math-review.md) |
| 8 | [Capstone: matrix-factorization recommender](weeks/week08-recommender/README.md) | 7.2, 8.1, 9.8 | MovieLens 100K | [week 8](docs/week8-math-review.md) |

**PyTorch** shows up gradually: as an autograd answer key in weeks 1–2, as the main comparison in week 5, and as an alternative implementation in weeks 6–8. Every week ends with a *PyTorch corner*, and [docs/pytorch-primer.md](docs/pytorch-primer.md) collects the differences from NumPy that will bite you.

## Weekly workflow

1. **Review the math** in `docs/weekN-math-review.md`. Skim the lessons you feel shaky on.
2. **Read/watch** the one or two "primary" resources in the week's README (not all of them).
3. **Derive on paper first**: gradients, Hessians, update rules. Write the shapes next to every symbol.
4. **Implement** following `code-outline.md`. Start on a tiny synthetic problem where you know the answer.
5. **Verify** against the library answer key. If they disagree, your code is wrong until proven otherwise.
6. **Write up** a short "what I learned" section at the bottom of the week's README (results table + 1–2 plots + surprises).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch is only needed from week 5 on (it's large). If `pip install torch` is slow or you want a specific CPU/GPU build, use the selector at <https://pytorch.org/get-started/locally/>. A CPU build is plenty for everything here.

Work in Jupyter (`jupyter lab`) for exploration, but move finished functions into a `.py` file in the week's folder so later weeks can `import` them (for example, the gradient checker from week 2 is reused in weeks 5 and 6).

Datasets download into `data/` (or scikit-learn's own cache in `~/scikit_learn_data`). Both are git-ignored, so don't commit data.

## Where to go deeper

See [docs/resources.md](docs/resources.md) for the full list. The best companions across the whole sequence:

- **Mathematics for Machine Learning** (Deisenroth, Faisal, Ong), free PDF at <https://mml-book.github.io/>. Chapters 9–12 are literally weeks 1, 3, 6, and 7 of this plan.
- **Andrej Karpathy, Neural Networks: Zero to Hero**, <https://karpathy.ai/zero-to-hero.html>, for week 5 and beyond.
- **Dive into Deep Learning**, <https://d2l.ai/>, free, with code in PyTorch.
- **An Introduction to Statistical Learning with Python (ISLP)**, free PDF at <https://www.statlearning.com/>. Gentle statistical view of weeks 1–4 and 7–8.

*Links written October 2026. Not every one could be re-checked when this was written, so if one breaks, search the resource title. They are all long-lived, free resources.*
