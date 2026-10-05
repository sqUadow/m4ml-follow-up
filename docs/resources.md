# Resources

Everything referenced in the weekly READMEs, in one place. ★ = the primary resource for this plan. You don't need all of these; each week names one or two to focus on.

## Books (all free online)

| Book | Link | Best for weeks |
|---|---|---|
| ★ **Mathematics for Machine Learning**, Deisenroth, Faisal & Ong (MML) | <https://mml-book.github.io/> (PDF: <https://mml-book.github.io/book/mml-book.pdf>) | all. Ch. 9 = wk 1, Ch. 10 = wk 3, Ch. 7 + 12 = wk 6, Ch. 11 = wk 7 |
| MML companion notebooks (linear regression, PCA, GMM; with solutions) | <https://github.com/mml-book/mml-book.github.io/tree/master/tutorials> | 1, 3, 7 |
| ★ **An Introduction to Statistical Learning, Python edition** (ISLP), James, Witten, Hastie, Tibshirani & Taylor | <https://www.statlearning.com/> | 1 (Ch. 3), 2 & 4 (Ch. 4), 3 & 7 & 8 (Ch. 12) |
| **Dive into Deep Learning** (d2l), Zhang, Lipton, Li & Smola | <https://d2l.ai/> | 1, 2, 5, 6, 8 (PyTorch code) |
| **Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares** (VMLS), Boyd & Vandenberghe | <https://web.stanford.edu/~boyd/vmls/> | 1, 8 (least squares, regularized LS) |
| **Pattern Recognition and Machine Learning** (PRML), Bishop | <https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/> | 2 (§4.3), 4 (§4.2), 6 (§7.1), 7 (Ch. 9) |
| **Probabilistic Machine Learning: An Introduction**, Murphy | <https://probml.github.io/pml-book/book1.html> | reference for everything |
| **Deep Learning**, Goodfellow, Bengio & Courville | <https://www.deeplearningbook.org/> | 5, 6 (Ch. 4, 6, 8) |

## Courses and lecture series

| Course | Link | Best for weeks |
|---|---|---|
| ★ **Neural Networks: Zero to Hero**, Andrej Karpathy | <https://karpathy.ai/zero-to-hero.html> · code: <https://github.com/karpathy/nn-zero-to-hero> | 5 (and beyond) |
| micrograd lecture: "The spelled-out intro to neural networks and backpropagation" | <https://www.youtube.com/watch?v=VMj-3S1tku0> · repo: <https://github.com/karpathy/micrograd> | 5 |
| **Stanford CS229** (Machine Learning) lecture notes | <https://cs229.stanford.edu/main_notes.pdf> · course: <https://cs229.stanford.edu/> | 1, 2, 4, 6, 7 |
| **Stanford CS231n** course notes | <https://cs231n.github.io/> | 2 (linear classifiers), 5 (backprop), 6 (optimizers) |
| **MIT 18.065** Matrix Methods in Data Analysis, Signal Processing, and Machine Learning (Strang) | <https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/> | 1, 3, 6, 8 |
| **Machine Learning Specialization**, Andrew Ng (Coursera, free to audit) | <https://www.coursera.org/specializations/machine-learning-introduction> | 1, 2, 5, 7, 8 (gentle intuition) |
| **Computational Linear Algebra**, Rachel Thomas (fast.ai) | <https://github.com/fastai/numerical-linear-algebra> | 3, 8, and your ongoing linear algebra |
| **3Blue1Brown**: Neural networks / Essence of linear algebra | <https://www.3blue1brown.com/topics/neural-networks> · <https://www.3blue1brown.com/topics/linear-algebra> | 3, 5 (visual intuition) |

## Articles and papers

| Article | Link | Week |
|---|---|---|
| "Why Momentum Really Works", Goh (Distill) | <https://distill.pub/2017/momentum/> | 6 |
| "An overview of gradient descent optimization algorithms", Ruder | <https://www.ruder.io/optimizing-gradient-descent/> | 6 |
| "Adam: A Method for Stochastic Optimization", Kingma & Ba | <https://arxiv.org/abs/1412.6980> | 6 |
| "Netflix Update: Try This at Home", Simon Funk (the original SGD matrix factorization post) | <https://sifter.org/~simon/journal/20061211.html> | 8 |
| "Matrix Factorization Techniques for Recommender Systems", Koren, Bell & Volinsky (IEEE Computer, 2009) | search the title | 8 |
| "Eigenfaces for Recognition", Turk & Pentland (1991) | search the title | 3 |

## Library docs (your answer keys)

- NumPy: <https://numpy.org/doc/stable/> (see also [numpy-refresher.md](numpy-refresher.md))
- SciPy `optimize` / `stats` / `special`: <https://docs.scipy.org/doc/scipy/reference/>
- scikit-learn user guide: <https://scikit-learn.org/stable/user_guide.html>
- statsmodels OLS: <https://www.statsmodels.org/stable/regression.html>
- PyTorch: <https://docs.pytorch.org/docs/stable/> (see also [pytorch-primer.md](pytorch-primer.md))

## Datasets used

| Dataset | Week | How to load |
|---|---|---|
| California housing | 1 | `sklearn.datasets.fetch_california_housing()` (downloads ~0.5 MB once) |
| Breast cancer Wisconsin (diagnostic) | 2 | `sklearn.datasets.load_breast_cancer()` (bundled, no download) |
| Digits 8×8 | 2, 3, 5 | `sklearn.datasets.load_digits()` (bundled) |
| Olivetti faces | 3 | `sklearn.datasets.fetch_olivetti_faces()` (downloads once) |
| Sample photos (`china.jpg`, `flower.jpg`) | 3 | `sklearn.datasets.load_sample_image("china.jpg")` (bundled) |
| SMS Spam Collection (UCI) | 4 | <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> (download snippet in week 4 outline) |
| Iris | 4, 7 | `sklearn.datasets.load_iris()` (bundled) |
| MNIST | 5 | `sklearn.datasets.fetch_openml("mnist_784", version=1, as_frame=False)` (~15 MB) |
| Old Faithful geyser | 7 | `seaborn.load_dataset("geyser")` or the CSV at <https://raw.githubusercontent.com/mwaskom/seaborn-data/master/geyser.csv> |
| MovieLens 100K | 8 | <https://grouplens.org/datasets/movielens/100k/> (zip: <https://files.grouplens.org/datasets/movielens/ml-100k.zip>) |

## Where to go after week 8

- Karpathy's makemore lectures (rest of Zero to Hero), which build up to a GPT.
- CS229 problem sets (on the course site), for the same algorithms with more rigorous derivations.
- A beginner Kaggle competition using only your own implementations: [Titanic](https://www.kaggle.com/competitions/titanic) (your week 2 code) or [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (your week 1 code + ridge).
- Multivariable calculus and more linear algebra: MIT 18.06 / 18.065, or the next Math Academy course.
