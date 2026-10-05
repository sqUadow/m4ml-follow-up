# Week 4: Probabilistic classifiers (Naive Bayes and Gaussian discriminant analysis)

**Goal:** Build *generative* classifiers from Bayes' theorem: a multinomial Naive Bayes spam filter, and Gaussian discriminant analysis (LDA with a shared covariance, QDA with per-class covariances). Then compare their decision boundaries with week 2's *discriminative* logistic regression.

**Time:** ~6–10 hours. **Code outline:** [code-outline.md](code-outline.md)

## Math to review

Full lesson list: [docs/week4-math-review.md](../../docs/week4-math-review.md). The key ideas:

| Idea | Math4ML lesson | You'll use it to... |
|---|---|---|
| Law of total probability, Bayes' theorem (extended) | 10.1.1–10.1.3 | `P(class \| x) ∝ P(x \| class)·P(class)`, normalized over classes |
| Covariance, correlation, covariance matrix | 11.4.1–11.4.4 | estimate per-class covariance matrices |
| Bivariate normal, combining normals, i.i.d. normals | 11.5.2–11.5.5 | write the class-conditional density `N(x; μ_c, Σ_c)` |
| MLE (from week 2) | 12.2.7 | the class means/covariances/word frequencies *are* MLEs |
| Independence (prereq) | 11.1.4, 11.2.3 | the "naive" assumption: features independent given the class |

**Derive before coding:**

1. Write Bayes' rule for `P(c | x)` with `K` classes using the extended law of total probability in the denominator.
2. **Naive Bayes for text:** with word counts `x_j` and per-class word probabilities `θ_{c,j}`, show `log P(c | x) = log π_c + Σ_j x_j log θ_{c,j} + const`. Why work in logs? (Multiply 50 probabilities of ~0.001 and see.)
3. MLE of `θ_{c,j}` is (count of word j in class c) / (total words in class c). What goes wrong for a word never seen in spam? Add Laplace smoothing `α`.
4. **GDA:** write `log N(x; μ, Σ)` for d dimensions. Show that if all classes share `Σ`, the quadratic term `xᵀΣ⁻¹x` cancels between classes, so the decision boundary is **linear** (LDA). With different `Σ_c`, it's **quadratic** (QDA).
5. Bonus: show that with shared `Σ`, `P(c=1 | x)` is exactly `σ(wᵀx + b)` for some `w, b`. So LDA and logistic regression have the same *form* but fit it differently.

## Learn / read (pick 1–2)

- ★ **CS229 notes, "Generative learning algorithms"** chapter (GDA, Naive Bayes, Laplace smoothing; it even uses spam as the example): <https://cs229.stanford.edu/main_notes.pdf>
- ★ **ISLP, §4.4 "Generative Models for Classification"** (LDA, QDA, Naive Bayes) and §4.5 (comparison with logistic regression): <https://www.statlearning.com/>
- **Bishop PRML, §4.2 "Probabilistic Generative Models"** and §2.3 (the Gaussian): <https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/>
- **MML book, §6.5 "Gaussian Distribution"** (marginals, conditionals, products): <https://mml-book.github.io/>
- **d2l, Naive Bayes** appendix (does it on MNIST): <https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/naive-bayes.html>

## Datasets

**SMS Spam Collection (UCI)**: 5,574 SMS messages labeled `ham`/`spam` (about 13% spam). Page: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>. Download snippet in the [code outline](code-outline.md#step-1-get-the-sms-data). Each line of the `SMSSpamCollection` file is `label<TAB>message`.

**Iris** (bundled): 150 flowers, 4 features, 3 species. Use two features (e.g. petal length and width) for decision-boundary plots.

```python
from sklearn.datasets import load_iris
iris = load_iris(); X, y = iris.data[:, 2:4], iris.target     # petal length, petal width
```

**Synthetic Gaussians**: sample from known `N(μ_c, Σ_c)` with `rng.multivariate_normal` to check that your estimates recover the true parameters.

## Tasks

### Part A: Naive Bayes spam filter
- [ ] Download and load the SMS data; look at class balance and a few examples of each class.
- [ ] Write your own tokenizer (lowercase, regex for words) and build a vocabulary from the **training split only**.
- [ ] Build a document-term count matrix `X` (`n_docs × vocab_size`). It's fine to compare against `sklearn.feature_extraction.text.CountVectorizer` afterwards.
- [ ] Fit multinomial Naive Bayes: class priors `π_c` and smoothed word probabilities `θ_{c,j}`, **in log space**.
- [ ] Predict with one matrix multiply: `log π + X @ log θᵀ`, then `argmax`.
- [ ] Evaluate: accuracy, confusion matrix, precision and recall for spam. Why is accuracy alone misleading here?
- [ ] Compare to `sklearn.naive_bayes.MultinomialNB(alpha=1.0)`; predictions should match exactly with the same vocabulary and α.
- [ ] Print the 15 most "spammy" words (largest `log θ_spam − log θ_ham`). Do they make sense?

### Part B: Gaussian discriminant analysis
- [ ] Implement the multivariate normal log-density yourself (use `slogdet` and `solve`, vectorized over all points). Check against `scipy.stats.multivariate_normal.logpdf`.
- [ ] Implement your own sample covariance and check against `np.cov(..., rowvar=False)`.
- [ ] **Synthetic check:** generate 2 classes from known Gaussians, fit QDA, and confirm the estimated `μ_c, Σ_c` are close to the truth (closer as n grows).
- [ ] Fit **QDA** (per-class `Σ_c`) and **LDA** (pooled shared `Σ`) on 2-D iris.
- [ ] Plot decision regions for LDA, QDA, and your week 2 logistic regression (one-vs-rest or softmax) side by side, plus 1σ/2σ ellipses for each class Gaussian.
- [ ] Compare to `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` / `QuadraticDiscriminantAnalysis` (predicted labels and `predict_proba`).

### Stretch
- [ ] **Gaussian Naive Bayes** = QDA with *diagonal* covariances. Implement it as a one-line change and compare boundaries.
- [ ] Verify bonus derivation 5 numerically: compute the `w, b` implied by your LDA fit and compare `σ(wᵀx + b)` to your LDA `P(c=1|x)` on a 2-class problem.
- [ ] Use all 4 iris features and report test accuracy for LDA vs. QDA vs. logistic regression with a small training set (10 per class) and a large one. Which wins when data is scarce, and why?
- [ ] Spam filter: try Bernoulli NB (word presence instead of counts) and compare.

## Done when

- Your NB spam filter matches `MultinomialNB` and you can report spam precision/recall.
- Your MVN log-density matches SciPy; LDA/QDA match sklearn.
- One figure shows LDA vs. QDA vs. logistic regression decision boundaries on the same data.

## Reflection questions

1. Naive Bayes' independence assumption is obviously false for words ("free" and "prize" co-occur). Why does it still classify well? Are its *probabilities* well calibrated? (Look at `predict_proba` values.)
2. QDA has many more parameters than LDA. Count them for `d` features and `K` classes. When would you prefer LDA?
3. Generative (LDA) vs. discriminative (logistic regression): which one uses more assumptions, and when does that help vs. hurt?

## What I learned

<!-- Your write-up goes here. -->
