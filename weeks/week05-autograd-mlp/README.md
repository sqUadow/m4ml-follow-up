# Week 5: Autograd engine and a NumPy neural network

**Goal:** (A) build a tiny scalar autograd engine (micrograd) to see the multivariable chain rule as code, (B) write a 2-layer MLP in NumPy with hand-derived **matrix** backprop, verified against finite differences and your engine, (C) train it on MNIST, then rebuild it in PyTorch.

**Time:** ~8–12 hours (the micrograd video alone is 2.5 h). **Code outline:** [code-outline.md](code-outline.md)

## Math to review

Full lesson list: [docs/week5-math-review.md](../../docs/week5-math-review.md). The key ideas:

| Idea | Math4ML lesson | You'll use it to... |
|---|---|---|
| Multivariable chain rule | 9.2.13 | backprop: `∂L/∂x = Σ_paths ∂L/∂u · ∂u/∂x` |
| Jacobian of a vector function | 9.4.1, 9.4.4 | each layer is a function `Rⁿ → Rᵐ` with a Jacobian |
| Total derivatives and their chain rule | 9.5.1–9.5.2 | composing layer Jacobians |
| Vector and matrix gradients | 9.5.3–9.5.5 | `∂L/∂W` for `Z = XW + b` without writing out a 4-D Jacobian |

**Derive before coding** (write shapes on everything; batch of `n`, input `d`, hidden `h`, `k` classes):

1. Forward: `Z1 = X W1 + b1` → `A1 = ReLU(Z1)` → `Z2 = A1 W2 + b2` → `P = softmax(Z2)` → `L = −(1/n) Σᵢ log P[i, yᵢ]`.
2. Show `∂L/∂Z2 = (P − Y)/n` where `Y` is one-hot. (You did this in week 2's stretch.)
3. For `Z = A W + b`: given `G = ∂L/∂Z` (shape `n×m`), derive `∂L/∂W`, `∂L/∂b`, `∂L/∂A`. **Trick:** the result must have the same shape as the thing you differentiate with respect to; that usually leaves only one way to arrange the transposes.
4. ReLU backward: `∂L/∂Z1 = ∂L/∂A1 ⊙ 1[Z1 > 0]`.
5. In micrograd, why does `backward()` *accumulate* (`+=`) gradients instead of assigning them? (Hint: what if a variable is used twice, as in `a * a`?)

## Learn / watch (pick 1–2, but do the first)

- ★ **Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd"** (2h25m): <https://www.youtube.com/watch?v=VMj-3S1tku0> · reference code: <https://github.com/karpathy/micrograd> · series page: <https://karpathy.ai/zero-to-hero.html>. Code along; don't just watch.
- ★ **CS231n notes, "Backpropagation, Intuitions"**: <https://cs231n.github.io/optimization-2/>. Includes the "gradients for vectorized operations" section that is exactly derivation 3.
- **MML book, §5.6 "Backpropagation and Automatic Differentiation"**: <https://mml-book.github.io/>
- **d2l §5.3 "Forward Propagation, Backward Propagation, and Computational Graphs"**: <https://d2l.ai/chapter_multilayer-perceptrons/backprop.html>
- **CS231n notes, "Neural Networks Part 3"** (gradient checks, sanity checks, babysitting training): <https://cs231n.github.io/neural-networks-3/>
- **3Blue1Brown, Neural networks series** (chapters 1–4) for visual intuition: <https://www.3blue1brown.com/topics/neural-networks>
- Stretch: Karpathy's **"Building makemore Part 4: Becoming a Backprop Ninja"**, manual tensor-level backprop through a real network (in the [Zero to Hero](https://karpathy.ai/zero-to-hero.html) series).

## Datasets

**MNIST**: 70,000 handwritten digits, 28×28 grayscale, flattened to 784 features.

```python
from sklearn.datasets import fetch_openml
X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)   # ~15 MB download, cached
X = X.astype(np.float32) / 255.0           # pixels to [0, 1]
y = y.astype(np.int64)                     # labels come back as STRINGS ('5'), so convert
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]   # the standard split
```

Alternative (if you've installed torchvision): `torchvision.datasets.MNIST(root="data", download=True)`.

For fast debugging, use **Digits** (`load_digits()`, 8×8 images, 1,797 samples) before scaling up.

## Tasks

### Part A: micrograd
- [ ] Follow the video and build a `Value` class supporting `+ − × ÷`, `**` (constant power), `exp`, `tanh`, `relu`, and `backward()` via topological sort.
- [ ] Check gradients on a small expression against (a) your hand calculation, (b) finite differences, (c) PyTorch (see the PyTorch corner).
- [ ] Build `Neuron`, `Layer`, `MLP` on top of `Value`, and train on a tiny dataset (e.g. 4 points from the video, or `make_moons(n_samples=100)`).
- [ ] Draw the computation graph for `L = (a*b + c)**2`, and label each node's local derivative and its `grad` after `backward()`.

### Part B: Matrix backprop in NumPy
- [ ] Implement `forward`, `softmax_cross_entropy` (returning loss **and** `dZ2`), and `backward` returning a dict of gradients.
- [ ] **Gradient check** every parameter (W1, b1, W2, b2) on a tiny network (d=5, h=4, k=3, n=7) in **float64**. Relative error should be < 1e-7.
- [ ] **Cross-check with your micrograd:** build the same tiny network with `Value` objects, copy in identical weights, and confirm the gradients match your matrix backprop.
- [ ] Train on Digits with mini-batch SGD as a quick sanity check (should reach >95% test accuracy quickly).

### Part C: MNIST
- [ ] Train on MNIST: hidden size 128–256, He initialization, mini-batches of 64–128, a few epochs of SGD. Plot train loss per step and train/test accuracy per epoch.
- [ ] Target: **~97–98% test accuracy** with a single hidden layer.
- [ ] Sanity checks from CS231n: initial loss ≈ `log(10) ≈ 2.30`; you can overfit 100 training examples to ~100% accuracy.
- [ ] Show a grid of misclassified test digits with predicted vs. true labels.

### Part D: PyTorch version
- [ ] Rebuild the same MLP with `nn.Module` and train it with `torch.optim.SGD`. Same architecture and hyperparameters should give similar accuracy.
- [ ] Copy your NumPy weights into the PyTorch model and confirm the loss **and gradients** match on one batch (watch the weight transpose!).

### Stretch
- [ ] Add momentum to your NumPy SGD (preview of week 6).
- [ ] Add L2 weight decay and/or a second hidden layer.
- [ ] Visualize the first-layer weights `W1[:, j].reshape(28, 28)` for a few hidden units.
- [ ] Watch "Becoming a Backprop Ninja" and do its exercises.

## Done when

- micrograd gradients match PyTorch on a test expression.
- Matrix backprop passes the gradient check and matches micrograd on a tiny net.
- NumPy MLP gets ~97%+ on MNIST; the PyTorch version matches your gradients exactly on a shared batch.

## Reflection questions

1. micrograd does one Python operation per scalar; your NumPy MLP does one per *matrix*. Roughly how many times faster is the NumPy forward/backward for a 784→128→10 net on a batch of 64? (Time it if you build the micrograd version.)
2. Why does initializing all weights to zero fail for an MLP but work fine for logistic regression?
3. Your backward pass is a product of Jacobians applied right-to-left (reverse mode). Why is reverse mode efficient when the output is a scalar loss and there are many parameters?

## What I learned

<!-- Your write-up goes here. -->
