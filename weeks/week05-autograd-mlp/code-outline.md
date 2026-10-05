# Week 5 code outline: micrograd, NumPy MLP, PyTorch MLP

Skeletons only. Syntax reminders: [numpy-refresher.md](../../docs/numpy-refresher.md) §8 (classes); PyTorch basics: [pytorch-primer.md](../../docs/pytorch-primer.md).

## Packages

```python
import math                                   # micrograd works on Python floats: math.exp, math.tanh
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, load_digits, make_moons
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

import sys; sys.path.insert(0, "../..")
from mlutils.gradcheck import numerical_gradient, relative_error
```

## Suggested layout

```
weeks/week05-autograd-mlp/
  micrograd.py       # Value, Neuron, Layer, MLP
  mlp_numpy.py       # init_params, forward, softmax_cross_entropy, backward, sgd_step, train
  week05.ipynb
```

---

## Part A: micrograd

### Python features you'll need

```python
class Vec:
    def __init__(self, x): self.x = x
    def __add__(self, other):        # self + other
        other = other if isinstance(other, Vec) else Vec(other)   # allow Vec + 3
        return Vec(self.x + other.x)
    def __radd__(self, other):       # 3 + self   (Python tries int.__add__ first, fails, then calls this)
        return self + other
    def __repr__(self): return f"Vec({self.x})"
```

Dunder methods you'll implement: `__add__ __radd__ __mul__ __rmul__ __neg__ __sub__ __rsub__ __truediv__ __rtruediv__ __pow__`. Most can be written in terms of `+`, `*`, `**` (e.g. `a - b = a + (-b)`, `a / b = a * b**-1`), so only a few need their own backward rule.

**Closures:** each operation defines a little `_backward` function that "remembers" its inputs and output:

```python
def make_multiplier(k):
    def f(x):
        return k * x          # f remembers k after make_multiplier returns
    return f
```

### Skeleton

```python
class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None      # leaf nodes have nothing to propagate
        self._prev = set(_children)
        self._op = _op                     # for debugging / drawing the graph

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            # local derivative of + w.r.t. each input, times out.grad (chain rule)
            # ACCUMULATE with +=
            ...
        out._backward = _backward
        return out

    def __mul__(self, other): ...
    def __pow__(self, k: float): ...       # k is a plain number, not a Value
    def exp(self): ...
    def tanh(self): ...
    def relu(self): ...

    def backward(self):
        """1. Build a topological order of all nodes reachable from self (DFS, visited set).
           2. self.grad = 1.0
           3. Call node._backward() for each node in REVERSED topological order."""
        ...
```

Then `Neuron(nin)` (weights `Value(random.uniform(-1, 1))`, bias, optional nonlinearity), `Layer(nin, nout)`, `MLP(nin, [nouts])`, each with a `.parameters()` method returning a flat list of `Value`s. A training step: `loss = sum((yp - yt)**2 for ...)`, zero all grads, `loss.backward()`, `p.data -= lr * p.grad`.

### Test expression (compare with PyTorch below)

```python
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
d = a * b + c
e = d.tanh() * a + b ** 2
e.backward()
print(a.grad, b.grad, c.grad)
```

---

## Part B: NumPy MLP with matrix backprop

### Shapes (write these at the top of your file)

| Symbol | Shape | Meaning |
|---|---|---|
| `X` | `(n, d)` | batch of inputs |
| `W1`, `b1` | `(d, h)`, `(h,)` | layer 1 |
| `Z1`, `A1` | `(n, h)` | pre-activation, ReLU output |
| `W2`, `b2` | `(h, k)`, `(k,)` | layer 2 |
| `Z2`, `P` | `(n, k)` | logits, softmax probabilities |
| `y` | `(n,)` int | class labels |

```python
def init_params(d: int, h: int, k: int, rng, dtype=np.float64) -> dict[str, np.ndarray]:
    """He init for the ReLU layer: W1 ~ N(0, 2/d). Small init for W2 (e.g. N(0, 1/h)). Biases zero.
    Return {"W1": ..., "b1": ..., "W2": ..., "b2": ...}."""
    ...

def forward(params: dict, X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return (logits Z2, cache). Cache everything backward needs: X, Z1, A1."""
    ...

def softmax_cross_entropy(Z2: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (mean loss, dZ2). Stable: subtract the row max before exp.
    log-softmax = Z - logsumexp(Z), then pick the correct-class entries with fancy indexing:
    logp[np.arange(n), y]"""
    ...

def backward(params: dict, cache: dict, dZ2: np.ndarray) -> dict[str, np.ndarray]:
    """Return grads with the SAME keys and shapes as params."""
    ...
```

<details>
<summary>Check your backprop formulas (peek after deriving)</summary>

```
dZ2 = (P − Y_onehot) / n           (n, k)
dW2 = A1ᵀ dZ2                      (h, k)
db2 = dZ2.sum(axis=0)              (k,)
dA1 = dZ2 W2ᵀ                      (n, h)
dZ1 = dA1 ⊙ (Z1 > 0)               (n, h)
dW1 = Xᵀ dZ1                       (d, h)
db1 = dZ1.sum(axis=0)              (h,)
```

The bias gradient is a sum over the batch because `b` was broadcast to every row in the forward pass. Broadcasting forward means summing backward.

</details>

### Gradient check on a tiny network

```python
rng = np.random.default_rng(0)
params = init_params(d=5, h=4, k=3, rng=rng)            # float64!
Xs, ys = rng.normal(size=(7, 5)), rng.integers(0, 3, size=7)

def loss_fn(params):
    Z2, _ = forward(params, Xs)
    return softmax_cross_entropy(Z2, ys)[0]

Z2, cache = forward(params, Xs)
_, dZ2 = softmax_cross_entropy(Z2, ys)
grads = backward(params, cache, dZ2)

for name in params:
    def f(p_val, name=name):                           # default-arg trick binds the loop variable
        return loss_fn({**params, name: p_val})        # dict unpacking: copy params, replace one entry
    num = numerical_gradient(f, params[name])          # your week-1 checker works on any shape
    print(name, relative_error(grads[name], num))
```

Note: ReLU has a kink at 0. If a check fails *slightly* on one entry, it may be a `Z1` value within `eps` of 0, not a bug.

### Cross-check with micrograd

Build the same 5→4→3 network out of `Value`s (wrap each entry of `W1`, `b1`, ... in a `Value`), compute the same loss (you'll need `exp`, `log`, or a `log` op you add), call `.backward()`, and compare each `Value.grad` to the matching entry of `grads`. This is slow, which is the point (see reflection question 1).

### Training loop

```python
def iterate_minibatches(X, y, batch_size, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], y[batch]                     # a generator: use it in a for-loop

def sgd_step(params, grads, lr):
    for k in params:
        params[k] -= lr * grads[k]                   # in-place update of the arrays in the dict

def accuracy(params, X, y) -> float:
    Z2, _ = forward(params, X)
    return (Z2.argmax(axis=1) == y).mean()
```

Training: `for epoch in range(E): for Xb, yb in iterate_minibatches(...): forward → loss → backward → sgd_step`. Log the loss every ~100 steps. A learning rate of ~0.1 with batch 64 and h=128 is a reasonable start for MNIST in float32.

Use `dtype=np.float32` for MNIST training (faster) and float64 only for gradient checks.

---

## Syntax reminders for this week

- Fancy indexing: `P[np.arange(n), y]` picks `P[0,y[0]], P[1,y[1]], ...` → shape `(n,)`.
- One-hot: `np.eye(k)[y]` → `(n, k)`.
- `{**d, "key": new}` makes a shallow copy of a dict with one entry replaced.
- `yield` makes a generator function; `for xb, yb in gen(...)` consumes it.
- `isinstance(x, Value)` for type checks; `random.uniform(-1, 1)` for Python-float randoms in micrograd.
- Misclassified grid: `wrong = np.flatnonzero(pred != y_test)[:25]`, then `show_grid` from week 3 with `shape=(28, 28)`.

---

## PyTorch corner

This is the week where PyTorch becomes a real tool. Read [pytorch-primer.md](../../docs/pytorch-primer.md) §3–7 first.

### 1. micrograd vs. PyTorch on the same expression (as in the video)

```python
a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
b = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
c = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
e = (a * b + c).tanh() * a + b ** 2
e.backward()
print(a.grad.item(), b.grad.item(), c.grad.item())     # compare with your Value grads
```

PyTorch does the same thing as micrograd (build a graph during the forward pass, then walk it in reverse) but on whole tensors, with C++/CUDA kernels.

### 2. The same MLP as an `nn.Module`

```python
class MLP(nn.Module):
    def __init__(self, d, h, k):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, k)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))     # logits

model = MLP(784, 128, 10)
loss_fn = nn.CrossEntropyLoss()                      # = log-softmax + NLL, mean over batch
opt = torch.optim.SGD(model.parameters(), lr=0.1)
```

The training loop is in the primer §6, using `DataLoader` (§7).

### 3. Prove your NumPy gradients match PyTorch exactly

```python
torch.set_default_dtype(torch.float64)
tm = MLP(5, 4, 3)
with torch.no_grad():                                    # copying weights shouldn't be tracked
    tm.fc1.weight.copy_(torch.from_numpy(params["W1"].T))   # NumPy (d,h) -> PyTorch (h,d)
    tm.fc1.bias.copy_(torch.from_numpy(params["b1"]))
    tm.fc2.weight.copy_(torch.from_numpy(params["W2"].T))
    tm.fc2.bias.copy_(torch.from_numpy(params["b2"]))

loss = nn.functional.cross_entropy(tm(torch.from_numpy(Xs)), torch.from_numpy(ys))
loss.backward()
np.allclose(tm.fc1.weight.grad.numpy().T, grads["W1"])  # transpose back!
```

**Distinctions to watch:**
- **Weight layout:** `nn.Linear(d, h).weight` is `(h, d)` and computes `x @ W.T + b`. Your NumPy `W1` is `(d, h)` and computes `X @ W1`. Transpose when copying weights or comparing grads.
- **Loss input:** `nn.CrossEntropyLoss` wants **raw logits** and **integer** labels (`torch.long`). Passing softmax probabilities "works" (no error) but silently trains worse.
- **Mean vs. sum:** `CrossEntropyLoss` averages over the batch, like your `(P − Y)/n`.
- **Initialization:** `nn.Linear` uses Kaiming-*uniform*, not your He-*normal*. Final accuracies will be similar but not identical.
- **`zero_grad()`:** forgetting it makes gradients accumulate across steps, which is the same `+=` you wrote in micrograd. Now you know why it exists.
- **dtype:** MNIST from NumPy is float32 if you cast it as above, matching PyTorch's default. Labels from `y.astype(np.int64)` become `torch.long` automatically via `torch.from_numpy`.
- **Evaluation:** wrap test-set evaluation in `with torch.no_grad():` and call `model.eval()`.
