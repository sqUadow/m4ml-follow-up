# PyTorch primer for NumPy users

PyTorch tensors behave a lot like NumPy arrays, plus two extras: **automatic differentiation** and **GPU support**. In this repo PyTorch is mostly the *answer key* for your hand-derived gradients, and from week 5 on it's an alternative implementation. Each week's code outline ends with a "PyTorch corner" that builds on this page.

Official starting points: [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), [Autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [`torch.linalg` docs](https://docs.pytorch.org/docs/stable/linalg.html). For a book-length treatment in PyTorch, [d2l.ai](https://d2l.ai/) uses it throughout.

## 1. NumPy -> PyTorch translation table

| NumPy | PyTorch | Note |
|---|---|---|
| `np.array(x)` | `torch.tensor(x)` | `torch.tensor` copies. (`torch.Tensor` with a capital T is the class; don't call it as a constructor.) |
| `np.zeros((n, d))` | `torch.zeros(n, d)` | sizes can be separate args or a tuple |
| `rng.normal(size=(n, d))` | `torch.randn(n, d)` | seed with `torch.manual_seed(0)` |
| `x.astype(np.float32)` | `x.float()` / `x.to(torch.float64)` | |
| `axis=0` | `dim=0` | most functions accept both names, but docs use `dim` |
| `keepdims=True` | `keepdim=True` | docs spell it without the "s" (recent versions accept both) |
| `x.reshape(-1, 1)` | `x.reshape(-1, 1)` or `x.view(-1, 1)` / `x.unsqueeze(1)` | `view` needs contiguous memory |
| `np.concatenate` / `np.hstack` | `torch.cat` / `torch.hstack` | |
| `A @ B`, `A.T` | `A @ B`, `A.T` | for batched 3-D tensors use `A.mT` |
| `np.linalg.solve(A, b)` | `torch.linalg.solve(A, b)` | |
| `np.linalg.lstsq(X, y)` | `torch.linalg.lstsq(X, y).solution` | returns a named tuple; grab `.solution` |
| `U, s, Vt = np.linalg.svd(X, full_matrices=False)` | `U, S, Vh = torch.linalg.svd(X, full_matrices=False)` | same ordering (descending) |
| `np.linalg.eigh(S)` | `torch.linalg.eigh(S)` | ascending eigenvalues, like NumPy |
| `np.linalg.slogdet(A)` | `torch.linalg.slogdet(A)` | |
| `scipy.special.logsumexp(a, axis=1)` | `torch.logsumexp(a, dim=1)` | |
| `scipy.special.expit(z)` | `torch.sigmoid(z)` | |
| `x.max(axis=1)` | `x.max(dim=1)` | **returns `(values, indices)`**, unlike NumPy |
| `np.argmax(x, axis=1)` | `x.argmax(dim=1)` | |
| `float(arr.sum())` | `t.sum().item()` | `.item()` turns a 1-element tensor into a Python number |

**Converting between the two**

```python
t = torch.from_numpy(arr)      # shares memory with arr (changes to one show in the other)
arr = t.numpy()                # also shares memory; fails if t requires grad or is on a GPU
arr = t.detach().cpu().numpy() # the always-safe version
```

## 2. Default dtype: float32 vs float64

NumPy defaults to **float64**; PyTorch defaults to **float32**. `torch.from_numpy` keeps float64.

- Mixing a float64 tensor with a float32 model raises `RuntimeError: expected ... Float but found Double`. Fix with `.float()` on the data.
- For **gradient checking** and comparing to your NumPy results to ~1e-8, use float64 everywhere (`torch.set_default_dtype(torch.float64)` at the top of a checking notebook).
- For **training networks** (week 5+), float32 is normal and faster. Expect agreement with NumPy only to ~1e-5.

## 3. Autograd in five lines

```python
w = torch.zeros(3, requires_grad=True)      # leaf tensor we want gradients for
loss = ((X @ w - y) ** 2).mean()            # build the computation (graph is recorded)
loss.backward()                             # d loss / d w, stored in w.grad
print(w.grad)                               # compare to YOUR hand-derived gradient
```

Things that trip people up:

1. **`.backward()` needs a scalar.** If your output is a vector, reduce it first (`.sum()` / `.mean()`).
2. **Gradients accumulate.** Calling `backward()` twice adds into `.grad`. Zero it each step: `w.grad = None` (or `w.grad.zero_()`, or `optimizer.zero_grad()`).
3. **Updates must not be tracked.** Do manual parameter updates inside `torch.no_grad()`:
   ```python
   with torch.no_grad():
       w -= lr * w.grad
   w.grad = None
   ```
4. **The graph is freed after `backward()`.** Recompute the forward pass each iteration (you would anyway).
5. **`.detach()`** gives a tensor that shares data but is cut out of the graph (use it for logging/plotting).

Higher-order and checking tools:

```python
torch.autograd.functional.hessian(f, w)         # full Hessian of a scalar function (week 2 Newton, week 6)
torch.autograd.gradcheck(f, (w,), eps=1e-6)      # finite-difference check; needs float64 inputs
torch.func.grad(f)(w)                            # functional-style gradient (JAX-like)
```

## 4. `nn.Module`, layers, and the weight-layout gotcha

```python
import torch.nn as nn

layer = nn.Linear(in_features=784, out_features=128)
layer.weight.shape   # torch.Size([128, 784])  <- (out, in)!
layer.bias.shape     # torch.Size([128])
# forward computes:  x @ layer.weight.T + layer.bias      for x of shape (batch, 784)
```

If your NumPy MLP stores `W1` as `(784, 128)` and computes `X @ W1`, then the matching PyTorch weight is `W1.T`. Copy weights between them with `layer.weight.data = torch.from_numpy(W1.T).float()` when you want identical starting points.

A minimal model:

```python
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                    # required
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))   # return LOGITS, not probabilities

model = MLP(784, 128, 10)
sum(p.numel() for p in model.parameters())   # parameter count
```

`nn.Linear` initializes weights with a Kaiming-uniform scheme, not zeros or `randn`. If you want to match your NumPy init, overwrite them.

## 5. Loss functions: what they expect

| You computed by hand | PyTorch | Input | Target | Gotcha |
|---|---|---|---|---|
| `mean((y_hat - y)**2)` | `nn.MSELoss()` | predictions | same shape float | mean, no 1/2 factor; `reduction="sum"` for a sum |
| logistic NLL `-[y log p + (1-y) log(1-p)]` | `nn.BCEWithLogitsLoss()` | **logits** `z` (before sigmoid) | float 0./1. | numerically stable. Avoid `nn.BCELoss` on `sigmoid(z)` |
| softmax cross-entropy | `nn.CrossEntropyLoss()` | **logits** `(n, k)` | **integer class ids** `(n,)` dtype `long` | applies log-softmax for you. Don't softmax first! |
| `-log p[y]` from log-probs | `nn.NLLLoss()` | log-probabilities | integer ids | pairs with `log_softmax` |

All default to `reduction="mean"` over the batch. If your NumPy loss is a **sum**, your gradient will be `n` times larger than PyTorch's.

## 6. Optimizers and the training loop

```python
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)   # or torch.optim.Adam(..., lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    model.train()
    for xb, yb in loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()      # 1. clear old grads
        loss.backward()      # 2. compute new grads
        opt.step()           # 3. update params
    model.eval()
    with torch.no_grad():    # no graph during evaluation
        acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
```

- `weight_decay=λ` in `SGD`/`Adam` adds `λ·w` to the gradient, which is equivalent to a `(λ/2)·||w||²` penalty in the loss. Use `AdamW` for "decoupled" weight decay with Adam.
- `model.train()` / `model.eval()` only matter for layers like dropout or batch norm, but get in the habit.

## 7. Data loading

```python
from torch.utils.data import TensorDataset, DataLoader

X_t = torch.from_numpy(X).float()
y_t = torch.from_numpy(y).long()              # class ids must be long for CrossEntropyLoss
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=128, shuffle=True)
```

## 8. Probability distributions (weeks 4 and 7)

```python
from torch.distributions import Normal, MultivariateNormal, Bernoulli, Categorical

MultivariateNormal(loc=mu, covariance_matrix=Sigma).log_prob(X)   # (n,) log densities
Bernoulli(logits=z).log_prob(y)
```

These are useful answer keys alongside `scipy.stats.multivariate_normal.logpdf`.

## 9. Devices (optional)

```python
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device); xb = xb.to(device)
```

Everything in this repo runs fine on CPU. Data and model must be on the same device.
