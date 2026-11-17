"""
Appendix N, Table tab:mnist_ce: MNIST target alignment, output freezing vs full inference
(NO weight rescaling).

Deep LINEAR nets, norm-maintaining Gaussian init N(0,1/fan_out), softmax cross-entropy, online (B=1).
Three learners, initialised IDENTICALLY and shown the SAME samples:
  BP          : standard backprop
  PC full-inf : PC with the CE loss inside the energy (softmax re-linearised at equilibrium => S_H)
  PC frozen   : PC with a FIXED logit-space error g = y - p (L2 clamp to yhat+g => S_H = S)
None use the Theorem-1/2 rescaling.

Outputs target alignment cos(dyhat, g) at init over 8 seeds for each network.
MNIST is downloaded to $DATA_ROOT (default ./data) if it is not already there.

Run:  python mnist_ce_freezing.py
"""
import os
import numpy as np
import torch, torchvision
import torchvision.transforms as T

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)
DEV = "cpu"
ROOT = os.environ.get("DATA_ROOT", "./data")
DIMS = [784, 64, 64, 64, 10]
L = len(DIMS) - 1
B = 50
INFER_ITERS, INFER_LR = 150, 0.1      # enough for the linear-net inference to equilibrate (TA converged by ~100)


def load_mnist(n_train=6000, n_eval=2000):
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    tr = torchvision.datasets.MNIST(ROOT, train=True, download=True, transform=tfm)
    te = torchvision.datasets.MNIST(ROOT, train=False, download=True, transform=tfm)
    def stack(ds, n):
        X = torch.stack([ds[i][0].view(-1) for i in range(n)]).double()
        y = torch.tensor([ds[i][1] for i in range(n)])
        return X.to(DEV), y.to(DEV)
    return (*stack(tr, n_train), *stack(te, n_eval))


def onehot(y):
    return torch.eye(10, dtype=torch.float64, device=y.device)[y]


def init_W(seed):
    # NORM-MAINTAINING Gaussian init  W ~ N(0, 1/fan_out), i.e. std = 1/sqrt(DIMS[l]).
    # This gives E||Wx||^2 = ||x||^2 for ANY layer shape (unlike N(0,1/fan_in), which shrinks the
    # norm by fan_out/fan_in across a non-square layer, e.g. 784->256). It keeps activity norms
    # constant across depth AND carries genuine interference (W_{L:l+1}W_{L:l+1}^T != I, unlike
    # exact-orthogonal), which is the regime where PC's S^-1 and output freezing actually matter.
    g = torch.Generator().manual_seed(seed); W = [None]
    for l in range(1, L + 1):
        W.append((torch.randn(DIMS[l], DIMS[l - 1], generator=g) / DIMS[l] ** 0.5).to(DEV))
    return W


def feedforward(W, X):
    a = [X]
    for l in range(1, L + 1):
        a.append(a[-1] @ W[l].T)
    return a


def Wabove(W, l):
    P = torch.eye(DIMS[L], dtype=torch.float64, device=DEV)
    for k in range(L, l, -1):
        P = P @ W[k]
    return P


def pc_infer(W, X, Y, kind):
    """Equilibrium hidden states. kind='frozen' (L2 clamp to yhat+g) or 'full' (CE in energy)."""
    xhat = feedforward(W, X)
    g = Y - torch.softmax(xhat[L], 1)
    zt = (xhat[L] + g).detach()
    s = [xhat[l].clone().requires_grad_(True) for l in range(1, L)]
    opt = torch.optim.Adam(s, lr=INFER_LR)
    for _ in range(INFER_ITERS):
        opt.zero_grad()
        acts = [X] + s
        E = sum(0.5 * ((acts[l] - acts[l - 1] @ W[l].T) ** 2).sum() for l in range(1, L))
        lo = acts[L - 1] @ W[L].T
        E = E + (0.5 * ((zt - lo) ** 2).sum() if kind == "frozen"
                 else torch.nn.functional.cross_entropy(lo, Y.argmax(1), reduction="sum"))
        E.backward(); opt.step()
    return [X] + [si.detach() for si in s], g, zt


def pc_updates(W, X, Y, kind):
    xstar, g, zt = pc_infer(W, X, Y, kind)
    dW = {}
    for l in range(1, L):
        eps = xstar[l] - xstar[l - 1] @ W[l].T
        dW[l] = eps.T @ xstar[l - 1] / B
    lo = xstar[L - 1] @ W[L].T
    epsL = (zt - lo) if kind == "frozen" else (Y - torch.softmax(lo, 1))
    dW[L] = epsL.T @ xstar[L - 1] / B
    return dW, feedforward(W, X), g


def bp_updates(W, X, Y):
    xhat = feedforward(W, X)
    g = Y - torch.softmax(xhat[L], 1)
    dW = {l: (Wabove(W, l).T @ g.T @ xhat[l - 1]) / B for l in range(1, L + 1)}
    return dW, xhat, g


def method_updates(W, X, Y, method):
    if method == "BP":
        return bp_updates(W, X, Y)
    return pc_updates(W, X, Y, "full" if method == "PC full-inf" else "frozen")


def target_alignment(W, X, Y, method):
    dW, xhat, g = method_updates(W, X, Y, method)
    dyhat = sum(xhat[l - 1] @ dW[l].T @ Wabove(W, l).T for l in range(1, L + 1))
    cs = (dyhat * g).sum(1) / (dyhat.norm(dim=1) * g.norm(dim=1) + 1e-12)
    return cs.mean().item()


@torch.no_grad()
def evaluate(W, Xte, yte):
    logits = feedforward(W, Xte)[L]
    return (torch.nn.functional.cross_entropy(logits, yte).item(),
            (logits.argmax(1) == yte).double().mean().item())


def train(method, lr, Xtr, ytr, Xte, yte, updates, seed):
    W = init_W(seed)                                   # identical init across methods (same seed)
    gen = torch.Generator().manual_seed(1234 + seed)   # identical batch stream across methods
    for _ in range(updates):
        idx = torch.randint(0, Xtr.shape[0], (B,), generator=gen)
        dW, _, _ = method_updates(W, Xtr[idx], onehot(ytr[idx]), method)
        for l in range(1, L + 1):
            if torch.isfinite(dW[l]).all():
                W[l] = W[l] + lr * dW[l]
    return evaluate(W, Xte, yte)


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_mnist()
    METHODS = ["BP", "PC full-inf", "PC frozen"]
    INFER_ITERS = 400          # rebinds the module global; functions read it at call time
    B = 1                      # ONLINE: batch adds inter-sample interference (needs Thm-2 scaling)
    print("MNIST target alignment cos(dyhat, g), g = y - softmax(yhat), softmax CE, online (B=1)")
    print(f"Init: norm-maintaining Gaussian N(0,1/fan_out). UNSCALED. inference {INFER_ITERS} steps, 8 seeds.\n")
    for dims in [[784, 256, 256, 10], [784, 64, 64, 64, 10]]:
        DIMS = dims; L = len(dims) - 1     # rebind module globals used by init_W / feedforward / Wabove
        print(f"net {dims}:")
        for m in METHODS:
            vals = []
            for s in range(8):
                W = init_W(s)
                gen = torch.Generator().manual_seed(9 + s)
                idx = torch.randint(0, Xtr.shape[0], (B,), generator=gen)
                vals.append(target_alignment(W, Xtr[idx], onehot(ytr[idx]), m))
            vals = np.array(vals)
            print(f"   {m:>12}: {vals.mean():>6.3f} ± {vals.std():>5.3f}")
        print()
