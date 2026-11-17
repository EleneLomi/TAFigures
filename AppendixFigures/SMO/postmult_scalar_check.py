"""
Appendix O, Table tab:postmult: post-multiplicative scalar analysis.

The PC change-in-prediction (Eq. 10) is  sum_l W_{L:l+1} W_{L:l+1}^T S^{-1} r * (x*_{l-1}^T x_hat_{l-1}).
The only place the post-inference activity enters is the SCALAR  s_l := x*_{l-1}^T x_hat_{l-1}.
Using the equilibrium activity (Eq. 33)  x*_{l-1} = x_hat_{l-1} + C_{l-1} r,

    s_l = ||x_hat_{l-1}||^2  +  r^T C_{l-1}^T x_hat_{l-1} .
          ^ BP factor (>0)       ^ correction, linear in r, either sign

Claim tested:
 * With a norm-preserving init the activity norm is stable, ||r|| << ||x_hat||, so s_l ~ ||x_hat||^2 > 0
   for every layer -> low-variance, near-optimal target alignment.
 * With Kaiming the activity norm DECAYS with depth, so in deep layers ||r|| ~ ||x_hat|| and the
   correction can dominate and make s_l < 0. A sign-flipped scalar makes that layer's term point
   against r, which is the source of the HIGH VARIANCE of PC target alignment in Fig. 2A.

Run: python postmult_scalar_check.py
"""
import torch
import numpy as np
torch.set_default_dtype(torch.float64)

d = 512  # input = hidden = output width (Fig. 2 default)


def make_net(depth, init, seed):
    """depth = number of hidden layers; L = depth+1 weight layers, all d x d."""
    g = torch.Generator().manual_seed(seed)
    L = depth + 1
    W = [None]
    for _ in range(1, L + 1):
        if init == "kaiming":                       # PyTorch default for nn.Linear
            b = (1.0 / d) ** 0.5
            W.append((torch.rand(d, d, generator=g) * 2 - 1) * b)
        else:                                        # norm-preserving:  W ~ N(0, 1/d)
            W.append(torch.randn(d, d, generator=g) / d ** 0.5)
    return W, L


def analyse(depth, init, seed):
    W, L = make_net(depth, init, seed)
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(d, generator=g)
    y = torch.randn(d, generator=g)
    # feedforward activities x_hat[l] = W_{l:1} x
    xhat = [x.clone()]
    for l in range(1, L + 1):
        xhat.append(W[l] @ xhat[-1])
    r = y - xhat[L]

    def Wprod(a, b):                                 # W_{a:b} = W[a] ... W[b]  (a>=b), else I
        P = torch.eye(d)
        for k in range(b, a + 1):
            P = W[k] @ P
        return P
    Wa = [Wprod(L, l + 1) for l in range(L + 1)]      # Wa[l] = W_{L:l+1}
    S = sum(Wa[l] @ Wa[l].T for l in range(1, L + 1)) # = I + sum_{l=1}^{L-1} W_{L:l+1}W_{L:l+1}^T
    Sinv = torch.linalg.inv(S)

    # equilibrium activity (Appendix E):  x*_l = x_hat_l + (I + sum_{k=2}^{l} W_{l:k}W_{l:k}^T) W_{L:l+1}^T S^{-1} r
    xstar = [x.clone()]
    for l in range(1, L):
        C = torch.eye(d)
        for k in range(2, l + 1):
            Wlk = Wprod(l, k)
            C = C + Wlk @ Wlk.T
        C = C @ Wprod(L, l + 1).T @ Sinv
        xstar.append(xhat[l] + C @ r)

    # sanity check: x* must be a stationary point of the energy, dE/dx_l = e_l - W_{l+1}^T e_{l+1} = 0
    full = xstar + [y]
    for l in range(1, L):
        res = (full[l] - W[l] @ full[l - 1]) - W[l + 1].T @ (full[l + 1] - W[l + 1] @ full[l])
        assert res.norm() < 1e-8 * full[l].norm(), f"x*_{l} is not an equilibrium (|dE/dx|={res.norm():.2e})"

    # post-multiplicative scalars s_l = x*_{l-1}^T x_hat_{l-1},  l = 1..L
    s = [float(xstar[l - 1] @ xhat[l - 1]) for l in range(1, L)]      # l-1 = 0..L-2 (hidden pathway)
    act = [float(xhat[l - 1] @ xhat[l - 1]) for l in range(1, L)]     # ||x_hat_{l-1}||^2 (BP factor)
    # PC change in prediction and target alignment
    dy = sum(Wa[l] @ Wa[l].T @ Sinv @ r * (xstar[l - 1] @ xhat[l - 1]) for l in range(1, L + 1))
    TA = float(r @ dy / (r.norm() * dy.norm()))
    # BP change in prediction and target alignment (no S^-1; BP post-factor x_hat^T x_hat)
    dy_bp = sum(Wa[l] @ Wa[l].T @ r * (xhat[l - 1] @ xhat[l - 1]) for l in range(1, L + 1))
    TA_bp = float(r @ dy_bp / (r.norm() * dy_bp.norm()))
    return dict(TA=TA, TA_bp=TA_bp, s=s, act=act, rnorm=float(r.norm()),
                min_act=min(act), frac_neg=np.mean([si < 0 for si in s]))


if __name__ == "__main__":
    print(f"{'init':>15} {'depth':>5} | {'PC TA mean±std':>16} | {'BP TA mean±std':>16} | "
          f"{'frac layers s_l<0':>17} | {'min ||x_hat||^2':>14} | {'||r||':>7}")
    for init in ["norm-preserving", "kaiming"]:
        for depth in [1, 4, 8, 12]:
            res = [analyse(depth, init, s) for s in range(10)]
            TA = np.array([r["TA"] for r in res])
            TAb = np.array([r["TA_bp"] for r in res])
            fn = np.mean([r["frac_neg"] for r in res])
            ma = np.mean([r["min_act"] for r in res])
            rn = np.mean([r["rnorm"] for r in res])
            print(f"{init:>15} {depth:>5} | {TA.mean():>7.3f} ± {TA.std():>5.3f} | "
                  f"{TAb.mean():>7.3f} ± {TAb.std():>5.3f} | "
                  f"{fn:>17.2f} | {ma:>14.2e} | {rn:>7.2f}")
