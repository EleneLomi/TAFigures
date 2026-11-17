"""
Appendix P, Table supp:s_only_ablation: isolate-S ablation.

Instead of normalising the post-multiplication term to obtain scaled PC, replace
PC's post-multiplication term (post-inference activity, x*_{l-1}^T x_hat_{l-1}) with BP's
(pre-inference activity, x_hat_{l-1}^T x_hat_{l-1}). This isolates the contribution of the PRE-
multiplication reconditioning S^{-1} (PC) vs the weight product W_{L:l+1}W_{L:l+1}^T (BP),
because the two variants then share the *same* post-multiplicative scalar.

Three single-step change-in-prediction variants (Eqs. 6/10), with r = y - yhat:
  BP        : dy = sum_l W_{L:l+1}W_{L:l+1}^T  r        (x_hat_{l-1}^T x_hat_{l-1})
  S-only    : dy = sum_l W_{L:l+1}W_{L:l+1}^T  S^-1 r   (x_hat_{l-1}^T x_hat_{l-1})   <-- ablation
  PC (full) : dy = sum_l W_{L:l+1}W_{L:l+1}^T  S^-1 r   (x*_{l-1}^T   x_hat_{l-1})

"S-only" swaps ONLY the pre-multiplication (adds S^-1) while keeping BP's post term, so the gap
BP -> S-only is attributable to S^-1 alone, and S-only -> PC is the extra effect of the
post-inference activity term. We report target alignment cos(r, dy) vs depth.

Reuses the equilibrium-activity machinery of SMO/postmult_scalar_check.py.

Run:  python isolate_S_ablation.py
"""
import torch
import numpy as np
torch.set_default_dtype(torch.float64)

d = 512


def make_net(depth, init, seed):
    g = torch.Generator().manual_seed(seed)
    L = depth + 1
    W = [None]
    for _ in range(1, L + 1):
        if init == "kaiming":
            b = (1.0 / d) ** 0.5
            W.append((torch.rand(d, d, generator=g) * 2 - 1) * b)
        else:
            W.append(torch.randn(d, d, generator=g) / d ** 0.5)
    return W, L


def analyse(depth, init, seed):
    W, L = make_net(depth, init, seed)
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(d, generator=g)
    y = torch.randn(d, generator=g)
    xhat = [x.clone()]
    for l in range(1, L + 1):
        xhat.append(W[l] @ xhat[-1])
    r = y - xhat[L]

    def Wprod(a, b):
        P = torch.eye(d)
        for k in range(b, a + 1):
            P = W[k] @ P
        return P
    Wa = [Wprod(L, l + 1) for l in range(L + 1)]           # Wa[l] = W_{L:l+1}
    S = sum(Wa[l] @ Wa[l].T for l in range(1, L + 1))
    Sinv = torch.linalg.inv(S)

    # equilibrium activities (Appendix E):  x*_l = x_hat_l + (I + sum_{k=2}^{l} W_{l:k}W_{l:k}^T) W_{L:l+1}^T S^{-1} r
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

    cos = lambda a: float(r @ a / (r.norm() * a.norm())) if a.norm() > 0 else 0.0

    dy_bp     = sum(Wa[l] @ Wa[l].T @ r          * (xhat[l - 1]  @ xhat[l - 1]) for l in range(1, L + 1))
    dy_sonly  = sum(Wa[l] @ Wa[l].T @ Sinv @ r   * (xhat[l - 1]  @ xhat[l - 1]) for l in range(1, L + 1))
    dy_pc     = sum(Wa[l] @ Wa[l].T @ Sinv @ r   * (xstar[l - 1] @ xhat[l - 1]) for l in range(1, L + 1))
    return cos(dy_bp), cos(dy_sonly), cos(dy_pc)


if __name__ == "__main__":
    print("Target alignment cos(r, dy).  S-only = BP's post term + PC's S^-1 pre term "
          "(isolates S^-1).\n")
    for init in ["norm-preserving", "kaiming"]:
        print(f"[{init} init]")
        print(f"{'depth':>5} | {'BP':>14} | {'S-only (isolate S)':>18} | {'PC (full)':>14}")
        for depth in [1, 2, 4, 8, 12]:
            res = np.array([analyse(depth, init, s) for s in range(10)])
            bp, so, pc = res.mean(0)
            bps, sos, pcs = res.std(0)
            print(f"{depth:>5} | {bp:>6.3f} ± {bps:>5.3f} | {so:>8.3f} ± {sos:>5.3f}   | "
                  f"{pc:>6.3f} ± {pcs:>5.3f}")
        print()
    print("Interpretation: BP -> S-only isolates the effect of the S^-1 pre-multiplication "
          "(shared post\nterm); S-only -> PC is the extra effect of the post-inference "
          "activity. If S-only ~ PC and\nboth >> BP, the reconditioning S^-1 is the dominant "
          "driver of PC's target-alignment advantage.")
