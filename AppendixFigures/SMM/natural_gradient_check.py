"""
Appendix M: natural-gradient equivalence check.

Does PC with weight-update scaling produce the SAME PARAMETER UPDATE as the natural gradient,
or only the same prediction change?

Claim (appendix derivation): in the DLN batch limit the *effective total update* on the
collapsed map  W_{L:1}  obeys   dW_{L:1}/dt = alpha (Vbar - W_{L:1}),  which is the EXACT
natural gradient for the DLN regression problem (Vbar = Y X^-1 = least-squares map).

We test this by reproducing the update rule:
each layer's local PC gradient is post-multiplied by  A_k = pinv(X*_{k-1}^T) @ pinv(X_hat_{k-1}),
then we accumulate the induced change of the product map
    dWprod = sum_k  W_{L:k+1} dW_k W_{k-1:1}
and compare its DIRECTION and MAGNITUDE to the exact natural gradient (Vbar - W).

Config: square DLN (all widths = d), batch B = d so that X is invertible, Theorem-2 exactness
holds (B <= min width), and the natural gradient is exactly alpha(Vbar - W) with no projection
ambiguity. Real PC inference (Adam on the errors) to equilibrium, several depths and seeds.

Run:  python natural_gradient_check.py
"""
import torch
import numpy as np
torch.set_default_dtype(torch.float64)

d = 24                     # all layer widths = d
B = d                      # batch = width  => X invertible, Thm-2 exact (B <= width)
INFER_ITERS = 6000
INFER_LR = 0.02


def build(depth, seed):
    """depth hidden layers -> L = depth+1 weight layers, all d x d, norm-preserving init."""
    g = torch.Generator().manual_seed(seed)
    L = depth + 1
    W = [None] + [torch.randn(d, d, generator=g) / d ** 0.5 for _ in range(L)]
    return W, L


def Wprod(W, a, b):                       # W_{a:b} = W[a] ... W[b] (a>=b), else I
    P = torch.eye(d)
    for k in range(b, a + 1):
        P = W[k] @ P
    return P


def pc_infer(W, L, X, Y):
    """Batched real PC inference: minimise 0.5*sum||e_l||^2 + 0.5||Y-Yhat||^2 over errors.
    X,Y are (d x B). Returns equilibrium input-side activities Xstar[k] (d x B), k=0..L-1."""
    # feedforward activities
    xhat = [X.clone()]
    for l in range(1, L + 1):
        xhat.append(W[l] @ xhat[-1])
    # errors e_1..e_{L-1} on the hidden states (columns = samples)
    e = [torch.zeros(d, B, requires_grad=True) for _ in range(1, L)]
    opt = torch.optim.Adam(e, lr=INFER_LR)
    for _ in range(INFER_ITERS):
        opt.zero_grad()
        s = [X]
        for l in range(1, L):
            s.append(e[l - 1] + W[l] @ s[-1])
        yhat = W[L] @ s[-1]
        E = 0.5 * sum((ei ** 2).sum() for ei in e) + 0.5 * ((Y - yhat) ** 2).sum()
        E.backward()
        opt.step()
    with torch.no_grad():
        s = [X]
        for l in range(1, L):
            s.append(e[l - 1].detach() + W[l] @ s[-1])
        conv = float(sum((ei.grad ** 2).sum() for ei in e) ** 0.5)   # residual grad norm
    return [si.detach() for si in s] + [W[L] @ s[-1]], xhat, conv     # xstar[0..L], xhat[0..L]


def run(depth, seed):
    W, L = build(depth, seed)
    g = torch.Generator().manual_seed(5000 + seed)
    X = torch.randn(d, B, generator=g)
    Y = torch.randn(d, B, generator=g)
    xstar, xhat, conv = pc_infer(W, L, X, Y)
    R = Y - xhat[L]                                    # feedforward residual (d x B)

    # ---- scaled-PC (released rule): dW_k = [ (1/B) E_k X*_{k-1}^T ] @ A_k ,
    #      A_k = pinv(X*_{k-1}^T) @ pinv(X_hat_{k-1}) ----
    def A(k):
        Xs_km1 = xstar[k - 1]                          # (d x B) equilibrium input activities
        Xh_km1 = xhat[k - 1]                           # (d x B) feedforward input activities
        return torch.linalg.pinv(Xs_km1.T) @ torch.linalg.pinv(Xh_km1)   # (d x d)

    # local PC errors eps_k at equilibrium: eps_k = x*_k - W_k x*_{k-1} (hidden); top uses R*
    eps = {}
    for k in range(1, L):
        eps[k] = xstar[k] - W[k] @ xstar[k - 1]        # (d x B)
    eps[L] = Y - xstar[L]                              # top error at equilibrium logits

    dW_pc, dW_bp = {}, {}
    for k in range(1, L + 1):
        raw = (eps[k] @ xstar[k - 1].T) / B            # (d x d) mean local PC update
        dW_pc[k] = raw @ A(k)                          # scaled-PC (natural) update
        dW_bp[k] = (Wprod(W, L, k + 1).T @ R @ xhat[k - 1].T) / B   # BP update

    def induced(dW):                                   # change of product map W_{L:1}
        return sum(Wprod(W, L, k + 1) @ dW[k] @ Wprod(W, k - 1, 1) for k in range(1, L + 1))

    dWprod_pc, dWprod_bp = induced(dW_pc), induced(dW_bp)

    # ---- exact natural gradient (B=d, X invertible): NG = Vbar - Wprod, Vbar = Y X^-1 ----
    Wprod_full = Wprod(W, L, 1)
    Vbar = Y @ torch.linalg.inv(X)
    NG = Vbar - Wprod_full
    Gplain = R @ X.T / B                               # Euclidean (plain) product gradient dir

    cos = lambda Amat, Bmat: float((Amat.flatten() @ Bmat.flatten()) /
                                   (Amat.norm() * Bmat.norm()))
    # best-fit scalar alpha and relative residual  ||dWprod_pc - alpha*NG|| / ||dWprod_pc||
    alpha = float((dWprod_pc.flatten() @ NG.flatten()) / (NG.flatten() @ NG.flatten()))
    resid = float((dWprod_pc - alpha * NG).norm() / dWprod_pc.norm())

    # Theorem-2 prediction-change sanity: dyhat_b ∝ r_b for every sample
    dYhat = dWprod_pc @ X                              # (d x B)
    per_sample = [cos(dYhat[:, b:b + 1], R[:, b:b + 1]) for b in range(B)]

    return dict(conv=conv,
                cos_pc_NG=cos(dWprod_pc, NG),
                cos_bp_NG=cos(dWprod_bp, NG),
                cos_pc_plain=cos(dWprod_pc, Gplain),
                cos_bp_plain=cos(dWprod_bp, Gplain),
                resid=resid,
                ta_min=min(per_sample), ta_mean=float(np.mean(per_sample)))


if __name__ == "__main__":
    print(f"square DLN, width d={d}, batch B={B} (=width, Thm-2 exact). "
          f"Vbar = Y X^-1, NG dir = Vbar - W_prod.\n")
    print(f"{'depth':>5} | {'cos(scaledPC, NG)':>17} | {'cos(BP, NG)':>11} | "
          f"{'cos(scaledPC,plain)':>19} | {'||dWpc-aNG||/||dWpc||':>21} | "
          f"{'per-sample TA (min/mean)':>24} | {'infer conv':>10}")
    for depth in [2, 4, 8]:
        res = [run(depth, s) for s in range(5)]
        m = lambda k: np.mean([r[k] for r in res])
        print(f"{depth:>5} | {m('cos_pc_NG'):>17.5f} | {m('cos_bp_NG'):>11.4f} | "
              f"{m('cos_pc_plain'):>19.4f} | {m('resid'):>21.2e} | "
              f"{m('ta_min'):>10.5f} /{m('ta_mean'):>11.5f} | {m('conv'):>10.1e}")
    print("\nInterpretation: cos(scaledPC, NG) ~ 1 and residual ~ 0  =>  scaled PC's TOTAL "
          "parameter\nupdate EQUALS the natural gradient (not merely the same prediction "
          "change). BP is not\naligned with NG. Per-sample TA ~ 1 is the Theorem-2 sanity check.")
