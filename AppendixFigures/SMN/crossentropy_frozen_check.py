"""
Appendix N, Table tab:toy_ce: does PC's interference cancellation transfer to cross-entropy?

With PC inference run to equilibrium on the CE loss term itself, the softmax is re-linearised through
the inference and injects its curvature H = diag(p)-pp^T, giving the loss-reconditioned matrix
S_H = I + H(S-I) != S, so S^-1 no longer cancels the weight interference exactly.

Claim tested here: that obstruction is an artefact of putting the CE loss INSIDE the energy. If the
softmax-CE error is used as a FIXED logit-space error  g = y - p  (computed at the feedforward pass),
i.e. the logits are soft-clamped toward the target logit  yhat + g  with a SQUARED (L2) top term,
then the top curvature is the identity, S_H = S, and PC's S^-1 cancellation applies EXACTLY as for
squared error -- rescaled PC should reach optimal logit-space target alignment  cos(dyhat, g) = 1.

We compare five CE variants (all vs the SAME g = y - p):
  BP                                  : cos(g, dyhat_BP)
  PC, full CE inference, unscaled     : loss inside energy
  PC, full CE inference, scaled       : loss inside energy, Theorem-1 factor (the S_H case)
  PC, frozen error, unscaled          : top term 0.5||(yhat+g) - logits||^2
  PC, frozen error, scaled            : as above, Theorem-1 factor  -> ~1.00 expected

Run:  python crossentropy_frozen_check.py
"""
import torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

widths = [10, 20, 20, 20, 10]
L = len(widths) - 1
cos = lambda a, b: (a @ b / (a.norm() * b.norm())).item()


def Wabove(W, l):                                   # W_{L:l+1} = W[L]...W[l+1]
    P = torch.eye(widths[L])
    for k in range(L, l, -1):
        P = P @ W[k]
    return P


def run(seed):
    torch.manual_seed(seed)
    W = [None] + [0.5 * torch.randn(widths[l], widths[l - 1]) for l in range(1, L + 1)]
    x = torch.randn(widths[0])
    xhat = [x.clone()]
    for l in range(1, L + 1):
        xhat.append(W[l] @ xhat[-1])
    yhat = xhat[L]
    Wa = [Wabove(W, l) for l in range(L + 1)]

    # softmax-CE output error at the feedforward pass
    y = torch.zeros(widths[L]); y[torch.randint(0, widths[L], (1,)).item()] = 1.0
    p = torch.softmax(yhat, 0)
    g = y - p                                        # = -dL/d(logits)

    def dyhat(dW):
        return sum(Wa[l] @ dW[l] @ xhat[l - 1] for l in range(1, L + 1))

    # ---- BP ----
    dW_bp = {l: Wa[l].T @ g.outer(xhat[l - 1]) for l in range(1, L + 1)}
    ta_bp = cos(g, dyhat(dW_bp))

    def infer(top_kind, iters=5000, lr=0.05):
        """Minimise 0.5*sum||e_l||^2 + top term, over hidden states; return x*_0..x*_{L-1}."""
        z = [xhat[l].clone().requires_grad_(True) for l in range(1, L)]
        opt = torch.optim.Adam(z, lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            acts = [x] + z
            E = sum(0.5 * ((acts[l] - W[l] @ acts[l - 1]) ** 2).sum() for l in range(1, L))
            logits = W[L] @ acts[L - 1]
            if top_kind == "ce":                     # loss inside the energy (S_H case)
                E = E - (y * torch.log_softmax(logits, 0)).sum()
            else:                                    # frozen L2 clamp toward target logit yhat+g
                E = E + 0.5 * ((yhat + g) - logits).pow(2).sum()
            E.backward(); opt.step()
        return [x] + [zi.detach() for zi in z]

    def pc_ta(top_kind):
        """Returns (unscaled, Theorem-1 scaled) target alignment from one equilibrium."""
        xstar = infer(top_kind)
        eps = {l: xstar[l] - W[l] @ xstar[l - 1] for l in range(1, L)}
        logits_star = W[L] @ xstar[L - 1]
        g_top = (yhat + g - logits_star) if top_kind == "frozen" else (y - torch.softmax(logits_star, 0))
        dW = {l: eps[l].outer(xstar[l - 1]) for l in range(1, L)}
        dW[L] = g_top.outer(xstar[L - 1])
        # Theorem-1 rescaling: divide layer-l update by x*_{l-1}^T xhat_{l-1}
        dW_sc = {l: dW[l] / (xstar[l - 1] @ xhat[l - 1]) for l in range(1, L + 1)}
        return cos(g, dyhat(dW)), cos(g, dyhat(dW_sc))

    return (ta_bp, *pc_ta("ce"), *pc_ta("frozen"))


if __name__ == "__main__":
    import numpy as np
    res = np.array([run(s) for s in range(8)])
    m, s = res.mean(0), res.std(0)
    print("Cross-entropy: target alignment cos(g, dyhat),  g = y - softmax(yhat),  8 seeds\n")
    print(f"  BP                                : {m[0]:.3f} ± {s[0]:.3f}")
    print(f"  PC unscaled, full CE inference    : {m[1]:.3f} ± {s[1]:.3f}")
    print(f"  PC scaled, full CE inference      : {m[2]:.3f} ± {s[2]:.3f}   (the S_H = I+H(S-I) case)")
    print(f"  PC unscaled, frozen error         : {m[3]:.3f} ± {s[3]:.3f}")
    print(f"  PC scaled, frozen error           : {m[4]:.3f} ± {s[4]:.3f}   (fixed g, quadratic top => S_H=S)")
    print("\nIf the frozen-error variant reaches ~1.0, PC's S^-1 interference cancellation transfers"
          "\nto cross-entropy in LOGIT space (dyhat ∝ g = y - p), exactly as for squared error.")
