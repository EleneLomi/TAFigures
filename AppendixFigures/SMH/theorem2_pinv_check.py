"""
Appendix H, Table tab:pinv: numerical verification of the pseudo-inverse form of Theorem 2.

The rescaling factor A_l must make the inter-sample interference matrix a scalar multiple of the
identity:   X*^T A_l X_hat = c * I_B   (any c>0; the scalar is absorbed into the learning rate, and
target alignment -- a cosine -- is scale-invariant, so c*I is already OPTIMAL alignment).

The covariance rescaling (PC_MLP.get_activity_covariance in Figure4/figure4.py) computes
    A_l = pinv( z0.T @ z / N )   with z0 = feedforward (x_hat), z = post-inference (x*), (batch,width)
        = pinv( (1/B) sum_b  x_hat_b  x*_b^T )          # a MEAN OUTER PRODUCT (activity cross-cov)
i.e. the Moore-Penrose pseudo-inverse of a mean outer product -- exactly Theorem 2's object, with the
ordinary inverse replaced by the pseudo-inverse.

Two things this checks:
  (1) pinv of the mean outer product DOES give optimal alignment for B <= d (it is c*I), and fails
      only for B > d (genuine rank obstruction).
  (2) ORDER matters: the object must be sum_b x_hat_b x*_b^T (feedforward (x) post-inference), i.e.
      z0.T @ z. The transposed order sum_b x*_b x_hat_b^T (as printed in the theorem) does NOT give
      c*I even after pseudo-inversion. The code uses the correct (working) order.
Also verifies the reverse-order pseudo-inverse law, so pinv(z0.T@z) == pinv(z*^T)@pinv(x_hat).

Run:  python theorem2_pinv_check.py
"""
import numpy as np


def dev_from_scalar_I(M):
    """Relative deviation of M from c*I  (0 => M = c*I => perfect target alignment). Returns (dev,c)."""
    B = M.shape[0]
    c = np.trace(M) / B
    return np.linalg.norm(M - c * np.eye(B)) / (abs(c) * np.sqrt(B)), c


def candidates(Xh, Xs, B):
    # PyTorch convention: activity matrices are (batch, width)
    z0, z = Xh.T, Xs.T                       # z0 = feedforward, z = post-inference
    return {
        "code   pinv(z0.T @ z / N)      [= pinv(sum x_hat x*^T)]": np.linalg.pinv(z0.T @ z / B),
        "written pinv((1/B) sum x* x_hat^T)  [theorem as printed]": np.linalg.pinv(Xs @ Xh.T / B),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("X*^T A X_hat  vs  c*I   (dev=0 => OPTIMAL target alignment; scalar c absorbed in LR)\n")
    print(f"{'d':>4} {'B':>4} {'regime':>18} | {'code order':>22} | {'written order':>16}")
    for d, B in [(64, 16), (128, 32), (64, 63), (64, 64), (64, 80), (16, 64)]:
        Xh = rng.standard_normal((d, B))
        Xs = Xh + 0.5 * rng.standard_normal((d, B))          # x* correlated with x_hat, as in a net
        cand = candidates(Xh, Xs, B)
        devc, cc = dev_from_scalar_I(Xs.T @ cand["code   pinv(z0.T @ z / N)      [= pinv(sum x_hat x*^T)]"] @ Xh)
        devw, cw = dev_from_scalar_I(Xs.T @ cand["written pinv((1/B) sum x* x_hat^T)  [theorem as printed]"] @ Xh)
        regime = "B<=d (works)" if B <= d else "B>d (rank-limited)"
        print(f"{d:>4} {B:>4} {regime:>18} | {devc:>13.2e} (c={cc:>5.1f}) | {devw:>16.2e}")

    # reverse-order pseudo-inverse law: pinv(z0.T @ z) == pinv(z*^T) @ pinv(x_hat)  (B<=d)
    d, B = 64, 16
    Xh = rng.standard_normal((d, B)); Xs = Xh + 0.5 * rng.standard_normal((d, B))
    z0, z = Xh.T, Xs.T
    lhs = np.linalg.pinv(z0.T @ z)
    rhs = np.linalg.pinv(z) @ np.linalg.pinv(z0.T)
    print(f"\nreverse-order law ||pinv(z0.T@z) - pinv(z)@pinv(z0.T)|| = {np.linalg.norm(lhs - rhs):.2e}"
          "  => the single-pinv-of-covariance and the product-of-pinvs forms are the SAME object.")
