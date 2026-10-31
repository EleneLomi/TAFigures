import math
import os
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# ============================================================
# BACKPROPAGATION MODEL (MATCHING PC ARCHITECTURE)
# ============================================================

# ============================================================
# PATHS / GLOBAL SETTINGS
# ============================================================
base_dir = Path(__file__).resolve().parent
script_dir = base_dir

epsilon = 0.0
alpha = 0.0  # Retained from the original experiment (unused).

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
seeds = [0, 1]#, 2, 3, 4]
input_dim = 256
output_dim = 256
n_train = 7096
n_test = 2048
batch_size = 100

hidden_dim = 256
nm_layers = 5

e_lr = 0.01
iters_values = [1, 2, 5, 50, 200, 300, 500, 5000]
alignment_trials = 3
weight_lr = 1e-2

optimizer_cls = torch.optim.SGD
optimizer_kwargs = dict(momentum=0.0)
weight_decay = 0.0
init_name = "norm_preservation"  # Or "kaiming_relu".

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# INITIALIZATION / SCALING HELPERS
# ============================================================
def stable_inverse(cov):
    return torch.linalg.pinv(cov, rtol=1e-6)


def choose_epsilon_from_eigs(cov, mode="relative", alpha=1e-5,
                             kappa_target=1e5):
    eigs = torch.linalg.eigvalsh(cov.double())
    lam_min = eigs.min()
    lam_max = eigs.max()
    if mode == "relative":
        eps = alpha * lam_max
    elif mode == "floor":
        eps = torch.clamp(alpha * lam_max - lam_min, min=0.0)
    elif mode == "condition":
        eps = (lam_max - kappa_target * lam_min) / (kappa_target - 1.0)
        eps = torch.clamp(eps, min=0.0)
    else:
        raise ValueError("mode must be 'relative', 'floor', or 'condition'")
    return eps


def apply_init(model: nn.Module, init_name: str):
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                continue
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            if init_name == "kaiming_uniform":
                limit = math.sqrt(1.0 / fan_in)
                nn.init.uniform_(m.weight, a=-limit, b=limit)
            elif init_name == "norm_preservation":
                nn.init.normal_(m.weight, mean=0.0,
                                std=math.sqrt(1.0 / fan_out))
            elif init_name == "lecun_normal":
                nn.init.normal_(m.weight, mean=0.0,
                                std=math.sqrt(1.0 / fan_in))
            elif init_name == "kaiming_relu":
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
            else:
                raise ValueError(f"Unknown init: {init_name}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ============================================================
# NONLINEAR RESCALED PREDICTIVE-CODING MODEL
# ============================================================
class PC_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers=2,
                 bias=False, scale_method=None, iters=100, e_lr=0.05):
        super().__init__()
        self.nm_layers = nm_layers
        self.iters = iters
        self.e_lr = e_lr
        self.errors = None

        # ONLY linear layers here: one PC error state per hidden layer.
        self.layers = nn.ModuleList()
        for _ in range(nm_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            input_dim = hidden_dim
        self.layers.append(nn.Linear(input_dim, output_dim, bias=bias))

        self._hook_handles = []
        self.scalings = None
        self.z0 = None
        self.z = None
        scale_methods = {
            "norm": self.get_activity_norm,
            "covariance": self.get_activity_covariance,
        }
        if scale_method is not None and scale_method not in scale_methods:
            raise ValueError(f"Unknown scale_method: {scale_method}")
        self.get_activity = (scale_methods[scale_method]
                             if scale_method is not None else None)

    @staticmethod
    def hidden_prediction(layer, x):
        return F.relu(layer(x))

    def feedforward(self, x):
        """Ordinary nonlinear network output, without PC state changes."""
        for layer in self.layers[:-1]:
            x = self.hidden_prediction(layer, x)
        return self.layers[-1](x)

    def forward(self, x: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y is None:
            return self.feedforward(x)
        self.minimize_error_energy(x, y)
        return self.y_pred(x)

    def output_loss(self, y_hat, y):
        return 0.5 * F.mse_loss(y_hat, y, reduction="sum")

    def y_pred(self, x):
        """Prediction with additive errors AFTER each hidden ReLU."""
        s_i = x
        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i = self.hidden_prediction(layer_i, s_i) + e_i
        return self.layers[-1](s_i)

    def E(self, x, y):
        E_errors = 0.5 * sum(torch.sum(e ** 2) for e in self.errors)
        return E_errors + self.output_loss(self.y_pred(x), y)

    def E_local(self, x, y):
        E = 0.0
        s_i = x
        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = self.hidden_prediction(layer_i, s_i)
            s_i = (e_i + s_i_pred).detach()
            E = E + 0.5 * F.mse_loss(s_i_pred, s_i, reduction="sum")
        y_hat = self.layers[-1](s_i)
        return E + self.output_loss(y_hat, y)

    @torch.no_grad()
    def init_zero_errors(self, x):
        tmp = x
        errors = []
        for layer_i in self.layers[:-1]:
            tmp = self.hidden_prediction(layer_i, tmp)
            errors.append(torch.zeros_like(tmp, requires_grad=True))
            tmp = tmp.detach()
        self.errors = errors

    def minimize_error_energy(self, x, y):
        # Freeze weights during PC inference; restore their original settings.
        params = list(self.layers.parameters())
        prior_requires_grad = [p.requires_grad for p in params]
        try:
            for p in params:
                p.requires_grad_(False)
            self.init_zero_errors(x)
            if self.iters > 0:
                error_optim = torch.optim.SGD(self.errors, lr=self.e_lr)
                for _ in range(self.iters):
                    error_optim.zero_grad()
                    energy = self.E(x, y)
                    energy.backward()
                    error_optim.step()
        finally:
            for p, original in zip(params, prior_requires_grad):
                p.requires_grad_(original)

    def loss(self, x, y):
        batch_size = x.shape[0]
        self.minimize_error_energy(x, y)
        E_final = self.E_local(x, y)
        loss = E_final / batch_size
        if self.get_activity is not None:
            self.get_activity(x)
            self.register_hook()
        y_hat = self.y_pred(x)
        return loss, y_hat

    @torch.no_grad()
    def get_activities(self, x):
        # z0[l] and z[l] are inputs to Linear layer l.
        z0 = [x]
        s_i = x
        for layer_i in self.layers[:-1]:
            s_i = self.hidden_prediction(layer_i, s_i)
            z0.append(s_i)
        self.z0 = z0

        z = [x]
        s_i = x
        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i = self.hidden_prediction(layer_i, s_i) + e_i
            z.append(s_i.detach())
        self.z = z

    def get_activity_norm(self, x):
        self.get_activities(x)
        self.scalings = [
            (1.0 / ((z * z0).sum(-1).mean().item() + epsilon))
            * torch.eye(z.shape[-1], device=z.device, dtype=z.dtype)
            for z, z0 in zip(self.z, self.z0)
        ]

    def get_activity_covariance(self, x):
        self.get_activities(x)
        covariances = [
            z0.double().T @ z.double() / z.shape[0]
            for z0, z in zip(self.z0, self.z)
        ]
        self.scalings = [stable_inverse(cov).float() for cov in covariances]

    def register_hook(self):
        self.remove_hooks()
        for layer_idx, layer in enumerate(self.layers):
            handle = layer.weight.register_hook(
                lambda grad, idx=layer_idx: grad @ self.scalings[idx]
            )
            self._hook_handles.append(handle)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []


# ============================================================
# REPRODUCIBILITY / SYNTHETIC DATA
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_synthetic(seed: int, rank_data: int):
    """Same rank-controlled LINEAR teacher as the original experiment."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(rank_data, input_dim, generator=g) / math.sqrt(input_dim)
    B = torch.randn(output_dim, rank_data, generator=g) / math.sqrt(rank_data)
    W_data = B @ A
    X_train = torch.randn(n_train, input_dim, generator=g)
    X_test = torch.randn(n_test, input_dim, generator=g)
    Y_train = X_train @ W_data.T
    Y_test = X_test @ W_data.T
    return X_train, Y_train, X_test, Y_test, W_data


def make_batches(X, Y, steps, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, X.size(0), (steps, batch_size), generator=g)
    return [(X[idx[t]].to(device), Y[idx[t]].to(device))
            for t in range(steps)]


# ============================================================
# TRAINING / TARGET ALIGNMENT
# ============================================================
def train_step(model, xb, yb, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss, _ = model.loss(xb, yb)
    loss.backward()
    optimizer.step()
    model.remove_hooks()
    return float(loss.item())


@torch.no_grad()
def target_alignment(y, y_pre, y_post, eps=1e-12):
    r = y - y_pre
    delta_y = y_post - y_pre
    numerator = (r * delta_y).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(r, dim=-1)
        * torch.linalg.vector_norm(delta_y, dim=-1)
    )
    valid = denominator > eps
    if not valid.any():
        return float("nan")
    return (numerator[valid] / denominator[valid]).mean().item()

class BP_MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        nm_layers=1,
        bias=False,
    ):
        super().__init__()

        self.nm_layers = nm_layers
        self.layers = nn.ModuleList()

        for _ in range(nm_layers):
            self.layers.append(
                nn.Linear(input_dim, hidden_dim, bias=bias)
            )
            input_dim = hidden_dim

        self.layers.append(
            nn.Linear(input_dim, output_dim, bias=bias)
        )

    def forward(self, x):

        # ReLU after each hidden layer, exactly as in PC_MLP
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Linear output layer
        return self.layers[-1](x)

    def loss(self, x, y):

        batch_size = x.shape[0]
        y_hat = self.forward(x)

        error = (
            0.5
            * F.mse_loss(y_hat, y, reduction="sum")
            / batch_size
        )

        return error, y_hat

# ============================================================
# MAIN EXPERIMENT
# PC RESCALED vs PC UNSCALED vs BACKPROPAGATION
# ============================================================

def main():

    alignment_rows = []

    print(f"Using device: {device}")
    print(f"Relaxation steps: {iters_values}")
    print(f"Fixed weight learning rate: {weight_lr}")
    print(f"Trials per seed: {alignment_trials}")
    print("Hidden activation: ReLU")

    scale_conditions = {
        "PC (rescaled, ReLU)": "covariance",
        "PC (unscaled, ReLU)": None,
    }

    for seed in seeds:

        print(f"\n========== seed {seed} ==========", flush=True)

        set_seed(seed)

        # ----------------------------------------------------
        # Synthetic data
        # ----------------------------------------------------

        X_train, Y_train, X_test, Y_test, W_data = make_synthetic(
            seed=seed,
            rank_data=15,
        )

        X_train = X_train.to(device)
        Y_train = Y_train.to(device)

        # ----------------------------------------------------
        # One common initialization for this seed
        # ----------------------------------------------------

        base_model = PC_MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            nm_layers=nm_layers,
            iters=1,
            e_lr=e_lr,
        ).to(device)

        apply_init(base_model, init_name)

        base_state = {
            k: v.detach().clone()
            for k, v in base_model.state_dict().items()
        }

        # ----------------------------------------------------
        # Fixed minibatches for this seed
        # ----------------------------------------------------

        batches = make_batches(
            X_train,
            Y_train,
            steps=alignment_trials,
            seed=seed + 999,
        )

        for trial, (xb, yb) in enumerate(batches):

            # ================================================
            # BACKPROPAGATION
            # ================================================

            bp_model = BP_MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                nm_layers=nm_layers,
                bias=False,
            ).to(device)

            # Both models use identically named Linear layers.
            # Therefore the full state can be copied directly.
            bp_model.load_state_dict(base_state, strict=True)

            # Confirm identical feedforward predictions
            # before either model is updated.
            with torch.no_grad():

                pc_initial = base_model.feedforward(xb)
                bp_initial = bp_model(xb)

                assert torch.allclose(
                    pc_initial,
                    bp_initial,
                    rtol=1e-5,
                    atol=1e-6,
                ), "PC and BP initial predictions differ!"

            # Optimizer
            bp_optimizer = optimizer_cls(
                bp_model.parameters(),
                lr=weight_lr,
                weight_decay=weight_decay,
                **optimizer_kwargs,
            )

            # -----------------------------------------------
            # Prediction BEFORE BP update
            # -----------------------------------------------

            with torch.no_grad():
                bp_y_pre = bp_model(xb).detach().clone()

            # -----------------------------------------------
            # ONE BP weight update
            # -----------------------------------------------

            bp_optimizer.zero_grad(set_to_none=True)

            bp_loss, _ = bp_model.loss(xb, yb)

            bp_loss.backward()

            bp_optimizer.step()

            # -----------------------------------------------
            # Prediction AFTER BP update
            # -----------------------------------------------

            with torch.no_grad():
                bp_y_post = bp_model(xb).detach().clone()

            # -----------------------------------------------
            # Target alignment for BP
            # -----------------------------------------------

            bp_ta = target_alignment(
                y=yb,
                y_pre=bp_y_pre,
                y_post=bp_y_post,
            )

            alignment_rows.append(
                {
                    "seed": seed,
                    "trial": trial,
                    "iters": np.nan,
                    "condition": "BP (ReLU)",
                    "target_alignment": bp_ta,
                    "loss": float(bp_loss.item()),
                    "weight_lr": weight_lr,
                    "e_lr": np.nan,
                    "hidden_dim": hidden_dim,
                }
            )

            print(
                f"{'BP (ReLU)':22s}  "
                f"trial={trial:02d}  "
                f"TA={bp_ta: .6f}",
                flush=True,
            )

            # ================================================
            # PREDICTIVE CODING
            # ================================================

            for n_iters in iters_values:

                for condition_name, scale_method in scale_conditions.items():

                    # ---------------------------------------
                    # Fresh PC model for this condition
                    # ---------------------------------------

                    model = PC_MLP(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        nm_layers=nm_layers,
                        scale_method=scale_method,
                        iters=n_iters,
                        e_lr=e_lr,
                    ).to(device)

                    # Identical initial weights for all
                    # PC conditions and BP
                    model.load_state_dict(
                        base_state,
                        strict=True,
                    )

                    optimizer = optimizer_cls(
                        model.parameters(),
                        lr=weight_lr,
                        weight_decay=weight_decay,
                        **optimizer_kwargs,
                    )

                    # ---------------------------------------
                    # Feedforward prediction BEFORE update
                    # ---------------------------------------

                    with torch.no_grad():
                        y_pre = (
                            model.feedforward(xb)
                            .detach()
                            .clone()
                        )

                    # ---------------------------------------
                    # ONE PC weight update
                    # ---------------------------------------

                    loss = train_step(
                        model,
                        xb,
                        yb,
                        optimizer,
                    )

                    # ---------------------------------------
                    # Feedforward prediction AFTER update
                    # ---------------------------------------

                    with torch.no_grad():
                        y_post = (
                            model.feedforward(xb)
                            .detach()
                            .clone()
                        )

                    # ---------------------------------------
                    # Target alignment
                    # ---------------------------------------

                    ta = target_alignment(
                        y=yb,
                        y_pre=y_pre,
                        y_post=y_post,
                    )

                    # ---------------------------------------
                    # Store results
                    # ---------------------------------------

                    alignment_rows.append(
                        {
                            "seed": seed,
                            "trial": trial,
                            "iters": n_iters,
                            "condition": condition_name,
                            "target_alignment": ta,
                            "loss": loss,
                            "weight_lr": weight_lr,
                            "e_lr": e_lr,
                            "hidden_dim": hidden_dim,
                        }
                    )

                    print(
                        f"{condition_name:22s}  "
                        f"trial={trial:02d}  "
                        f"iters={n_iters:4d}  "
                        f"TA={ta: .6f}",
                        flush=True,
                    )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    df_alignment = pd.DataFrame(alignment_rows)

    raw_csv_path = (
        base_dir / "target_alignment_vs_relaxation_raw.csv"
    )

    df_alignment.to_csv(
        raw_csv_path,
        index=False,
    )

    # ========================================================
    # AGGREGATE ACROSS SEEDS
    # ========================================================

    # First average repeated trials within each seed.
    # Then calculate mean and SEM across seeds.

    seed_means = (
        df_alignment
        .groupby(
            ["condition", "iters", "seed"],
            dropna=False,
        )["target_alignment"]
        .mean()
        .reset_index()
    )

    agg_alignment = (
        seed_means
        .groupby(
            ["condition", "iters"],
            dropna=False,
        )["target_alignment"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    agg_alignment["sem"] = (
        agg_alignment["std"]
        / np.sqrt(agg_alignment["count"])
    )

    summary_csv_path = (
        base_dir / "target_alignment_vs_relaxation_summary.csv"
    )

    agg_alignment.to_csv(
        summary_csv_path,
        index=False,
    )

    print("\n========================================")
    print("Target alignment vs relaxation steps")
    print("========================================")

    print(
        agg_alignment.to_string(index=False)
    )

    # ========================================================
    # PLOT TARGET ALIGNMENT VS RELAXATION STEPS
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(4.8, 3.5)
    )

    colors = {
        "PC (rescaled, ReLU)": "tab:blue",
        "PC (unscaled, ReLU)": "tab:orange",
    }

    # --------------------------------------------------------
    # PLOT PC CURVES
    # --------------------------------------------------------

    for condition_name, color in colors.items():

        df_plot = (
            agg_alignment[
                agg_alignment["condition"] == condition_name
            ]
            .sort_values("iters")
        )

        x = df_plot["iters"].to_numpy(dtype=float)

        y = df_plot["mean"].to_numpy(dtype=float)

        sem = (
            df_plot["sem"]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        # Mean target alignment
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=condition_name,
        )

        # SEM shading
        ax.fill_between(
            x,
            np.maximum(y - sem, -1.0),
            np.minimum(y + sem, 1.0),
            color=color,
            alpha=0.2,
        )

    # ========================================================
    # BACKPROPAGATION BASELINE
    # ========================================================

    bp_row = agg_alignment[
        agg_alignment["condition"] == "BP (ReLU)"
    ].iloc[0]

    bp_mean = float(bp_row["mean"])

    bp_sem = (
        float(bp_row["sem"])
        if pd.notna(bp_row["sem"])
        else 0.0
    )

    # Horizontal BP reference
    ax.axhline(
        y=bp_mean,
        color="tab:green",
        linestyle="-.",
        linewidth=1.8,
        label="BP (ReLU)",
    )

    # BP SEM shading
    ax.axhspan(
        max(bp_mean - bp_sem, -1.0),
        min(bp_mean + bp_sem, 1.0),
        color="tab:green",
        alpha=0.12,
    )

    # ========================================================
    # AXIS FORMATTING
    # ========================================================

    # Symlog allows relaxation step 0 to appear.
    ax.set_xscale(
        "symlog",
        linthresh=1,
    )

    ax.set_xticks(iters_values)

    ax.set_xticklabels(
        [str(v) for v in iters_values],
        rotation=45,
    )

    # Full range of cosine similarity
    ax.set_ylim(-1.0, 1.05)
    ax.set_xlim(1,5010)

    # Reference at zero alignment
    ax.axhline(
        y=0.0,
        linestyle=":",
        color="grey",
        linewidth=1.0,
    )

    # Perfect alignment reference
    ax.axhline(
        y=1.0,
        linestyle="--",
        color="grey",
        linewidth=0.8,
        alpha=0.6,
    )

    ax.set_xlabel("PC relaxation steps")
    ax.set_ylabel("Target alignment")

    ax.legend(
        frameon=False,
        fontsize=8,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.8)

    # ========================================================
    # SAVE SVG FIGURE
    # ========================================================

    plt.tight_layout()

    figure_path = (
        script_dir / "target_alignment_vs_relaxation.svg"
    )

    plt.savefig(
        figure_path,
        format="svg",
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # PRINT OUTPUT LOCATIONS
    # ========================================================

    print("\nSaved:")
    print(f"  raw data: {raw_csv_path}")
    print(f"  summary:  {summary_csv_path}")
    print(f"  figure:   {figure_path}")


# ============================================================
# RUN EXPERIMENT
# ============================================================

if __name__ == "__main__":
    main()