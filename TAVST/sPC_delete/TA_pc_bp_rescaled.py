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
# PATHS / GLOBAL SETTINGS
# ============================================================

base_dir = Path(__file__).parent
script_dir = os.path.dirname(os.path.abspath(__file__))

epsilon = 0.0
alpha = 0.0  # set >0 if you want covariance regularisation


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Use several seeds for a more reliable mean/SEM.
# For a very quick test, use: seeds = [0]
seeds = [0, 10]  # , 2, 3, 4

# Synthetic-data dimensions.
input_dim = 2
output_dim = 2
n_train = 7096
n_test = 2048
batch_size = 128#28

# Network architecture.
hidden_dim = 2
nm_layers = 1

# Predictive-coding relaxation hyperparameters.
# e_lr is the relaxation/error-state learning rate inside PC inference.
e_lr = 0.01

# Relaxation-step values tested for PC.
iters_values = [0,1]

# Number of different minibatches tested for each seed.
# Every condition sees the SAME minibatch within a trial.
alignment_trials = 3

# Keep the network weight learning rate fixed across conditions.
weight_lr = 1e-2

# Optimizer for network weights.
optimizer_cls = torch.optim.SGD
optimizer_kwargs = dict(momentum=0.0)
weight_decay = 0.0

# Initialization.
init_name = "norm_preservation"

# Device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# INITIALIZATION / SCALING HELPERS
# ============================================================

def stable_inverse(cov):
    return torch.linalg.pinv(cov, rcond=1e-6)


def choose_epsilon_from_eigs(
    cov,
    mode="relative",
    alpha=1e-5,
    kappa_target=1e5,
):
    eigs = torch.linalg.eigvalsh(cov.double())
    lam_min = eigs.min()
    lam_max = eigs.max()

    if mode == "relative":
        eps = alpha * lam_max
    elif mode == "floor":
        floor = alpha * lam_max
        eps = torch.clamp(floor - lam_min, min=0.0)
    elif mode == "condition":
        eps = (lam_max - kappa_target * lam_min) / (kappa_target - 1.0)
        eps = torch.clamp(eps, min=0.0)
    else:
        raise ValueError("mode must be 'relative', 'floor', or 'condition'")

    return eps


def apply_init(model: nn.Module, init_name: str):
    """Initialise all linear/conv layers with the requested scheme."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)

                if init_name == "kaiming_uniform":
                    limit = math.sqrt(1.0 / fan_in)
                    nn.init.uniform_(m.weight, a=-limit, b=limit)
                elif init_name == "norm_preservation":
                    std = math.sqrt(1.0 / fan_out)
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                elif init_name == "lecun_normal":
                    std = math.sqrt(1.0 / fan_in)
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                else:
                    raise ValueError(f"Unknown init: {init_name}")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ============================================================
# BACKPROPAGATION MODEL
# Supports unscaled BP and covariance-rescaled BP.
# ============================================================

class BP_MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        nm_layers=2,
        bias=False,
        scale_method=None,
    ):
        super().__init__()
        self.nm_layers = nm_layers
        self.layers = nn.ModuleList()

        for _ in range(nm_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            input_dim = hidden_dim

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

        self._hook_handles = []
        self.scalings = None
        self.z0 = None

        scale_methods = {
            "norm": self.get_activity_norm,
            "covariance": self.get_activity_covariance,
        }

        if scale_method is not None:
            self.get_activity = scale_methods[scale_method]
        else:
            self.get_activity = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, x, y):
        batch_size = x.shape[0]
        y_hat = self.forward(x)
        error = 0.5 * nn.MSELoss(reduction="sum")(y_hat, y) / batch_size

        if self.get_activity is not None:
            self.get_activity(x, y)
            self.register_hook()

        return error, y_hat

    def get_activities(self, x, y):
        del y  # unused; kept for interface symmetry
        x_fp = x.clone()
        self.z0 = [x_fp]

        for layer in self.layers[:-1]:
            x_fp = layer(x_fp)
            self.z0.append(x_fp)

    def get_activity_norm(self, x, y):
        self.get_activities(x, y)

        self.scalings = [
            (
                1.0 / ((z0 * z0).sum(-1).mean().item() + epsilon)
            )
            * torch.eye(
                z0.shape[-1],
                device=z0.device,
                dtype=z0.dtype,
            )
            for z0 in self.z0
        ]

    def get_activity_covariance(self, x, y):
        self.get_activities(x, y)

        covariances = [
            z.double().T @ z.double() / z.shape[0]
            for z in self.z0
        ]

        self.scalings = []

        for cov in covariances:
        #    eps = choose_epsilon_from_eigs(
        #        cov,
        #        mode="floor",
        #        alpha=alpha,
        #    )
        #    reg_cov = cov + eps * torch.eye(
        #        cov.shape[0],
        #        device=cov.device,
        #        dtype=cov.dtype,
        #    )
            self.scalings.append(stable_inverse(cov).float())

    def register_hook(self):
        self.remove_hooks()

        for layer_idx, layer in enumerate(self.layers):
            handle = layer.weight.register_hook(
                lambda grad, idx=layer_idx: grad @ self.scalings[idx]
            )
            self._hook_handles.append(handle)

    def remove_hooks(self):
        for h in getattr(self, "_hook_handles", []):
            h.remove()
        self._hook_handles = []


# ============================================================
# PREDICTIVE-CODING MODEL
# Supports unscaled PC and covariance-rescaled PC.
# ============================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

#def stable_inverse(matrix, eps=1e-6):
#    """Regularised inverse of a square matrix."""
#    I = torch.eye(
#        matrix.shape[-1],
#        device=matrix.device,
#        dtype=matrix.dtype,
#    )
#    return torch.linalg.inv(matrix + eps * I)

#def stable_inverse(matrix, eps=1e-6):
#    return matrix


class PC_MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        nm_layers=2,
        bias=False,
        scale_method=None,
        iters=100,
        e_lr=0.05,
    ):
        super().__init__()

        self.nm_layers = nm_layers
        self.iters = iters
        self.e_lr = e_lr

        self.layers = nn.ModuleList()

        for _ in range(nm_layers):
            self.layers.append(
                nn.Linear(input_dim, hidden_dim, bias=bias)
            )
            input_dim = hidden_dim

        self.layers.append(
            nn.Linear(input_dim, output_dim, bias=bias)
        )

        self._hook_handles = []

        self.scalings = None
        self.z0 = None
        self.z = None
        self.errors = None

        scale_methods = {
            "norm": self.get_activity_norm,
            "covariance": self.get_activity_covariance,
        }

        if scale_method is not None:
            self.get_activity = scale_methods[scale_method]
        else:
            self.get_activity = None

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y is None:
            self.init_activities(x)
        else:
            self.minimize_error_energy(x, y)

        return self.y_pred(x)

    def output_loss(self, y_hat, y):
        return 0.5 * F.mse_loss(
            y_hat, y, reduction="sum"
        )

    def y_pred(self, x):
        """Prediction using current hidden activities."""
        if self.z is None:
            self.init_activities(x)

        return self.layers[-1](self.z[-1])

    @torch.no_grad()
    def init_activities(self, x):
        """Initialise hidden activities at feedforward values."""
        z = [x.detach()]
        s_i = x.detach()

        for layer in self.layers[:-1]:
            s_i = layer(s_i).detach()
            z.append(s_i)

        self.z0 = [
            s.detach().clone() for s in z
        ]

        self.z = z

    def E(self, x, y):
        """Standard PC energy."""
        E = 0.0
        errors = []

        for i, layer in enumerate(self.layers[:-1]):
            prediction = layer(self.z[i])
            error = self.z[i + 1] - prediction

            errors.append(error)
            E = E + 0.5 * torch.sum(error ** 2)

        y_hat = self.layers[-1](self.z[-1])
        output_error = y - y_hat

        errors.append(output_error)

        E = E + 0.5 * torch.sum(output_error ** 2)

        self.errors = errors

        return E

    def E_local(self, x, y):
        """Local energy for weight learning."""
        E = 0.0

        for i, layer in enumerate(self.layers[:-1]):
            z_prev = self.z[i].detach()
            z_next = self.z[i + 1].detach()

            prediction = layer(z_prev)

            E = E + 0.5 * F.mse_loss(
                prediction,
                z_next,
                reduction="sum",
            )

        y_hat = self.layers[-1](
            self.z[-1].detach()
        )

        E = E + self.output_loss(y_hat, y)

        return E

    def minimize_error_energy(self, x, y):
        """Run exactly self.iters activity inference steps."""
        self.init_activities(x)

        params = list(self.layers.parameters())
        old_flags = [p.requires_grad for p in params]

        try:
            for p in params:
                p.requires_grad_(False)

            with torch.enable_grad():
                z_hidden = [
                    z_i.detach().requires_grad_(True)
                    for z_i in self.z[1:]
                ]

                for _ in range(self.iters):
                    self.z = [x.detach()] + z_hidden

                    E = self.E(x, y)

                    grads = torch.autograd.grad(
                        E,
                        z_hidden,
                    )

                    z_hidden = [
                        (
                            z_i - self.e_lr * grad_i
                        ).detach().requires_grad_(True)
                        for z_i, grad_i in zip(
                            z_hidden, grads
                        )
                    ]

            self.z = [x.detach()] + [
                z_i.detach() for z_i in z_hidden
            ]

        finally:
            for p, flag in zip(params, old_flags):
                p.requires_grad_(flag)

        # Store errors at the final inferred state.
        with torch.no_grad():
            self.E(x, y)

    def loss(self, x, y):
        batch_size = x.shape[0]

        # 1. Standard PC inference.
        self.minimize_error_energy(x, y)

        # 2. Local weight-learning objective.
        E_final = self.E_local(x, y)
        loss = E_final / batch_size

        # 3. Optional gradient rescaling.
        self.remove_hooks()

        if self.get_activity is not None:
            self.get_activity(x)
            self.register_hook()

        # 4. Prediction from inferred activities.
        with torch.no_grad():
            y_hat = self.y_pred(x)

        return loss, y_hat

    def get_activities(self, x):
        """Feedforward and relaxed activities."""
        if self.z0 is None or self.z is None:
            raise RuntimeError(
                "Run inference before computing activities."
            )

        self.z0 = [
            z.detach() for z in self.z0
        ]

        self.z = [
            z.detach() for z in self.z
        ]

    def get_activity_norm(self, x):
        self.get_activities(x)

        #epsilon = 1e-8

        self.scalings = [
            (
                1.0
                / (
                    (z * z0).sum(-1).mean().item()
                  #  + epsilon
                )
            )
            * torch.eye(
                z.shape[-1],
                device=z.device,
                dtype=z.dtype,
            )
            for z, z0 in zip(self.z, self.z0)
        ]

    def get_activity_covariance(self, x):
        """Cross-covariance gradient rescaling."""
        self.get_activities(x)

        covariances = [
            z0.double().T @ z.double() / z.shape[0]
            for z0, z in zip(self.z0, self.z)
        ]

        self.scalings = []

        for cov in covariances:
            self.scalings.append(
                stable_inverse(cov).to(
                    device=self.z[0].device,
                    dtype=self.z[0].dtype,
                )
            )

    def register_hook(self):
        self.remove_hooks()

        for layer_idx, layer in enumerate(self.layers):
            handle = layer.weight.register_hook(
                lambda grad, idx=layer_idx:
                    grad @ self.scalings[idx]
            )
            self._hook_handles.append(handle)

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()

        self._hook_handles = []

# ============================================================
# REPRODUCIBILITY / DATA HELPERS
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
    """
    Create train/test data with

        y = W_data x

    where W_data has controlled rank <= rank_data.
    """
    g = torch.Generator().manual_seed(seed)

    A = torch.randn(
        rank_data,
        input_dim,
        generator=g,
    ) / math.sqrt(input_dim)

    B = torch.randn(
        output_dim,
        rank_data,
        generator=g,
    ) / math.sqrt(rank_data)

    W_data = B @ A

    X_train = torch.randn(
        n_train,
        input_dim,
        generator=g,
    )

    X_test = torch.randn(
        n_test,
        input_dim,
        generator=g,
    )

    Y_train = X_train @ W_data.T
    Y_test = X_test @ W_data.T

    return X_train, Y_train, X_test, Y_test, W_data


def make_batches(X, Y, steps, seed):
    """Pre-sample minibatches so all conditions see the same data."""
    g = torch.Generator().manual_seed(seed)

    idx = torch.randint(
        0,
        X.size(0),
        (steps, batch_size),
        generator=g,
    )

    batches = []

    for t in range(steps):
        xb = X[idx[t]].to(device)
        yb = Y[idx[t]].to(device)
        batches.append((xb, yb))

    return batches


# ============================================================
# TRAINING / TARGET-ALIGNMENT HELPERS
# ============================================================

def train_step(model, xb, yb, optimizer):
    """Perform one network-weight update."""
    optimizer.zero_grad(set_to_none=True)

    loss, _ = model.loss(xb, yb)
    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def target_alignment(y, y_pre, y_post, eps=1e-12):
    r"""
    Target alignment:

                       r^T Delta y_hat
        TA = -----------------------------------------
             ||r||_2 ||Delta y_hat||_2

    where r = y - y_pre and Delta y_hat = y_post - y_pre.
    """
    r = y - y_pre
    delta_y = y_post - y_pre

    numerator = (r * delta_y).sum(dim=-1)
    r_norm = torch.linalg.vector_norm(r, dim=-1)
    delta_norm = torch.linalg.vector_norm(delta_y, dim=-1)
    denominator = r_norm * delta_norm

    valid = denominator > eps

    if not valid.any():
        return float("nan")

    alignment = numerator[valid] / denominator[valid]
    return alignment.mean().item()


def copy_linear_weights(source_model, target_model):
    """Copy linear-layer parameters by layer order."""
    source_layers = [
        layer for layer in source_model.modules()
        if isinstance(layer, nn.Linear)
    ]
    target_layers = list(target_model.layers)

    assert len(source_layers) == len(target_layers), (
        "Source and target models must have the same number of linear layers."
    )

    with torch.no_grad():
        for source_layer, target_layer in zip(source_layers, target_layers):
            assert source_layer.weight.shape == target_layer.weight.shape, (
                "Source and target weight shapes differ."
            )

            target_layer.weight.copy_(source_layer.weight)

            if target_layer.bias is not None:
                if source_layer.bias is not None:
                    target_layer.bias.copy_(source_layer.bias)
                else:
                    target_layer.bias.zero_()


# ============================================================
# MAIN EXPERIMENT
# PC RESCALED vs PC UNSCALED vs BP RESCALED vs BP UNSCALED
# ============================================================

alignment_rows = []

print(f"Using device: {device}")
print(f"Relaxation steps: {iters_values}")
print(f"Fixed weight learning rate: {weight_lr}")
print(f"Trials per seed: {alignment_trials}")

pc_conditions = {
    "PC (rescaled)": "covariance",
    "PC (unscaled)": None,
}

bp_conditions = {
    "BP (rescaled)": "covariance",
    "BP (unscaled)": None,
}

for seed in seeds:
    print(f"\n========== seed {seed} ==========")
    set_seed(seed)

    # --------------------------------------------------------
    # Synthetic data
    # --------------------------------------------------------

    X_train, Y_train, X_test, Y_test, W_data = make_synthetic(
        seed=seed,
        rank_data=15,
    )

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    # --------------------------------------------------------
    # Common initialization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fixed minibatches for this seed
    # --------------------------------------------------------

    batches = make_batches(
        X_train,
        Y_train,
        steps=alignment_trials,
        seed=seed + 999,
    )

    for trial, (xb, yb) in enumerate(batches):

        # ====================================================
        # BACKPROPAGATION: RESCALED AND UNSCALED
        # ====================================================

        for bp_condition_name, bp_scale_method in bp_conditions.items():
            bp_model = BP_MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                nm_layers=nm_layers,
                bias=False,
                scale_method=bp_scale_method,
            ).to(device)

            # Exact same initial weights as PC for this seed.
            copy_linear_weights(base_model, bp_model)

            bp_optimizer = optimizer_cls(
                bp_model.parameters(),
                lr=weight_lr,
                weight_decay=weight_decay,
                **optimizer_kwargs,
            )

            # Prediction before BP update.
            with torch.no_grad():
                bp_y_pre = bp_model(xb).detach().clone()

            # One BP weight update. For BP (rescaled), loss()
            # installs the covariance-rescaling hooks before backward().
            bp_optimizer.zero_grad(set_to_none=True)
            bp_loss, _ = bp_model.loss(xb, yb)
            bp_loss.backward()
            bp_optimizer.step()

            # Prediction after BP update.
            with torch.no_grad():
                bp_y_post = bp_model(xb).detach().clone()

            bp_ta = target_alignment(
                y=yb,
                y_pre=bp_y_pre,
                y_post=bp_y_post,
            )

            alignment_rows.append(
                {
                    "seed": seed,
                    "trial": trial,
                    "iters": 0,
                    "condition": bp_condition_name,
                    "target_alignment": bp_ta,
                    "loss": bp_loss.item(),
                    "weight_lr": weight_lr,
                    "e_lr": e_lr,
                    "hidden_dim": hidden_dim,
                }
            )

            print(
                f"{bp_condition_name:15s}  "
                f"trial={trial:02d}  "
                f"TA={bp_ta: .6f}"
            )

        # ====================================================
        # PREDICTIVE CODING: RESCALED AND UNSCALED
        # ====================================================

        for n_iters in iters_values:
            for condition_name, scale_method in pc_conditions.items():
                model_kwargs = dict(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    nm_layers=nm_layers,
                    iters=n_iters,
                    e_lr=e_lr,
                )

                if scale_method is not None:
                    model_kwargs["scale_method"] = scale_method

                model = PC_MLP(**model_kwargs).to(device)

                # Exact same initial weights for every condition / iters value.
                model.load_state_dict(base_state, strict=True)

                optimizer = optimizer_cls(
                    model.parameters(),
                    lr=weight_lr,
                    weight_decay=weight_decay,
                    **optimizer_kwargs,
                )

                # Feedforward prediction before update.
                with torch.no_grad():
                    y_pre = model.forward(xb).detach().clone()

                # One PC weight update.
                loss = train_step(
                    model,
                    xb,
                    yb,
                    optimizer,
                )

                # Feedforward prediction after update.
                with torch.no_grad():
                    y_post = model.forward(xb).detach().clone()

                ta = target_alignment(
                    y=yb,
                    y_pre=y_pre,
                    y_post=y_post,
                )

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
                    f"{condition_name:15s}  "
                    f"trial={trial:02d}  "
                    f"iters={n_iters:4d}  "
                    f"TA={ta: .6f}"
                )


# ============================================================
# SAVE RAW RESULTS
# ============================================================

df_alignment = pd.DataFrame(alignment_rows)

raw_csv_path = base_dir / "target_alignment_vs_relaxation_raw.csv"
df_alignment.to_csv(raw_csv_path, index=False)


# ============================================================
# AGGREGATE ACROSS SEEDS
# ============================================================

# First average trials within each seed.
seed_means = (
    df_alignment
    .groupby(["condition", "iters", "seed"])["target_alignment"]
    .mean()
    .reset_index()
)

# Then aggregate across independent seeds.
agg_alignment = (
    seed_means
    .groupby(["condition", "iters"])["target_alignment"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

agg_alignment["sem"] = (
    agg_alignment["std"]
    / np.sqrt(agg_alignment["count"])
)

summary_csv_path = base_dir / "target_alignment_vs_relaxation_summary.csv"
agg_alignment.to_csv(summary_csv_path, index=False)

print("\n========================================")
print("Target alignment vs relaxation steps")
print("========================================")
print(agg_alignment.to_string(index=False))


# ============================================================
# PLOT TARGET ALIGNMENT VS RELAXATION STEPS
# ============================================================

fig, ax = plt.subplots(figsize=(3.5, 2.6))

pc_colors = {
    "PC (rescaled)": "tab:blue",
    "PC (unscaled)": "tab:orange",
}

# ------------------------------------------------------------
# Plot PC curves
# ------------------------------------------------------------

for condition_name, color in pc_colors.items():
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

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=1.8,
        markersize=4,
        color=color,
        label=condition_name,
    )

    ax.fill_between(
        x,
        np.maximum(y - sem, -1.0),
        np.minimum(y + sem, 1.0),
        color=color,
        alpha=0.2,
    )


# ============================================================
# PLOT BOTH BACKPROPAGATION BASELINES
# ============================================================

bp_plot_conditions = {
    "BP (rescaled)": {
        "color": "tab:green",
        "linestyle": "-.",
    },
    "BP (unscaled)": {
        "color": "tab:red",
        "linestyle": ":",
    },
}

for condition_name, style in bp_plot_conditions.items():
    bp_rows = agg_alignment[
        agg_alignment["condition"] == condition_name
    ]

    if bp_rows.empty:
        continue

    bp_row = bp_rows.iloc[0]
    bp_mean = float(bp_row["mean"])
    bp_sem = (
        float(bp_row["sem"])
        if pd.notna(bp_row["sem"])
        else 0.0
    )

    ax.axhline(
        y=bp_mean,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=1.8,
        label=condition_name,
    )

    ax.axhspan(
        max(bp_mean - bp_sem, -1.0),
        min(bp_mean + bp_sem, 1.0),
        color=style["color"],
        alpha=0.10,
    )


# ============================================================
# AXIS FORMATTING
# ============================================================

# symlog preserves the intended log-like spacing while allowing iters=0.
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlim(0,1)
ax.set_ylim(0, 1.05)

ax.axhline(
    y=0.0,
    linestyle=":",
    color="grey",
    linewidth=1.0,
)

ax.axhline(
    y=1.0,
    linestyle="--",
    color="grey",
    linewidth=0.8,
    alpha=0.6,
)

ax.set_xlabel("PC relaxation steps")
ax.set_ylabel("Target alignment")

ax.legend(frameon=False)
ax.grid(True, alpha=0.25)

for spine in ax.spines.values():
    spine.set_color("#CCCCCC")
    spine.set_linewidth(0.8)


# ============================================================
# SAVE FIGURE AS SVG
# ============================================================

plt.tight_layout()

figure_path = os.path.join(
    script_dir,
    "target_alignment_vs_relaxation.svg",
)

plt.savefig(
    figure_path,
    format="svg",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# PRINT OUTPUT LOCATIONS
# ============================================================

print("\nSaved:")
print(f"  raw data: {raw_csv_path}")
print(f"  summary:  {summary_csv_path}")
print(f"  figure:   {figure_path}")
