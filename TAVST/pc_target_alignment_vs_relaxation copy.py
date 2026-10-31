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
alpha = 0#1e-5


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Use several seeds for a more reliable mean/SEM.
# For a very quick test, use: seeds = [0]
seeds = [0, 1, 2, 3, 4]

# Synthetic-data dimensions, matching your existing Figure 4 setup.
input_dim = 256
output_dim = 256
n_train = 7096
n_test = 2048
batch_size = 100 #20#64

# Network architecture.
hidden_dim = 256
nm_layers = 5

# Predictive-coding relaxation hyperparameters.
# e_lr is the relaxation/error-state learning rate inside PC inference.
e_lr = .01#0.01

# This is the x-axis of the experiment.
iters_values = [0,1,2,5,50,200,300,500,5000]#, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Number of different minibatches tested for each seed.
# Every iters value sees the SAME minibatch within a trial.
alignment_trials = 3#20

# IMPORTANT: keep the weight learning rate fixed across iters.
# Replace this with the best LR you want to use for PC (scaled lr),
# if you already know it from your LR sweep.
weight_lr = 1e-2

# Optimizer for the network weights.
optimizer_cls = torch.optim.SGD
optimizer_kwargs = dict(momentum=0.0)
weight_decay = 0.0

# Initialization used in your current Figure 4 code.
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
    """
    Initializations copied from your Figure 4 setup.
    """
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
# RESCALED PC MODEL
# ============================================================

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
        self.errors = None

        self.layers = nn.ModuleList()

        for _ in range(nm_layers):
            self.layers.append(
                nn.Linear(input_dim, hidden_dim, bias=bias)
            )
           # self.layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.layers.append(
            nn.Linear(hidden_dim, output_dim, bias=bias)
        )

        self._hook_handles = []
        self.scalings = None
        self.z0 = None
        self.z = None

        scale_methods = {
            "norm": self.get_activity_norm,
            "covariance": self.get_activity_covariance,
        }

        if scale_method is not None:
            self.get_activity = scale_methods[scale_method]
        else:
            self.get_activity = None

    #def forward(self, x):
    #    """
    #    Pure feedforward prediction W_{L:1} x.
    #    This is the prediction used in the target-alignment definition.
    #    """
    #    for layer in self.layers:
    #        x = layer(x)
    #    return x
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:#(self, x: torch.Tensor) -> torch.Tensor:
        #x = self._prepare_x(x)

        if y is None:
            self.init_zero_errors(x)
        else:
            self.minimize_error_energy(x, y)

        return self.y_pred(x)
    # ADDED ABOVE

    def output_loss(self, y_hat, y):
        return 0.5 * F.mse_loss(y_hat, y, reduction="sum")

    def y_pred(self, x):
        """
        Prediction using the current PC error states.
        """
        s_i = x

        for e_i, layer_i in zip(self.errors + [0.0], self.layers):
            s_i = e_i + layer_i(s_i)

        return s_i

    def E(self, x, y):
        """
        Energy minimized during PC relaxation.
        """
        E_errors = 0.5 * sum(torch.sum(e ** 2) for e in self.errors)
        return E_errors + self.output_loss(self.y_pred(x), y)

    def E_local(self, x, y):
        """
        Local energy used to obtain weight gradients after relaxation.
        """
        E = 0.0
        s_i = x

        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)
            s_i = (e_i + s_i_pred).detach()
            E += 0.5 * F.mse_loss(
                s_i_pred,
                s_i,
                reduction="sum",
            )

        y_hat = self.layers[-1](s_i)
        return E + self.output_loss(y_hat, y)

    @torch.no_grad()
    def init_zero_errors(self, x):
        tmp = x
        errors = []

        for layer_i in self.layers[:-1]:
            tmp = layer_i(tmp)
            err = torch.zeros_like(tmp, requires_grad=True)
            errors.append(err)
            tmp = tmp.detach()

        self.errors = errors

    def minimize_error_energy(self, x, y):
        """
        Run self.iters relaxation steps while keeping weights fixed.
        """
        for p in self.layers.parameters():
            p.requires_grad_(False)

        self.init_zero_errors(x)

        error_optim = torch.optim.SGD(
            self.errors,
            lr=self.e_lr,
        )

        for _ in range(self.iters):
            error_optim.zero_grad()
            E = self.E(x, y)
            E.backward()
            error_optim.step()

        for p in self.layers.parameters():
            p.requires_grad_(True)

    def loss(self, x, y):
        batch_size = x.shape[0]

        # 1) Relax PC error states for exactly self.iters steps.
        self.minimize_error_energy(x, y)

        # 2) Compute local weight-learning objective.
        E_final = self.E_local(x, y)
        loss = E_final / batch_size

        # 3) Build rescaling matrices from feedforward/relaxed activities.
        #    These hooks rescale weight gradients during loss.backward().
        if self.get_activity is not None:
            self.get_activity(x)
            self.register_hook()

        y_hat = self.y_pred(x)
        return loss, y_hat

    def get_activities(self, x):
        # Feedforward activities z0.
        z0 = []
        s_i = x.clone()
        z0.append(s_i)

        for layer in self.layers[:-1]:
            s_i = layer(s_i)
            z0.append(s_i)

        self.z0 = z0

        # Relaxed activities z.
        z = []
        s_i = x
        z.append(s_i)

        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)
            s_i = (e_i + s_i_pred).detach()
            z.append(s_i)

        self.z = z

    def get_activity_norm(self, x):
        self.get_activities(x)

        self.scalings = [
            (
                1.0
                / ((z * z0).sum(-1).mean().item() + epsilon)
            )
            * torch.eye(
                z.shape[-1],
                device=z.device,
                dtype=z.dtype,
            )
            for z, z0 in zip(self.z, self.z0)
        ]

    def get_activity_covariance(self, x):
        """
        Covariance-based rescaling from your original PC_MLP.
        """
        self.get_activities(x)

        covariances = [
            z0.double().T @ z.double() / z.shape[0]
            #identity matrix shapes as z.double().T @ z0.double() / (z.shape[0] * z.shape[1])
            #torch.eye(z.shape[1], device=z.device, dtype=z.dtype)
            #z.double().T @ z0.double() / z.shape[0]
            for z0, z in zip(self.z0, self.z)
        ]

        self.scalings = []

        for cov in covariances:
            #eps = choose_epsilon_from_eigs(
            #    cov,
            #    mode="floor",
            #    alpha=alpha,
            #)

            reg_cov = cov #+ eps * torch.eye(
            #    cov.shape[0],
            #    device=cov.device,
            #    dtype=cov.dtype,
            #)

            self.scalings.append(
                stable_inverse(reg_cov).float()
            )

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
    """
    Pre-sample minibatches so every value of iters sees exactly
    the same minibatches.
    """
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
    """
    One network-weight update.

    For PC_MLP, model.loss() first runs `model.iters` relaxation
    steps, then backward() computes the weight gradients, and
    optimizer.step() applies the weight update.
    """
   # model.train()
    optimizer.zero_grad(set_to_none=True)

    loss, _ = model.loss(xb, yb)
    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def target_alignment(y, y_pre, y_post, eps=1e-12):
    r"""
    Target alignment from the definition in your screenshot:

        r = y - y_pre
        Delta y_hat = y_post - y_pre

                         r^T Delta y_hat
        TA = -----------------------------------------
             ||r||_2 ||Delta y_hat||_2

    We compute the cosine separately for every example in the
    minibatch and then average over the minibatch.
    """
    r = y - y_pre
    delta_y = y_post - y_pre

    numerator = (r * delta_y).sum(dim=-1)

    r_norm = torch.linalg.vector_norm(r, dim=-1)
    delta_norm = torch.linalg.vector_norm(delta_y, dim=-1)

    denominator = r_norm * delta_norm

    # Avoid undefined 0/0 values if an update happens to cause
    # essentially no prediction change for a sample.
    valid = denominator > eps

    if not valid.any():
        return float("nan")

    alignment = numerator[valid] / denominator[valid]

    return alignment.mean().item()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

alignment_rows = []

print(f"Using device: {device}")
print(f"Relaxation steps: {iters_values}")
print(f"Fixed weight learning rate: {weight_lr}")
print(f"Trials per seed: {alignment_trials}")

for seed in seeds:
    print(f"\n========== seed {seed} ==========")

    set_seed(seed)

    # Same synthetic-data convention as your original Figure 4 code:
    # rank_data = 15.
    X_train, Y_train, X_test, Y_test, W_data = make_synthetic(
        seed=seed,
        rank_data=15,
    )

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    # --------------------------------------------------------
    # One common initialization for this seed.
    # Every value of iters starts from this exact state.
    # --------------------------------------------------------
    base_model = PC_MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        nm_layers=nm_layers,
        # scale_method="covariance",  # <-- RESCALED MODEL
        iters=1,
        e_lr=e_lr,
    ).to(device)

    apply_init(base_model, init_name)

    base_state = {
        k: v.detach().clone()
        for k, v in base_model.state_dict().items()
    }

    # Fixed set of minibatches for this seed.
    batches = make_batches(
        X_train,
        Y_train,
        steps=alignment_trials,
        seed=seed + 999,
    )

    for trial, (xb, yb) in enumerate(batches):
        for n_iters in iters_values:

            # ------------------------------------------------
            # Fresh model for this condition.
            # This is crucial: every n_iters value starts from
            # exactly the same W_pre for this seed/trial.
            # ------------------------------------------------
            model = PC_MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                nm_layers=nm_layers,
                scale_method="covariance",  # <-- rescaling ON
                iters=n_iters,
                e_lr=e_lr,
            ).to(device)

            model.load_state_dict(base_state, strict=True)

            optimizer = optimizer_cls(
                model.parameters(),
                lr=weight_lr,
                weight_decay=weight_decay,
                **optimizer_kwargs,
            )

            # ------------------------------------------------
            # Feedforward prediction BEFORE the weight update.
            # Use the SAME minibatch that generates the update.
            # ------------------------------------------------
            with torch.no_grad():
                y_pre = model.forward(xb).detach().clone()

            # ------------------------------------------------
            # ONE weight update after n_iters PC relaxation steps.
            # ------------------------------------------------
            loss = train_step(
                model,
                xb,
                yb,
                optimizer,
            )

            # ------------------------------------------------
            # Feedforward prediction AFTER the weight update.
            # ------------------------------------------------
            with torch.no_grad():
                y_post = model.forward(xb).detach().clone()

            # ------------------------------------------------
            # Target alignment of this one update.
            # ------------------------------------------------
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
                    "target_alignment": ta,
                    "loss": loss,
                    "weight_lr": weight_lr,
                    "e_lr": e_lr,
                    "hidden_dim": hidden_dim,
                }
            )

            print(
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
# AGGREGATE ACROSS SEEDS / TRIALS
# ============================================================

agg_alignment = (
    df_alignment
    .groupby("iters")["target_alignment"]
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

x = agg_alignment["iters"].to_numpy(dtype=float)
y = agg_alignment["mean"].to_numpy(dtype=float)
sem = agg_alignment["sem"].fillna(0.0).to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(4.2, 3.3))

ax.plot(
    x,
    y,
    marker="o",
    label="PC (rescaled)",
)

ax.fill_between(
    x,
    np.maximum(y - sem, -1.0),
    np.minimum(y + sem, 1.0),
    alpha=0.2,
)

# Log x-axis because relaxation-step values span several orders of magnitude.
ax.set_xscale("log")

# Target alignment is a cosine similarity.
ax.set_ylim(.8, 1.05)

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

plt.tight_layout()

figure_path = os.path.join(
    script_dir,
    "target_alignment_vs_relaxation.svg",
)

plt.savefig(
    figure_path,
    dpi=300,
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
