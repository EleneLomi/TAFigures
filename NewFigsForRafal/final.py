import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

torch.manual_seed(0)

epsilon = 1e-8

# Input and target
x = torch.tensor([[1.0]])
target = torch.tensor([[-1.0, 1.0]])

# Training settings
epochs = 2000
lr = 0.01


# ============================================================
# Backpropagation model
# ============================================================

class BP_MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=1, output_dim=2, bias=False):
        super().__init__()

        self.hidden = nn.Linear(
            input_dim,
            hidden_dim,
            bias=bias
        )

        self.output = nn.Linear(
            hidden_dim,
            output_dim,
            bias=bias
        )

        # Initialise all weights to 1
        nn.init.constant_(self.hidden.weight, 1.0)
        nn.init.constant_(self.output.weight, 1.0)

    def forward(self, x):
        h = self.hidden(x)
        y_hat = self.output(h)
        return y_hat


# ============================================================
# Predictive coding model
# ============================================================

class PC_MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        nm_layers=1,
        bias=False,
        scale_method=None,
    ):
        super().__init__()

        self.nm_layers = nm_layers

        self.layers = nn.ModuleList()

        # ----------------------------------------------------
        # Hidden layers
        # ----------------------------------------------------

        for _ in range(nm_layers):

            self.layers.append(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                    bias=bias
                )
            )

            input_dim = hidden_dim

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        self.layers.append(
            nn.Linear(
                hidden_dim,
                output_dim,
                bias=bias
            )
        )

        # ----------------------------------------------------
        # Initialise every weight to exactly 1
        # ----------------------------------------------------

        for layer in self.layers:
            nn.init.constant_(layer.weight, 1.0)

        # ----------------------------------------------------
        # Activity scaling
        # ----------------------------------------------------

        scale_methods = {
            "norm": self.get_activity_norm,
        }

        if scale_method is not None:
            self.get_activity = scale_methods[scale_method]
        else:
            self.get_activity = None

        self._hook_handles = []


    # ========================================================
    # Normal forward pass
    # ========================================================

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x


    # ========================================================
    # Forward pass with PC output covariance
    # ========================================================

    def forward_with_variances(self, x, y):

        # Start with identity output covariance
        variance = torch.eye(
            y.shape[-1],
            device=x.device,
            dtype=x.dtype
        )

        # Output weight matrix
        previous_activity = self.layers[-1].weight

        # Add final hidden layer contribution
        variance = (
            variance
            + previous_activity @ previous_activity.T
        )

        # Extra hidden layers, if present
        for layer in self.layers[-2:0:-1]:

            previous_activity = (
                previous_activity @ layer.weight
            )

            variance = (
                variance
                + previous_activity @ previous_activity.T
            )

        # Full input -> output weight product
        previous_activity = (
            previous_activity
            @ self.layers[0].weight
        )

        # Prediction
        y_hat = x @ previous_activity.T

        # ----------------------------------------------------
        # Stable precision calculation
        # ----------------------------------------------------

        try:

            L = torch.linalg.cholesky(variance)

        except RuntimeError:

            # If covariance becomes non-SPD,
            # return zero precision so learning stops
            return (
                y_hat,
                torch.zeros_like(variance)
            )

        precision = torch.cholesky_inverse(L)

        return y_hat, precision


    # ========================================================
    # Predictive-coding loss
    # ========================================================

    def loss(self, x, y):

        batch_size = x.shape[0]

        y_hat, precision = (
            self.forward_with_variances(x, y)
        )

        error = y - y_hat

        loss = (
            0.5
            * (
                (error @ precision)
                * error
            ).sum()
            / batch_size
        )

        # If activity scaling is enabled,
        # compute scaling matrices and install hooks
        if self.get_activity is not None:

            self.get_activity(
                x,
                y,
                y_hat,
                precision
            )

            self.register_hook()

        return loss, y_hat


    # ========================================================
    # Compute inferred activities
    # ========================================================

    def get_activities(
        self,
        x,
        y,
        y_hat,
        precision
    ):

        # Output residual
        r = y - y_hat

        # ----------------------------------------------------
        # Feedforward activities z0
        # ----------------------------------------------------

        x_fp = x.clone()

        self.z0 = [x_fp]

        for layer in self.layers[:-1]:

            x_fp = layer(x_fp)

            self.z0.append(x_fp)

        # ----------------------------------------------------
        # Products of downstream weights
        # ----------------------------------------------------

        weights_prod = [
            self.layers[-1].weight
        ]

        for layer in reversed(
            self.layers[1:-1]
        ):

            new_prod = (
                weights_prod[-1]
                @ layer.weight
            )

            weights_prod.append(new_prod)

        # ----------------------------------------------------
        # Prediction-error corrections
        # ----------------------------------------------------

        errors = [
            r @ precision @ wp
            for wp in weights_prod
        ]

        # ----------------------------------------------------
        # Inferred activities z
        # ----------------------------------------------------

        z = [x]

        for e, layer in zip(
            reversed(errors),
            self.layers[:-1]
        ):

            z_new = layer(z[-1]) + e

            z.append(z_new)

        self.z = z


    # ========================================================
    # Activity norm scaling
    # ========================================================

    def get_activity_norm(
        self,
        x,
        y,
        y_hat,
        precision
    ):

        self.get_activities(
            x,
            y,
            y_hat,
            precision
        )

        self.scalings = []

        for z, z0 in zip(
            self.z,
            self.z0
        ):

            # Scalar overlap / activity norm
            activity_scale = (
                (z * z0)
                .sum(-1)
                .mean()
                .item()
            )

            scaling = (
                1.0
                / (activity_scale + epsilon)
            )

            S = (
                scaling
                * torch.eye(
                    z.shape[-1],
                    device=z.device,
                    dtype=z.dtype
                )
            )

            self.scalings.append(S)


    # ========================================================
    # Gradient hooks
    # ========================================================

    def register_hook(self):

        self.remove_hooks()

        with torch.no_grad():

            for layer_idx, layer in enumerate(
                self.layers
            ):

                handle = (
                    layer.weight.register_hook(
                        lambda grad,
                        idx=layer_idx:
                        grad @ self.scalings[idx]
                    )
                )

                self._hook_handles.append(
                    handle
                )


    def remove_hooks(self):

        for h in getattr(
            self,
            "_hook_handles",
            []
        ):
            h.remove()

        self._hook_handles = []


# ============================================================
# Training functions
# ============================================================

def train_bp(
    model,
    x,
    y,
    lr=0.01,
    epochs=2000
):

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    trajectory = []
    losses = []

    for epoch in range(epochs):

        optimizer.zero_grad()

        y_hat = model(x)

        trajectory.append(
            y_hat.detach()
            .squeeze()
            .clone()
        )

        error = y - y_hat

        loss = (
            0.5
            * (error ** 2).sum()
        )

        losses.append(
            loss.item()
        )

        loss.backward()

        optimizer.step()

    return (
        torch.stack(trajectory),
        losses
    )


def train_pc(
    model,
    x,
    y,
    lr=0.01,
    epochs=2000
):

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    trajectory = []
    losses = []

    for epoch in range(epochs):

        optimizer.zero_grad()

        loss, y_hat = model.loss(
            x,
            y
        )

        trajectory.append(
            y_hat.detach()
            .squeeze()
            .clone()
        )

        losses.append(
            loss.item()
        )

        loss.backward()

        optimizer.step()

    return (
        torch.stack(trajectory),
        losses
    )


# ============================================================
# Create the three models
# ============================================================

bp_model = BP_MLP(
    input_dim=1,
    hidden_dim=1,
    output_dim=2,
    bias=False
)


pc_model = PC_MLP(
    input_dim=1,
    hidden_dim=1,
    output_dim=2,
    nm_layers=1,
    bias=False,
    scale_method=None
)


pc_norm_model = PC_MLP(
    input_dim=1,
    hidden_dim=1,
    output_dim=2,
    nm_layers=1,
    bias=False,
    scale_method="norm"
)


# ============================================================
# Check identical initial predictions
# ============================================================

print("Initial predictions")
print("-------------------")

print(
    "BP:",
    bp_model(x).detach().numpy()
)

print(
    "PC:",
    pc_model(x).detach().numpy()
)

print(
    "PC + norm:",
    pc_norm_model(x).detach().numpy()
)


# ============================================================
# Train all three
# ============================================================

bp_trajectory, bp_losses = train_bp(
    bp_model,
    x,
    target,
    lr=lr,
    epochs=epochs
)


pc_trajectory, pc_losses = train_pc(
    pc_model,
    x,
    target,
    lr=lr,
    epochs=epochs
)


pc_norm_trajectory, pc_norm_losses = train_pc(
    pc_norm_model,
    x,
    target,
    lr=lr,
    epochs=epochs
)


# ============================================================
# Final predictions
# ============================================================

print("\nFinal predictions")
print("-----------------")

print(
    "BP:",
    bp_model(x).detach().numpy()
)

print(
    "PC:",
    pc_model(x).detach().numpy()
)

print(
    "PC + norm:",
    pc_norm_model(x).detach().numpy()
)


# ============================================================
# Plot prediction trajectories
# ============================================================

plt.figure(figsize=(6, 5))


# ------------------------------------------------------------
# BP
# ------------------------------------------------------------

plt.plot(
    bp_trajectory[:, 0].numpy(),
    bp_trajectory[:, 1].numpy(),
    linewidth=2,
    label="BP"
)


# ------------------------------------------------------------
# PC
# ------------------------------------------------------------

plt.plot(
    pc_trajectory[:, 0].numpy(),
    pc_trajectory[:, 1].numpy(),
    linewidth=2,
    label="PC"
)


# ------------------------------------------------------------
# PC + activity norm scaling
# ------------------------------------------------------------

plt.plot(
    pc_norm_trajectory[:, 0].numpy(),
    pc_norm_trajectory[:, 1].numpy(),
    linewidth=2,
    label="PC + activity norm"
)


# ------------------------------------------------------------
# Ideal trajectory
# ------------------------------------------------------------

plt.plot(
    [1, -1],
    [1, 1],
    "--",
    linewidth=1.5,
    label="ideal"
)


# ------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------

plt.scatter(
    1,
    1,
    s=60,
    zorder=10
)

plt.annotate(
    "initialisation",
    xy=(1, 1),
    xytext=(0.35, 1.06)
)


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

plt.scatter(
    -1,
    1,
    marker="x",
    s=100,
    linewidth=2,
    zorder=10
)

plt.annotate(
    "target",
    xy=(-1, 1),
    xytext=(-1.05, 0.93)
)


# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

plt.xlabel(
    r"$\hat{y}_1$",
    fontsize=13
)

plt.ylabel(
    r"$\hat{y}_2$",
    fontsize=13
)

plt.xlim(
    -1.1,
    1.1
)

plt.ylim(
    0.0,
    1.15
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# Optional: compare the losses
# ============================================================

plt.figure(figsize=(6, 4))

plt.plot(
    bp_losses,
    label="BP"
)

plt.plot(
    pc_losses,
    label="PC"
)

plt.plot(
    pc_norm_losses,
    label="PC + activity norm"
)

plt.xlabel(
    "Training step"
)

plt.ylabel(
    "Loss"
)

plt.yscale(
    "log"
)

plt.legend()

plt.tight_layout()

plt.show()