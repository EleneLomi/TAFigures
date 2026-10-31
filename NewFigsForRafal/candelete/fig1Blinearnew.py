import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# Data
# ============================================================

torch.manual_seed(0)

# Input x = 1
x = torch.tensor([[1.0]])

# Target = [-1, 1]
target = torch.tensor([[-1.0, 1.0]])


# ============================================================
# Backpropagation model
# ============================================================

class BP_MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=1, output_dim=2, bias=False):
        super().__init__()

        self.hidden = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.output = nn.Linear(hidden_dim, output_dim, bias=bias)

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
        input_dim=1,
        hidden_dim=1,
        output_dim=2,
        nm_layers=1,
        bias=False,
    ):
        super().__init__()

        self.nm_layers = nm_layers
        self.layers = nn.ModuleList()

        # Hidden layers
        for _ in range(nm_layers):
            self.layers.append(
                nn.Linear(input_dim, hidden_dim, bias=bias)
            )
            input_dim = hidden_dim

        # Output layer
        self.layers.append(
            nn.Linear(hidden_dim, output_dim, bias=bias)
        )

        # Initialise all weights to 1
        for layer in self.layers:
            nn.init.constant_(layer.weight, 1.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_with_variances(self, x, y):

        # Start with identity covariance at output
        variance = torch.eye(
            y.shape[-1],
            device=x.device,
            dtype=x.dtype
        )

        # Start from final-layer weights
        previous_activity = self.layers[-1].weight

        # Add variance contribution from final hidden layer
        variance = (
            variance
            + previous_activity @ previous_activity.T
        )

        # If there are extra hidden layers, propagate backwards
        for layer in self.layers[-2:0:-1]:

            previous_activity = (
                previous_activity @ layer.weight
            )

            variance = (
                variance
                + previous_activity @ previous_activity.T
            )

        # Full linear mapping from input to output
        previous_activity = (
            previous_activity @ self.layers[0].weight
        )

        # Mean prediction
        y_hat = x @ previous_activity.T

        # Precision = inverse covariance
        precision = torch.linalg.inv(variance)

        return y_hat, precision

    def loss(self, x, y):

        batch_size = x.shape[0]

        y_hat, precision = self.forward_with_variances(x, y)

        error = y - y_hat

        # Precision-weighted quadratic prediction error
        loss = (
            0.5
            * ((error @ precision) * error).sum()
            / batch_size
        )

        return loss, y_hat


# ============================================================
# Training functions
# ============================================================

def train_bp(model, x, y, lr=0.01, epochs=2000):

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    trajectory = []
    losses = []

    for epoch in range(epochs):

        optimizer.zero_grad()

        y_hat = model(x)

        # Save prediction before the update
        trajectory.append(
            y_hat.detach().squeeze().clone()
        )

        # Standard squared error
        error = y - y_hat

        loss = 0.5 * (error ** 2).sum()

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    trajectory = torch.stack(trajectory)

    return trajectory, losses


def train_pc(model, x, y, lr=0.01, epochs=2000):

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    trajectory = []
    losses = []

    for epoch in range(epochs):

        optimizer.zero_grad()

        loss, y_hat = model.loss(x, y)

        # Save prediction before the update
        trajectory.append(
            y_hat.detach().squeeze().clone()
        )

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    trajectory = torch.stack(trajectory)

    return trajectory, losses


# ============================================================
# Create models
# ============================================================

bp_model = BP_MLP(
    input_dim=1,
    hidden_dim=1,
    output_dim=2,
    bias=False,
)

pc_model = PC_MLP(
    input_dim=1,
    hidden_dim=1,
    output_dim=2,
    nm_layers=1,     # one hidden layer
    bias=False,
)


# ============================================================
# Check initial predictions
# ============================================================

print("Initial BP prediction:")
print(bp_model(x).detach())

print("\nInitial PC prediction:")
print(pc_model(x).detach())


# ============================================================
# Train
# ============================================================

epochs = 2000
lr = 0.01

bp_trajectory, bp_losses = train_bp(
    bp_model,
    x,
    target,
    lr=lr,
    epochs=epochs,
)

pc_trajectory, pc_losses = train_pc(
    pc_model,
    x,
    target,
    lr=lr,
    epochs=epochs,
)


# ============================================================
# Print final predictions
# ============================================================

print("\nFinal BP prediction:")
print(bp_model(x).detach())

print("\nFinal PC prediction:")
print(pc_model(x).detach())


# ============================================================
# Plot both trajectories
# ============================================================

plt.figure(figsize=(6, 5))


# Backpropagation trajectory
plt.plot(
    bp_trajectory[:, 0].numpy(),
    bp_trajectory[:, 1].numpy(),
    linewidth=2,
    label="BP",
)


# Predictive coding trajectory
plt.plot(
    pc_trajectory[:, 0].numpy(),
    pc_trajectory[:, 1].numpy(),
    linewidth=2,
    label="PC",
)


# Ideal trajectory:
# y2 stays fixed at 1 while y1 moves from 1 to -1
plt.plot(
    [1, -1],
    [1, 1],
    "--",
    linewidth=1.5,
    label="ideal",
)


# Initial point
plt.scatter(
    1,
    1,
    s=60,
    zorder=10,
)

plt.annotate(
    "initialisation",
    xy=(1, 1),
    xytext=(0.35, 1.06),
)


# Target point
plt.scatter(
    -1,
    1,
    marker="x",
    s=100,
    linewidth=2,
    zorder=10,
)

plt.annotate(
    "target",
    xy=(-1, 1),
    xytext=(-1.05, 0.93),
)


# Labels
plt.xlabel(r"$\hat{y}_1$", fontsize=13)
plt.ylabel(r"$\hat{y}_2$", fontsize=13)

plt.xlim(-1.1, 1.1)
plt.ylim(0.0, 1.15)

plt.legend()
plt.tight_layout()

plt.show()


# ============================================================
# Optional: plot training losses
# ============================================================

plt.figure(figsize=(6, 4))

plt.plot(bp_losses, label="BP loss")
plt.plot(pc_losses, label="PC loss")

plt.xlabel("Training step")
plt.ylabel("Loss")

plt.yscale("log")

plt.legend()
plt.tight_layout()

plt.show()