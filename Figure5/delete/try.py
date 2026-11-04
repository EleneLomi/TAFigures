
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

# ============================================================
# Data
# ============================================================

x = torch.tensor(1.0)
target = torch.tensor([-1.0, 1.0])

epsilon = 0# 1e-8


# ============================================================
# Scaling function
# ============================================================

def activity_scaling(activity, method=None):
    if method is None:
        return torch.ones_like(activity)

    if method in ("norm", "covariance"):
        return 1.0 / (activity.pow(2) + epsilon)

    raise ValueError(f"Unknown scaling method: {method}")


# ============================================================
# BACKPROPAGATION
# ============================================================

def train_bp(
    x,
    target,
    n_steps=500,
    lr=0.01,
    scale_method=None,
):
    w_in = torch.tensor(1.0, requires_grad=True)
    w_out = torch.ones(2, requires_grad=True)

    trajectory = []

    for step in range(n_steps):

        # Forward pass
        h = w_in * x
        y_hat = w_out * h

        trajectory.append(y_hat.detach().clone())

        # Reconstruction / prediction loss
        loss = 0.5 * ((y_hat - target) ** 2).sum()

        # Backpropagation
        loss.backward()

        # Scaling factors
        with torch.no_grad():
            S_in = activity_scaling(x, scale_method)
            S_out = activity_scaling(h.detach(), scale_method)

            # Scaled weight updates
            w_in -= lr * w_in.grad * S_in
            w_out -= lr * w_out.grad * S_out

            # Clear gradients
            w_in.grad.zero_()
            w_out.grad.zero_()

    return torch.stack(trajectory)


# ============================================================
# PREDICTIVE CODING
# ============================================================

def train_pc(
    x,
    target,
    n_steps=500,
    weight_lr=0.01,
    inference_lr=0.1,
    inference_steps=50,
    scale_method=None,
):
    w_in = torch.tensor(1.0)
    w_out = torch.ones(2)

    trajectory = []

    for step in range(n_steps):

        # 1. Feedforward prediction
        h_ff = w_in * x
        y_hat = w_out * h_ff

        trajectory.append(y_hat.clone())

        # 2. Initialise hidden activity
        h = h_ff.clone()

        # 3. Inference
        for _ in range(inference_steps):

            e_h = h - w_in * x
            e_y = target - w_out * h

            grad_h = e_h - torch.sum(w_out * e_y)

            h = h - inference_lr * grad_h

        # 4. Prediction errors after inference
        e_h = h - w_in * x
        e_y = target - w_out * h

        # 5. Activity-based scaling
        S_in = activity_scaling(x, scale_method)
        S_out = activity_scaling(h, scale_method)

        # 6. Scaled local PC updates
        w_in = w_in + weight_lr * S_in * e_h * x
        w_out = w_out + weight_lr * S_out * e_y * h

    return torch.stack(trajectory)


# ============================================================
# Experiments
# ============================================================

n_steps = 1000
lr = 0.01

bp_trajectory = train_bp(
    x, target,
    n_steps=n_steps,
    lr=lr,
    scale_method=None,
)

bp_scaled_trajectory = train_bp(
    x, target,
    n_steps=n_steps,
    lr=lr,
    scale_method="norm",
)

inference_steps_list = [1, 2, 5, 10, 20, 50, 100, 300]

pc_trajectories = {}
pc_scaled_trajectories = {}

for n_inf in inference_steps_list:

    pc_trajectories[n_inf] = train_pc(
        x, target,
        n_steps=n_steps,
        weight_lr=lr,
        inference_lr=0.1,
        inference_steps=n_inf,
        scale_method=None,
    )

    pc_scaled_trajectories[n_inf] = train_pc(
        x, target,
        n_steps=n_steps,
        weight_lr=lr,
        inference_lr=0.1,
        inference_steps=n_inf,
        scale_method="norm",
    )


# ============================================================
# Plotting
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

for ax, trajectories, bp_traj, title in [
    (axes[0], pc_trajectories, bp_trajectory, "Unscaled"),
    (axes[1], pc_scaled_trajectories, bp_scaled_trajectory, "Activity-scaled"),
]:

    ax.plot(
        bp_traj[:, 0],
        bp_traj[:, 1],
        label="BP",
        linewidth=3,
        color="black",
    )

    for n_inf, trajectory in trajectories.items():
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            linewidth=1.8,
            label=f"PC ({n_inf} steps)",
        )

    ax.scatter(1, 1, marker="o", s=80, label="Initialisation", zorder=10)
    ax.scatter(-1, 1, marker="x", s=120, linewidth=2,
               label="Target", zorder=10)

    ax.plot([-1, 1], [1, 1], "--", linewidth=1, label="Ideal")

    ax.set_xlabel(r"$\hat{y}_1$")
    ax.set_ylabel(r"$\hat{y}_2$")
    ax.set_title(title)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0.0, 1.15)
    ax.grid(True, alpha=0.25)

axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()
