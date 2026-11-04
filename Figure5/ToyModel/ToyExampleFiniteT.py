import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

torch.manual_seed(0)

# ============================================================
# Data
# ============================================================

x = torch.tensor(1.0)
target = torch.tensor([-1.0, 1.0])

EPS = 1e-8


# ============================================================
# Scaling functions
# ============================================================

def safe_inverse(value, eps=EPS):
    """
    Sign-preserving inverse, with protection near zero.
    """
    safe_value = torch.where(
        value >= 0,
        value.clamp(min=eps),
        value.clamp(max=-eps),
    )
    return 1.0 / safe_value


def bp_scaling(activity):
    """
    BP: inverse squared feedforward activity.
    """
    return safe_inverse(activity ** 2)


def pc_scaling(pre_activity, post_activity):
    """
    PC: inverse product of pre- and post-inference activities.
    """
    return safe_inverse(pre_activity * post_activity)


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
    """
    Network:
        x --> h --> [y1, y2]

    All weights initialized to 1.
    No biases.

    scale_method:
        None       : standard BP
        "norm"     : inverse squared activity scaling
        "covariance": same as norm for scalar activities
    """

    w_in = torch.tensor(1.0, requires_grad=True)
    w_out = torch.ones(2, requires_grad=True)

    trajectory = []

    for step in range(n_steps):

        # ----------------------------------------------------
        # 1. Forward pass
        # ----------------------------------------------------

        h = w_in * x
        y_hat = w_out * h

        trajectory.append(y_hat.detach().clone())

        # ----------------------------------------------------
        # 2. Squared error
        # ----------------------------------------------------

        loss = 0.5 * ((y_hat - target) ** 2).sum()

        # ----------------------------------------------------
        # 3. Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # 4. Scaling and weight updates
        # ----------------------------------------------------

        with torch.no_grad():

            if scale_method is None:
                S_in = 1.0
                S_out = 1.0

            elif scale_method in ("norm", "covariance"):
                S_in = bp_scaling(x)
                S_out = bp_scaling(h.detach())

            else:
                raise ValueError(
                    f"Unknown scaling method: {scale_method}"
                )

            w_in -= lr * S_in * w_in.grad
            w_out -= lr * S_out * w_out.grad

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
    """
    Network:
        x --> h --> [y1, y2]

    All weights initialized to 1.
    No biases.

    PC scaling:
        S_in  = 1 / (x_pre * x_post)
        S_out = 1 / (h_pre * h_post)

    Since x is clamped:
        x_pre = x_post = x
    """

    w_in = torch.tensor(1.0)
    w_out = torch.ones(2)

    trajectory = []

    for step in range(n_steps):

        # ====================================================
        # 1. Feedforward prediction
        # ====================================================

        h_pre = w_in * x
        y_hat = w_out * h_pre

        # Record prediction BEFORE inference
        trajectory.append(y_hat.clone())

        # ====================================================
        # 2. Initialise hidden activity
        # ====================================================

        h = h_pre.clone()

        # ====================================================
        # 3. Inference phase
        # ====================================================

        for _ in range(inference_steps):

            # Hidden prediction error
            e_h = h - w_in * x

            # Output prediction error
            e_y = target - w_out * h

            # Gradient of free energy wrt h
            grad_h = e_h - torch.sum(w_out * e_y)

            # Inference update
            h = h - inference_lr * grad_h

        # Hidden activity AFTER inference
        h_post = h

        # ====================================================
        # 4. Prediction errors after inference
        # ====================================================

        e_h = h_post - w_in * x
        e_y = target - w_out * h_post

        # ====================================================
        # 5. Scaling factors
        # ====================================================

        if scale_method is None:
            S_in = 1.0
            S_out = 1.0

        elif scale_method in ("norm", "covariance"):

            # Input activity is unchanged by inference
            S_in = pc_scaling(x, x)

            # Inverse product of PRE and POST inference
            # hidden activities
            S_out = pc_scaling(h_pre, h_post)

        else:
            raise ValueError(
                f"Unknown scaling method: {scale_method}"
            )

        # ====================================================
        # 6. Local predictive coding weight updates
        # ====================================================

        w_in = w_in + weight_lr * S_in * e_h * x
        w_out = w_out + weight_lr * S_out * e_y * h_post

    return torch.stack(trajectory)


# ============================================================
# Experiment settings
# ============================================================

n_steps = 1000
lr = 0.01

inference_steps_list = [1, 2, 5, 10, 300]


# ============================================================
# Train BP
# ============================================================

bp_trajectory = train_bp(
    x=x,
    target=target,
    n_steps=n_steps,
    lr=lr,
    scale_method=None,
)

bp_scaled_trajectory = train_bp(
    x=x,
    target=target,
    n_steps=n_steps,
    lr=lr,
    scale_method="norm",
)


# ============================================================
# Train PC for different inference steps
# ============================================================

pc_trajectories = {}
pc_scaled_trajectories = {}

for n_inference_steps in inference_steps_list:

    # Standard PC
    pc_trajectories[n_inference_steps] = train_pc(
        x=x,
        target=target,
        n_steps=n_steps,
        weight_lr=lr,
        inference_lr=0.1,
        inference_steps=n_inference_steps,
        scale_method=None,
    )

    # Scaled PC
    pc_scaled_trajectories[n_inference_steps] = train_pc(
        x=x,
        target=target,
        n_steps=n_steps,
        weight_lr=lr,
        inference_lr=0.1,
        inference_steps=n_inference_steps,
        scale_method="norm",
    )


# ============================================================
# Plot trajectories
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(7, 2.6),
    sharex=True,
    sharey=True,
)

experiments = [
    (
        axes[0],
        bp_trajectory,
        pc_trajectories,
        "Standard BP vs PC",
    ),
    (
        axes[1],
        bp_scaled_trajectory,
        pc_scaled_trajectories,
        "Scaled BP vs Scaled PC",
    ),
]

for ax, bp_traj, pc_trajs, title in experiments:

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    ax.plot(
        bp_traj[:, 0],
        bp_traj[:, 1],
        label="BP",
        linewidth=2,
        color="orange",
    )

    # --------------------------------------------------------
    # Predictive coding
    # --------------------------------------------------------

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(
        vmin=min(pc_trajs),
        vmax=max(pc_trajs),
    )

    for n_inference_steps, trajectory in pc_trajs.items():
        idx = inference_steps_list.index(n_inference_steps)

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=cmap(0.3 + 0.6 * idx / (len(inference_steps_list) - 1)),
            linewidth=1,
            label=f"PC (T={n_inference_steps})",
        )

    #for n_inference_steps, trajectory in pc_trajs.items():
#
  #      ax.plot(
  #          trajectory[:, 0],
  #          trajectory[:, 1],
  #          linewidth=1,
   #         label=f"PC (T={n_inference_steps})",
 #       )
#
    # --------------------------------------------------------
    # Initialisation
    # --------------------------------------------------------

    ax.scatter(
        1,
        1,
        marker="o",
        s=80,
        label="Initialisation",
        color="black",
        zorder=10,
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    ax.scatter(
        -1,
        1,
        marker="x",
        s=120,
        linewidth=2,
        label="Target",
        color="black",
        zorder=10,
    )

    # --------------------------------------------------------
    # Ideal path
    # --------------------------------------------------------

    ax.plot(
        [-1, 1],
        [1, 1],
        "--",
        linewidth=1,
        color="grey",
        label="Ideal",
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlabel(r"$\hat{y}_1$")
    ax.set_ylabel(r"$\hat{y}_2$")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0.6, 1.15)

    # Light grey subplot borders
    for spine in ax.spines.values():
        spine.set_color("#D3D3D3")
        spine.set_linewidth(0.8)

    ax.set_title(title)
    ax.grid(alpha=0.25)

axes[1].legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)

plt.tight_layout()
#save as svg file
plt.savefig("figure.svg", format="svg", bbox_inches="tight")
plt.show()
#plt.show()