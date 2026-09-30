import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

# ============================================================
# Data
# ============================================================

x = torch.tensor(1.0)
target = torch.tensor([-1.0, 1.0])


# ============================================================
# BACKPROPAGATION
# ============================================================

def train_bp(
    x,
    target,
    n_steps=500,
    lr=0.01,
):
    """
    Network:

        x --> h --> [y1, y2]

    All weights initialized to 1.
    No biases.
    """

    # weights
    w_in = torch.tensor(1.0, requires_grad=True)
    w_out = torch.ones(2, requires_grad=True)

    trajectory = []

    for step in range(n_steps):

        # -------------------------
        # forward pass
        # -------------------------
        h = w_in * x
        y_hat = w_out * h

        trajectory.append(y_hat.detach().clone())

        # squared error
        loss = 0.5 * ((y_hat - target) ** 2).sum()

        # -------------------------
        # backpropagation
        # -------------------------
        loss.backward()

        with torch.no_grad():
            w_in -= lr * w_in.grad
            w_out -= lr * w_out.grad

        # clear gradients
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
):
    """
    Predictive coding version of the same network:

        x --> h --> [y1, y2]

    Prediction errors:

        e_h = h - w_in*x

        e_y = target - w_out*h

    During each training step:

    1. Run a normal forward pass.
    2. Clamp output activity to the target.
    3. Iteratively infer the hidden activity h.
    4. Update weights using local prediction errors.

    All weights start at 1.
    No biases.
    """

    # --------------------------------------------------------
    # weights start at exactly 1
    # --------------------------------------------------------

    w_in = torch.tensor(1.0)
    w_out = torch.ones(2)

    trajectory = []

    for step in range(n_steps):

        # ====================================================
        # 1. Normal feedforward prediction
        # ====================================================

        h_ff = w_in * x
        y_hat = w_out * h_ff

        # Record the actual network prediction BEFORE
        # clamping the output to the target
        trajectory.append(y_hat.clone())

        # ====================================================
        # 2. Initialise hidden activity from feedforward pass
        # ====================================================

        h = h_ff.clone()

        # ====================================================
        # 3. Inference phase
        # ====================================================
        #
        # Output activity is clamped:
        #
        #       y = target
        #
        # Free energy:
        #
        # F = 1/2 (h - w_in*x)^2
        #     + 1/2 ||target - w_out*h||^2
        #
        # We gradient-descent on h.
        # ====================================================

        for _ in range(inference_steps):

            # prediction error at hidden layer
            e_h = h - w_in * x

            # prediction error at output layer
            e_y = target - w_out * h

            # gradient of F with respect to h
            #
            # dF/dh =
            #     e_h - sum(w_out * e_y)
            #
            grad_h = e_h - torch.sum(w_out * e_y)

            # inference update
            h = h - inference_lr * grad_h

        # ====================================================
        # 4. Compute prediction errors after inference
        # ====================================================

        e_h = h - w_in * x
        e_y = target - w_out * h

        # ====================================================
        # 5. Local predictive-coding weight updates
        # ====================================================
        #
        # No error is backpropagated through the network here.
        #
        # Each weight uses its local pre-synaptic activity
        # multiplied by its local prediction error.
        # ====================================================

        w_in = w_in + weight_lr * e_h * x

        w_out = w_out + weight_lr * e_y * h

    return torch.stack(trajectory)


# ============================================================
# Train both models
# ============================================================

bp_trajectory = train_bp(
    x=x,
    target=target,
    n_steps=1000,
    lr=0.01,
)

pc_trajectory = train_pc(
    x=x,
    target=target,
    n_steps=1000,
    weight_lr=0.01,
    inference_lr=0.1,
    inference_steps=50,
)
# NOTE YOU CAN INSTEAD USE EQUILIBRATED ENERGY

# ============================================================
# Plot both trajectories
# ============================================================

plt.figure(figsize=(6, 5))

# Backpropagation
plt.plot(
    bp_trajectory[:, 0],
    bp_trajectory[:, 1],
    label="BP",
    linewidth=2,
)

# Predictive coding
plt.plot(
    pc_trajectory[:, 0],
    pc_trajectory[:, 1],
    label="PC",
    linewidth=2,
)

# initialization
plt.scatter(
    1,
    1,
    marker="o",
    s=80,
    label="initialisation",
)

# target
plt.scatter(
    -1,
    1,
    marker="x",
    s=120,
    linewidth=2,
    label="target",
)

# ideal path
plt.plot(
    [-1, 1],
    [1, 1],
    "--",
    linewidth=1,
    label="ideal",
)

plt.xlabel(r"$\hat{y}_1$")
plt.ylabel(r"$\hat{y}_2$")

plt.xlim(-1.1, 1.1)
plt.ylim(0.0, 1.15)

plt.legend()
plt.tight_layout()
plt.show()