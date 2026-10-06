# ============================================================
# TARGET ALIGNMENT FOR BACKPROP AFTER ONE WEIGHT UPDATE
# ============================================================

import math
import random
import numpy as np
import torch
import torch.nn as nn

seed = 0
input_dim = 20
hidden_dim = 20
output_dim = 20
nm_layers = 1
n_train = 7096
n_test = 2048
batch_size = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BP_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers=1, bias=False):
        super().__init__()
        self.nm_layers = nm_layers
        self.layers = nn.ModuleList()

        for _ in range(nm_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            input_dim = hidden_dim

        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, x, y):
        batch_size = x.shape[0]
        y_hat = self.forward(x)
        error = 0.5 * nn.MSELoss(reduction="sum")(y_hat, y) / batch_size
        return error, y_hat


def apply_init(model: nn.Module, init_name: str):
    for m in model.modules():
        if isinstance(m, nn.Linear):
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
                raise ValueError(f"Unknown initialization: {init_name}")

            if m.bias is not None:
                nn.init.zeros_(m.bias)


def make_synthetic(seed: int, rank_data: int):
    g = torch.Generator().manual_seed(seed)

    A = torch.randn(rank_data, input_dim, generator=g) / math.sqrt(input_dim)
    B = torch.randn(output_dim, rank_data, generator=g) / math.sqrt(rank_data)
    W_data = B @ A

    X_train = torch.randn(n_train, input_dim, generator=g)
    X_test = torch.randn(n_test, input_dim, generator=g)

    Y_train = X_train @ W_data.T
    Y_test = X_test @ W_data.T

    return X_train, Y_train, X_test, Y_test, W_data


@torch.no_grad()
def target_alignment(y, y_pre, y_post, eps=1e-12):
    r = y - y_pre
    delta_y = y_post - y_pre

    numerator = (r * delta_y).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(r, dim=-1)
        * torch.linalg.vector_norm(delta_y, dim=-1)
    )

    alignment_per_sample = numerator / denominator.clamp_min(eps)
    return alignment_per_sample.mean().item()


set_seed(seed)

X_train, Y_train, X_test, Y_test, W_data = make_synthetic(
    seed=seed,
    rank_data=hidden_dim,
)

X_train = X_train.to(device)
Y_train = Y_train.to(device)

g = torch.Generator().manual_seed(seed + 123)
idx = torch.randint(
    low=0,
    high=X_train.size(0),
    size=(batch_size,),
    generator=g,
)

xb = X_train[idx]
yb = Y_train[idx]

bp_model = BP_MLP(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    nm_layers=nm_layers,
    bias=False,
).to(device)

apply_init(bp_model, "norm_preservation")

bp_lr = 0.01
optimizer = torch.optim.SGD(
    bp_model.parameters(),
    lr=bp_lr,
    momentum=0.0,
)

with torch.no_grad():
    y_pre = bp_model.forward(xb).detach().clone()

bp_model.train()
optimizer.zero_grad(set_to_none=True)
loss, _ = bp_model.loss(xb, yb)
loss.backward()
optimizer.step()

with torch.no_grad():
    y_post = bp_model.forward(xb).detach().clone()

ta_bp = target_alignment(
    y=yb,
    y_pre=y_pre,
    y_post=y_post,
)

print(
    f"Backprop target alignment after one weight update: {ta_bp:.6f}"
)
