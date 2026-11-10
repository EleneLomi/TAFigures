"""
Figure 3, panel A -- target alignment after a SINGLE training step vs the condition
number of the weight matrices, for BP, PC and their rescaled counterparts (online
learning, batch size 1). The "Whole training" panels B-D are produced by ../online.py,
../Deep/online_deep.py and ../../Autoencoder.

Origin: thesis_weight_scaling_fig.ipynb
    models / set_condition_number   cell 1 (only the 'norm' rescaling is kept)
    experiment                      cell 11
    plot                            cell 12, third figure ("combined")

Added relative to the notebook: a fixed seed, CSV output and a panel-labelled file name.

Usage:  python panel_A_condition_number.py
Output: output_data/condition_number.csv, panel_A_condition_number.svg and .png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

SEED = 0
N_REPEATS = 10
INPUT_DIM = HIDDEN_DIM = OUTPUT_DIM = 512
NM_LAYERS = 1
BATCH_SIZE = 1
LR = 1e-4
CONDITION_NUMBERS = [1, 2, 10, 50, 1000, 10000, 100000, 1e7, 1e9, 1e12]

base_dir = Path(__file__).parent
data_dir = base_dir / "output_data"
data_dir.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_theme(style="whitegrid")
PALETTE = {"BP": "#ff7f0e", "PC": "#1f77b4"}


def set_condition_number(cn, model):
    """Replace each layer's singular values with a linear ramp of condition number cn,
    keeping the singular vectors and the Frobenius norm of the layer."""
    with torch.no_grad():
        for layer in model.layers:
            U, S, Vh = torch.linalg.svd(layer.weight.data)
            S_new = torch.linspace(S.max(), S.max() / cn, S.shape[0], device=S.device)
            S_new = S_new * (torch.norm(S) / torch.norm(S_new))
            layer.weight.data = U @ torch.diag(S_new) @ Vh


class LinearMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers, rescale=False, bias=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(nm_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            input_dim = hidden_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        self.rescale = rescale
        self._hook_handles = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def feedforward_activities(self, x):
        z0 = [x.clone()]
        for layer in self.layers[:-1]:
            z0.append(layer(z0[-1]))
        return z0

    def register_scalings(self, scalings):
        """Multiply each layer's gradient by its layer-specific learning-rate scaling."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = [
            layer.weight.register_hook(lambda grad, idx=idx: grad @ scalings[idx])
            for idx, layer in enumerate(self.layers)
        ]


class BP_MLP(LinearMLP):
    def loss(self, x, y):
        y_hat = self.forward(x)
        if self.rescale:
            self.register_scalings([
                (1 / (z0 * z0).sum(-1).mean().item()) * torch.eye(z0.shape[-1], device=z0.device, dtype=z0.dtype)
                for z0 in self.feedforward_activities(x)
            ])
        return 0.5 * ((y_hat - y) ** 2).sum()


class PC_MLP(LinearMLP):
    """Linear PC at the inference equilibrium, using the closed-form precision."""

    def forward_with_variances(self, x, y):
        variance = torch.eye(y.shape[-1], device=x.device, dtype=x.dtype)
        previous_activity = self.layers[-1].weight
        variance += previous_activity @ previous_activity.t()
        for layer in self.layers[-2:0:-1]:
            previous_activity = previous_activity @ layer.weight
            variance += previous_activity @ previous_activity.t()
        previous_activity = previous_activity @ self.layers[0].weight
        precision = torch.cholesky_inverse(torch.linalg.cholesky(variance))
        return x @ previous_activity.T, precision

    def equilibrium_activities(self, x, y_hat, y, precision):
        r = y - y_hat
        weights_prod = [self.layers[-1].weight]
        for layer in reversed(self.layers[1:-1]):
            weights_prod.append(weights_prod[-1] @ layer.weight)
        errors = [r @ precision @ wp for wp in weights_prod]
        z = [x]
        for e, layer in zip(reversed(errors), self.layers[:-1]):
            z.append(layer(z[-1]) + e)
        return z

    def loss(self, x, y):
        y_hat, precision = self.forward_with_variances(x, y)
        if self.rescale:
            z0s = self.feedforward_activities(x)
            zs = self.equilibrium_activities(x, y_hat, y, precision)
            self.register_scalings([
                (1 / (z * z0).sum(-1).mean().item()) * torch.eye(z.shape[-1], device=z.device, dtype=z.dtype)
                for z, z0 in zip(zs, z0s)
            ])
        return 0.5 * ((y - y_hat) @ precision * (y - y_hat)).sum()


def train_one_step(model, x, y, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    model.loss(x, y).backward()
    optimizer.step()


def run():
    rows = []
    kwargs = dict(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, nm_layers=NM_LAYERS)
    for repeat in tqdm(range(N_REPEATS), desc="panel A (condition number)"):
        x = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)
        y = (torch.randn(BATCH_SIZE, INPUT_DIM) @ torch.randn(OUTPUT_DIM, INPUT_DIM).T / INPUT_DIM**0.5).to(device)
        for cn in CONDITION_NUMBERS:
            model_bp = BP_MLP(**kwargs)
            set_condition_number(cn, model_bp)
            models = [
                ("BP", "default", model_bp),
                ("BP", "scaled lr", BP_MLP(**kwargs, rescale=True)),
                ("PC", "default", PC_MLP(**kwargs)),
                ("PC", "scaled lr", PC_MLP(**kwargs, rescale=True)),
            ]
            for algorithm, method, model in models:
                if model is not model_bp:
                    model.load_state_dict(model_bp.state_dict())
                model.to(device)
                y_pred = model(x).detach()
                r = y - y_pred
                train_one_step(model, x, y, LR)
                delta_y = model(x).detach() - y_pred
                similarity = nn.CosineSimilarity(dim=1, eps=1e-6)(delta_y, r).mean().item()
                rows.append({"repeat": repeat, "condition number": cn, "algorithm": algorithm,
                             "method": method, "similarity": similarity})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    df = run()
    df.to_csv(data_dir / "condition_number.csv", index=False)

    plt.figure(figsize=(4, 3))
    sns.lineplot(data=df, x="condition number", y="similarity", hue="algorithm", style="method", palette=PALETTE)
    plt.xscale("log")
    plt.xlabel("Condition number")
    plt.ylabel(r"Cosine similarity(r, $\Delta y_{pred}$)")
    plt.title("Fig. 3A: online learning, batch size 1", fontsize=9)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    stem = base_dir / "panel_A_condition_number"
    plt.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
