"""
Figure 4, panel A -- target alignment after a SINGLE training step vs batch size, for
BP, PC (solid) and their covariance-rescaled counterparts (dashed). The "Whole training"
panels B-D are produced by ../figure4.py, ../Deep/figure4_deep.py and ../../Autoencoder.

Origin: thesis_weight_scaling_fig.ipynb
    models                          cell 1 (only the 'covariance' rescaling is kept)
    experiment + plot               cells 40, 41 (batch sizes, models, combined plot)
    target and learning rate        cell 14 (teacher target, lr proportional to 1/batch size)

No single notebook cell reproduces this panel. Cell 40 as saved has y = randn and
lr = 1e-4, which gives the collapsing curves stored in cell 41. The combination below was
identified by comparing candidates with the curves in ../figure4.svg:
    y = x @ W.T  (a realisable teacher, resampled for each batch size)
    lr = 1e-4 / batch_size   (needed because the loss is summed over the batch)
The y limits (0.75, 1.05) are read off the published panel.

Added relative to the notebook: a fixed seed, CSV output and a panel-labelled file name.

Usage:  python panel_A_batch_size.py
Output: output_data/batch_size.csv, panel_A_batch_size.svg and .png
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

SEED = 1
N_REPEATS = 10
INPUT_DIM = HIDDEN_DIM = OUTPUT_DIM = 512
NM_LAYERS = 1
BASE_LR = 1e-4
BATCH_SIZES = [2, 32, 64, 128, 256, 480, 550, 1000, 2048, 4096]

base_dir = Path(__file__).parent
data_dir = base_dir / "output_data"
data_dir.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_theme(style="whitegrid")
PALETTE = {"BP": "#ff7f0e", "PC": "#1f77b4"}


def stable_inverse(cov):
    return torch.linalg.pinv(cov, rcond=1e-6)


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
        """Post-multiply each layer's gradient by its inverse activity covariance."""
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
            with torch.no_grad():
                covariances = [z.double().T @ z.double() / z.shape[0] for z in self.feedforward_activities(x)]
                self.register_scalings([stable_inverse(cov).float() for cov in covariances])
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
            with torch.no_grad():
                z0s = self.feedforward_activities(x)
                zs = self.equilibrium_activities(x, y_hat, y, precision)
                covariances = [z0.double().T @ z.double() / z.shape[0] for z0, z in zip(z0s, zs)]
                self.register_scalings([stable_inverse(cov).float() for cov in covariances])
        return 0.5 * ((y - y_hat) @ precision * (y - y_hat)).sum()


def train_one_step(model, x, y, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    model.loss(x, y).backward()
    optimizer.step()


def run():
    rows = []
    kwargs = dict(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, nm_layers=NM_LAYERS)
    for repeat in tqdm(range(N_REPEATS), desc="panel A (batch size)"):
        for bs in BATCH_SIZES:
            W = (torch.randn(OUTPUT_DIM, INPUT_DIM).T / INPUT_DIM**0.5).to(device)
            x = torch.randn(bs, INPUT_DIM).to(device)
            y = x @ W.T

            model_bp = BP_MLP(**kwargs)
            models = [
                ("BP", "default", model_bp),
                ("BP", "scaled", BP_MLP(**kwargs, rescale=True)),
                ("PC", "default", PC_MLP(**kwargs)),
                ("PC", "scaled", PC_MLP(**kwargs, rescale=True)),
            ]
            for algorithm, method, model in models:
                if model is not model_bp:
                    model.load_state_dict(model_bp.state_dict())
                model.to(device)
                y_pred = model(x).detach()
                r = y - y_pred
                train_one_step(model, x, y, BASE_LR / bs)
                delta_y = model(x).detach() - y_pred
                similarity = nn.CosineSimilarity(dim=1, eps=1e-6)(delta_y, r).mean().item()
                rows.append({"repeat": repeat, "batch_size": bs, "algorithm": algorithm,
                             "method": method, "similarity": similarity})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    df = run()
    df.to_csv(data_dir / "batch_size.csv", index=False)

    plt.figure(figsize=(3.5, 3))
    sns.lineplot(data=df, x="batch_size", y="similarity", hue="algorithm", style="method", palette=PALETTE)
    plt.vlines(x=HIDDEN_DIM, ymin=0, ymax=1.05, colors="gray", linestyles="dashed", label="")
    plt.ylim(0.75, 1.05)
    plt.xscale("log")
    plt.xlabel("Batch size")
    plt.ylabel(r"Cosine similarity(r, $\Delta y_{pred}$)")
    plt.title("Fig. 4A: linear network, one hidden layer", fontsize=9)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    stem = base_dir / "panel_A_batch_size"
    plt.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
