"""
Figure 2, panels A, B, C, D -- target alignment after a SINGLE training step.
(The "Whole training" panels E-H are produced by ../figure2.py and ../Deep/deep.py.)

    Panel A : Kaiming init (PyTorch default)     alignment vs model depth
    Panel B : Kaiming init (PyTorch default)     alignment vs hidden layer width
    Panel C : Norm-preserving init               alignment vs model depth
    Panel D : Norm-preserving init               alignment vs hidden layer width

"Norm preservation" in the paper is W_ij ~ N(0, 1/n_out), called `gaus` in the notebook
(which also saves it as "lecun"). It is NOT the notebook's orthogonal init, which gives
a target alignment of exactly 1 for square networks.

Origin: thesis_weight_init_fig.ipynb
    models / initialisation     cell 2
    width experiment + plots    cells 8, 9   (panels B, D)
    depth experiment + plots    cells 11, 12 (panels A, C)
The notebook also ran an orthogonal init; it is in no panel of Figure 2 and is left out.

Differences from the notebook:
  * WIDTH_NM_LAYERS = 1. The saved cell 8 has nm_layers = 2, which does not reproduce
    panels B and D; one hidden layer does, and is the paper's stated default architecture.
  * Added a fixed seed, CSV output and panel-labelled file names.

Usage:  python panels_ABCD_single_step.py
Output: output_data/depth.csv, output_data/width.csv,
        panel_{A,B,C,D}_*.svg and .png next to this file
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

SEED = 0
N_REPEATS = 10
INPUT_DIM = OUTPUT_DIM = 512
BATCH_SIZE = 1

# Panels A, C
DEPTHS = np.arange(1, 9)
DEPTH_HIDDEN_DIM = 512
DEPTH_LR = 1e-4

# Panels B, D
WIDTHS = [128, 256, 512, 1024]
WIDTH_NM_LAYERS = 1   # the notebook's cell 8 has 2; see the header
WIDTH_LR = 1e-5

base_dir = Path(__file__).parent
data_dir = base_dir / "output_data"
data_dir.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_theme(style="whitegrid")
PALETTE = {"BP": "#ff7f0e", "PC": "#1f77b4"}


def init_model_norm_preserving(model):
    """W_ij ~ N(0, 1/n_out); `init_model_gaus` in the notebook."""
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.normal_(mean=0.0, std=1.0 / layer.weight.size(0) ** 0.5)
            if layer.bias is not None:
                layer.bias.zero_()


class BP_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers=2, bias=False):
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
        y_hat = self.forward(x)
        return 0.5 * nn.MSELoss(reduction="sum")(y_hat, y), y_hat


class PC_MLP(BP_MLP):
    """Linear PC at the inference equilibrium, using the closed-form precision."""

    def forward_with_variances(self, x, y):
        variance = torch.eye(y.shape[-1], device=x.device)
        previous_activity = self.layers[-1].weight
        variance += previous_activity @ previous_activity.t()
        for layer in self.layers[-2:0:-1]:
            previous_activity = previous_activity @ layer.weight
            variance += previous_activity @ previous_activity.t()
        previous_activity = previous_activity @ self.layers[0].weight
        return x @ previous_activity.T, torch.linalg.inv(variance)

    def loss(self, x, y):
        y_hat, precision = self.forward_with_variances(x, y)
        return 0.5 * ((y - y_hat) @ precision * (y - y_hat)).sum(), y_hat


def train_one_step(model, x, y, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    loss, _ = model.loss(x, y)
    loss.backward()
    optimizer.step()


def alignment_for_architecture(x, y, hidden_dim, nm_layers, lr):
    """One step of BP and PC from a shared init; returns one row per (init, algorithm)."""
    rows = []
    for init_name in ["kaiming", "norm_preserving"]:
        kwargs = dict(input_dim=INPUT_DIM, hidden_dim=hidden_dim, output_dim=OUTPUT_DIM, nm_layers=nm_layers)
        model_bp, model_pc = BP_MLP(**kwargs), PC_MLP(**kwargs)
        if init_name == "norm_preserving":
            init_model_norm_preserving(model_bp)
        model_pc.load_state_dict(model_bp.state_dict())

        for algorithm, model in [("BP", model_bp), ("PC", model_pc)]:
            model.to(device)
            y_pred = model(x).detach()
            r = y - y_pred
            train_one_step(model, x, y, lr)
            delta_y = model(x).detach() - y_pred
            similarity = nn.CosineSimilarity(dim=1, eps=1e-6)(delta_y, r).mean().item()
            rows.append({"init": init_name, "algorithm": algorithm, "similarity": similarity})
    return rows


def sample_data():
    x = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)
    y = (torch.randn(BATCH_SIZE, INPUT_DIM) @ torch.randn(OUTPUT_DIM, INPUT_DIM).T / INPUT_DIM**0.5).to(device)
    return x, y


def run_depth():
    rows = []
    for repeat in tqdm(range(N_REPEATS), desc="panels A, C (depth)"):
        x, y = sample_data()
        for depth in DEPTHS:
            for row in alignment_for_architecture(x, y, DEPTH_HIDDEN_DIM, int(depth), DEPTH_LR):
                rows.append({"repeat": repeat, "depth": int(depth), **row})
    return pd.DataFrame(rows)


def run_width():
    rows = []
    for repeat in tqdm(range(N_REPEATS), desc="panels B, D (width)"):
        x, y = sample_data()
        for hidden_dim in WIDTHS:
            for row in alignment_for_architecture(x, y, hidden_dim, WIDTH_NM_LAYERS, WIDTH_LR):
                rows.append({"repeat": repeat, "hidden_dim": hidden_dim, **row})
    return pd.DataFrame(rows)


def plot_panel(df, init_name, x, xlabel, ylim, panel, title, square_line=False):
    plt.figure(figsize=(2.5, 3))
    sns.lineplot(data=df[df["init"] == init_name], x=x, y="similarity", hue="algorithm", palette=PALETTE)
    if square_line:  # dashed grey line: square network (hidden width = input width)
        plt.vlines(x=INPUT_DIM, ymin=0, ymax=1.05, colors="gray", linestyles="dashed", label="")
    plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel(r"Cosine similarity(r, $\Delta y_{pred}$)")
    plt.title(f"Fig. 2{panel}: {title}", fontsize=9)
    if panel != "A":  # the paper shows the legend in panel A only
        plt.legend().remove()
    stem = base_dir / f"panel_{panel}_{init_name}_{x}"
    plt.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(SEED)

    df_depth = run_depth()
    df_depth.to_csv(data_dir / "depth.csv", index=False)
    df_width = run_width()
    df_width.to_csv(data_dir / "width.csv", index=False)

    plot_panel(df_depth, "kaiming", "depth", "Model depth", (0.5, 1.05), "A", "Kaiming init, depth")
    plot_panel(df_width, "kaiming", "hidden_dim", "Hidden layer width", (0.6, 1.05), "B", "Kaiming init, width", square_line=True)
    plot_panel(df_depth, "norm_preserving", "depth", "Model depth", (0.5, 1.05), "C", "Norm-preserving init, depth")
    plot_panel(df_width, "norm_preserving", "hidden_dim", "Hidden layer width", (0.6, 1.05), "D", "Norm-preserving init, width", square_line=True)
