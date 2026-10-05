import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from lightning import LightningModule


def flatten_if_needed(x):
    if x.ndim > 2:
        return x.view(x.size(0), -1)
    return x


epsilon = 1e-12
alpha = 1e-5


def stable_inverse(cov):
    return torch.linalg.pinv(cov, rcond=1e-10)


def choose_epsilon_from_eigs(cov, mode="relative", alpha=1e-5, kappa_target=1e5):
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


class BPMLP(LightningModule):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        w_lr: float,
        flatten_input: bool = True,
        num_classes: int = 10,
        scale_method: Optional[str] = None,
        epsilon: float = 1e-8,
        alpha: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["architecture"])

        self.layers = torch.nn.ModuleList(architecture)
        self.errors = None

        self.net = nn.Sequential(*architecture)
        self.w_lr = w_lr
        self.flatten_input = flatten_input
        self.num_classes = num_classes

        self.epsilon = epsilon
        self.alpha = alpha

        self._hook_handles = []
        self.scalings = None
        self.z0 = None

        scale_methods = {
            "norm": self.get_activity_norm,
            "covariance": self.get_activity_covariance,
        }

        self.get_activity = scale_methods.get(scale_method, None)

    def _prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.flatten_input and x.ndim > 2:
            x = x.view(x.size(0), -1)
        return x

    def _prepare_y(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            return F.one_hot(y, num_classes=self.num_classes).to(dtype=torch.float32)
        return y

    def forward(self, x):
        if self.flatten_input:
            x = flatten_if_needed(x)
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.w_lr)

    def get_activities(self, x):
        x = self._prepare_x(x)
        x_fp = x.clone()
        self.z0 = [x_fp] + [x_fp := layer(x_fp) for layer in self.layers[:-1]]

    def get_activity_norm(self, x):
        self.get_activities(x)
        # ADDED SMALL CONSTANT TO AVOID DIVISION BY ZERO
        self.scalings = [(1 / ((z0 * z0).sum(-1).mean().item() + epsilon)) * torch.eye(z0.shape[-1], device=z0.device, dtype=z0.dtype) for z0 in self.z0]

    def get_activity_covariance(self, x):
        self.get_activities(x)
        covariances = [z.double().T @ z.double() / z.shape[0] for z in self.z0]
        self.scalings = []
        for cov in covariances:
            eps = choose_epsilon_from_eigs(cov, mode="floor", alpha=alpha)
            reg_cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            self.scalings.append(stable_inverse(reg_cov).float())

    def register_hook(self):
        self.remove_hooks()

        weight_layers = []
        for block in self.layers:
            for m in block.modules():
                if m is not block and hasattr(m, "weight") and m.weight is not None:
                    weight_layers.append(m)

        for layer_idx, layer in enumerate(weight_layers):
            def hook_fn(grad, idx=layer_idx):
                return grad @ self.scalings[idx]

            handle = layer.weight.register_hook(hook_fn)
            self._hook_handles.append(handle)

    def remove_hooks(self):
        for h in getattr(self, "_hook_handles", []):
            h.remove()
        self._hook_handles = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        target = self._prepare_y(y)
        outputs = self(x)

        loss = 0.5 * F.mse_loss(outputs, target, reduction="sum")
        if self.get_activity is not None:
            self.get_activity(x)
            self.register_hook()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss
