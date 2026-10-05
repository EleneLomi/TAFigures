from typing import Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule

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


class PCMLP(LightningModule):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        iters: int,
        e_lr: float,
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

        self.iters = iters
        self.e_lr = e_lr
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

    def output_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * F.mse_loss(y_hat, y, reduction="sum")

    def y_pred(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_x(x)
        s_i = x
        for e_i, layer_i in zip(self.errors + [0.0], self.layers):
            s_i = e_i + layer_i(s_i)
        return s_i

    def configure_optimizers(self):
        return torch.optim.SGD(self.layers.parameters(), lr=self.w_lr)

    def E(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Full PC energy:
          sum of squared latent errors + output MSE
        """
        E_errors = 0.5 * sum(torch.sum(e**2) for e in self.errors)
        return E_errors + self.output_loss(self.y_pred(x), y)

    def E_local(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Local energy used for weight gradients.
        No backprop through inferred states.
        """
        x = self._prepare_x(x)
        E = 0.0
        s_i = x

        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)
            s_i = (e_i + s_i_pred).detach()
            E += 0.5 * F.mse_loss(s_i_pred, s_i, reduction="sum")

        y_hat = self.layers[-1](s_i)
        return E + self.output_loss(y_hat, y)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._prepare_x(x)

        if y is None:
            self.init_zero_errors(x)
        else:
            self.minimize_error_energy(x, self._prepare_y(y))

        return self.y_pred(x)

    def minimize_error_energy(self, x: torch.Tensor, y: torch.Tensor):
        """
        Infer latent error variables with weights frozen.
        Output is clamped via the supervised MSE term.
        """
        for p in self.layers.parameters():
            p.requires_grad_(False)

        self.init_zero_errors(x)
        error_optim = torch.optim.SGD(self.errors, lr=self.e_lr)

        for _ in range(self.iters):
            error_optim.zero_grad()
            E = self.E(x, y)
            E.backward()
            error_optim.step()

        self.log("E_errors", E.detach(), prog_bar=True, on_step=True, on_epoch=True)

        for p in self.layers.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def init_zero_errors(self, x: torch.Tensor):
        """
        Create trainable error tensors of the correct hidden-state shapes.
        """
        x = self._prepare_x(x)
        tmp = x
        errors = []

        for layer_i in self.layers[:-1]:
            tmp = layer_i(tmp)
            err = torch.zeros_like(tmp, requires_grad=True)
            errors.append(err)
            tmp = tmp.detach()

        self.errors = errors

    def get_activities(self, x: torch.Tensor):
        x = self._prepare_x(x)

        # ---- Pre-relaxation activities (pure feedforward) ----
        z0 = []
        s_i = x.clone()
        z0.append(s_i)  # input to layer 0

        for layer in self.layers[:-1]:
            s_i = layer(s_i)
            z0.append(s_i)  # input to next layer

        self.z0 = z0

        # ---- Post-relaxation activities (using inferred errors) ----
        z = []
        s_i = x
        z.append(s_i)  # input to layer 0

        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)
            s_i = (e_i + s_i_pred).detach()
            z.append(s_i)  # input to next layer after relaxation

        self.z = z

    def get_activity_norm(self, x: torch.Tensor):
        self.get_activities(x)
        self.scalings = [(1 / ((z * z0).sum(-1).mean().item() + epsilon)) * torch.eye(z.shape[-1], device=z.device, dtype=z.dtype) for z, z0 in zip(self.z, self.z0)]

    def get_activity_covariance(self, x: torch.Tensor):
        self.get_activities(x)
        covariances = [z0.double().T @ z.double() / z.shape[0] for z0, z in zip(self.z0, self.z)]
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

    def on_after_backward(self):
        self.remove_hooks()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self._prepare_x(x)
        y = self._prepare_y(y)

        # PC inference
        self.minimize_error_energy(x, y)

        # Weight objective
        E_final = self.E_local(x, y)
        loss = E_final / x.shape[0]

        # Optional activity-based gradient rescaling
        if self.get_activity is not None:
            self.get_activity(x)
            self.register_hook()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self._prepare_x(x)
        y = self._prepare_y(y)

        self.init_zero_errors(x)
        y_hat = self.y_pred(x)
        loss = self.output_loss(y_hat, y) / x.shape[0]

        self.log("val_loss", loss.detach(), prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self._prepare_x(x)
        y = self._prepare_y(y)

        self.init_zero_errors(x)
        y_hat = self.y_pred(x)
        loss = self.output_loss(y_hat, y) / x.shape[0]

        self.log("test_loss", loss.detach(), prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = self._prepare_x(x)
        y = self._prepare_y(y)

        self.init_zero_errors(x)
        y_hat = self.y_pred(x)
        loss = self.output_loss(y_hat, y) / x.shape[0]
        print("Output loss =", loss.item())

        return {"y_hat": y_hat, "y": y}
