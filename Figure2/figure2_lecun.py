# This file is for the main figures in the text
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os

epsilon = 1e-12#3#12
alpha = 1e-5#4

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))


def init_model_gaus(model):
    # LeCun initialization - Should be better than default init
    with torch.no_grad():
        for layer in model.layers:
            n_out = layer.weight.size(0)
            sigma = 1.0 / (n_out ** 0.5)
            layer.weight.normal_(mean=0.0, std=sigma)
            if layer.bias is not None:
                layer.bias.zero_()

            
def init_model_orthogonal(model):
    with torch.no_grad():
        for layer in model.layers:
            nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                layer.bias.zero_()

def stable_inverse(cov):
    return torch.linalg.pinv(cov, rcond=1e-6) #NOTE THAT I CHANGED THIS 

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


###################################  Implementation without calculating activities ################################### 

class BP_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers=2, bias=False):
        super(BP_MLP, self).__init__()
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
        error = 0.5 * nn.MSELoss(reduction='sum')(y_hat, y) / batch_size
        return error, y_hat
    
# See Francesco's paper for loss function
class PC_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers=2, bias=False):
        super(PC_MLP, self).__init__()
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
        batch_size = x.shape[0]
        y_hat, precision = self.forward_with_variances(x, y)
        loss = 0.5*((y - y_hat) @ precision * (y - y_hat)).sum() / batch_size
        return loss, y_hat
    
def train(model, x, y, lr=1e-3, epochs=1, verbose=True):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dl = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in dl:
        optimizer.zero_grad()
        loss, y_hat = model.loss(x, y)
        loss.backward()
        optimizer.step()


######################################### Implementation with calculating activities (for calculating proposed scaling factors) #########################################

class BP_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers=2, bias=False, scale_method=None):
        super(BP_MLP, self).__init__()
        self.nm_layers = nm_layers
        self.layers = nn.ModuleList()
        for _ in range(nm_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            input_dim = hidden_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

        scale_methods = {
            'norm': self.get_activity_norm,
            'covariance': self.get_activity_covariance
        }
        if scale_method is not None:
            self.get_activity = scale_methods[scale_method]
        else:
            self.get_activity = None


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def loss(self, x, y):
        batch_size = x.shape[0]
        y_hat = self.forward(x)
        error = 0.5 * nn.MSELoss(reduction='sum')(y_hat, y) / batch_size
        if self.get_activity is not None:
            self.get_activity(x, y)
            self.register_hook()
        return error, y_hat
    
    def get_activities(self, x, y):
        x_fp = x.clone()
        self.z0 = [x_fp] + [x_fp:=layer(x_fp) for layer in self.layers[:-1]]

    def get_activity_norm(self, x, y):
        self.get_activities(x, y)
        # ADDED SMALL CONSTANT TO AVOID DIVISION BY ZERO
        self.scalings = [(1 / ((z0*z0).sum(-1).mean().item() + epsilon)) * torch.eye(z0.shape[-1], device=z0.device, dtype=z0.dtype) for z0 in self.z0]

    def get_activity_covariance(self, x, y):
        self.get_activities(x, y)
        covariances = [z.double().T @ z.double() / z.shape[0] for z in self.z0]
        self.scalings = []
        for cov in covariances:
            eps = choose_epsilon_from_eigs(cov, mode="floor", alpha=alpha)
            reg_cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            self.scalings.append(stable_inverse(reg_cov).float())
      #  self.scalings = [
      #      stable_inverse(cov + epsilon * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)).float() 
      #      for cov in covariances]
       # self.scalings = [(stable_inverse(cov)).float() for cov in covariances]
        #+epsilon*torch.ones_like(cov)

    def register_hook(self):
        self.remove_hooks()
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                handle = layer.weight.register_hook(lambda grad, idx=layer_idx: grad @ self.scalings[idx])
                self._hook_handles.append(handle)

    def remove_hooks(self):
        for h in getattr(self, "_hook_handles", []):
            h.remove()
        self._hook_handles = []


class PC_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nm_layers=2, bias=False, scale_method=None):
        super(PC_MLP, self).__init__()
        self.nm_layers = nm_layers
        self.layers = nn.ModuleList()
        for _ in range(nm_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            input_dim = hidden_dim
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))

        scale_methods = {
            'norm': self.get_activity_norm,
            'covariance': self.get_activity_covariance,
        }
        if scale_method is not None:
            self.get_activity = scale_methods[scale_method]
        else:
            self.get_activity = None
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward_with_variances(self, x, y):
        variance = torch.eye(y.shape[-1], device=x.device, dtype=x.dtype)
        previous_activity = self.layers[-1].weight
        variance += previous_activity @ previous_activity.t()
        for layer in self.layers[-2:0:-1]:
            previous_activity = previous_activity @ layer.weight
            variance += previous_activity @ previous_activity.t()
        previous_activity = previous_activity @ self.layers[0].weight
        # invert variance
       # eigvals = torch.linalg.eigvalsh(variance) 
       # print("min eig:", eigvals.min().item(), "max eig:", eigvals.max().item())
       # print("cond ~", (eigvals.max() / eigvals.min().clamp_min(1e-30)).item())

        try:
            L = torch.linalg.cholesky(variance)
        except RuntimeError:
        # Not numerically SPD -> signal "stop learning"
            pred = x @ previous_activity.T
            return pred, torch.zeros_like(variance)  # precision unavailable, not OK

  #      L = torch.linalg.cholesky(variance)
        precision = torch.cholesky_inverse(L)
        return x @ previous_activity.T, precision

    def loss(self, x, y):
        batch_size = x.shape[0]
        y_hat, precision = self.forward_with_variances(x, y)
        loss = 0.5*((y - y_hat) @ precision * (y - y_hat)).sum() / batch_size
        if self.get_activity is not None:
            self.get_activity(x, y, y_hat, precision)
            self.register_hook()
        return loss, y_hat
    
    def get_activities(self, x, y, y_hat, precision):
        r = y - y_hat
        # init/fp activities
        x_fp = x.clone()
        self.z0 = [x_fp] + [x_fp:=layer(x_fp) for layer in self.layers[:-1]] # feed forward activity
        # compute weight products
        weights_prod = [self.layers[-1].weight]
        for layer in reversed(self.layers[1:-1]):
            new_prod = weights_prod[-1] @ layer.weight
            weights_prod.append(new_prod)

        errors = [r @ precision @ wp for wp in weights_prod]
        z = [x]
        for e, layer in zip(reversed(errors), self.layers[:-1]):
            z_new = layer(z[-1]) + e
            z = z + [z_new]
        self.z = z

    def get_activity_norm(self, x, y, y_hat, precision):
        self.get_activities(x, y, y_hat, precision)
        self.scalings = [(1 / ((z*z0).sum(-1).mean().item()+ epsilon)) * torch.eye(z.shape[-1], device=z.device, dtype=z.dtype) for z, z0 in zip(self.z, self.z0)]

    def get_activity_covariance(self, x, y, y_hat, precision):
        self.get_activities(x, y, y_hat, precision)
        covariances = [z0.double().T @ z.double() / z.shape[0] for z0, z in zip(self.z0, self.z)]
        #covariances = [z0.double().T @ z.double() / z.shape[0] for z0, z in zip(self.z0, self.z)]
        self.scalings = []
        for cov in covariances:
            eps = choose_epsilon_from_eigs(cov, mode="floor", alpha=alpha)
            reg_cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            self.scalings.append(stable_inverse(reg_cov).float())
    #    self.scalings = [
    #        stable_inverse(cov + epsilon * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)).float() 
    #        for cov in covariances]
    #    self.scalings = [(stable_inverse(cov)).float() for cov in covariances]
    #+epsilon*torch.ones_like(cov)

    def register_hook(self):
        self.remove_hooks()
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                handle = layer.weight.register_hook(lambda grad, idx=layer_idx: grad @ self.scalings[idx])
                self._hook_handles.append(handle)

    def remove_hooks(self):
        for h in getattr(self, "_hook_handles", []):
            h.remove()
        self._hook_handles = []


def train(model, x, y, lr=1e-3, epochs=1, verbose=True):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dl = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in dl:
        optimizer.zero_grad()
        loss, y_hat = model.loss(x, y)
        loss.backward()
        optimizer.step()



############################# Example usecase #############################
"""
Width sweep + LR sweep per width + plot best test_MSE curve for each width.

Assumptions:
- You have two model classes available: BP_MLP and PC_MLP
- Each model implements: loss(xb, yb) -> (loss_tensor, aux_dict_or_any)
- Models are stacks of nn.Linear layers (bias OK), and we compare effective W_eff to W_data.

If your models are elsewhere, change the import line below.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- your models must be available -------------------
# from your_code import BP_MLP, PC_MLP
# e.g.:
# from models import BP_MLP, PC_MLP


# ===================== CONFIG =====================
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
numruns = len(seeds)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# synthetic data dims
input_dim  = 20
output_dim = 20 #10
n_train    = 7096
n_test     = 2048
batch_size = 64 #64

# widths to sweep
widths    = [15, 20, 40] #[5, 10, 20, 50, 100]  # edit as needed
nm_layers = 1#5

# training budgets
max_steps_sweep = 100#00 #2000
max_steps_curve = 100#00 #2000
eval_every      = 10#00

# LR grid (base grid; we optionally amplify for some models)
lrs = np.linspace(10**(-4), 10**(-1.7), 10)#np.linspace(10**(-3.5), 10**(-0.02), 10) #(-2.9, -0.04, 15)  # ~ 1e-4 ... 1.6e-1

# Optional LR scaling knobs (keep if you want)
amplify_lr_BP_scaled = .8#3#2       # for "(scaled lr)" variants
amplify_lr_PC_scaled = 15#3#50       # for "(scaled lr)" variants
amplify_lr_PC = 15#3#50    # for BP only (if you want)

# optimizer choice
optimizer_cls = torch.optim.SGD
optimizer_kwargs = dict(momentum=0.0)
weight_decay = 0.0


# ===================== INIT HELPERS =====================
def lecun_normal_(tensor: torch.Tensor):
    fan_in = nn.init._calculate_fan_in_and_fan_out(tensor)[0]
    std = math.sqrt(1.0 / fan_in)
    return nn.init.normal_(tensor, mean=0.0, std=std)

def apply_init(model: nn.Module, init_name: str):
    """
    init_name: 'kaiming_uniform' or 'lecun_normal'
    """
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if init_name == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
            elif init_name == "lecun_normal":
                lecun_normal_(m.weight)
            else:
                raise ValueError(f"Unknown init: {init_name}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ===================== REPRO HELPERS =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_synthetic(seed: int):
    """
    Create train/test data with y = W_data x
    """
    g = torch.Generator().manual_seed(seed)

    # W_data: [output_dim, input_dim]
    W_data = torch.randn(output_dim, input_dim, generator=g) / math.sqrt(input_dim)
    print('This is the mapping to learn', W_data)

    X_train = torch.randn(n_train, input_dim, generator=g)
    X_test  = torch.randn(n_test,  input_dim, generator=g)

    Y_train = X_train @ W_data.T
    Y_test  = X_test  @ W_data.T

    return X_train, Y_train, X_test, Y_test, W_data

def make_batches(X, Y, steps, seed):
    """
    Pre-sample a sequence of minibatches so every model/LR sees the same data order.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, X.size(0), (steps, batch_size), generator=g)
    batches = []
    for t in range(steps):
        xb = X[idx[t]].to(device)
        yb = Y[idx[t]].to(device)
        batches.append((xb, yb))
    return batches


# ===================== METRIC: compare effective W to W_data =====================
@torch.no_grad()
def effective_matrix_from_linears(model: nn.Module, device=None, dtype=None):
    """
    Returns W_eff = W_last @ ... @ W_first using only nn.Linear layers.
    Assumes each Linear is y = W x + b, with W shape (out, in).
    Biases are ignored in W_eff.
    """
    Ws = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            Ws.append(m.weight.detach())

    if len(Ws) == 0:
        raise ValueError("No nn.Linear layers found.")

    W_eff = Ws[0]
    if device is not None:
        W_eff = W_eff.to(device)
    if dtype is not None:
        W_eff = W_eff.to(dtype)

    for W in Ws[1:]:
        if device is not None:
            W = W.to(device)
        if dtype is not None:
            W = W.to(dtype)
        W_eff = W @ W_eff  # later layer on the left

    return W_eff

@torch.no_grad()
def test_mse(model, W_data):
    W_eff = effective_matrix_from_linears(model, device=W_data.device, dtype=W_data.dtype)
    if W_eff.shape != W_data.shape:
        raise ValueError(f"Shape mismatch: W_eff {W_eff.shape} vs W_data {W_data.shape}")
    return F.mse_loss(W_eff, W_data, reduction="mean").item() #THINK THIS SHOULD BE SUM


# ===================== TRAIN STEP =====================
def train_step(model, xb, yb, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = model.loss(xb, yb)
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ===================== MODEL FACTORY =====================
def fresh_models(base_state, hidden_dim):
    """
    Create BP / BP_scaled / PC / PC_scaled and load identical weights (base_state).
    """
    bp        = BP_MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, nm_layers=nm_layers)
  #  bp_scaled = BP_MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, nm_layers=nm_layers, scale_method="covariance")
    pc        = PC_MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, nm_layers=nm_layers)
  #  pc_scaled = PC_MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, nm_layers=nm_layers, scale_method="covariance")

    models = [
        ("BP", bp),
   #     ("BP (scaled lr)", bp_scaled),
        ("PC", pc),
   #     ("PC (scaled lr)", pc_scaled),
    ]
    for _, m in models:
        m.load_state_dict(base_state, strict=True)
        m.to(device)
    return models


# ===================== EXPERIMENT =====================
inits = ["lecun_normal"] #["kaiming_uniform", "lecun_normal"]

lr_sweep_rows = []     # per seed, width, init, model, lr -> test_mse
curve_rows    = []     # per seed, width, init, model, step -> test_mse
best_lr_rows  = []     # per seed, width, init, model -> best lr

for seed in seeds:
    set_seed(seed)

    X_train, Y_train, X_test, Y_test, W_data = make_synthetic(seed)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    W_data = W_data.to(device)

    for hidden_dim in widths:
        print(hidden_dim)
        if hidden_dim == 15:
            max_steps_sweep = 10#00 #2000
            max_steps_curve = 10#00 #2000
            eval_every      = 2#00

            # LR grid (base grid; we optionally amplify for some models)
            # lrs = np.linspace(10**(-3.5), 10**(-0.02), 5) #(-2.9, -0.04, 15)  # ~ 1e-4 ... 1.6e-1
            lrs = np.logspace((-3.5), (-0.02), 80)
           

            # for 1 hidden layer it was 1,1,1
            # Optional LR scaling knobs (keep if you want)
            amplify_lr_BP_scaled = 1#2       # for "(scaled lr)" variants
            amplify_lr_PC_scaled = 1#50       # for "(scaled lr)" variants
            amplify_lr_PC = 1#25#3#50    # for BP only (if you want)
        
        if hidden_dim == 20:
            max_steps_sweep = 300#00 #2000
            max_steps_curve = 300#00 #2000
            eval_every      = 5#00

            # LR grid (base grid; we optionally amplify for some models)
            #lrs = np.linspace(10**(-3.5), 10**(-0.02), 5) #(-2.9, -0.04, 15)  # ~ 1e-4 ... 1.6e-1
            lrs = np.logspace((-3.5), (0.1), 80)

            # Optional LR scaling knobs (keep if you want)
            amplify_lr_BP_scaled = .2#2       # for "(scaled lr)" variants
            amplify_lr_PC_scaled = 2.5#50       # for "(scaled lr)" variants
            amplify_lr_PC = 1.4#5#3#50    # for BP only (if you want)
        
        if hidden_dim == 40:
            max_steps_sweep = 50#00 #2000
            max_steps_curve = 50#00 #2000
            eval_every      = 5#00

            # LR grid (base grid; we optionally amplify for some models)
            #lrs = np.linspace(10**(-3.5), 10**(-0.02), 5) #(-2.9, -0.04, 15)  # ~ 1e-4 ... 1.6e-1
            lrs = np.logspace((-3.5), (-0.02), 80)

            # Optional LR scaling knobs (keep if you want)
            amplify_lr_BP_scaled = 1#2       # for "(scaled lr)" variants
            amplify_lr_PC_scaled = 1#50       # for "(scaled lr)" variants
            amplify_lr_PC = 1#3#50    # for BP only (if you want)

        batches_sweep = make_batches(X_train, Y_train, max_steps_sweep, seed=seed+111)
        batches_curve = make_batches(X_train, Y_train, max_steps_curve, seed=seed+111) #222)

        for init_name in inits:
            # --- base init state for this (seed, width, init) ---
            base = BP_MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, nm_layers=nm_layers).to(device)
            apply_init(base, init_name)
            base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

            # --- LR sweep per model ---
            best_lrs = {}

            for model_name, model in fresh_models(base_state, hidden_dim):
                best_err = float("inf")
                best_lr  = None

                for lr0 in lrs:
                    # reset
                    model.load_state_dict(base_state, strict=True)

                    # choose LR (optionally amplify)
                    lr = float(lr0)
                    if model_name == "PC":
                        lr *= amplify_lr_PC
                    if model_name == "PC (scaled lr)":
                        lr *= amplify_lr_PC_scaled
                    if model_name == "BP (scaled lr)":
                        lr *= amplify_lr_BP_scaled
                    #if "scaled" in model_name:
                    #    lr *= amplify_lr

                    opt = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay, **optimizer_kwargs)

                    # train for max_steps_sweep
                    for xb, yb in batches_sweep:
                        train_step(model, xb, yb, opt)

                    err = test_mse(model, W_data)

                    lr_sweep_rows.append({
                        "seed": seed, "width": hidden_dim, "init": init_name,
                        "model": model_name, "lr": float(lr0), "lr_used": lr, "test_mse": err
                    })

                    if err < best_err:
                        best_err = err
                        best_lr  = lr

                best_lrs[model_name] = best_lr
                best_lr_rows.append({
                    "seed": seed, "width": hidden_dim, "init": init_name,
                    "model": model_name, "best_lr": best_lr
                })

            # --- Curves with best LR per model ---
            for model_name, model in fresh_models(base_state, hidden_dim):
                lr = best_lrs[model_name]
                model.load_state_dict(base_state, strict=True)
                opt = optimizer_cls(model.parameters(), lr=float(lr), weight_decay=weight_decay, **optimizer_kwargs)

                # step 0
                curve_rows.append({
                    "seed": seed, "width": hidden_dim, "init": init_name, "model": model_name,
                    "steps": 0, "test_mse": test_mse(model, W_data)
                })

                #print(model,'and', test_mse(model, W_data))

                for step, (xb, yb) in enumerate(batches_curve, start=1):
                    train_step(model, xb, yb, opt)
                    eval_interval = round(eval_every/2) if step <= 100 else eval_every 
                    if (step % eval_interval == 0) or (step == max_steps_curve):
                        curve_rows.append({
                            "seed": seed, "width": hidden_dim, "init": init_name, "model": model_name,
                            "steps": step, "test_mse": test_mse(model, W_data)
                        })


# ===================== AGGREGATE (mean/std over seeds) =====================
df_lr   = pd.DataFrame(lr_sweep_rows)
df_best = pd.DataFrame(best_lr_rows)
df_cur  = pd.DataFrame(curve_rows)

agg_lr = (
    df_lr.groupby(["width", "init", "model", "lr_used"])["test_mse"] #.
         .agg(["mean", "std"])
         .reset_index()
         .rename(columns={"mean": "mse_mean", "std": "mse_std"})
)

agg_cur = (
    df_cur.groupby(["width", "init", "model", "steps"])["test_mse"]
          .agg(["mean", "std"])
          .reset_index()
          .rename(columns={"mean": "mse_mean", "std": "mse_std"})
)


# ===================== PLOT: best curves, one line per width =====================
# --- choose orderings ---
inits_order = ["lecun_normal"]#["kaiming_uniform", "lecun_normal"]
models_order = ["BP", "BP (scaled lr)", "PC", "PC (scaled lr)"]
widths_order = sorted(agg_cur["width"].unique())

# --- style map: color by family, linestyle by scaled/unscaled ---
style = {
    "BP":              dict(color="tab:orange",   linestyle="-"),
    "BP (scaled lr)":  dict(color="tab:orange",   linestyle="--", alpha = 0.3),
    "PC":              dict(color="tab:blue", linestyle="-"),
    "PC (scaled lr)":  dict(color="tab:blue", linestyle="--", alpha = 0.3),
}

# --- make figure: rows = widths, cols = inits ---
fig, axes = plt.subplots(
    len(widths_order), len(inits_order),
    figsize=(3.5 * len(inits_order), 2.8 * len(widths_order)),
    #figsize=(5, 2.8 * len(widths_order)),
    sharex=True, sharey=True
)

# Handle edge cases where rows/cols = 1
if len(widths_order) == 1 and len(inits_order) == 1:
    axes = np.array([[axes]])
elif len(widths_order) == 1:
    axes = np.array([axes])
elif len(inits_order) == 1:
    axes = np.array([[ax] for ax in axes])

for r, width in enumerate(widths_order):
    for c, init_name in enumerate(inits_order):
        ax = axes[r, c]

        sub = agg_cur[(agg_cur["width"] == width) & (agg_cur["init"] == init_name)]

        for model_name in models_order:
            d = sub[sub["model"] == model_name].sort_values("steps").copy()
            if d.empty:
                continue

            x = pd.to_numeric(d["steps"], errors="coerce").to_numpy()
            y = pd.to_numeric(d["mse_mean"], errors="coerce").to_numpy()
            s = pd.to_numeric(d["mse_std"],  errors="coerce").fillna(0.0).to_numpy()

            # Clean invalid values for log plotting
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            s = np.asarray(s, dtype=float)

            sem = s / math.sqrt(numruns)
            lower = np.maximum(y - sem, 1e-16)
            upper = np.maximum(y + sem, 1e-16)

            ax.plot(
                x, y,
                linewidth=2.5, markersize=5.5,
                label=model_name,
                **style[model_name]
            )
            ax.fill_between(x, lower, upper, alpha=0.15, color=style[model_name]["color"])

        # Axes formatting
        ax.tick_params(axis='both')
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')   # light gray
            spine.set_linewidth(0.8) 
        # Add a vertical dashed line at x = 20

        # Titles / labels
        if r == 0:
            ax.set_title(init_name.replace("_", " "))
        if c == 0:
            ax.set_ylabel(r"$\|W_{\mathrm{data}} - W_{L:1}\|_2^2$")
           # ax.set_ylabel(f"width={width}\nTest MSE")
        if r == len(widths_order) - 1:
            ax.set_xlabel("Training steps")

# Single legend for the whole figure
handles, labels = axes[0, 0].get_legend_handles_labels()

plt.tight_layout(rect=[0, 0, 1, 1])
# Save inside the same folder (figure2)
file_path = os.path.join(script_dir, "Training_curves_final.png")
plt.savefig(file_path, dpi=300)

# --- make figure: rows = widths, cols = inits ---
fig, axes = plt.subplots(
    len(widths_order), len(inits_order),
    figsize=(3.5 * len(inits_order), 2.8 * len(widths_order)),
    #figsize=(12, 2.8 * len(widths_order)),
    sharex=True, sharey=True
)
style = {
    "BP":              dict(color="tab:orange",   linestyle="-",  marker="o"),
    "BP (scaled lr)":  dict(color="tab:orange",   linestyle="--", marker="o", alpha = 0.3),
    "PC":              dict(color="tab:blue", linestyle="-",  marker="o"),
    "PC (scaled lr)":  dict(color="tab:blue", linestyle="--", marker="o", alpha = 0.3),
}

# Handle edge cases where rows/cols = 1
if len(widths_order) == 1 and len(inits_order) == 1:
    axes = np.array([[axes]])
elif len(widths_order) == 1:
    axes = np.array([axes])
elif len(inits_order) == 1:
    axes = np.array([[ax] for ax in axes])

for r, width in enumerate(widths_order):
    for c, init_name in enumerate(inits_order):
        ax = axes[r, c]

        sub = agg_lr[(agg_lr["width"] == width) & (agg_lr["init"] == init_name)]

        for model_name in models_order:
            d = sub[sub["model"] == model_name].sort_values("lr_used").copy() #.
            if d.empty:
                continue

            x = pd.to_numeric(d["lr_used"], errors="coerce").to_numpy() #.
            y = pd.to_numeric(d["mse_mean"], errors="coerce").to_numpy()
            s = pd.to_numeric(d["mse_std"],  errors="coerce").fillna(0.0).to_numpy()

            # Clean invalid values for log plotting
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            s = np.asarray(s, dtype=float)

            sem = s / math.sqrt(numruns)
            lower = np.maximum(y - sem, 1e-16)
            upper = np.maximum(y + sem, 1e-16)

            ax.plot(
                x, y,
                linewidth=2.5, markersize=4.5,
                label=model_name,
                **style[model_name]
            )
            ax.fill_between(x, lower, upper, alpha=0.15, color=style[model_name]["color"])

        # Axes formatting
        ax.set_xscale("log")
        ax.set_yscale("log")
        #ax.set_ylim(1e-16, 1e-1)
      #  ax.set_xlim(-1, max_steps_sweep)
       # ax.set_ylim(1e-4, 1e-1)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')   # light gray
            spine.set_linewidth(0.8) 

        # Titles / labels
        if r == 0:
            ax.set_title(init_name.replace("_", " "))
        if c == 0:
            ax.set_ylabel(r"$\|W_{\mathrm{data}} - W_{L:1}\|_2^2$")
            #ax.set_ylabel(f"width={width}\nTest MSE")
        if r == len(widths_order) - 1:
            ax.set_xlabel("Learning rates (log)")

# Single legend for the whole figure
handles, labels = axes[0, 0].get_legend_handles_labels()

plt.tight_layout(rect=[0, 0, 1, 1])
file_path_lrsweep = os.path.join(script_dir, "lr_sweeps_final.png")
plt.savefig(file_path_lrsweep, dpi=300)

# --- compute best MSE per (width, init, model) ---
best = (
    agg_cur
    .groupby(["width", "init", "model"])
    .agg(best_mse=("mse_mean", "min"))
    .reset_index()
)

# --- make figure: one column per init ---
fig, axes = plt.subplots(1, len(inits_order),
                         figsize = (3.5 * len(inits_order), 3.1),
                         #figsize=(6 * len(inits_order), 4),
                         sharey=True)

if len(inits_order) == 1:
    axes = [axes]

for c, init_name in enumerate(inits_order):
    ax = axes[c]

    sub = best[best["init"] == init_name]

    for model_name in models_order:
        d = sub[sub["model"] == model_name].sort_values("width")
        if d.empty:
            continue

        x = d["width"].to_numpy(dtype=float)
        y = d["best_mse"].to_numpy(dtype=float)

      #  mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
      #  x, y = x[mask], y[mask]
       # if len(x) == 0:
        #    continue

        ax.plot(
            x, y,
            linewidth=2.5,
            markersize=6,
            label=model_name,
            **style[model_name]
        )

    ax.set_yscale("log")
    ax.set_xlabel("Width")
    ax.set_title(init_name.replace("_", " "))
    ax.grid(True, alpha=0.25)
    ax.axvline(x=20, linestyle='dashed', color='grey')
    for spine in ax.spines.values():
            spine.set_color('#CCCCCC')   # light gray
            spine.set_linewidth(0.8) 

axes[0].set_ylabel(r"$\|W_{\mathrm{data}} - W_{L:1}\|_2^2$") # set_ylabel("Best Test MSE")

# single legend
handles, labels = axes[0].get_legend_handles_labels()

plt.tight_layout(rect=[0, 0, 1, 0.92])
file_path_width = os.path.join(script_dir, "widths_sweep_final.png")
plt.savefig(file_path_width, dpi=300)

# ===================== OPTIONAL: print best LR summary =====================
print("\nBest LR per (width, init, model): mean ± std over seeds")
summary = (
    df_best.groupby(["width", "init", "model"])["best_lr"]
           .agg(["mean", "std"])
           .reset_index()
           .sort_values(["width", "init", "model"])
)
print(summary.to_string(index=False))