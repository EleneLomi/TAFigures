import math
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Sampler
from lightning import Callback, LightningModule, Trainer, seed_everything
from torchvision import datasets, transforms

from pc_e_scaled_fixed import PCAutoencoder as PCEAutoencoder


# ===
# Config
# ===
SEED = 42
RUN_SEEDS = [0]#, 1]#, 3, 4]

DATA_ROOT = "/Users/elenelominadze/TargetAlignment/data"

BATCH_SIZE = 64
SWEEP_STEPS = 100
CURVE_STEPS = 100
EVAL_EVERY = 10

PC_E_LR = 0.05
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]


# ===
# Data
# ===
def make_mnist_datasets(root=DATA_ROOT):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # keep inputs in [0,1] for autoencoder with sigmoid output
    ])

    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=False,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=False,
        transform=transform,
    )

    return train_dataset, test_dataset


# ===
# Fixed minibatch machinery
# ===
class FixedBatchSampler(Sampler):
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


def make_fixed_batches(dataset_size, batch_size, num_steps, seed):
    """
    Precompute exactly num_steps batches of indices.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    batches = []
    order = torch.randperm(dataset_size, generator=g).tolist()
    ptr = 0

    while len(batches) < num_steps:
        if ptr + batch_size > len(order):
            order = torch.randperm(dataset_size, generator=g).tolist()
            ptr = 0

        batch = order[ptr:ptr + batch_size]
        batches.append(batch)
        ptr += batch_size

    return batches


def make_loader_from_fixed_batches(dataset, fixed_batches, num_workers=0):
    return DataLoader(
        dataset,
        batch_sampler=FixedBatchSampler(fixed_batches),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ===
# Initialization
# ===
class MyLinear(nn.Linear):
    def reset_parameters(self):
        fan_in = self.weight.size(1)
        std = math.sqrt(2.0 / fan_in) #WE NEED TO MULTIPLY BY SQRT 2
        ###std = math.sqrt(1.0 / fan_in)  # LeCun normal
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class LinearOutput(nn.Linear):
    def reset_parameters(self):
        fan_in = self.weight.size(1)
        std = math.sqrt(1.0 / fan_in)  # LeCun normal
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

def make_architecture():
    # Small autoencoder: 784 -> 128 -> 32 -> 128 -> 784
    return [
        nn.Sequential(MyLinear(28 * 28, 128), nn.ReLU()),#nn.GELU()),
        nn.Sequential(MyLinear(128, 32), nn.ReLU()),# nn.GELU()),
        nn.Sequential(MyLinear(32, 128), nn.ReLU()), #nn.GELU()),
        nn.Sequential(LinearOutput(128, 28 * 28), nn.Sigmoid()),
    ]


def clone_architecture_from_reference(reference_arch):
    new_arch = make_architecture()
    for new_layer, ref_layer in zip(new_arch, reference_arch):
        new_layer.load_state_dict(deepcopy(ref_layer.state_dict()))
    return new_arch


# ===
# Helpers
# ===
def flatten_if_needed(x):
    if x.ndim > 2:
        return x.view(x.size(0), -1)
    return x


# ===
# BP autoencoder model
# ===
class BPAutoencoder(LightningModule):
    def __init__(self, architecture, w_lr=1e-3, flatten_input=True):
        super().__init__()
        self.save_hyperparameters(ignore=["architecture"])
        self.net = nn.Sequential(*architecture)
        self.w_lr = w_lr
        self.flatten_input = flatten_input

    def forward(self, x):
        if self.flatten_input:
            x = flatten_if_needed(x)
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.w_lr)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        target = flatten_if_needed(x) if self.flatten_input else x
        outputs = self(x)

        loss = 0.5 * F.mse_loss(outputs, target, reduction="sum") #F.mse_loss(outputs, target) 
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss


# ===
# Model factories
# ===
def make_bp_model(w_lr, reference_arch):
    arch = clone_architecture_from_reference(reference_arch)
    return BPAutoencoder(
        architecture=arch,
        w_lr=w_lr,
        flatten_input=True,
    )


def make_pc_model(w_lr, reference_arch):
    arch = clone_architecture_from_reference(reference_arch)
    return PCEAutoencoder(
        architecture=arch,
        iters=32,
        e_lr=PC_E_LR,
        w_lr=w_lr,
        #scale_method='covariance', #ADDED HERE
       # use_cross_entropy=False,
        flatten_input=True,
    )

def make_pc_model_scaled(w_lr, reference_arch):
    arch = clone_architecture_from_reference(reference_arch)
    return PCEAutoencoder(
        architecture=arch,
        iters=32,
        e_lr=PC_E_LR,
        w_lr=w_lr,
        scale_method='covariance', #ADDED HERE
       # use_cross_entropy=False,
        flatten_input=True,
    )


# ===
# Evaluation
# ===
@torch.no_grad()
def compute_test_loss(model, test_loader):
    """
    Average reconstruction MSE over the full test set.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_n = 0
    device = model.device

    for x, _ in test_loader:
        x = x.to(device)
        target = flatten_if_needed(x)
        if isinstance(model, PCEAutoencoder):
        #    print("got to here")
            model.forward(x=x)
            outputs = model.y_pred(x)
        else:
            outputs = model(x)

        loss = F.mse_loss(outputs, target, reduction="sum")
        total_loss += loss.item()
        total_n += x.size(0)
        finalsize = x.size(0)

    if was_training:
        model.train()

    return total_loss / total_n


# ===
# Callback for training curves
# ===
class TestLossCallback(Callback):
    def __init__(self, test_loader, eval_every=50, name="model"):
        super().__init__()
        self.test_loader = test_loader
        self.eval_every = eval_every
        self.name = name
        self.steps = []
        self.test_losses = []

    def _evaluate(self, trainer, pl_module):
        loss = compute_test_loss(pl_module, self.test_loader)
        self.steps.append(trainer.global_step)
        self.test_losses.append(loss)
        print(f"[{self.name}] step={trainer.global_step:4d} test_loss={loss:.6f}")

    def on_fit_start(self, trainer, pl_module):
        self._evaluate(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > 0 and trainer.global_step % self.eval_every == 0:
            self._evaluate(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        if len(self.steps) == 0 or self.steps[-1] != trainer.global_step:
            self._evaluate(trainer, pl_module)


# ===
# Trainer helper
# ===
def make_trainer(max_steps, callbacks=None, is_pc=False):
    return Trainer(
        max_steps=max_steps,
        accelerator="auto",
        devices=1,
        log_every_n_steps=min(50, max_steps),
        callbacks=callbacks or [],
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        inference_mode=False if is_pc else True,
    )


# ===
# Multi-run matched LR sweep
# ===
def sweep_learning_rates_matched_multi(
    train_dataset,
    test_loader,
    lrs,
    max_steps,
    batch_size,
    run_seeds,
):
    bp_results = []
    pc_results = []
    pc_results_scaled = []

    for lr in lrs:
        bp_losses = []
        pc_losses = []
        pc_losses_scaled = []

        for run_seed in run_seeds:
            print(f"\n=== LR {lr:.5g} | seed {run_seed} ===")
            seed_everything(run_seed)

            # same initialization for BP and PC within this seed
            reference_arch = make_architecture()

            # same minibatch order for BP and PC within this seed
            fixed_batches = make_fixed_batches(
                dataset_size=len(train_dataset),
                batch_size=batch_size,
                num_steps=max_steps,
                seed=run_seed,
            )

            train_loader_bp = make_loader_from_fixed_batches(train_dataset, fixed_batches, num_workers=0)
            train_loader_pc = make_loader_from_fixed_batches(train_dataset, fixed_batches, num_workers=0)
            ## ADDED
            train_loader_pc_scaled = make_loader_from_fixed_batches(train_dataset, fixed_batches, num_workers=0)


            bp_model = make_bp_model(lr, reference_arch)
            pc_model = make_pc_model(lr, reference_arch)
            ## ADDED
            pc_model_scaled = make_pc_model_scaled(lr, reference_arch)

            trainer_bp = make_trainer(max_steps=max_steps, is_pc=False)
            trainer_pc = make_trainer(max_steps=max_steps, is_pc=True)
            trainer_pc_scaled = make_trainer(max_steps=max_steps, is_pc=True)

            print(f"[seed={run_seed}] Training BP | lr={lr:.5g}")
            trainer_bp.fit(bp_model, train_dataloaders=train_loader_bp)
            bp_test_loss = compute_test_loss(bp_model, test_loader)
            bp_losses.append(bp_test_loss)

            print(f"[seed={run_seed}] Training PC | lr={lr:.5g}")
            trainer_pc.fit(pc_model, train_dataloaders=train_loader_pc)
            pc_test_loss = compute_test_loss(pc_model, test_loader)
            pc_losses.append(pc_test_loss)

            print(f"[seed={run_seed}] Training PC scaled | lr={lr:.5g}")
            trainer_pc_scaled.fit(pc_model_scaled, train_dataloaders=train_loader_pc_scaled)
            pc_test_loss_scaled = compute_test_loss(pc_model_scaled, test_loader)
            pc_losses_scaled.append(pc_test_loss_scaled)

            print(
                f"[seed={run_seed}] lr={lr:.5g} | "
                f"BP test loss={bp_test_loss:.6f} | "
                f"PC test loss={pc_test_loss:.6f}"
            )

        bp_mean = float(torch.tensor(bp_losses).mean())
        bp_std = float(torch.tensor(bp_losses).std(unbiased=False))

        pc_mean = float(torch.tensor(pc_losses).mean())
        pc_std = float(torch.tensor(pc_losses).std(unbiased=False))

        pc_mean_scaled = float(torch.tensor(pc_losses_scaled).mean())
        pc_std_scaled = float(torch.tensor(pc_losses_scaled).std(unbiased=False))

        bp_results.append({
            "lr": lr,
            "mean_test_loss": bp_mean,
            "std_test_loss": bp_std,
            "all_test_losses": bp_losses,
        })

        pc_results.append({
            "lr": lr,
            "mean_test_loss": pc_mean,
            "std_test_loss": pc_std,
            "all_test_losses": pc_losses,
        })

        pc_results_scaled.append({
            "lr": lr,
            "mean_test_loss": pc_mean_scaled,
            "std_test_loss": pc_std_scaled,
            "all_test_losses": pc_losses_scaled,
        })


        print(
            f"\nLR {lr:.5g} summary | "
            f"BP mean={bp_mean:.6f} ± {bp_std:.6f} | "
            f"PC mean={pc_mean:.6f} ± {pc_std:.6f}\n"
            f"PC scaled mean={pc_mean_scaled:.6f} ± {pc_std_scaled:.6f}\n"
        )

    bp_best = min(bp_results, key=lambda d: d["mean_test_loss"])
    pc_best = min(pc_results, key=lambda d: d["mean_test_loss"])
    pc_best_scaled = min(pc_results_scaled, key=lambda d: d["mean_test_loss"])

    print(f"Best BP lr: {bp_best['lr']:.5g} with mean test loss={bp_best['mean_test_loss']:.6f}")
    print(f"Best PC lr: {pc_best['lr']:.5g} with mean test loss={pc_best['mean_test_loss']:.6f}")
    print(f"Best PC scaled lr: {pc_best_scaled['lr']:.5g} with mean test loss={pc_best_scaled['mean_test_loss']:.6f}")

    return bp_results, bp_best, pc_results, pc_best, pc_results_scaled, pc_best_scaled


# ===
# Multi-run final curves
# ===
def run_final_curves_multi(
    train_dataset,
    test_loader,
    bp_best_lr,
    pc_best_lr,
    pc_best_lr_scaled,
    curve_steps,
    batch_size,
    eval_every,
    run_seeds,
):
    bp_runs = []
    pc_runs = []
    pc_runs_scaled = []

    for run_seed in run_seeds:
        print(f"\n=== Final curves | seed {run_seed} ===")
        seed_everything(run_seed)

        # same initialization for BP and PC within this seed
        reference_arch = make_architecture()

        # same minibatch order for BP and PC within this seed
        fixed_batches = make_fixed_batches(
            dataset_size=len(train_dataset),
            batch_size=batch_size,
            num_steps=curve_steps,
            seed=run_seed,
        )

        train_loader_bp = make_loader_from_fixed_batches(train_dataset, fixed_batches, num_workers=0)
        train_loader_pc = make_loader_from_fixed_batches(train_dataset, fixed_batches, num_workers=0)
        train_loader_pc_scaled = make_loader_from_fixed_batches(train_dataset, fixed_batches, num_workers=0)

        bp_model = make_bp_model(bp_best_lr, reference_arch)
        pc_model = make_pc_model(pc_best_lr, reference_arch)
        pc_model_scaled = make_pc_model_scaled(pc_best_lr_scaled, reference_arch)

        bp_callback = TestLossCallback(
            test_loader=test_loader,
            eval_every=eval_every,
            name=f"BP seed {run_seed}",
        )
        pc_callback = TestLossCallback(
            test_loader=test_loader,
            eval_every=eval_every,
            name=f"PC seed {run_seed}",
        )
        pc_callback_scaled = TestLossCallback(
            test_loader=test_loader,
            eval_every=eval_every,
            name=f"PC scaled seed {run_seed}",
        )

        trainer_bp = make_trainer(
            max_steps=curve_steps,
            callbacks=[bp_callback],
            is_pc=False,
        )
        trainer_pc = make_trainer(
            max_steps=curve_steps,
            callbacks=[pc_callback],
            is_pc=True,
        )
        trainer_pc_scaled = make_trainer(
            max_steps=curve_steps,
            callbacks=[pc_callback_scaled],
            is_pc=True,
        )

        print(f"[seed={run_seed}] Training BP with best LR={bp_best_lr:.5g}")
        trainer_bp.fit(bp_model, train_dataloaders=train_loader_bp)

        print(f"[seed={run_seed}] Training PC with best LR={pc_best_lr:.5g}")
        trainer_pc.fit(pc_model, train_dataloaders=train_loader_pc)

        print(f"[seed={run_seed}] Training PC scaled with best LR={pc_best_lr_scaled:.5g}")
        trainer_pc_scaled.fit(pc_model_scaled, train_dataloaders=train_loader_pc_scaled)

        bp_runs.append({
            "steps": bp_callback.steps,
            "test_losses": bp_callback.test_losses,
        })
        pc_runs.append({
            "steps": pc_callback.steps,
            "test_losses": pc_callback.test_losses,
        })
        pc_runs_scaled.append({
            "steps": pc_callback_scaled.steps,
            "test_losses": pc_callback_scaled.test_losses,
        })

    return bp_runs, pc_runs, pc_runs_scaled


# ===
# Aggregation
# ===
def aggregate_curve_runs(runs):
    steps = runs[0]["steps"]
    losses = torch.tensor([r["test_losses"] for r in runs], dtype=torch.float32)

    mean = losses.mean(dim=0).tolist()
    std = losses.std(dim=0, unbiased=False).tolist()

    return steps, mean, std


# ===
# Plotting
# ===
def plot_lr_sweep(bp_results, pc_results, pc_results_scaled):
    plt.figure(figsize=(2.8, 3))

    plt.errorbar(
        [r["lr"] for r in bp_results],
        [r["mean_test_loss"] for r in bp_results],
        yerr=[r["std_test_loss"] for r in bp_results],
        marker="o",
    #    linewidth=2.5,
    #    capsize=4,
        label="BP",
        color = '#ff7f0e',
    )

    plt.errorbar(
        [r["lr"] for r in pc_results],
        [r["mean_test_loss"] for r in pc_results],
        yerr=[r["std_test_loss"] for r in pc_results],
        marker="o",
    #    linewidth=2.5,
    #    capsize=4,
        label="PC",
        color = '#1f77b4',
    )

    plt.errorbar(
        [r["lr"] for r in pc_results_scaled],
        [r["mean_test_loss"] for r in pc_results_scaled],
        yerr=[r["std_test_loss"] for r in pc_results_scaled],
        marker="o",
    #    linewidth=2.5,
    #    capsize=4,
        label="PC scaled",
        color = '#1f77b4', 
        linestyle = "dashed",
        alpha=0.5,
    )

    plt.tick_params(axis='both')#, labelsize=18)
    plt.xscale("log")
    plt.xlabel("Learning rate")#,fontsize=18)
    plt.ylabel("Final test reconstruction loss")#,fontsize=18)
    plt.title("MNIST autoencoder")#,fontsize=18)
    plt.legend()
    # Lighten the box (spines)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.show()
    plt.savefig('MNIST_autoencoder_lrsweep.png', dpi=300)


def plot_test_loss_curves_multi(bp_runs, pc_runs, pc_runs_scaled):
    bp_steps, bp_mean, bp_std = aggregate_curve_runs(bp_runs)
    pc_steps, pc_mean, pc_std = aggregate_curve_runs(pc_runs)
    pc_steps_scaled, pc_mean_scaled, pc_std_scaled = aggregate_curve_runs(pc_runs_scaled)

    plt.figure(figsize=(2.8, 3))

    plt.plot(bp_steps, bp_mean, label="BP", color = '#ff7f0e')#, linewidth=2.5)
    plt.fill_between(
        bp_steps,
        [m - s for m, s in zip(bp_mean, bp_std)],
        [m + s for m, s in zip(bp_mean, bp_std)],
        alpha=0.2,
    )

    plt.plot(pc_steps, pc_mean, label="PC", color = '#1f77b4')#, linewidth=2.5)
    plt.fill_between(
        pc_steps,
        [m - s for m, s in zip(pc_mean, pc_std)],
        [m + s for m, s in zip(pc_mean, pc_std)],
        alpha=0.2,
    )

    plt.plot(pc_steps_scaled, pc_mean_scaled, label="PC_scaled", color = '#1f77b4', linestyle = "dashed", alpha=0.5)#, linewidth=2.5)
    plt.fill_between(
        pc_steps_scaled,
        [m - s for m, s in zip(pc_mean_scaled, pc_std_scaled)],
        [m + s for m, s in zip(pc_mean_scaled, pc_std_scaled)],
        alpha=0.2,
    )

    plt.tick_params(axis='both')#, labelsize=18)
    plt.xlabel("Training steps")#, fontsize=18)
    plt.ylabel("Test reconstruction loss")#, fontsize=18)
    plt.title("MNIST autoencoder")#,fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Lighten the box (spines)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    plt.tight_layout()
    #plt.show()
    plt.savefig('MNIST_autoencoder.png', dpi=300)


# ===
# Main
# ===
def main():
    seed_everything(SEED)

    train_dataset, test_dataset = make_mnist_datasets()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # ----
    # 1) LR sweep with matched initialization + minibatches
    # ----
    bp_results, bp_best, pc_results, pc_best, pc_results_scaled, pc_best_scaled  = sweep_learning_rates_matched_multi(
        train_dataset=train_dataset,
        test_loader=test_loader,
        lrs=LRS,
        max_steps=SWEEP_STEPS,
        batch_size=BATCH_SIZE,
        run_seeds=RUN_SEEDS,
    )

    plot_lr_sweep(bp_results, pc_results, pc_results_scaled)

    # ----
    # 2) Final curves with best LRs and matched minibatches
    # ----
    bp_runs, pc_runs, pc_runs_scaled = run_final_curves_multi(
        train_dataset=train_dataset,
        test_loader=test_loader,
        bp_best_lr=bp_best["lr"],
        pc_best_lr=pc_best["lr"],
        pc_best_lr_scaled=pc_best_scaled["lr"],
        curve_steps=CURVE_STEPS,
        batch_size=BATCH_SIZE,
        eval_every=EVAL_EVERY,
        run_seeds=RUN_SEEDS,
    )

    plot_test_loss_curves_multi(bp_runs, pc_runs, pc_runs_scaled)


if __name__ == "__main__":
    main()