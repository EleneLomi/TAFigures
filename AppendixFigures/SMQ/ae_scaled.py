"""
Appendix Q, Table tab:ae: scaled PC-vs-BP autoencoder learning-speed study.

Sweeps over:
  * dataset:    CIFAR10 / CIFAR100 / STL10  (same AE reconstruction task, harder = STL10/CIFAR100)
  * depth:      shallow ([1024]) vs deep ([1024,512,256]) encoder (mirrored decoder)
  * bottleneck: swept
  * seeds:      multiple, aggregated to mean +/- std (error bars)

Fixed choices:
  * orthogonal / norm-preserving init  -> keeps PC in its ideal regime AND lets inference equilibrate
  * Adam inference, fresh each weight update (warm-start across shuffled minibatches hurts)
  * fixed inference-step count (no early stop), energy_scale = 1 (weight LR swept, best kept)

Outputs per (dataset,depth): final-MSE-vs-bottleneck (BP vs PC, mean+/-std bands) + learning curves.
Datasets are downloaded to $DATA_ROOT (default ./data) if they are not already there.
See run_all.sh for the exact commands behind the table.
"""

import argparse, csv, json, os, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision
import torchvision.transforms.v2 as v2
from torch import nn
from pc_e import PCE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")


# ---------------------------------------------------------------- data
def load_dataset(name, n_val=2000):
    tfm = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                      v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def stack(ds, n=None):
        loader = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, num_workers=4)
        xs = [xb for xb, _ in loader]
        x = torch.cat(xs, 0)
        return x if n is None else x[:n]

    if name == "CIFAR10":
        tr = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tfm)
        te = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tfm)
    elif name == "CIFAR100":
        tr = torchvision.datasets.CIFAR100(DATA_ROOT, train=True, download=True, transform=tfm)
        te = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=tfm)
    elif name == "STL10":
        tr = torchvision.datasets.STL10(DATA_ROOT, split="train", download=True, transform=tfm)
        te = torchvision.datasets.STL10(DATA_ROOT, split="test", download=True, transform=tfm)
    else:
        raise ValueError(name)

    x_train = stack(tr).to(DEVICE)
    x_val = stack(te, n_val).to(DEVICE)
    chw = tuple(x_train.shape[1:])
    return x_train, x_val, chw


# ---------------------------------------------------------------- architecture
def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_ae(chw, hidden_dims, bottleneck, seed=0):
    """Bias-free MLP autoencoder: in -> hidden_dims -> bottleneck -> reversed(hidden_dims) -> in."""
    torch.manual_seed(seed)
    in_dim = int(np.prod(chw))
    dims = [in_dim] + list(hidden_dims) + [bottleneck]
    enc_sizes = list(zip(dims[:-1], dims[1:]))          # encoder linear shapes
    dec_sizes = [(b, a) for (a, b) in reversed(enc_sizes)]  # mirror

    layers = [nn.Sequential(nn.Flatten(), nn.Linear(enc_sizes[0][0], enc_sizes[0][1], bias=False), nn.ReLU())]
    for a, b in enc_sizes[1:]:
        layers.append(nn.Sequential(nn.Linear(a, b, bias=False), nn.ReLU()))
    for i, (a, b) in enumerate(dec_sizes):
        if i < len(dec_sizes) - 1:
            layers.append(nn.Sequential(nn.Linear(a, b, bias=False), nn.ReLU()))
        else:
            layers.append(nn.Sequential(nn.Linear(a, b, bias=False), nn.Unflatten(1, chw)))
    for blk in layers:
        blk.apply(orthogonal_init)
    return layers


# ---------------------------------------------------------------- helpers
@torch.no_grad()
def val_mse(ff, x_val, batch=1024):
    tot, n = 0.0, 0
    for i in range(0, x_val.shape[0], batch):
        xb = x_val[i:i + batch]
        tot += torch.nn.functional.mse_loss(ff(xb), xb, reduction="sum").item()
        n += xb.numel()
    return tot / n


def batches(x_train, bs, n_updates, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    n = x_train.shape[0]
    for _ in range(n_updates):
        idx = torch.randint(0, n, (bs,), generator=g, device=DEVICE)
        yield x_train[idx]


# ---------------------------------------------------------------- BP
def train_bp(chw, hidden, bottleneck, w_lr, x_train, x_val, bs, n_updates, eval_every, seed,
             w_opt="sgd"):
    model = nn.ModuleList(build_ae(chw, hidden, bottleneck, seed)).to(DEVICE)

    def ff(x):
        s = x
        for l in model:
            s = l(s)
        return s

    opt = (torch.optim.Adam(model.parameters(), lr=w_lr) if w_opt == "adam"
           else torch.optim.SGD(model.parameters(), lr=w_lr, momentum=0.0))
    curve = [(0, val_mse(ff, x_val))]
    for step, xb in enumerate(batches(x_train, bs, n_updates, seed), 1):
        opt.zero_grad()
        loss = 0.5 * torch.nn.functional.mse_loss(ff(xb), xb, reduction="sum") / bs
        if not torch.isfinite(loss):
            curve.append((step, float("inf"))); break
        loss.backward(); opt.step()
        if step % eval_every == 0:
            curve.append((step, val_mse(ff, x_val)))
    return curve


# ---------------------------------------------------------------- PC (orthogonal init)
# WEIGHT optimiser: w_opt in {"sgd","adam"} (default SGD, momentum 0).
# INFERENCE optimiser: infer_opt in {"adam","sgd"}.
def train_pc(chw, hidden, bottleneck, w_lr, iters, e_lr, x_train, x_val, bs, n_updates, eval_every,
             seed, infer_opt="adam", w_opt="sgd"):
    arch = build_ae(chw, hidden, bottleneck, seed)
    pce = PCE(arch, iters=iters, e_lr=e_lr, w_lr=w_lr, output_loss="mse",
              nm_batches=n_updates, nm_epochs=1, w_optimizer="sgd", w_scheduler="none").to(DEVICE)
    pce.log = lambda *a, **k: None
    pce.energy_scale = 1.0
    opt = (torch.optim.Adam(pce.layers.parameters(), lr=w_lr) if w_opt == "adam"
           else torch.optim.SGD(pce.layers.parameters(), lr=w_lr, momentum=0.0))

    def ff(x):
        return pce.y_pred_feedforward(x)

    def make_e_opt(errs):
        return (torch.optim.Adam(errs, lr=e_lr) if infer_opt == "adam"
                else torch.optim.SGD(errs, lr=e_lr))

    curve = [(0, val_mse(ff, x_val))]
    last_conv = float("nan")
    for step, xb in enumerate(batches(x_train, bs, n_updates, seed), 1):
        for p in pce.layers.parameters():
            p.requires_grad_(False)
        pce.init_zero_errors(xb)
        e_opt = make_e_opt(pce.errors)
        for _ in range(iters):
            e_opt.zero_grad(); E = pce.E(xb, xb); E.backward(); e_opt.step()
        for p in pce.layers.parameters():
            p.requires_grad_(True)
        opt.zero_grad()
        E_total, _ = pce.E_local(xb, xb)
        loss = E_total / bs
        if not torch.isfinite(loss):
            curve.append((step, float("inf"))); break
        loss.backward(); opt.step()
        if step % eval_every == 0:
            curve.append((step, val_mse(ff, x_val)))
            last_conv = sum(pce.grad_norm_ratio(e, a).mean().item()
                            for e, a in zip(pce.errors, pce.activities)) / max(1, len(pce.errors))
    return curve, last_conv


DEPTHS = {"shallow": [1024], "deep": [1024, 512, 256]}


# ---------------------------------------------------------------- driver
# Protocol: (1) grid-search the weight LR with a SINGLE seed (grid_seed) to pick the best LR
#           per config; (2) re-run that best LR over ALL seeds and report mean±std.
def grid_then_seeds(run_fn, lrs, grid_seed, seeds):
    """run_fn(lr, seed) -> (curve, conv).  Returns dict with sweep (1 seed) + seed_runs (best LR)."""
    sweep, best_lr, best_final = [], None, float("inf")
    for lr in lrs:
        curve, conv = run_fn(lr, grid_seed)
        final = curve[-1][1]
        if not np.isfinite(final):
            final = float("inf")
        sweep.append({"w_lr": lr, "final": final, "conv": conv})
        if final < best_final:
            best_final, best_lr = final, lr
    if best_lr is None:                      # every LR diverged -> fall back (result stays inf)
        best_lr = lrs[0]
        print(f"[warn] all LRs diverged; falling back to lr={best_lr:g}", flush=True)
    seed_runs = []
    for s in seeds:
        curve, conv = run_fn(best_lr, s)
        final = curve[-1][1]
        seed_runs.append({"seed": s, "final": (final if np.isfinite(final) else float("inf")),
                          "conv": conv, "curve": curve})
    finals = [r["final"] for r in seed_runs if np.isfinite(r["final"])]
    return {"best_w_lr": best_lr, "grid_seed": grid_seed, "sweep": sweep, "seed_runs": seed_runs,
            "mean_final": float(np.mean(finals)) if finals else float("inf"),
            "std_final": float(np.std(finals)) if finals else float("inf"),
            "n_seeds": len(finals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output_data")
    ap.add_argument("--datasets", nargs="+", default=["CIFAR10", "CIFAR100"])
    ap.add_argument("--depths", nargs="+", default=["shallow", "deep"])
    ap.add_argument("--bottlenecks", type=int, nargs="+", default=[256, 128, 64, 32, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="seeds for the final mean±std at the best LR")
    ap.add_argument("--grid_seed", type=int, default=None,
                    help="single seed used for the LR grid search (default = seeds[0])")
    ap.add_argument("--pc_iters", type=int, nargs="+", default=[1, 100])
    ap.add_argument("--wlrs", type=float, nargs="+", default=[3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--bp_wlrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--e_lr", type=float, default=0.01)
    ap.add_argument("--infer_opt", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--bp_opt", choices=["sgd", "adam"], default="sgd",
                    help="weight optimiser for the BP baseline")
    ap.add_argument("--pc_opt", choices=["sgd", "adam"], default="sgd",
                    help="weight optimiser for PC (inference optimiser is set by --infer_opt)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--n_updates", type=int, default=300)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--tag", default="scaled")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    grid_seed = args.grid_seed if args.grid_seed is not None else args.seeds[0]
    t0 = time.time()
    results = {"config": vars(args), "runs": []}

    for dset in args.datasets:
        print(f"[data] loading {dset} ...", flush=True)
        x_train, x_val, chw = load_dataset(dset)
        print(f"[data] {dset} train={tuple(x_train.shape)} val={tuple(x_val.shape)} ({time.time()-t0:.0f}s)", flush=True)

        for depth in args.depths:
            hidden = DEPTHS[depth]
            for bn in args.bottlenecks:
                # BP (grid LR on 1 seed, then best over seeds)
                res = grid_then_seeds(
                    lambda lr, s: (train_bp(chw, hidden, bn, lr, x_train, x_val,
                                            args.batch_size, args.n_updates, args.eval_every, s,
                                            w_opt=args.bp_opt), None),
                    args.bp_wlrs, grid_seed, args.seeds)
                res.update({"dataset": dset, "depth": depth, "bottleneck": bn, "method": "BP", "iters": 0})
                results["runs"].append(res)
                print(f"[BP ] {dset} {depth} bn={bn} best_lr={res['best_w_lr']:g} "
                      f"mean={res['mean_final']:.4f}±{res['std_final']:.4f} ({time.time()-t0:.0f}s)", flush=True)
                # PC per #inference-steps
                for it in args.pc_iters:
                    res = grid_then_seeds(
                        lambda lr, s: train_pc(chw, hidden, bn, lr, it, args.e_lr, x_train, x_val,
                                               args.batch_size, args.n_updates, args.eval_every, s,
                                               infer_opt=args.infer_opt, w_opt=args.pc_opt),
                        args.wlrs, grid_seed, args.seeds)
                    res.update({"dataset": dset, "depth": depth, "bottleneck": bn, "method": "PC", "iters": it})
                    results["runs"].append(res)
                    conv = res["seed_runs"][0]["conv"]
                    print(f"[PC ] {dset} {depth} bn={bn} iters={it} best_lr={res['best_w_lr']:g} "
                          f"mean={res['mean_final']:.4f}±{res['std_final']:.4f} "
                          f"conv={conv:.3f} ({time.time()-t0:.0f}s)", flush=True)

    jp = os.path.join(args.out, f"results_{args.tag}.json")
    json.dump(results, open(jp, "w"))
    print(f"[save] {jp}", flush=True)
    write_summary(results, args.out, args.tag)
    write_sweep_csv(results, args.out, args.tag)
    make_plots(results, args.out, args.tag)
    make_sweep_plot(results, args.out, args.tag)
    print(f"[done] total {time.time()-t0:.0f}s", flush=True)


def write_summary(results, out, tag):
    """One row per config: best LR + mean±std over seeds."""
    p = os.path.join(out, f"summary_{tag}.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "depth", "bottleneck", "method", "iters",
                    "best_w_lr", "mean_final_mse", "std_final_mse", "n_seeds"])
        for r in results["runs"]:
            w.writerow([r["dataset"], r["depth"], r["bottleneck"], r["method"], r["iters"],
                        r["best_w_lr"], f"{r['mean_final']:.6f}", f"{r['std_final']:.6f}", r["n_seeds"]])
    print(f"[save] {p}", flush=True)


def write_sweep_csv(results, out, tag):
    """Full LR grid (single grid seed): one row per (config, weight-LR) tried, plus per-seed
    final MSE at the best LR (is_best=1)."""
    p = os.path.join(out, f"sweep_{tag}.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "depth", "bottleneck", "method", "iters", "phase",
                    "seed", "w_lr", "final_mse", "is_best", "conv"])
        for r in results["runs"]:
            best = r["best_w_lr"]
            for s in r["sweep"]:                       # grid search, single seed
                fm = s["final"]
                w.writerow([r["dataset"], r["depth"], r["bottleneck"], r["method"], r["iters"],
                            "grid", r["grid_seed"], s["w_lr"],
                            (f"{fm:.6f}" if np.isfinite(fm) else "inf"),
                            int(s["w_lr"] == best),
                            ("" if s["conv"] is None else f"{s['conv']:.4f}")])
            for sr in r["seed_runs"]:                  # best LR, all seeds
                fm = sr["final"]
                w.writerow([r["dataset"], r["depth"], r["bottleneck"], r["method"], r["iters"],
                            "best_lr_seeds", sr["seed"], best,
                            (f"{fm:.6f}" if np.isfinite(fm) else "inf"), 1,
                            ("" if sr["conv"] is None else f"{sr['conv']:.4f}")])
    print(f"[save] {p}", flush=True)


def _rec(runs, dataset, depth, method, iters, bn):
    for r in runs:
        if (r["dataset"] == dataset and r["depth"] == depth and r["method"] == method
                and r["iters"] == iters and r["bottleneck"] == bn):
            return r
    return None


def make_sweep_plot(results, out, tag):
    """final MSE vs weight-LR (single grid seed), grid = depth x bottleneck, per dataset."""
    runs = results["runs"]
    for dset in sorted({r["dataset"] for r in runs}):
        depths = sorted({r["depth"] for r in runs if r["dataset"] == dset})
        bns = sorted({r["bottleneck"] for r in runs if r["dataset"] == dset}, reverse=True)
        series = [("BP", 0)] + sorted({("PC", r["iters"]) for r in runs if r["method"] == "PC"})
        fig, axes = plt.subplots(len(depths), len(bns),
                                 figsize=(4.0 * len(bns), 3.6 * len(depths)), squeeze=False)
        for i, dep in enumerate(depths):
            for j, bn in enumerate(bns):
                ax = axes[i][j]
                for method, it in series:
                    r = _rec(runs, dset, dep, method, it, bn)
                    if not r:
                        continue
                    lrs = [s["w_lr"] for s in r["sweep"]]
                    fin = [s["final"] if np.isfinite(s["final"]) else np.nan for s in r["sweep"]]
                    lbl = "BP" if method == "BP" else f"PC (iters={it})"
                    ax.plot(lrs, fin, "o-", label=lbl, lw=1.6)
                    ax.plot([r["best_w_lr"]], [r["mean_final"]], "*", ms=11, color="crimson")
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlabel("weight LR"); ax.set_ylabel("final val MSE")
                ax.set_title(f"{dset} / {dep} / bn={bn}", fontsize=9); ax.legend(fontsize=7)
        fig.suptitle(f"{dset}: weight-LR grid search (single seed; ★ = best LR, mean over seeds)")
        fig.tight_layout()
        p = os.path.join(out, f"sweep_{tag}_{dset}.png")
        fig.savefig(p, dpi=130); print(f"[plot] {p}", flush=True)


def make_plots(results, out, tag):
    runs = results["runs"]
    dsets = sorted({r["dataset"] for r in runs})
    depths = sorted({r["depth"] for r in runs})
    bns = sorted({r["bottleneck"] for r in runs})
    pc_iters = sorted({r["iters"] for r in runs if r["method"] == "PC"})
    fig, axes = plt.subplots(len(dsets), len(depths), figsize=(5.2 * len(depths), 4.2 * len(dsets)), squeeze=False)
    for i, d in enumerate(dsets):
        for j, dep in enumerate(depths):
            ax = axes[i][j]
            for method, it, style in [("BP", 0, "ko-")] + [("PC", k, "o-") for k in pc_iters]:
                recs = [_rec(runs, d, dep, method, it, bn) for bn in bns]
                xs = [bn for bn, r in zip(bns, recs) if r]
                m = np.array([r["mean_final"] for r in recs if r]); s = np.array([r["std_final"] for r in recs if r])
                if not len(xs):
                    continue
                lbl = "BP" if method == "BP" else f"PC (iters={it})" + (" ~equilib" if it == max(pc_iters) else "")
                ax.plot(xs, m, style, label=lbl)
                ax.fill_between(xs, m - s, m + s, alpha=0.15)
            ax.set_xscale("log", base=2); ax.set_yscale("log")
            ax.set_xlabel("bottleneck width"); ax.set_ylabel("final val MSE")
            ax.set_title(f"{d} / {dep} AE"); ax.legend(fontsize=8)
    fig.suptitle("PC vs BP on autoencoding: final reconstruction MSE vs bottleneck (mean±std over seeds)")
    fig.tight_layout()
    p = os.path.join(out, f"scaled_vs_bottleneck_{tag}.png")
    fig.savefig(p, dpi=130); print(f"[plot] {p}", flush=True)


if __name__ == "__main__":
    main()
