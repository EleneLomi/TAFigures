#!/usr/bin/env bash
# Regenerates Table tab:ae (Appendix Q) with the settings stored with the published results.
# The table uses the BP rows and the PC rows with iters=100; iters=1 is run for comparison.
# Run from this folder. Set DATA_ROOT to reuse existing datasets (default ./data).
set -e
cd "$(dirname "$0")"

python ae_scaled.py --datasets CIFAR10 --depths shallow deep \
  --bottlenecks 256 64 16 --seeds 0 1 2 --pc_iters 1 100 \
  --wlrs 3e-3 1e-2 3e-2 1e-1 --bp_wlrs 3e-4 1e-3 3e-3 1e-2 3e-2 \
  --batch_size 256 --n_updates 300 --eval_every 50 --tag cifar10

python ae_scaled.py --datasets CIFAR100 --depths shallow deep \
  --bottlenecks 256 64 16 --seeds 0 1 2 --pc_iters 1 100 \
  --wlrs 3e-3 1e-2 3e-2 1e-1 --bp_wlrs 3e-4 1e-3 3e-3 1e-2 3e-2 \
  --batch_size 256 --n_updates 300 --eval_every 50 --tag cifar100

python ae_scaled.py --datasets STL10 --depths shallow deep \
  --bottlenecks 256 64 16 --seeds 0 1 2 --pc_iters 1 100 \
  --wlrs 1e-3 3e-3 1e-2 --bp_wlrs 1e-4 3e-4 1e-3 \
  --batch_size 128 --n_updates 200 --eval_every 50 --tag stl10
