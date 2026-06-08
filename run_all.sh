#!/usr/bin/env bash
# Single-GPU driver. All six training runs read configs/base_1gpu.yaml; only
# model / aug_level / copy_paste / train_split / out_dir vary per run, so the
# comparison is fair by construction.
#
# Usage: bash run_all.sh [GPU_ID] [SEED]   (defaults: GPU 0, seed 0)
set -e
GPU=${1:-0}
SEED=${2:-0}
BASE=configs/base_1gpu.yaml
export CUDA_VISIBLE_DEVICES=$GPU
echo "[run_all] GPU=$GPU seed=$SEED base=$BASE"

python -m scripts.prepare_data --root data --validate

train () {  # $1=model $2=aug_level $3=copy_paste $4=train_split $5=out_dir
  python -m scripts.train --config $BASE \
    model=$1 aug_level=$2 copy_paste=$3 train_split=$4 out_dir=$5 seed=$SEED
}

# ---------------- Experiment (i): spatial / cross-dataset ----------------
train erm    none   False mocs_train runs/spatial_erm_s$SEED
train aug    strong True  mocs_train runs/spatial_aug_s$SEED
train causal strong True  mocs_train runs/spatial_causal_s$SEED
for m in erm aug causal; do
  python -m scripts.eval_spatial --ckpt runs/spatial_${m}_s$SEED/final.pth \
      --model $m --out runs/spatial_${m}_s$SEED/spatial.json
done

# ---------------- Experiment (ii): temporal / cross-stage ----------------
train erm    none   False mocs_train_earthmoving runs/temporal_erm_s$SEED
train aug    strong True  mocs_train_earthmoving runs/temporal_aug_s$SEED
train causal strong True  mocs_train_earthmoving runs/temporal_causal_s$SEED
for m in erm aug causal; do
  python -m scripts.eval_temporal --ckpt runs/temporal_${m}_s$SEED/final.pth \
      --model $m --out runs/temporal_${m}_s$SEED/temporal.json
done

echo "[run_all] done (GPU=$GPU seed=$SEED)"
