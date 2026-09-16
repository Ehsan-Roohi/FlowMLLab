#!/bin/bash
#SBATCH --job-name=w15-deeponet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=week15_deeponet-%j.out

set -euo pipefail
BASE=/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-geomdeeponet
RUN="$BASE/week15-ordinary-deeponet-${SLURM_JOB_ID}"
mkdir -p "$RUN"
"$BASE/conda-tf220-clean/bin/python" "$BASE/train_week15_vanilla_deeponet.py" \
  --dataset "$BASE/dataset.npz" \
  --output "$RUN" \
  --epochs 400 --steps-per-epoch 16 --batch-size 8192 --seeds 17 29 43
printf '%s\n' "$RUN" > "$BASE/LATEST_WEEK15_ORDINARY_DEEPONET"
