#!/bin/bash
#SBATCH --job-name=chris-fourway
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --mail-type=NONE
set -euo pipefail

ROOT=/project/pi_roohie_umass_edu
CODE=$ROOT/FlowMLLab_chris_capacity_20260916/qa
SMALL=$ROOT/FlowMLLab_chris_deep_matrix_20260914/case-0
LARGE=$ROOT/FlowMLLab_chris_capacity_20260916/re100_d5_w64_n200k
CFD=$ROOT/FlowMLLab_re100_d5_mesh_20260914
PYTHON=$ROOT/FlowMLLab_chris_deep_tf_20260913/venv/bin/python

export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
"$PYTHON" "$CODE/compare_chris_capacity_fourway.py" \
    --pinn-small "$SMALL/audit-72680-n32/pinn/field.npz" \
    --pinn-large "$LARGE/evaluation/field.npz" \
    --chunk "$CFD/nektar-v2/n32/run/chunk-0119" \
    --foam "$CFD/openfoam-continue/production-34" \
    --loss-small "$SMALL/data/loss.dat" \
    --loss-large "$LARGE/data/loss.dat" \
    --output "$LARGE/fourway-comparison"

