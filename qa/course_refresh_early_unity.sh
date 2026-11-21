#!/bin/bash
#SBATCH --job-name=flowml-early-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
set -euo pipefail
root="${EARLY_QA_ROOT:?}"
cd "$root/repo"
py="/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115/week04-lab2-qa-20261120/venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1
export FLOWMLLAB_EARLY_TMP="$root/tmp/task_${SLURM_ARRAY_TASK_ID}"
export TMPDIR="$FLOWMLLAB_EARLY_TMP/temp"
mkdir -p "$TMPDIR"
export PYTHONPATH="$root/repo"
notebooks=(
notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb
notebooks/week05_06/P1_Re_Generalization.ipynb
notebooks/week05_06/P2_Physics_Guided_DNN.ipynb
notebooks/week05_06/P3_POD_Study.ipynb
notebooks/week05_06/P4_Uncertainty_Study.ipynb
)
"$py" qa/course_refresh_early_execute.py "${notebooks[$SLURM_ARRAY_TASK_ID]}"
"$py" -m pip freeze > "$root/task_${SLURM_ARRAY_TASK_ID}_environment.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$root/task_${SLURM_ARRAY_TASK_ID}_COMPLETE"