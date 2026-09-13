#!/bin/bash
#SBATCH --job-name=chris-original-env
#SBATCH --partition=cpu-preempt
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=chris-environment-%j.log
set -euo pipefail
: "${CHRIS_PYTHON:?}"
export PIP_NO_CACHE_DIR=1
"$CHRIS_PYTHON" -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
"$CHRIS_PYTHON" -m pip install matplotlib pytorch_optimizer scikit-optimize
"$CHRIS_PYTHON" -m pip check
"$CHRIS_PYTHON" -c 'import sys, torch, skopt, pytorch_optimizer, matplotlib; assert sys.version_info >= (3,12); assert torch.__version__.startswith("2.10."); assert hasattr(torch.autograd.graph,"set_warn_on_accumulate_grad_stream_mismatch"); print("DEPENDENCY_IMPORTS_OK",sys.version,torch.__version__)'
"$CHRIS_PYTHON" -m pip freeze
