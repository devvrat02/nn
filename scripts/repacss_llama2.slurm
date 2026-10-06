#!/bin/bash
#SBATCH --job-name=tracer-llama2
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the TRACER directory}"
source "${TRACER_CONDA_INIT:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${TRACER_CONDA_ENV:-tracer-llama2}"
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
# Respect Slurm's CUDA_VISIBLE_DEVICES. cuda:0 is the allocated GPU.
if [[ $# -eq 0 ]]; then
    echo "Usage: sbatch scripts/repacss_llama2.slurm <local_workflow command> [arguments]" >&2
    exit 2
fi
nvidia-smi
if [[ "$1" == "verify" ]]; then
    # Same dataset/verifier/budget as the requested run; stop before partial output
    # if any baseline prompt cannot fit Llama 2's native context.
    srun python local_workflow.py context-check "${@:2}" --model-profile llama2-7b --device-map cuda
fi
srun python local_workflow.py "$@" --model-profile llama2-7b --device-map cuda
