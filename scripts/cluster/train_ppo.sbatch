#!/bin/bash
# PPO training on the MIT Engaging cluster, inside the project's Apptainer
# image (Drake's wheels need a newer glibc than the cluster's).  Submit from
# the repo root; every run resumes from the latest checkpoint in OUT, so
# resubmitting the same command continues a run that hit the time limit.
#
#   sbatch scripts/cluster/train_ppo.sbatch
#   OUT=data/work/ppo_bench TIMESTEPS=50000 N_ENVS=32 sbatch -t 00:30:00 scripts/cluster/train_ppo.sbatch
#
# Each environment process holds about 0.4 GB and the learner about 4 GB;
# leave a few cores for the learner process.
#SBATCH -J train-ppo
#SBATCH -p mit_normal
#SBATCH -c 96
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.log

set -euo pipefail
OUT=${OUT:-data/work/ppo}
TIMESTEPS=${TIMESTEPS:-3000000}
N_ENVS=${N_ENVS:-90}
IMAGE=${IMAGE:-$HOME/uv-py314.sif}

# One numerical-library thread per environment process.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

module load apptainer
mkdir -p logs "$OUT"
echo "job $SLURM_JOB_ID on $(hostname): $N_ENVS envs, $TIMESTEPS steps, out $OUT, commit $(git rev-parse --short HEAD)"
apptainer exec "$IMAGE" uv run --no-sync python main.py train-ppo \
    --timesteps "$TIMESTEPS" --n-envs "$N_ENVS" --out "$OUT" --resume
