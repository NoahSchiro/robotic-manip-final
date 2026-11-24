#!/bin/bash
# Package E: placement episodes, one (seed, condition) per array task
# (src/evaluation/placement.py).  Task i runs seed SEED0 + i / N_CONDITIONS under
# condition i % N_CONDITIONS, so the array size must be seeds x conditions.
# Submit from the repo root; a task whose output exists is skipped, so
# resubmitting (or a preemption requeue) only fills in the rest.
#
# Benchmark (3 seeds x 3 conditions):
#   CONDITIONS="true_eta none learned:3" sbatch -a 0-8 scripts/cluster/placement.sbatch
# Full run (30 seeds x 17 conditions = 510 tasks):
#   sbatch -a 0-509%64 scripts/cluster/placement.sbatch
#
# Episodes run at the tightest tolerance of interest (0.5 cm, 5 deg) with a
# 20-push cap; looser tolerances and smaller budgets are scored afterwards.
#SBATCH -J placement
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.log

set -euo pipefail
OUT=${OUT:-data/results/placement}
MODEL=${MODEL:-data/models/ppo_policy.zip}
SEED0=${SEED0:-95000}
IMAGE=${IMAGE:-$HOME/uv-py314.sif}
CONDITIONS=${CONDITIONS:-"true_eta none learned:1 learned:2 learned:3 learned:4 learned:5 random:1 random:2 random:3 random:4 random:5 long_fast:1 long_fast:2 long_fast:3 long_fast:4 long_fast:5"}

read -r -a LIST <<<"$CONDITIONS"
N=${#LIST[@]}
SEED=$((SEED0 + SLURM_ARRAY_TASK_ID / N))
CONDITION=${LIST[$((SLURM_ARRAY_TASK_ID % N))]}
REPORT="$OUT/${SEED}_${CONDITION/:/_}.json"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs "$OUT"
if [ -e "$REPORT" ]; then
    echo "$REPORT exists; skipping"
    exit 0
fi
echo "task $SLURM_ARRAY_TASK_ID: seed $SEED, $CONDITION on $(hostname), commit $(git rev-parse --short HEAD)"
module load apptainer
apptainer exec "$IMAGE" uv run --no-sync python main.py placement-eval \
    --seed "$SEED" --condition "$CONDITION" --model "$MODEL" --workers 15 \
    --out "$REPORT.partial"
mv "$REPORT.partial" "$REPORT"
