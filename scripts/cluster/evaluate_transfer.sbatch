#!/bin/bash
# Drake-world transfer check of the trained probing policy, one paired episode
# per array task: learned policy, random lattice pushes, and long-fast lattice
# pushes on identical conditions (src/evaluation.py), plus the surrogate-world
# reference.  Submit from the repo root; a task whose output exists is
# skipped, so resubmitting (or a preemption requeue) only fills in the rest.
#
#   sbatch scripts/cluster/evaluate_transfer.sbatch
#   FIXED_PUSHES=3 OUT=data/transfer_fixed3 sbatch scripts/cluster/evaluate_transfer.sbatch
#   uv run python -m scripts.merge_evaluations data/transfer/episode_*.json --out data/transfer/summary.json
#
# FIXED_PUSHES makes every strategy push exactly that many times; otherwise
# the policy decides when to stop and the baselines copy its count.
#
# Each task runs ~9 pushes in the truth sim with full-grid Drake belief
# updates on its workers (~0.3 GB each).
#SBATCH -J transfer
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -a 0-39%40
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.log

set -euo pipefail
OUT=${OUT:-data/transfer}
MODEL=${MODEL:-data/ppo/ppo_final.zip}
SEED0=${SEED0:-90000}
IMAGE=${IMAGE:-$HOME/uv-py314.sif}
FIXED_PUSHES=${FIXED_PUSHES:-}
SEED=$((SEED0 + SLURM_ARRAY_TASK_ID))
REPORT="$OUT/episode_$SEED.json"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs "$OUT"
if [ -e "$REPORT" ]; then
    echo "$REPORT exists; skipping"
    exit 0
fi
echo "task $SLURM_ARRAY_TASK_ID (seed $SEED, fixed pushes ${FIXED_PUSHES:-none}) on $(hostname), commit $(git rev-parse --short HEAD)"
module load apptainer
apptainer exec "$IMAGE" uv run --no-sync python main.py evaluate \
    --model "$MODEL" --episodes 1 --seed "$SEED" --workers 15 --out "$REPORT.partial" \
    ${FIXED_PUSHES:+--fixed-pushes "$FIXED_PUSHES"}
mv "$REPORT.partial" "$REPORT"
