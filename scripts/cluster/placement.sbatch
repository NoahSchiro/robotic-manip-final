#!/bin/bash
# Placement episodes, one (seed, condition) per array task
# (src/evaluation/placement.py).  Episode j runs seed j / N_CONDITIONS (from
# SEED_FILE, or N_SEEDS consecutive seeds from SEED0) under condition
# j % N_CONDITIONS.  Array task i runs episodes i * EPISODES_PER_TASK onward, so
# the array size is ceil(seeds x conditions / EPISODES_PER_TASK); packing several
# episodes per task keeps the job count under the per-user submit limit.
# Submit from the repo root; a task whose output exists is skipped, so
# resubmitting (or a preemption requeue) only fills in the rest.
#
# Benchmark (3 seeds x 3 conditions):
#   CONDITIONS="true_eta none learned:3" sbatch -a 0-8 scripts/cluster/placement.sbatch
# Headroom of probing (30 seeds x 2 conditions per task; TASK is near or far,
# see src/experiment.py TASKS):
#   TASK=near CONDITIONS="true_eta none" OUT=data/results/headroom/near sbatch -a 0-59 scripts/cluster/placement.sbatch
#   TASK=far CONDITIONS="true_eta none" OUT=data/results/headroom/far sbatch -a 0-59 scripts/cluster/placement.sbatch
# Fixed wrong eta on the same seeds (does eta matter, or is the estimator good?):
#   TASK=near CONDITIONS="nominal_eta farthest_eta" OUT=data/results/headroom/near sbatch -a 0-59 scripts/cluster/placement.sbatch
#   TASK=far CONDITIONS="nominal_eta farthest_eta" OUT=data/results/headroom/far sbatch -a 0-59 scripts/cluster/placement.sbatch
# Probers before MPC on the same seeds (30 seeds x 9 conditions per task):
#   PROBES="learned:1 learned:2 learned:3 random:1 random:2 random:3 long_fast:1 long_fast:2 long_fast:3"
#   EPISODES_PER_TASK=4 TASK=near CONDITIONS="$PROBES" OUT=data/results/probing/near sbatch -a 0-67 scripts/cluster/placement.sbatch
#   EPISODES_PER_TASK=4 TASK=far CONDITIONS="$PROBES" OUT=data/results/probing/far sbatch -a 0-67 scripts/cluster/placement.sbatch
# Low friction (far task, 30 seeds with true mu <= 0.3, from scripts/select_seeds.py; 13 conditions):
#   EPISODES_PER_TASK=4 SEED_FILE=data/results/low_mu/seeds.txt TASK=far CONDITIONS="true_eta none nominal_eta farthest_eta $PROBES" OUT=data/results/low_mu/far sbatch -a 0-97 scripts/cluster/placement.sbatch
# High friction for contrast (30 seeds with true mu >= 0.6; the eta conditions only):
#   EPISODES_PER_TASK=4 SEED_FILE=data/results/high_mu/seeds.txt TASK=far CONDITIONS="true_eta none nominal_eta farthest_eta" OUT=data/results/high_mu/far sbatch -a 0-29 scripts/cluster/placement.sbatch
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
TASK=${TASK:-near}
IMAGE=${IMAGE:-$HOME/uv-py314.sif}
CONDITIONS=${CONDITIONS:-"true_eta none learned:1 learned:2 learned:3 learned:4 learned:5 random:1 random:2 random:3 random:4 random:5 long_fast:1 long_fast:2 long_fast:3 long_fast:4 long_fast:5"}

read -r -a LIST <<<"$CONDITIONS"
N=${#LIST[@]}
if [ -n "${SEED_FILE:-}" ]; then
    mapfile -t SEEDS <"$SEED_FILE"
else
    mapfile -t SEEDS < <(seq "$SEED0" $((SEED0 + ${N_SEEDS:-30} - 1)))
fi
TOTAL=$((${#SEEDS[@]} * N))
PER_TASK=${EPISODES_PER_TASK:-1}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs "$OUT"
module load apptainer
for ((k = 0; k < PER_TASK; k++)); do
    I=$((SLURM_ARRAY_TASK_ID * PER_TASK + k))
    if ((I >= TOTAL)); then
        break
    fi
    SEED=${SEEDS[$((I / N))]}
    CONDITION=${LIST[$((I % N))]}
    REPORT="$OUT/${SEED}_${CONDITION/:/_}.json"
    if [ -e "$REPORT" ]; then
        echo "$REPORT exists; skipping"
        continue
    fi
    echo "episode $I: $TASK seed $SEED, $CONDITION on $(hostname), commit $(git rev-parse --short HEAD)"
    apptainer exec "$IMAGE" uv run --no-sync python main.py placement-eval \
        --seed "$SEED" --condition "$CONDITION" --task "$TASK" --model "$MODEL" --workers 15 \
        --out "$REPORT.partial"
    mv "$REPORT.partial" "$REPORT"
done
