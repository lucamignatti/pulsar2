#!/bin/bash
#SBATCH --job-name=pulsar2-async
#SBATCH --nodes=2
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:6
#SBATCH --partition=el8
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/u/home/RLFR/RLFRkllj/scratch-shared/pulsar2_async_%j.out

set -e
export http_proxy="http://proxy:8888"
export https_proxy="$http_proxy"
module load spectrum-mpi cuda/11.2 gcc/15 2>/dev/null || true

export MPI_ROOT=/opt/ibm/spectrum_mpi
export OPAL_PREFIX=/opt/ibm/spectrum_mpi
export PATH="${MPI_ROOT}/bin:${PATH}"
export GCC15_LIB=/gpfs/u/software/dcs-rhel8/gcc/15.1.0/lib64
export NCCL_ROOT=/gpfs/u/home/RLFR/RLFRkllj/barn-shared/nccl_2.21.5-1+cuda11.0_ppc64le
export PYLIB=/gpfs/u/home/RLFR/RLFRkllj/scratch/aimosenv/miniconda3/envs/python310/lib
PRIV=/gpfs/u/home/RLFR/RLFRkllj/barn-shared/pulsar2-private
export LD_PRELOAD="${GCC15_LIB}/libstdc++.so.6:${GCC15_LIB}/libgcc_s.so.1"
export LD_LIBRARY_PATH="${GCC15_LIB}:${PRIV}/build:${NCCL_ROOT}/lib:${MPI_ROOT}/lib:${PYLIB}:/gpfs/u/home/RLFR/RLFRkllj/barn-shared/pytorch/torch/lib:/usr/local/cuda-11.2/lib64:${LD_LIBRARY_PATH:-}"

export GGL_ASYNC=1
export GGL_ASYNC_N_LEARNERS=${GGL_ASYNC_N_LEARNERS:-4}
# Weak scale: 160k rows/learner unless GGL_ASYNC_GLOBAL_BATCH is already set.
if [ -z "${GGL_ASYNC_GLOBAL_BATCH:-}" ]; then
  export GGL_ASYNC_GLOBAL_BATCH=$((160000 * GGL_ASYNC_N_LEARNERS))
fi
export GGL_ASYNC_MID_TRAJ_PULL=${GGL_ASYNC_MID_TRAJ_PULL:-1}
export GGL_NUM_GAMES=512
export GGL_CHECKPOINT_FOLDER=/gpfs/u/home/RLFR/RLFRkllj/scratch-shared/checkpoints_async_${SLURM_JOB_NUM_NODES:-}n
export GGL_PIPELINED_COLLECTION=0
export GGL_EMPTY_CACHE=0
export GGL_HULL=0
export GGL_TRAIN_AGAINST_OLD_CHANCE=0
export GGL_NEXTO_SERVE_FRAC=0
export GGL_INFER_FINITE_SYNC=0
unset GGL_DIST_BRINGUP
unset GGL_DIST_LOCKSTEP_CHECK

# Real metrics path: rank-0 MetricSender -> wandb. GGL_DIST_SMOKE forces sendMetrics=false.
unset GGL_DIST_SMOKE
if [ -f "$HOME/.wandb_api_key" ]; then
  export WANDB_API_KEY="$(tr -d '\r\n' < "$HOME/.wandb_api_key")"
fi
unset WANDB_MODE
export WANDB_DIR=/gpfs/u/home/RLFR/RLFRkllj/scratch-shared/wandb_async
export GGL_METRICS_RUN_NAME=${GGL_METRICS_RUN_NAME:-pulsar2-async}
export GGL_METRICS_GROUP_NAME=${GGL_METRICS_GROUP_NAME:-async-appo}
export PYTHONPATH="${PRIV}/GigaLearnCPP:${PYTHONPATH:-}"

cd "$PRIV"
mkdir -p "$GGL_CHECKPOINT_FOLDER" "$WANDB_DIR"

echo "=== SLURM ASYNC WANDB ==="
echo "job=${SLURM_JOB_ID:-} nodes=${SLURM_JOB_NUM_NODES:-} ntasks=${SLURM_NTASKS:-} nL=${GGL_ASYNC_N_LEARNERS}"
echo "nodelist=${SLURM_JOB_NODELIST:-}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "WANDB_API_KEY=$([ -n "${WANDB_API_KEY:-}" ] && echo set || echo MISSING)"
echo "GGL_DIST_SMOKE=${GGL_DIST_SMOKE:-<unset>}"
hostname
nvidia-smi -L

chmod +x ./pulsar2_numa_rank.sh
echo "=== GigaLearnBot async np=${SLURM_NTASKS} nL=${GGL_ASYNC_N_LEARNERS} ==="
mpirun -np "${SLURM_NTASKS}" \
  --bind-to none --map-by ppr:6:node \
  -x LD_PRELOAD -x LD_LIBRARY_PATH -x OPAL_PREFIX \
  -x GGL_ASYNC -x GGL_ASYNC_N_LEARNERS -x GGL_ASYNC_GLOBAL_BATCH -x GGL_ASYNC_MID_TRAJ_PULL -x GGL_OPTIM -x GGL_PPO_EPOCHS \
  -x GGL_NUM_GAMES -x GGL_CHECKPOINT_FOLDER \
  -x GGL_PIPELINED_COLLECTION -x GGL_EMPTY_CACHE -x GGL_HULL \
  -x GGL_TRAIN_AGAINST_OLD_CHANCE -x GGL_NEXTO_SERVE_FRAC -x GGL_INFER_FINITE_SYNC \
  -x GGL_METRICS_RUN_NAME -x GGL_METRICS_GROUP_NAME \
  -x WANDB_API_KEY -x WANDB_DIR -x PYTHONPATH -x http_proxy -x https_proxy \
  ./pulsar2_numa_rank.sh
echo PULSAR2_ASYNC_DONE
