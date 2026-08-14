#!/bin/bash
#SBATCH --job-name=pulsar2-a512
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:6
#SBATCH --partition=el8
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/u/home/RLFR/RLFRkllj/scratch-shared/pulsar2_dist6_%j.out

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

export GGL_NUM_GAMES=512
export GGL_CHECKPOINT_FOLDER=/gpfs/u/home/RLFR/RLFRkllj/scratch-shared/checkpoints_dist_6gpu_a512
export GGL_DIST_SMOKE=1
export GGL_DIST_LOCKSTEP_CHECK=1
export GGL_PIPELINED_COLLECTION=0
export GGL_EMPTY_CACHE=0
export GGL_HULL=0
export GGL_TRAIN_AGAINST_OLD_CHANCE=0
export GGL_NEXTO_SERVE_FRAC=0
export GGL_INFER_FINITE_SYNC=0
unset GGL_DIST_BRINGUP

cd "$PRIV"
mkdir -p "$GGL_CHECKPOINT_FOLDER"

echo "=== SLURM ==="
echo "job=${SLURM_JOB_ID:-} node=${SLURM_NODELIST:-} ntasks=${SLURM_NTASKS:-} arenas=${GGL_NUM_GAMES}"
hostname
nvidia-smi -L

chmod +x ./pulsar2_numa_rank.sh
mpirun -np "${SLURM_NTASKS}" \
  --bind-to none --report-bindings \
  -x LD_PRELOAD -x LD_LIBRARY_PATH -x OPAL_PREFIX \
  -x GGL_NUM_GAMES -x GGL_CHECKPOINT_FOLDER -x GGL_DIST_SMOKE -x GGL_DIST_LOCKSTEP_CHECK \
  -x GGL_PIPELINED_COLLECTION -x GGL_EMPTY_CACHE -x GGL_HULL \
  -x GGL_TRAIN_AGAINST_OLD_CHANCE -x GGL_NEXTO_SERVE_FRAC -x GGL_INFER_FINITE_SYNC \
  ./pulsar2_numa_rank.sh
echo PULSAR2_A512_DONE
