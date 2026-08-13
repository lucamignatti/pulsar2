#!/bin/bash
#SBATCH --job-name=pulsar2-bringup
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:6
#SBATCH --partition=el8
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/u/home/RLFR/RLFRkllj/barn-shared/pulsar2-private/pulsar2_bringup_%j.out

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
export GGL_DIST_BRINGUP=1

cd "$PRIV"

echo "=== SLURM ==="
echo "job=${SLURM_JOB_ID:-} node=${SLURM_NODELIST:-} ntasks=${SLURM_NTASKS:-}"
hostname
uname -m
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
ls -ld collision_meshes/soccar | cat

echo "=== GigaLearnBot GGL_DIST_BRINGUP np=${SLURM_NTASKS} ==="
mpirun -np "${SLURM_NTASKS}" -x LD_PRELOAD -x LD_LIBRARY_PATH -x OPAL_PREFIX -x GGL_DIST_BRINGUP \
  ./build/GigaLearnBot
echo PULSAR2_BRINGUP_OK
