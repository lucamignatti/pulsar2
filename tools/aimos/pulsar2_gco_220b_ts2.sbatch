#!/bin/bash
#SBATCH --job-name=gco-220b-ts2
#SBATCH --nodes=8-18
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:6
#SBATCH --partition=el8,dcs-2024
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/logs/gco_220b_ts2_%j.out

# ============================================================================
# 7.9-gco resume ("220b" per the user; the asset is the FULL 235,726,848,000-step
# checkpoint, the only resumable GCO bundle) on branch 22b-compat = private HEAD
# 133169b + the optimizer-state shape guard. Sync trainer, elastic 8-16 nodes.
#
# EXPERIMENT 2ts (2026-09-02): the 220b lineage continued at tickSkip 2 (60 Hz) from its 270.6B save.
# Arenas/rank 48 at 16 nodes: 96 players x 87 steps = 8352 rows -> the GAE window stays ~1.45 s of game
# time (ts8: 24 steps x 8 ticks = 1.6 s). GGL_TS_PER_VERSION 2e9 (ts2 steps are 4x shorter).
#
# v2 (2026-09-02 ~10:00, user-directed "learn faster"): pipelined collection ON, effective
# batch HALVED to 100,224 (2x updates/step), epochs 2->3, version ring ON (train vs old
# versions 0.30, GGL_TS_PER_VERSION=1B so the 32-deep ring spans ~30B steps instead of
# 800M), NEED table {32,24,16,12} with the per-rank split recomputed post-preflight.
#
# Semantics are the bundle's GCO_CONFIG.md contract, unchanged. The only knob that
# moved is the per-rank batch split, chosen so the EFFECTIVE batch the LR was
# tuned at is held exactly:
#   32n: 192 ranks x 1044 mb = 200,448 eff, 4176 rows/iter/rank, 87 arenas/rank
#   16n:  96 ranks x 2088 mb = 200,448 eff, 8352 rows/iter/rank, 174 arenas/rank
# -> same 801,792 fleet steps/iter, same 8 updates/iter, same 24-step GAE window.
# Cost: ~2x the per-iteration collect wall vs 32n (arenas per rank doubled).
#
# Engine: HEAD defaults (corrected physics; see docs/GCO_RESUME.md D1 - keep, do
# not revert). No GGL_DODGE_*/GGL_FLIP_RESET_FIX/GGL_OBS_FLAG_NOISE set.
# GGL_FRESH_OPTIM=0: the cluster libtorch remaps saved optimizer state correctly
# (the original run resumed these same archives at every hop); the desktop crash
# on these files is a libtorch-2.9 key-parsing artifact. The binary now verifies
# state shapes and resets per-model on mismatch (grep "optimizer state" in the log).
# ============================================================================

set -o pipefail   # NOT set -e: a failed hop must still let the chain continue.
ulimit -c 0       # core dumps filled the shared volume once already

export http_proxy="http://proxy:8888"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
module load spectrum-mpi cuda/11.2 gcc/15 2>/dev/null || true

export MPI_ROOT=/opt/ibm/spectrum_mpi
export OPAL_PREFIX=/opt/ibm/spectrum_mpi
export PATH="${MPI_ROOT}/bin:${PATH}"
export GCC15_LIB=/gpfs/u/software/dcs-rhel8/gcc/15.1.0/lib64
export NCCL_ROOT=/gpfs/u/home/RLFR/RLFRkllj/barn-shared/nccl_2.21.5-1+cuda11.0_ppc64le
export PYLIB=/gpfs/u/home/RLFR/RLFRkllj/scratch/aimosenv/miniconda3/envs/python310/lib
PRIV=/gpfs/u/home/RLFR/RLFRlcmg/scratch/pulsar2-private-luca
export LD_PRELOAD="${GCC15_LIB}/libstdc++.so.6:${GCC15_LIB}/libgcc_s.so.1"
export LD_LIBRARY_PATH="${GCC15_LIB}:${PRIV}/build:${NCCL_ROOT}/lib:${MPI_ROOT}/lib:${PYLIB}:/gpfs/u/home/RLFR/RLFRkllj/barn-shared/pytorch/torch/lib:/usr/local/cuda-11.2/lib64:${LD_LIBRARY_PATH:-}"

# ===== SEMANTIC CONTRACT (bundle GCO_CONFIG.md) - all six are required =====
export GGL_GCO=1                      # goal/concede only, exactly +/-150
export GGL_NO_REACH=1                 # checkpoint has no REACH_*; default-on would fresh-init 4 nets
export GGL_HULL=0                     # same story, 2 HULL_* models
export GGL_NO_VERSIONS=0              # v2: version ring + skill tracker ON (was 1 on the bundle)
export GGL_TS_PER_VERSION=2000000000  # v2: 1B steps/version; 32-deep ring ~ 30B-step window
export GGL_NEXTO_SERVE_FRAC=0         # pure mirror self-play
export GGL_TRAIN_AGAINST_OLD_CHANCE=0.30   # v2: train vs archived past selves (ExampleMain default)
export GGL_LEAGUE=0                   # default off; explicit so a future default flip cannot sneak in

# ===== BATCH (v2): effective batch 100,224 = HALF the 200,448 the LR was tuned at =====
# Deliberate: 2x optimizer updates per step (with epochs 3 -> 24 updates/iter vs the
# original 8). Per-rank split is recomputed after preflight from NEED (see table below);
# these are the 16-node values as a fallback only.
export GGL_NUM_GAMES=${GGL_NUM_GAMES:-174}
export GGL_TS_PER_ITR=${GGL_TS_PER_ITR:-8352}     # PER RANK
export GGL_MINIBATCH=${GGL_MINIBATCH:-1044}       # PER RANK; x96 ranks = 100,224
export GGL_PPO_EPOCHS=3

export GGL_FRESH_OPTIM=0
export GGL_ITER_PER_SAVE=${GGL_ITER_PER_SAVE:-500}
export GGL_CKPT_KEEP=${GGL_CKPT_KEEP:-5}
export GGL_BEST_KEEP=${GGL_BEST_KEEP:-0}
export GGL_CHECKPOINT_FOLDER=/gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/checkpoints_gco_220b_ts2
export GGL_TRAIN_TICK_SKIP=2   # 8->2 curriculum step: gammas/lambda/entropy re-derived in ExampleMain (wall-clock horizons kept)

export GGL_PIPELINED_COLLECTION=1   # v2: overlap collect (0.52s) with learn (0.37s)
export GGL_INFER_FINITE_SYNC=0
export GGL_MUON_SHARD=1               # sync trainer (the async deadlock does not apply)
export GGL_MOE=0
export GGL_CUDA_GRAPHS=0
export GGL_DIST_SMOKE=0               # NOT 1: GGL_DIST_SMOKE forces sendMetrics=false
export GGL_DIST_LOCKSTEP_CHECK=1
unset GGL_DIST_BRINGUP

# ===== WANDB (online through the proxy; rank 0 only) =====
# Under GCO the reward panel is one zero-sum term - read Episode Length (inverse goal
# rate) and the goal counters, never mean reward.
[ -n "${WANDB_API_KEY:-}" ] || eval "$(grep WANDB_API_KEY "$HOME/.bashrc" | tail -1)"
export WANDB_API_KEY
export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-180}
export WANDB_DIR=/gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/wandb_checkpoints_gco_220b_ts2
export GGL_RUN_NAME=${GGL_RUN_NAME:-7.9-gco-220b-ts2}
export GGL_METRICS_GROUP_NAME=${GGL_METRICS_GROUP_NAME:-gco-220b-exp}
export PYTHONPATH="$PRIV/build:$HOME/.local/lib/python3.10/site-packages"

cd "$PRIV" || exit 1
mkdir -p "$GGL_CHECKPOINT_FOLDER" "$WANDB_DIR" /gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/logs

echo "=== GCO-220b-ts2 job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} ntasks=${SLURM_NTASKS:-?} $(date) ==="
echo "nodelist=${SLURM_JOB_NODELIST:-}"
echo "ckpt=${GGL_CHECKPOINT_FOLDER} newest=$(ls "$GGL_CHECKPOINT_FOLDER" | grep -E '^[0-9]+$' | sort -n | tail -1)"
echo "arenas=${GGL_NUM_GAMES} tsPerItr=${GGL_TS_PER_ITR}/rank minibatch=${GGL_MINIBATCH}/rank freshOptim=${GGL_FRESH_OPTIM}"
echo "WANDB_API_KEY=$([ -n "${WANDB_API_KEY:-}" ] && echo set || echo MISSING) run=${GGL_RUN_NAME}"
git log --oneline -1
hostname
nvidia-smi -L | head -2

chmod +x ./pulsar2_numa_rank.sh ./gpucheck.sh ./gpuhealth 2>/dev/null || true

# ---- GPU PREFLIGHT: over-allocate (16-24), run on exactly NEED healthy nodes -----
# dcs208/dcs213 fail cudaGetDeviceCount while nvidia-smi looks healthy; dcs237-244
# pass the probe but kill the ORTE bootstrap (BANNED list, 4 hops burned 08-28);
# --exclude is blocked by policy. Probe every node with srun (NOT mpirun - an mpirun
# sweep bootstraps ORTE across the allocation and dies on a sick node). Both partitions
# are requested, so a dcs-2024 draw can carry up to 8 banned nodes - hence the 24 max.
# World size is pinned to NEED*6; if fewer than 16 but at least 12 nodes are healthy,
# degrade to the 12-node split that holds the SAME 200,448 effective batch
# (72 ranks x 2784; 232 arenas/rank; 11136 rows/iter/rank; same 24-step GAE window).
NEED=16
BANNED_NODES=${GGL_BANNED_NODES:-"dcs103 dcs165 dcs208 dcs213 dcs237 dcs238 dcs239 dcs240 dcs241 dcs242 dcs243 dcs244"}
SD="${SLURM_SUBMIT_DIR:-$PRIV}"
HOSTFILE="/tmp/hosts_${SLURM_JOB_ID}.txt"
HEALTHY=$(srun --ntasks-per-node=1 --gres=gpu:6 "$SD/gpucheck.sh" 2>/dev/null | sort -u)
for b in $BANNED_NODES; do HEALTHY=$(echo "$HEALTHY" | grep -vw "$b" || true); done
NHEALTH=$(echo "$HEALTHY" | grep -c .)
echo "[PREFLIGHT] allocated=${SLURM_JOB_NUM_NODES} healthy=${NHEALTH}"
echo "[PREFLIGHT] unhealthy/banned: $(comm -23 <(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -u) <(echo "$HEALTHY") | tr "\n" " ")"
# NEED table: largest of {32,24,16,12} <= healthy. Every row keeps the fleet at 16,704
# arenas / 801,792 steps per iteration / 100,224 effective batch / 24-step GAE window:
#   N   ranks  arenas/rank  rows/iter/rank  mb/rank
#   32  192    87           4176            522
#   24  144    116          5568            696
#   16  96     174          8352            1044
#   12  72     232          11136           1392
if   [ "$NHEALTH" -ge 16 ]; then NEED=16; export GGL_NUM_GAMES=48  GGL_TS_PER_ITR=8352  GGL_MINIBATCH=1044
elif [ "$NHEALTH" -ge 12 ]; then NEED=12; export GGL_NUM_GAMES=64  GGL_TS_PER_ITR=11136 GGL_MINIBATCH=1392
elif [ "$NHEALTH" -ge 8 ];  then NEED=8;  export GGL_NUM_GAMES=96  GGL_TS_PER_ITR=16704 GGL_MINIBATCH=2088
else
  echo "[PREFLIGHT] FATAL: only ${NHEALTH} healthy nodes, need >= 12; holding 30 min so the chain does not cascade"
  sleep 1800
  exit 1
fi
echo "$HEALTHY" | head -n "$NEED" | sed "s/$/ slots=6/" > "$HOSTFILE"
if [ ! -s "$HOSTFILE" ]; then
  echo "[PREFLIGHT] FATAL: hostfile $HOSTFILE is empty or unwritable"; sleep 1800; exit 1
fi
NP_USE=$(( NEED * 6 ))
echo "[PREFLIGHT] np=${NP_USE} arenas=${GGL_NUM_GAMES} tsPerItr=${GGL_TS_PER_ITR} mb=${GGL_MINIBATCH} hostfile=${HOSTFILE}: $(tr '\n' ' ' < "$HOSTFILE")"

# Node hygiene: crashed hops leave orphan ranks squatting GPU memory; sweep first.
srun --ntasks-per-node=1 bash -c 'pkill -9 -f "[b]uild/GigaLearnBot" || true; sleep 3' 2>/dev/null || true

# MILESTONE ARCHIVE: rotation (keep=5) destroys history. Archive the newest checkpoint's
# policy files (~23MB) once per hop, before training resumes. Never rotated.
A=/gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/checkpoints_gco_220b_ts2_archive
N=$(ls "$GGL_CHECKPOINT_FOLDER" 2>/dev/null | grep -E "^[0-9]+$" | sort -n | tail -1)
if [ -n "$N" ] && [ ! -d "$A/$N" ]; then
  mkdir -p "$A/$N.tmp" && cp "$GGL_CHECKPOINT_FOLDER/$N"/{POLICY.lt,SHARED_HEAD.lt,RUNNING_STATS.json} "$A/$N.tmp/" 2>/dev/null && mv "$A/$N.tmp" "$A/$N" && echo "ARCHIVED $N"
fi

# CASCADE GUARD: a hop that dies in under MIN_OK_SECS is a failed launch (sick node,
# ORTE bootstrap); retry, then HOLD the wall clock so the afterany chain does not burn
# every queued hop in minutes.
MIN_OK_SECS=${MIN_OK_SECS:-300}
HOP_T0=$(date +%s)
run_mpi() {
mpirun -np "${NP_USE}" --hostfile "${HOSTFILE}" \
  --bind-to none --map-by ppr:6:node \
  -x LD_PRELOAD -x LD_LIBRARY_PATH -x OPAL_PREFIX \
  -x http_proxy -x https_proxy -x HTTP_PROXY -x HTTPS_PROXY \
  -x WANDB_API_KEY -x WANDB_MODE -x WANDB_DIR -x WANDB_INIT_TIMEOUT -x PYTHONPATH \
  -x GGL_GCO -x GGL_NO_REACH -x GGL_HULL -x GGL_NO_VERSIONS \
  -x GGL_NEXTO_SERVE_FRAC -x GGL_TRAIN_AGAINST_OLD_CHANCE -x GGL_LEAGUE \
  -x GGL_NUM_GAMES -x GGL_TS_PER_ITR -x GGL_MINIBATCH -x GGL_FRESH_OPTIM -x GGL_PPO_EPOCHS -x GGL_TS_PER_VERSION -x GGL_TRAIN_TICK_SKIP \
  -x GGL_ITER_PER_SAVE -x GGL_CKPT_KEEP -x GGL_BEST_KEEP -x GGL_CHECKPOINT_FOLDER \
  -x GGL_PIPELINED_COLLECTION -x GGL_INFER_FINITE_SYNC -x GGL_MUON_SHARD \
  -x GGL_MOE -x GGL_CUDA_GRAPHS -x GGL_DIST_SMOKE -x GGL_DIST_LOCKSTEP_CHECK \
  -x GGL_RUN_NAME -x GGL_METRICS_GROUP_NAME \
  ./pulsar2_numa_rank.sh
}
RC=0
for ATTEMPT in 1 2 3; do
  echo "=== launch attempt ${ATTEMPT} $(date) ==="
  RC=0; run_mpi || RC=$?
  ELAPSED=$(( $(date +%s) - HOP_T0 ))
  if [ "$ELAPSED" -ge "$MIN_OK_SECS" ]; then break; fi
  echo "=== attempt ${ATTEMPT} died after ${ELAPSED}s (rc=${RC}); sweeping, retrying ==="
  srun --ntasks-per-node=1 bash -c 'pkill -9 -f "[b]uild/GigaLearnBot" || true; sleep 3' 2>/dev/null || true
  sleep 30
done
ELAPSED=$(( $(date +%s) - HOP_T0 ))
if [ "$ELAPSED" -lt "$MIN_OK_SECS" ]; then
  echo "=== hop failed to launch 3x (${ELAPSED}s); holding so the chain does not cascade ==="
  sleep 1800
fi
echo "PULSAR2_GCO220B_HOP_DONE rc=${RC} $(date)"
