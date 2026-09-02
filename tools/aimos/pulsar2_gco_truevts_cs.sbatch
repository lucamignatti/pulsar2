#!/bin/bash
#SBATCH --job-name=gco-truevts
#SBATCH --nodes=16-34
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:6
#SBATCH --partition=el8,dcs-2024
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/logs/gco_truevts_%j.out

# ============================================================================
# FROM-SCRATCH GCO cold start on branch vts-true-skip (private HEAD 133169b merged in
# at 26753b0): TRUE variable tickskip (env skips the policy-chosen 1/2/4/8/16 ticks,
# factored 90+5 head, per-tick gammas), APPO async trainer, LoRA league, reach/SIL/vdag/
# gap/goal-critic all from birth, all three team modes from birth (g_PhaseB pinned).
# User-directed 2026-09-02: "from scratch private run with vts and everything" - GCO
# sparse, not titan. NO pooling on this branch (GGL_POOL knobs of the gco tree do not
# exist here). Launcher derived from ~/pulsar2_gco_theta.sbatch (the proven elastic
# APPO recipe) with the VTS/step-replay and sidecar parts replaced.
# ============================================================================

set -o pipefail
ulimit -c 0
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
PRIV=/gpfs/u/home/RLFR/RLFRlcmg/scratch/pulsar2-vts
export LD_PRELOAD="${GCC15_LIB}/libstdc++.so.6:${GCC15_LIB}/libgcc_s.so.1"
export LD_LIBRARY_PATH="${GCC15_LIB}:${PRIV}/build:${NCCL_ROOT}/lib:${MPI_ROOT}/lib:${PYLIB}:/gpfs/u/home/RLFR/RLFRkllj/barn-shared/pytorch/torch/lib:/usr/local/cuda-11.2/lib64:${LD_LIBRARY_PATH:-}"

# ===== REWARD / OPPONENTS =====
export GGL_GCO=1                      # goal/concede only, +/-150, nothing else
export GGL_NEXTO_SERVE_FRAC=0         # pure self-play + league (async has no Nexto path anyway)
export GGL_TRAIN_AGAINST_OLD_CHANCE=0
export GGL_NO_VERSIONS=0              # version ring + Rating/Ref telemetry on
export GGL_NO_REACH=0                 # reach from birth (cold start: the validated regime)
export GGL_HULL=0
export GGL_NEXTO_EVAL_FRAC=0          # in-trainer NextoEval costs 5.5x; off
export GGL_NEXTOEVAL_SIDECAR=0        # sidecar mpirun cannot launch inside an allocation

# ===== TRUE VARIABLE TICKSKIP =====
export GGL_VTS=1
export GGL_VTS_HEAD=factored          # toy study: joint collapsed to 16-tick holds; factored kept a 4/8 mixture
export GGL_VTS_BUCKETS=1,2,4,8,16     # physics ticks per decision (120 .. 7.5 Hz)
# GGL_VTS_ENVWEIGHT / GGL_VTS_ENVSCALE deliberately UNSET: branch default envWeight 0 (no
# advantage skip bias) and entropySkipScale 0 (entropy on the control marginal only).
export GGL_THETA=0                    # theta-commit is disabled under GGL_VTS regardless
unset GGL_TICK_SKIP                   # has no meaning under GGL_VTS (fatal if set)

# ===== LEAGUE (theta-run settings; repulsion 0 under sparse GCO - see pulsar-gco-run notes) =====
export GGL_LEAGUE=1
export GGL_LEAGUE_DIVERSE=4
export GGL_LEAGUE_EXPLOITERS=2
export GGL_LEAGUE_RANK=4
export GGL_LEAGUE_FRAC=0.25
export GGL_LEAGUE_LR=2e-5
export GGL_LEAGUE_DIV_BETA=0.25
export GGL_LEAGUE_REPEL=0.0
export GGL_LEAGUE_REPELTARGET=0.28
export GGL_LEAGUE_REPELMAX=0.35
export GGL_LEAGUE_BINIT=1e-3
export GGL_LEAGUE_ACCUM=4
export GGL_LEAGUE_BDECAY=0.998
export GGL_LEAGUE_RESPAWN=0.15
export GGL_LEAGUE_RESPAWN_MIN=20
export GGL_LEAGUE_RESPAWN_EVERY=15
export GGL_LEAGUE_EXPBOOST=1.0
export GGL_LEAGUE_MINIBATCH=8192

# ===== APPO =====
export GGL_ASYNC=1
export GGL_ASYNC_GLOBAL_BATCH=800000  # decision rows per learn cycle (theta-run value)
export GGL_ASYNC_MAX_POLICY_LAG=64
# GGL_ASYNC_FRAGMENT_TICKS unset: under GGL_VTS the fragment cut is the branch's tick budget.
export GGL_PPO_EPOCHS=1               # measured best APPO throughput (5.17M SPS at epochs 1)
export GGL_NUM_GAMES=128              # per collector; all three team modes are pinned on
export GGL_MUON_SHARD=0               # MUST be 0 under async (sharded step deadlocks)
export GGL_PIPELINED_COLLECTION=0
export GGL_INFER_FINITE_SYNC=0
export GGL_CUDA_GRAPHS=0
export GGL_MOE=0
export GGL_DIST_SMOKE=0
export GGL_DIST_LOCKSTEP_CHECK=0
unset GGL_DIST_BRINGUP
export GGL_FRESH_OPTIM=0
export GGL_ITER_PER_SAVE=500
export GGL_CKPT_KEEP=3
export GGL_BEST_KEEP=0
export GGL_TS_PER_VERSION=1000000000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# FRESH LINEAGE. Shape-breaking head (95 outputs / 450 table): must never share a folder
# with a 90-way lineage. Verified absent before the first submit.
export GGL_CHECKPOINT_FOLDER=/gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/checkpoints_gco_truevts_cs
export GGL_RUN_NAME=7.9-gco-truevts-cs
export GGL_METRICS_GROUP_NAME=gco-truevts

# ===== WANDB (online through the proxy; rank 0 only) =====
[ -n "${WANDB_API_KEY:-}" ] || eval "$(grep WANDB_API_KEY "$HOME/.bashrc" | tail -1)"
export WANDB_API_KEY
export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=180
export WANDB__SERVICE_WAIT=300
export WANDB_DIR=/gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/wandb_gco_truevts
export PYTHONPATH="$PRIV/build:$HOME/.local/lib/python3.10/site-packages"

cd "$PRIV" || exit 1
mkdir -p "$GGL_CHECKPOINT_FOLDER" "$WANDB_DIR" /gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/logs

echo "=== GCO-TRUEVTS job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} $(date) ==="
echo "nodelist=${SLURM_JOB_NODELIST:-}"
echo "ckpt=${GGL_CHECKPOINT_FOLDER} newest=$(ls "$GGL_CHECKPOINT_FOLDER" | grep -E '^[0-9]+$' | sort -n | tail -1)"
echo "VTS=${GGL_VTS} head=${GGL_VTS_HEAD} buckets=${GGL_VTS_BUCKETS} league=${GGL_LEAGUE} batch=${GGL_ASYNC_GLOBAL_BATCH} epochs=${GGL_PPO_EPOCHS}"
echo "WANDB_API_KEY=$([ -n "${WANDB_API_KEY:-}" ] && echo set || echo MISSING) run=${GGL_RUN_NAME}"
git log --oneline -1
hostname
nvidia-smi -L | head -2

chmod +x ./pulsar2_numa_rank.sh ./gpucheck.sh ./gpuhealth 2>/dev/null || true

# ---- WATCHDOG (theta-run pattern): a sick allocation that never iterates, or a hung
# save, would otherwise hold the whole 6h wall. Kill the hop so the chain advances.
( WDLOG=/gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/logs/gco_truevts_${SLURM_JOB_ID}.out
  last=0; stall=0; started=0; boot=0
  while true; do
    sleep 300
    n=$(grep -c "ASYNC iter=" "$WDLOG" 2>/dev/null || true); n=${n:-0}
    if [ "$n" -gt 0 ]; then started=1; fi
    if [ "$started" = "0" ]; then
      boot=$((boot+1)); echo "[WATCHDOG] no first iteration yet ($((boot*5)) min)"
      if [ "$boot" -ge 4 ]; then echo "[WATCHDOG] FATAL: no first iteration after 20 min - killing job $SLURM_JOB_ID"; scancel "$SLURM_JOB_ID"; exit 0; fi
    fi
    if [ "$started" = "1" ] && [ "$n" -le "$last" ]; then
      stall=$((stall+1)); echo "[WATCHDOG] no new iterations for $((stall*5)) min (iters=$n)"
      if [ "$stall" -ge 5 ]; then echo "[WATCHDOG] FATAL: 25 min without progress - killing job $SLURM_JOB_ID"; scancel "$SLURM_JOB_ID"; exit 0; fi
    else stall=0; fi
    last=$n
  done ) &
WATCHDOG_PID=$!

# ---- GPU PREFLIGHT: take every healthy node of the draw (floor 16); learners = 1/3 ----
NEED=${GGL_NEED:-0}
BANNED_NODES=${GGL_BANNED_NODES:-"dcs103 dcs165 dcs208 dcs213 dcs237 dcs238 dcs239 dcs240 dcs241 dcs242 dcs243 dcs244"}
SD="${SLURM_SUBMIT_DIR:-$PRIV}"
HOSTFILE="/tmp/hosts_${SLURM_JOB_ID}.txt"
HEALTHY=$(srun --ntasks-per-node=1 --gres=gpu:6 "$SD/gpucheck.sh" 2>/dev/null | sort -u)
for b in $BANNED_NODES; do HEALTHY=$(echo "$HEALTHY" | grep -vw "$b" || true); done
NHEALTH=$(echo "$HEALTHY" | grep -c .)
[ "$NEED" -gt 0 ] || NEED=$NHEALTH
[ "$NEED" -le 32 ] || NEED=32
echo "[PREFLIGHT] allocated=${SLURM_JOB_NUM_NODES} healthy=${NHEALTH} need=${NEED}"
echo "[PREFLIGHT] unhealthy/banned: $(comm -23 <(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -u) <(echo "$HEALTHY") | tr "\n" " ")"
if [ "$NHEALTH" -lt "${GGL_NEED_FLOOR:-16}" ]; then
  echo "[PREFLIGHT] FATAL: only ${NHEALTH} healthy nodes, floor ${GGL_NEED_FLOOR:-16}; holding 30 min so the chain does not cascade"
  kill $WATCHDOG_PID 2>/dev/null; sleep 1800; exit 1
fi
[ "$NHEALTH" -lt "$NEED" ] && NEED=$NHEALTH
echo "$HEALTHY" | head -n "$NEED" | sed "s/$/ slots=6/" > "$HOSTFILE"
if [ ! -s "$HOSTFILE" ]; then echo "[PREFLIGHT] FATAL: hostfile empty"; kill $WATCHDOG_PID 2>/dev/null; sleep 1800; exit 1; fi
NP_USE=$(( NEED * 6 ))
export GGL_ASYNC_N_LEARNERS=${GGL_ASYNC_N_LEARNERS:-$(( NP_USE / 3 ))}
echo "[PREFLIGHT] np=${NP_USE} nL=${GGL_ASYNC_N_LEARNERS} hostfile=${HOSTFILE}: $(tr '\n' ' ' < "$HOSTFILE")"

srun --ntasks-per-node=1 bash -c 'pkill -9 -f "[b]uild/GigaLearnBot" || true; sleep 3' 2>/dev/null || true

# MILESTONE ARCHIVE (policy files only, never rotated)
A=/gpfs/u/home/RLFR/RLFRlcmg/scratch-shared/checkpoints_gco_truevts_cs_archive
N=$(ls "$GGL_CHECKPOINT_FOLDER" 2>/dev/null | grep -E "^[0-9]+$" | sort -n | tail -1)
if [ -n "$N" ] && [ ! -d "$A/$N" ]; then
  mkdir -p "$A/$N.tmp" && cp "$GGL_CHECKPOINT_FOLDER/$N"/{POLICY.lt,SHARED_HEAD.lt,RUNNING_STATS.json} "$A/$N.tmp/" 2>/dev/null && mv "$A/$N.tmp" "$A/$N" && echo "ARCHIVED $N"
fi

MIN_OK_SECS=${MIN_OK_SECS:-300}
HOP_T0=$(date +%s)
run_mpi() {
mpirun -np "${NP_USE}" --hostfile "${HOSTFILE}" \
  --bind-to none --map-by ppr:6:node \
  -x LD_PRELOAD -x LD_LIBRARY_PATH -x OPAL_PREFIX \
  -x http_proxy -x https_proxy -x HTTP_PROXY -x HTTPS_PROXY \
  -x WANDB_API_KEY -x WANDB_MODE -x WANDB_DIR -x WANDB_INIT_TIMEOUT -x WANDB__SERVICE_WAIT -x PYTHONPATH \
  -x GGL_GCO -x GGL_NEXTO_SERVE_FRAC -x GGL_TRAIN_AGAINST_OLD_CHANCE -x GGL_NO_VERSIONS -x GGL_NO_REACH -x GGL_HULL \
  -x GGL_NEXTO_EVAL_FRAC -x GGL_NEXTOEVAL_SIDECAR \
  -x GGL_VTS -x GGL_VTS_HEAD -x GGL_VTS_BUCKETS -x GGL_THETA \
  -x GGL_LEAGUE -x GGL_LEAGUE_DIVERSE -x GGL_LEAGUE_EXPLOITERS -x GGL_LEAGUE_RANK -x GGL_LEAGUE_FRAC -x GGL_LEAGUE_LR -x GGL_LEAGUE_DIV_BETA -x GGL_LEAGUE_REPEL -x GGL_LEAGUE_REPELTARGET -x GGL_LEAGUE_REPELMAX -x GGL_LEAGUE_BINIT -x GGL_LEAGUE_ACCUM -x GGL_LEAGUE_MINIBATCH -x GGL_LEAGUE_BDECAY -x GGL_LEAGUE_RESPAWN -x GGL_LEAGUE_RESPAWN_MIN -x GGL_LEAGUE_RESPAWN_EVERY -x GGL_LEAGUE_EXPBOOST \
  -x GGL_ASYNC -x GGL_ASYNC_N_LEARNERS -x GGL_ASYNC_GLOBAL_BATCH -x GGL_ASYNC_MAX_POLICY_LAG -x GGL_PPO_EPOCHS -x GGL_NUM_GAMES \
  -x GGL_MUON_SHARD -x GGL_PIPELINED_COLLECTION -x GGL_INFER_FINITE_SYNC -x GGL_CUDA_GRAPHS -x GGL_MOE \
  -x GGL_DIST_SMOKE -x GGL_DIST_LOCKSTEP_CHECK -x GGL_FRESH_OPTIM -x GGL_ITER_PER_SAVE -x GGL_CKPT_KEEP -x GGL_BEST_KEEP -x GGL_TS_PER_VERSION \
  -x PYTORCH_CUDA_ALLOC_CONF -x GGL_CHECKPOINT_FOLDER -x GGL_RUN_NAME -x GGL_METRICS_GROUP_NAME \
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
kill $WATCHDOG_PID 2>/dev/null
echo "PULSAR2_GCO_TRUEVTS_HOP_DONE rc=${RC} $(date)"
