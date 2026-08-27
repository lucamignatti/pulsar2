#!/bin/bash
# Launch wrapper for the GigaLearn RLBot v5 bot.
# The RLBotServer/RLBotSim runs this (run_command_linux) with cwd = this file's dir
# and sets RLBOT_SERVER_IP / RLBOT_SERVER_PORT / RLBOT_AGENT_ID in the environment.
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="/home/luca/Projects/pulsar2-3.0"
# GigaLearnRLBot is dynamically linked against libGigaLearnCPP, libRLBotCPP and libtorch.
export LD_LIBRARY_PATH="$REPO/build:$REPO/GigaLearnCPP/libtorch/lib:$LD_LIBRARY_PATH"
# Inference device. REVERTED TO CPU 2026-08-27 (user-directed).
#
# The GPU was chosen because CPU inference costs ~2.78 ms of the 8.33 ms ts1 tick budget,
# on the reasoning that "the GPU is idle during a match (the trainer lives on the
# cluster)". That reasoning has a hole: the GPU is NOT idle - Rocket League is rendering
# on it. A 230-dim MLP is a long chain of tiny kernels, and at 120 Hz those launches
# preempt the game's graphics work all match long. We are trading a cost we can measure
# (a few ms on ONE of 24 CPU cores) for contention on the single resource the game needs
# most, to protect the in-game rate that is the actual complaint.
#
# The CPU side is also much cheaper than it was when the GPU switch was made: the box was
# then losing ~a full core to a fork storm (34 leaked play.sh watchdogs, fixed 2026-08-27)
# and the bot was flushing a debug JSONL every tick (also fixed). The 2.78 ms figure was
# measured under that load and should be re-measured before it is quoted again.
#
# GGL_USE_GPU=1 ./play.sh ... still forces the GPU for an A/B.
export GGL_USE_GPU="${GGL_USE_GPU:-0}"
# SINGLE-THREADED CPU INFERENCE - REQUIRED, not a tuning preference.
#
# Nothing in the RLBot path had ever set a thread count (Learner.cpp and NextoEval.cpp
# both do; InferUnit and RLBotMain did not), so libtorch would default intra-op threads to
# the core count - 24 here. Splitting a BATCH-1 MLP's GEMMs across 24 threads is all
# overhead: the fork/join and barrier cost per layer dwarfs the arithmetic, and OpenMP's
# default spin-wait then burns every core in the gaps between ticks. Flipping to CPU
# without this would have made the lag WORSE than the GPU it replaced, which is exactly
# the failure it was supposed to fix.
#
# Set here (env, before the process starts) rather than only via at::set_num_threads,
# because OMP's thread pool is sized at library init and is not reliably shrinkable after.
# RLBotMain also calls at::set_num_threads(1) on the CPU path as a belt-and-braces guard.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# Do not let idle OMP workers spin-wait on cores the game wants.
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
# Per-decision debug JSONL (exact obs/mask/action + raw packet jump fields + air-state
# transitions) -> debug.<pid>.jsonl next to this script. NOT "a few MB": at ts1/120 Hz it
# serialises a 230-float obs every tick and reached 736 MB in one session (1.22 GB in
# another), so it is real per-tick CPU and real disk I/O - measured 193 KB/s of writes and
# ~280 snprintf calls per tick, all inside the packet callback.
#
# DEFAULT FLIPPED TO OFF 2026-08-27. It had been defaulting ON, so every ordinary game -
# including every match played to judge how the bot FEELS - paid the capture cost and left
# a multi-hundred-MB file behind. The capture is a MEASUREMENT tool: turn it on explicitly
# for the run that needs it (GGL_DEBUG_JSONL=1 ./play.sh ...), which is also the only way
# the resulting file is unambiguously attributable to an intended experiment.
export GGL_DEBUG_JSONL="${GGL_DEBUG_JSONL:-0}"
# Gap-verification handicaps, written by play.sh's mode tokens (see its comments for
# why a marker file instead of env prefixes: the env chain silently dropped them once).
if [ -f "$HERE/HANDICAPS" ]; then
	set -a; . "$HERE/HANDICAPS"; set +a
fi
# Capture each car's stdout+stderr + exit code to bot.<pid>.log (one file per Pulsar2 car,
# so 2v2/3v3 instances don't clobber each other). Segfault leaves exit 139; an uncaught
# C++ exception leaves its what() text here.
# Decision rate travels with the staged checkpoint (written by play.sh's sync_checkpoint).
# Without this the bot silently runs at its 15 Hz default on a 120 Hz (ts1) policy.
if [ -z "$GGL_TICK_SKIP" ] && [ -f "$HERE/checkpoint/TICKSKIP" ]; then
	export GGL_TICK_SKIP="$(cat "$HERE/checkpoint/TICKSKIP")"
fi

LOG="$HERE/bot.$$.log"
"$REPO/build/GigaLearnRLBot" "$HERE/checkpoint" > "$LOG" 2>&1
echo "=== GigaLearnRLBot exited with code $? ===" >> "$LOG"
