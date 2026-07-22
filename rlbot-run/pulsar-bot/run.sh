#!/bin/bash
# Launch wrapper for the GigaLearn RLBot v5 bot.
# The RLBotServer/RLBotSim runs this (run_command_linux) with cwd = this file's dir
# and sets RLBOT_SERVER_IP / RLBOT_SERVER_PORT / RLBOT_AGENT_ID in the environment.
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="/home/luca/Projects/pulsar2-3.0"
# GigaLearnRLBot is dynamically linked against libGigaLearnCPP, libRLBotCPP and libtorch.
export LD_LIBRARY_PATH="$REPO/build:$REPO/GigaLearnCPP/libtorch/lib:$LD_LIBRARY_PATH"
# Capture each car's stdout+stderr + exit code to bot.<pid>.log (one file per Pulsar2 car,
# so 2v2/3v3 instances don't clobber each other). Segfault leaves exit 139; an uncaught
# C++ exception leaves its what() text here.
LOG="$HERE/bot.$$.log"
"$REPO/build/GigaLearnRLBot" "$HERE/checkpoint" > "$LOG" 2>&1
echo "=== GigaLearnRLBot exited with code $? ===" >> "$LOG"
