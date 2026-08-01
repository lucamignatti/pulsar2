#!/bin/bash
# Play a 1v1: you (Human) vs Pulsar2 (our GigaLearn bot), in real Rocket League.
#
# SAFETY MODEL:
#   * This script NEVER launches Rocket League (match config uses launcher="NoLaunch").
#     YOU launch it from Steam and pick "Play without Easy Anti-Cheat".
#   * EAC discriminator (precise): EAC mode runs the game bootstrapper
#     "RocketLeague_EAC.exe" and the PERSISTENT runtime "EasyAntiCheat_EOS.exe".
#     Non-EAC mode runs plain "RocketLeague.exe". The EAC *installer*
#     "EasyAntiCheat_EOS_Setup.exe" can run in BOTH modes, so it is explicitly IGNORED
#     (matching it broadly is what false-aborted the previous session).
#   * If a real EAC process appears, the watchdog HARD-ABORTS (kills game+server+bot).
#
# ONE-TIME Steam setup: Rocket League > Properties > Launch Options:  -rlbot
set -u
cd "$(dirname "$0")"

# Args:  ./play.sh [match_config] [team_size] [eval]
#   match_config: toml name (".toml" optional). Default = match_vs_human (you vs the bot).
#   team_size:    1|2|3 (per side). Default 1. Empty slots fill with bots; in the human
#                 match your own team fills with Pulsar2 bots too.
#   eval:         the token "eval" (or "replay") turns on EVAL MODE: a replay of the
#                 match is saved (auto_save_replay=true). Matches are UNLIMITED-length,
#                 so nothing ends on its own - the replay is written when you END THE
#                 SESSION (Ctrl-C): the holder sends StopMatch, RL flushes the .replay,
#                 THEN the server is torn down. Watch as long as you like; stop to save.
#   team_size and eval may appear in either order after the config.
# Examples:  ./play.sh                          -> 1v1, you vs Pulsar2 (syncs to newest ckpt)
#            ./play.sh match_vs_element 3        -> 3v3, Pulsar2 vs Element (you spectate)
#            ./play.sh match_vs_nexto 1 eval     -> 1v1 vs Nexto, save a replay
#            ./play.sh match_vs_nexto 1 nosync  -> keep the staged checkpoint, do not sync
CONFIG="${1:-match_vs_human.toml}"
[ -f "$CONFIG" ] || CONFIG="$CONFIG.toml"
[ -f "$CONFIG" ] || { echo "Match config not found: ${1:-match_vs_human.toml}"; exit 1; }
[ $# -gt 0 ] && shift   # drop the config arg; scan the rest for team_size / eval
TEAM_SIZE=1
REPLAY=0
MODE=""
SYNC=1
for a in "$@"; do
	case "$a" in
		1|2|3)                 TEAM_SIZE="$a" ;;
		eval|replay|--eval|--replay) REPLAY=1 ;;
		# Gap-verification modes. Written to HANDICAPS marker files (and exported), because
		# env prefixes on this chain (play.sh -> RLBotServer -> launch manager -> bot) are
		# unverifiable and silently failed to propagate on 2026-07-31 - the debug log
		# showed our kickoff tape running in a run that was supposed to disable it.
		#   bugnexto:  PERFECT Pulsar (tape + argmax + fixed reconstruction) vs viz-style
		#              Nexto (flip-never-expires bug, no kickoff script).
		#   simparity: same Nexto, and Pulsar ALSO drops to sim conditions (no tape,
		#              sampling) - the full sim-reproduction.
		#   sample: ONE lever - Pulsar samples from the policy like every sim evaluation
		#           does, instead of the client's argmax default. Nexto untouched.
		bugnexto|simparity|sample) MODE="$a" ;;
		# Keep the bot on whatever checkpoint is already staged (see sync_checkpoint).
		nosync|--nosync)       SYNC=0 ;;
		*) echo "Unknown arg '$a' (expected a team size 1-3, 'eval', 'nosync', 'bugnexto', 'simparity' or 'sample')"; exit 1 ;;
	esac
done
export REPLAY

EAC_MATCH=""   # set by eac_active() to the offending "pid: cmdline" for logging
# NOTE: Proton hides the Windows .exe name from /proc/pid/comm (15-char truncated) and
# /proc/pid/exe (points at the wine loader). The FULL Windows path only shows up in
# /proc/pid/cmdline, so we must match on that.
eac_active() { # true iff a REAL anti-cheat process is running (not the installer)
	EAC_MATCH=""
	local self=$$
	for c in /proc/[0-9]*/cmdline; do
		local pid; pid=$(basename "$(dirname "$c")")
		[ "$pid" = "$self" ] && continue
		# `cat 2>/dev/null` (not shell `< "$c"`) so a process exiting mid-scan is silent.
		local line; line=$(cat "$c" 2>/dev/null | tr '\0' ' ') || continue
		case "$line" in *[Ss]etup*) continue ;; esac        # EAC installer - harmless, ignore
		case "$line" in
			*RocketLeague_EAC.exe*|*EasyAntiCheat_EOS.exe*|*EasyAntiCheatBootstrapper*)
				EAC_MATCH="$pid: ${line:0:100}"; return 0 ;;
		esac
	done
	return 1
}
# The trainer saturates all 24 cores (1024 env threads) and 10-13 GB of the 16 GB GPU.
# A match run underneath it corrupts BOTH sides:
#   * Python agents starve far harder than ours. Nexto rebuilds its obs and runs a
#     37-entity attention forward per packet on ONE thread, and warns at boot that it
#     needs a stable 120fps; Pulsar's bot is a C++ MLP. The match then measures
#     scheduler priority, not skill - this is what produced the bogus "62-38 vs Nexto"
#     on 2026-07-31.
#   * Rocket League itself holds ~1.5 GB of the same GPU and OOM-crashed the trainer
#     TWICE that day (18:31, 19:26) - the game appears by PID in the trainer's OOM dump.
# CLAUDE.md: never look for the trainer with `pgrep -f` (it matches its own command
# line); resolve /proc/<pid>/exe instead.
trainer_active() {
	TRAINER_MATCH=""
	local e t
	for e in /proc/[0-9]*/exe; do
		t=$(readlink "$e" 2>/dev/null) || continue
		# Rebuilding build/ under a live trainer is the DOCUMENTED-SAFE workflow
		# (trainerctl update does it), and it replaces the inode - so the running
		# process's exe link reads ".../GigaLearnBot (deleted)". Strip that suffix
		# before matching or this guard silently misses the most common case, which
		# is exactly what it did on first write.
		t=${t% (deleted)}
		case "$t" in
			*build/GigaLearnBot)
				TRAINER_MATCH="pid $(basename "$(dirname "$e")"): $t"; return 0 ;;
		esac
	done
	return 1
}
# --- live checkpoint sync -----------------------------------------------------
# pulsar-bot/checkpoint is a STATIC COPY. The viz hot-swaps to the newest save; this
# does not, so without a sync every real-game eval silently benchmarks an old bot -
# measured 2026-07-31, a 6.50B copy against a 9.15B live run, i.e. 2.65B of training
# invisible to every match played that night. Default ON; `nosync` to pin.
#
# CKPT_ROOT default is the LIVE LINEAGE and it drifts: the folder has been
# checkpoints_6M -> _resid -> _5.1 -> _5.2 -> _5.3, and every rename has silently
# broken something downstream (see the cold-start checklist). So a missing root is a
# LOUD failure listing candidates, never a quiet fallback to a dead lineage.
sync_checkpoint() {
	local root="${GGL_CKPT_ROOT:-../build/checkpoints_5.3}"
	local dest="pulsar-bot/checkpoint"

	if [ ! -d "$root" ]; then
		log "SYNC FAILED: checkpoint root '$root' does not exist."
		log "SYNC: candidates -"
		for d in ../build/checkpoints_*/; do [ -d "$d" ] && log "SYNC:   $d"; done
		log "SYNC: set GGL_CKPT_ROOT=<dir> (the live lineage folder has changed 5x)."
		return 1
	fi

	local have=""
	[ -f "$dest/STEPS.txt" ] && have=$(cat "$dest/STEPS.txt" 2>/dev/null)

	# Numbered dirs only: skips best_r*<ts> golden entries, policy_versions/ and any
	# in-flight *.tmp. Saves are atomic (write to .tmp then rename), so a dir that
	# exists under its final name is complete - but ROTATION (8 kept, ~10min window)
	# can delete it mid-copy, which is why we walk several candidates newest-first
	# instead of trusting the first one.
	local cands
	cands=$(ls "$root" 2>/dev/null | grep -E '^[0-9]+$' | sort -rn | head -5)
	if [ -z "$cands" ]; then
		log "SYNC FAILED: no numbered checkpoints in $root"
		return 1
	fi

	local ts
	for ts in $cands; do
		if [ "$ts" = "$have" ]; then
			log "SYNC: already current at $ts steps ($(echo "scale=2; $ts/1000000000" | bc 2>/dev/null || echo "?")B)"
			return 0
		fi
		local src="$root/$ts"
		local stage="pulsar-bot/.ckpt_stage.$$"
		rm -rf "$stage"; mkdir -p "$stage" || return 1

		# Only what InferUnit actually loads. RUNNING_STATS is optional (this lineage
		# trains on raw obs, standardizeObs=false) but copied when present so the dir
		# stays a faithful subset of one checkpoint.
		if cp "$src/POLICY.lt" "$stage/" 2>/dev/null \
			&& cp "$src/SHARED_HEAD.lt" "$stage/" 2>/dev/null \
			&& [ -s "$stage/POLICY.lt" ] && [ -s "$stage/SHARED_HEAD.lt" ]; then
			cp "$src/RUNNING_STATS.json" "$stage/" 2>/dev/null || true
			echo "$ts" > "$stage/STEPS.txt"
			mkdir -p "$dest"
			# Clear stale .lt first: the destination accumulated a full checkpoint's
			# worth of files (critics, optims) from an old copy, and leaving 6.5B
			# criticsnext to a 9.7B policy makes the dir lie about what it holds.
			rm -f "$dest"/*.lt "$dest"/RUNNING_STATS.json 2>/dev/null
			mv "$stage"/* "$dest"/ && rmdir "$stage"
			log "SYNC: ${have:-<none>} -> $ts steps ($(echo "scale=2; $ts/1000000000" | bc 2>/dev/null || echo "?")B)"
			log "SYNC: if the bot aborts with a size mismatch, RLBotMain.cpp's hardcoded"
			log "SYNC: net config no longer matches the trainer's (the standing hazard)."
			return 0
		fi
		rm -rf "$stage"
		log "SYNC: $ts vanished mid-copy (rotation), trying next-newest..."
	done

	log "SYNC FAILED: every candidate rotated away mid-copy; keeping ${have:-existing} checkpoint"
	return 1
}

rl_connected() { grep -q "Connected to Rocket League" core_play.log 2>/dev/null; }
# Each Pulsar2 car runs run.sh, which logs to pulsar-bot/bot.<pid>.log (one per car).
bot_spawned()  { grep -qs "Created RLBot bot" pulsar-bot/bot.*.log 2>/dev/null; }

log() { echo "$@"; echo "$(date '+%H:%M:%S') $*" >> watchdog.log; }

kill_all() {
	pkill -f 'RocketLeague_EAC\.exe' 2>/dev/null
	pkill -f 'EasyAntiCheat_EOS\.exe' 2>/dev/null
	pkill -f GigaLearnRLBot 2>/dev/null
	[ -n "${HOLDER:-}" ] && kill "$HOLDER" 2>/dev/null
	[ -n "${CORE:-}"   ] && kill "$CORE"   2>/dev/null
	pkill -f RLBotServer 2>/dev/null
}
abort_eac() {
	log "############################################################"
	log "#  EAC DETECTED - HARD ABORT.  matched process: $EAC_MATCH"
	log "#  Killing game + server + bot NOW. You launched WITH anti-cheat."
	log "#  Quit RL, relaunch, choose 'Play without Easy Anti-Cheat', re-run."
	log "############################################################"
	kill_all
	exit 1
}
# EVAL only: matches are unlimited-length, so a replay is only written when the
# holder sends StopMatch (auto_save_replay). RLBotServer must still be ALIVE when
# that happens, so we signal the holder and WAIT for it to finish saving BEFORE
# kill_all tears the server down. No-op outside eval or if the holder already exited.
# Proton prefix Demos folder for Rocket League (appid 252950) on this box.
REPLAY_DIR="/run/media/luca/biggestbox/SteamLibrary/steamapps/compatdata/252950/pfx/drive_c/users/steamuser/Documents/My Games/Rocket League/TAGame/Demos"
save_replay_if_eval() {
	[ "$REPLAY" = 1 ] || return 0
	[ -n "${HOLDER:-}" ] && kill -0 "$HOLDER" 2>/dev/null || return 0
	log "[eval] ending match & saving replay (holder StopMatch)..."
	kill -TERM "$HOLDER" 2>/dev/null
	for _ in $(seq 1 60); do          # up to ~12s (holder flushes for 5s then exits)
		kill -0 "$HOLDER" 2>/dev/null || break
		sleep 0.2
	done
	# Report the newest replay so it's clear the save landed (and where).
	newest=$(ls -t "$REPLAY_DIR"/*.replay 2>/dev/null | head -1)
	if [ -n "$newest" ] && [ -n "$(find "$newest" -mmin -1 2>/dev/null)" ]; then
		log "[eval] replay saved: $newest"
	else
		log "[eval] WARNING: no fresh .replay in $REPLAY_DIR - save may have failed (check holder.log)."
	fi
}
trap 'echo; echo "Ending session..."; save_replay_if_eval; kill_all; exit 0' INT TERM

pkill -f RLBotServer 2>/dev/null
sleep 1
: > core_play.log; : > watchdog.log; : > procs.log; rm -f pulsar-bot/bot.*.log pulsar-bot/debug.*.jsonl 2>/dev/null
# Marker files ALWAYS cleared first (a crashed handicap run must never leak its
# handicaps into the next normal match), then re-written only if this run asks.
rm -f nexto/HANDICAPS pulsar-bot/HANDICAPS 2>/dev/null
if [ "$MODE" = "bugnexto" ] || [ "$MODE" = "simparity" ]; then
	printf "NEXTO_VIZ_BUG=1\nNEXTO_NO_KICKOFF=1\n" > nexto/HANDICAPS
	export NEXTO_VIZ_BUG=1 NEXTO_NO_KICKOFF=1
	log "MODE $MODE: Nexto handicapped (viz flip bug + no kickoff script)"
fi
if [ "$MODE" = "simparity" ]; then
	printf "GGL_NO_KICKOFF_SCRIPT=1\nGGL_SAMPLE_ACTIONS=1\n" > pulsar-bot/HANDICAPS
	export GGL_NO_KICKOFF_SCRIPT=1 GGL_SAMPLE_ACTIONS=1
	log "MODE simparity: Pulsar also at sim conditions (no tape, sampling)"
fi
if [ "$MODE" = "sample" ]; then
	printf "GGL_SAMPLE_ACTIONS=1\n" > pulsar-bot/HANDICAPS
	export GGL_SAMPLE_ACTIONS=1
	log "MODE sample: Pulsar samples from the policy (every sim evaluation samples; the client's argmax default has never been evaluated anywhere else)"
fi
[ -n "$MODE" ] && log "RECEIPTS: check core_play.log for 'Nexto HANDICAPS' and pulsar-bot/bot.*.log for 'RLBot flags' - a missing receipt means the flag did NOT land"

# Before the server launches the bot, so the car spawns on the checkpoint we just staged.
# A failed sync is NOT fatal: playing the previous checkpoint beats refusing to play,
# and the log says loudly which one it is.
if [ "$SYNC" = 1 ]; then
	sync_checkpoint || log "SYNC: continuing on the previously staged checkpoint"
else
	log "SYNC: skipped (nosync); bot stays on $(cat pulsar-bot/checkpoint/STEPS.txt 2>/dev/null || echo '<unknown>') steps"
fi
log "Match: $CONFIG   team_size: $TEAM_SIZE   eval(replay): $([ "$REPLAY" = 1 ] && echo on || echo off)"
if eac_active; then log "Refusing to start: EAC already running ($EAC_MATCH)"; abort_eac; fi
if trainer_active && [ "${ALLOW_TRAINER:-0}" != "1" ]; then
	log "############################################################"
	log "#  TRAINER IS LIVE - $TRAINER_MATCH"
	log "#  Refusing to start. A match under the trainer measures scheduler"
	log "#  priority, not skill: Python agents (Nexto) starve far harder than"
	log "#  our C++ bot, and the game's GPU use has OOM-crashed the trainer."
	log "#"
	log "#    tools/trainerctl stop      # then re-run this; restart when done"
	log "#    ALLOW_TRAINER=1 ./play.sh ...   # override (result is NOT a measurement)"
	log "############################################################"
	exit 1
fi
if trainer_active; then
	log "WARNING: ALLOW_TRAINER=1 with the trainer live ($TRAINER_MATCH)."
	log "WARNING: both sides are starved and the game may OOM-crash the trainer."
	log "WARNING: whatever this match shows is NOT a valid measurement."
fi

# --- background process logger: snapshot RL/EAC procs every 1s (pure diagnostics) ----
(
	while true; do
		ts=$(date '+%H:%M:%S')
		for c in /proc/[0-9]*/cmdline; do
			line=$(cat "$c" 2>/dev/null | tr '\0' ' ') || continue
			case "$line" in *RocketLeague*|*EasyAntiCheat*|*RLBotServer*|*GigaLearnRLBot*)
				echo "$ts $(basename "$(dirname "$c")") ${line:0:110}" >> procs.log ;;
			esac
		done
		echo "$ts ----" >> procs.log
		sleep 1
	done
) &
PROCLOG=$!

RLBOT_LOG_LEVEL=debug ./RLBotServer > core_play.log 2>&1 &   # debug: show the match-start handshake
CORE=$!
log "RLBotServer starting (pid $CORE) - NoLaunch, will not open the game."
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ':23234' && break; sleep 0.5; done

# Match-manager holder: builds the match at the requested team size (filling empty/human
# slots with bots) and holds the connection. Full output -> holder.log.
.venv/bin/python -u run_match_holder.py "$CONFIG" "$TEAM_SIZE" > holder.log 2>&1 &
HOLDER=$!

echo ""
echo "==================================================================="
echo " NOW: launch Rocket League from Steam."
echo " At the prompt, choose  >>> Play without Easy Anti-Cheat <<<"
echo " Watchdog armed (precise: ignores the EAC *installer*, aborts only on"
echo " the real anti-cheat runtime). Diagnostics -> procs.log / holder.log."
echo "==================================================================="
echo ""

announced_conn=0; announced_bot=0
while kill -0 "$CORE" 2>/dev/null; do
	if eac_active; then abort_eac; fi
	if [ $announced_conn -eq 0 ] && rl_connected; then
		announced_conn=1; log "[safe] Rocket League connected in NON-EAC mode."
	fi
	if [ $announced_bot -eq 0 ] && bot_spawned; then
		announced_bot=1; log "[ok] Pulsar2 spawned into the match - GO PLAY. (Ctrl+C to end.)"
	fi
	sleep 1
done
log "[note] RLBotServer exited. Tearing down. (See core_play.log / holder.log / procs.log.)"
kill "$PROCLOG" 2>/dev/null
kill_all
