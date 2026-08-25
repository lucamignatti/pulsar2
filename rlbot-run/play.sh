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
#
#                 QUALIFIER GRIND (eval vs Nexto only): an eval run of a *nexto* config
#                 additionally arms the 42-before-28 race (override: QUAL_FOR/QUAL_AGAINST
#                 env; disable: 'noqual' token). The holder watches the score; reaching
#                 QUAL_FOR first ENDS THE SESSION ITSELF with the replay saved, and Nexto
#                 reaching QUAL_AGAINST first restarts the match (fresh 0-0) for the next
#                 attempt, forever, until an attempt lands. Leave it running; the replay
#                 in $REPLAY_DIR when it stops IS the qualifier evidence. Nexto is at
#                 full strength here (no handicap mode arms the qualifier).
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
TAPE=0   # kickoff tape default OFF; 'tape' token re-enables
NOQUAL=0 # 'noqual' disables the eval-vs-nexto qualifier race
# Which checkpoint to stage. "latest" (default) = newest numbered rotation checkpoint =
# the policy AS IT IS RIGHT NOW. "golden" = highest-rated best_r* entry.
#
# The default is deliberately LATEST, and it is the honest one for measurement. Golden
# entries are selected by Rating/1v1, which this project has established is inflated ~6x
# and is a treadmill (PolicyVersionManager::AddVersion copies the main's CURRENT rating
# into each new version, so the pool tracks the agent). Defaulting to golden would mean
# every real-match number silently reported a moment when the measuring stick happened to
# be favourable, biasing eval upward against a Nexto that is always playing its normal
# self. Use golden when you want to PLAY the best bot you have; use latest when you want
# to KNOW how strong it currently is.
# (Golden is also not safer against corruption: on 6.2, best_r2068_10525158912 existed as
# both a golden entry and a corrupt_ one, having been archived inside a bad save window.)
CKPT_PICK="latest"
for a in "$@"; do
	case "$a" in
		1|2|3)                 TEAM_SIZE="$a" ;;
		eval|replay|--eval|--replay) REPLAY=1 ;;
		golden|--golden)       CKPT_PICK="golden" ;;
		latest|--latest)       CKPT_PICK="latest" ;;
		# Gap-verification modes. Written to HANDICAPS marker files (and exported), because
		# env prefixes on this chain (play.sh -> RLBotServer -> launch manager -> bot) are
		# unverifiable and silently failed to propagate on 2026-07-31 - the debug log
		# showed our kickoff tape running in a run that was supposed to disable it.
		#   bugnexto:  PERFECT Pulsar (argmax + fixed reconstruction; tape follows the
		#              global default below - OFF unless 'tape' is passed) vs viz-style
		#              Nexto (flip-never-expires bug, no kickoff script), SOFTENED a
		#              further silent notch: NEXTO_BETA=0.85 - sampling at ~2.3x logit
		#              scale instead of argmax. Same net, same style, no visible
		#              randomness; it just occasionally takes a 2nd-choice action.
		#              NOTE this makes bugnexto a PLAY mode, no longer a pure
		#              gap-verification fixture; simparity keeps the honest viz-Nexto.
		#   simparity: viz-Nexto at full argmax strength (NO beta nerf - the sim's
		#              NextoOpponent.cpp argmaxes, so beta<1 would break reproduction),
		#              and Pulsar ALSO drops to sim conditions (no tape, sampling) -
		#              the full sim-reproduction.
		#   sample: ONE lever - Pulsar samples from the policy like every sim evaluation
		#           does, instead of the client's argmax default. Nexto untouched.
		# Gap-verification runs ALWAYS save a replay: they exist to produce evidence,
		# and a replay is the only record that survives the session (remember: the
		# match is unlimited-length, so the replay is written when you Ctrl-C).
		bugnexto|simparity|sample) MODE="$a"; REPLAY=1 ;;
		# Pulsar's kickoff tape is OFF BY DEFAULT (user preference 2026-08-11: the
		# policy plays its own kickoffs in every mode, including plain eval). This
		# token re-enables the scripted kickoff for a run.
		tape|--tape)           TAPE=1 ;;
		# Keep the bot on whatever checkpoint is already staged (see sync_checkpoint).
		nosync|--nosync)       SYNC=0 ;;
		# Disable the qualifier race that eval-vs-nexto arms by default.
		noqual|--noqual)       NOQUAL=1 ;;
		*) echo "Unknown arg '$a' (expected a team size 1-3, 'eval', 'nosync', 'tape', 'noqual', 'bugnexto', 'simparity' or 'sample')"; exit 1 ;;
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
# RLBot's FlatBuffers port. If anything already holds it, RLBotServer logs ONE line into
# core_play.log ("Port 23234 is already in use") and exits - and because the game had
# already connected by then, what you SEE is a successful launch followed by a generic
# "RLBotServer exited. Tearing down.", with no hint of the cause. That cost a full
# launch-the-game cycle on 2026-08-09 before the log was read.
#
# The usual culprit is NOT a stale RLBotServer (line ~305 pkills those): it is the VIZ
# RENDER SERVICE. build-viz is compiled with GGL_VIZ_RLBOT=ON so the viewer can hand its
# other team to a real RLBot bot, which means it binds this port for as long as it runs.
# A leftover bot.py from a previous session - including one in the SIBLING checkout, which
# play.sh also scans - does the same.
RLBOT_PORT=23234
PORT_HOLDER=""
port_held() {
	PORT_HOLDER=""
	command -v ss >/dev/null 2>&1 || return 1
	local out
	out=$(ss -lptn "sport = :$RLBOT_PORT" 2>/dev/null | tail -n +2) || return 1
	[ -n "$out" ] || return 1
	PORT_HOLDER=$(echo "$out" | sed 's/.*users:(//; s/)$//')
	return 0
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
# tickSkip is NOT stored in a checkpoint, and unlike the layer sizes (which the bot now
# derives from the weights) a mismatch is SILENT - the bot just decides at the wrong rate.
# 5.x lineages are ts8 (15 Hz); 6.0 is ts1 (120 Hz). Derive it from the lineage folder and
# stage it NEXT TO the checkpoint so the copy carries its own decision rate.
tick_skip_for_root() {
	# ORDER MATTERS: 7.0b is ts8 but would be caught by the 7.* rule below it.
	# The version number does NOT imply the rate -- 7.0/7.1/7.2 were ts1 and 7.0b went
	# back to ts8 as the validation seat for the composite critic, so this has to be an
	# explicit table, not a prefix guess. Getting it wrong is SILENT: the bot just decides
	# at the wrong rate and plays badly with no error anywhere.
	case "$1" in
		# MUST come before the 7.* rule. 7.8dense is the AiMOS dense control (2026-08-19),
		# trained at tickSkip 8 like the 5.x/7.0b seats - NOT ts1 like 7.0/7.1/7.2.
		*checkpoints_7.8dense*) echo 8 ;;
		# MUST come before the plain 7.9gco rule: "*checkpoints_7.9gco*" ALSO matches
		# "checkpoints_7.9gco_ts1", which would stamp the ts1 lineage as ts8 and run a
		# 120 Hz policy at 15 Hz - silent, no error, just a broken bot.
		# 7.9gco_ts1 = same gco lineage after the 2026-08-23 mid-run switch to tickSkip 1
		# (120 Hz, gamma 0.99971123 / 20s). Recovered to ~98% at ts1 within ~8h.
		*checkpoints_7.9gco_ts1*) echo 1 ;;
		# 7.9gco = the goal/concede-only sparse cold start (2026-08-21), ts8 recipe.
		*checkpoints_7.9gco*) echo 8 ;;
		*checkpoints_7.0b*) echo 8 ;;   # composite critic, ts8 validation seat
		*checkpoints_7.*)   echo 1 ;;   # 7.0 / 7.1 / 7.2 were all ts1
		*checkpoints_6.*)   echo 1 ;;   # 6.0 / 6.1 / 6.1b / 6.2 all ts1
		*)                  echo 8 ;;   # 5.x and earlier
	esac
}

sync_checkpoint() {
	# DEFAULT LINEAGE, 2026-08-19: the dense control from the AiMOS architecture A/B.
	# 36.7M dense (trunk 1280x3, policy 768x3), 13.0B steps, measured 91.6% goal share
	# vs Nexto (229-21, 250-goal clean eval, argmax/deployment, ts8) - the strongest bot
	# this project has produced and the first to beat Nexto convincingly. It replaces
	# checkpoints_7.0b here. The MoE line was retired the same day: on an identical
	# recipe the 1B MoE reached 3.6% while this reached 91.6%, because the MoE forward
	# costs 22ms + 0.2ms*experts with no token term (25k SPS vs dense's 515k), i.e. 20x
	# fewer experiences per hour. Override with GGL_CKPT_ROOT=<dir> as always.
	local root="${GGL_CKPT_ROOT:-../build/checkpoints_7.8dense}"
	local dest="pulsar-bot/checkpoint"

	if [ ! -d "$root" ]; then
		log "SYNC FAILED: checkpoint root '$root' does not exist."
		log "SYNC: candidates -"
		for d in ../build/checkpoints_*/; do [ -d "$d" ] && log "SYNC:   $d"; done
		log "SYNC: set GGL_CKPT_ROOT=<dir> (the live lineage folder has changed 5x)."
		return 1
	fi

	# Always (re)stamp the decision rate for this lineage, even on a nosync/no-op sync,
	# so a stale marker from a previous lineage can never outlive its checkpoint.
	local ts_for_root
	ts_for_root="${GGL_TICK_SKIP:-$(tick_skip_for_root "$root")}"
	mkdir -p "$dest" 2>/dev/null
	echo "$ts_for_root" > "$dest/TICKSKIP"
	log "SYNC: lineage '$root' -> tickSkip $ts_for_root ($(awk -v t="$ts_for_root" 'BEGIN{printf "%.0f", 120/t}') Hz)"

	# In-game display name, per lineage (bot.toml is shared by every root, so stamp it
	# at sync time like TICKSKIP - a stale name can never outlive its checkpoint).
	local bot_name
	case "$root" in
		*checkpoints_7.9gco*) bot_name="pulsar2-GCO" ;;   # sparse goal-only cold start
		*)                    bot_name="Pulsar2" ;;
	esac
	bot_name="${BOT_NAME:-$bot_name}"
	sed -i "s/^name = \".*\"/name = \"$bot_name\"/" pulsar-bot/bot.toml
	log "SYNC: bot name -> $bot_name"

	local have=""
	[ -f "$dest/STEPS.txt" ] && have=$(cat "$dest/STEPS.txt" 2>/dev/null)

	# Numbered dirs only: skips best_r*<ts> golden entries, policy_versions/ and any
	# in-flight *.tmp. Saves are atomic (write to .tmp then rename), so a dir that
	# exists under its final name is complete - but ROTATION (8 kept, ~10min window)
	# can delete it mid-copy, which is why we walk several candidates newest-first
	# instead of trusting the first one.
	local cands
	if [ "$CKPT_PICK" = "golden" ]; then
		# Golden entries are best_r<rating>_<ts>; rank by RATING (field 2 on '_'), not by
		# timestep, since that is what "best" means here. Map back to the plain <ts> name
		# used below by resolving through a parallel lookup: the loop stages from
		# "$root/$ts", so hand it the directory names directly instead.
		cands=$(ls "$root" 2>/dev/null | grep -E '^best_r[0-9]+_[0-9]+$' \
			| sort -t_ -k2.2 -rn | head -5)
		if [ -z "$cands" ]; then
			log "SYNC FAILED: 'golden' requested but no best_r* entries in $root"
			log "SYNC: golden entries only appear once the skill tracker has rated a save."
			return 1
		fi
		log "SYNC: picking GOLDEN (best-rated). NOTE Rating/1v1 is the inflated pool metric —"
		log "SYNC: this is 'the best bot I have', NOT 'how strong it is now'. Use the default"
		log "SYNC: (latest) for measurement."
	else
		cands=$(ls "$root" 2>/dev/null | grep -E '^[0-9]+$' | sort -rn | head -5)
		if [ -z "$cands" ]; then
			log "SYNC FAILED: no numbered checkpoints in $root"
			return 1
		fi
	fi

	local ts
	for ts in $cands; do
		# $ts is the DIRECTORY NAME, which for a golden pick is best_r<rating>_<steps>,
		# not a bare step count. Everything user-facing (STEPS.txt, the already-current
		# check, the log lines) must use the step number, or the staged copy claims to be
		# at step "best_r1700_13400211456" and the next run re-syncs forever.
		local steps="${ts##*_}"
		if [ "$steps" = "$have" ]; then
			log "SYNC: already current at $steps steps ($(echo "scale=2; $steps/1000000000" | bc 2>/dev/null || echo "?")B)"
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
			echo "$steps" > "$stage/STEPS.txt"
			mkdir -p "$dest"
			# Clear stale .lt first: the destination accumulated a full checkpoint's
			# worth of files (critics, optims) from an old copy, and leaving 6.5B
			# criticsnext to a 9.7B policy makes the dir lie about what it holds.
			rm -f "$dest"/*.lt "$dest"/RUNNING_STATS.json 2>/dev/null
			mv "$stage"/* "$dest"/ && rmdir "$stage"
			log "SYNC: ${have:-<none>} -> $steps steps ($(echo "scale=2; $steps/1000000000" | bc 2>/dev/null || echo "?")B)${CKPT_PICK:+ [$CKPT_PICK: $ts]}"
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
	# Markers must never outlive the run that wrote them. play.sh clears them at
	# startup, but pulsar-bot/run.sh and nexto/bot.py also read them and the VIZ
	# launches those directly - so a handicap run that ended without this cleanup
	# would silently disable the kickoff tape (or bug Nexto) in the next viz session.
	rm -f nexto/HANDICAPS pulsar-bot/HANDICAPS 2>/dev/null
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
# Proton prefix Demos folder for Rocket League (appid 252950).
# This was HARDCODED to the biggestbox library and was WRONG: that library holds an old,
# stale 252950 prefix (newest replay there is from 07-20), while Steam actually launches
# RL from the default ~/.local/share/Steam prefix. Consequences, both silent: the
# failed-attempt replay discard deleted nothing, and the end-of-match "wait for the
# .replay to appear" poll watched a folder that never changes, so a successfully saved
# replay still reported as missing. There are 4 Steam libraries on this box, so resolve
# it by DATA - the prefix with the most recently written .replay is the live one.
_resolve_replay_dir() {
	local best="" best_t=0 d t
	for lib in "$HOME/.local/share/Steam" /run/media/"$USER"/*/SteamLibrary; do
		d="$lib/steamapps/compatdata/252950/pfx/drive_c/users/steamuser/Documents/My Games/Rocket League/TAGame/Demos"
		[ -d "$d" ] || continue
		[ -z "$best" ] && best="$d"        # fall back to any existing prefix
		t=$(find "$d" -maxdepth 1 -name '*.replay' -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
		t=${t%%.*}
		if [ -n "$t" ] && [ "$t" -gt "$best_t" ] 2>/dev/null; then best_t=$t; best=$d; fi
	done
	printf '%s' "$best"
}
REPLAY_DIR="${REPLAY_DIR:-$(_resolve_replay_dir)}"
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
: > core_play.log; : > watchdog.log; : > procs.log; rm -f pulsar-bot/bot.*.log pulsar-bot/debug.*.jsonl QUALIFIED 2>/dev/null
# Marker files ALWAYS cleared first (a crashed handicap run must never leak its
# handicaps into the next normal match), then re-written only if this run asks.
rm -f nexto/HANDICAPS pulsar-bot/HANDICAPS 2>/dev/null
if [ "$MODE" = "bugnexto" ] || [ "$MODE" = "simparity" ]; then
	printf "NEXTO_VIZ_BUG=1\nNEXTO_NO_KICKOFF=1\n" > nexto/HANDICAPS
	export NEXTO_VIZ_BUG=1 NEXTO_NO_KICKOFF=1
	log "MODE $MODE: Nexto handicapped (viz flip bug + no kickoff script)"
fi
# bugnexto ONLY (not simparity - the sim argmaxes, so reproduction must too):
# the silent notch. beta<1 makes Nexto SAMPLE its policy with logits sharpened by
# log_3((1+b)/(1-b)) instead of argmaxing - indistinguishable to the eye, quietly
# weaker. Receipt: "Nexto HANDICAPS active: ... beta=0.85" in core_play.log.
if [ "$MODE" = "bugnexto" ]; then
	printf "NEXTO_BETA=0.85\n" >> nexto/HANDICAPS
	export NEXTO_BETA=0.85
	log "MODE bugnexto: Nexto softened (beta 0.85 sampling instead of argmax)"
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
# Kickoff tape default: OFF for every run (user preference 2026-08-11) - the policy
# plays its own kickoffs unless the 'tape' token was passed. Appended (not '>') so it
# composes with whatever a mode block wrote above; the grep guard skips the append when
# simparity already wrote the same flag. Receipt: "RLBot flags ... kickoff script OFF"
# in pulsar-bot/bot.*.log.
if [ "$TAPE" = 0 ]; then
	if ! grep -qs '^GGL_NO_KICKOFF_SCRIPT=1$' pulsar-bot/HANDICAPS; then
		printf "GGL_NO_KICKOFF_SCRIPT=1\n" >> pulsar-bot/HANDICAPS
	fi
	export GGL_NO_KICKOFF_SCRIPT=1
	log "Pulsar kickoff tape OFF (default; pass 'tape' to re-enable)"
else
	log "Pulsar kickoff tape ON ('tape' token)"
fi
[ -n "$MODE" ] && log "RECEIPTS: check core_play.log for 'Nexto HANDICAPS' and pulsar-bot/bot.*.log for 'RLBot flags' - a missing receipt means the flag did NOT land"

# Before the server launches the bot, so the car spawns on the checkpoint we just staged.
# A failed sync is NOT fatal: playing the previous checkpoint beats refusing to play,
# and the log says loudly which one it is.
# sync_checkpoint ONLY ever stages pulsar-bot/. Configs that field a STATIC bot dir
# (pulsar-gco-bot, pulsar-gco-ts1-bot) ignore it entirely, so reporting the sync as if it
# described the match is actively misleading - it once read "already current at 67.75B"
# for a match whose cars were the 404.5B ts1 policy and BonkDaddy. Only sync when this
# config actually uses pulsar-bot, and always print the REAL participants afterwards.
if grep -q 'config_file = "pulsar-bot/bot.toml"' "$CONFIG" 2>/dev/null; then
	if [ "$SYNC" = 1 ]; then
		sync_checkpoint || log "SYNC: continuing on the previously staged checkpoint"
	else
		log "SYNC: skipped (nosync); bot stays on $(cat pulsar-bot/checkpoint/STEPS.txt 2>/dev/null || echo '<unknown>') steps"
	fi
else
	log "SYNC: skipped - '$CONFIG' does not use pulsar-bot (static bot dirs only)."
fi
# Say what is ACTUALLY on the field: each car's dir, staged steps and decision rate.
log "PLAYERS:"
grep -oE 'config_file = "[^"]+"' "$CONFIG" 2>/dev/null | sed 's/.*"\(.*\)"/\1/' | while read -r bt; do
	bdir="$(dirname "$bt")"
	bname="$(grep -m1 '^name' "$bt" 2>/dev/null | sed 's/.*= *"\(.*\)"/\1/')"
	steps="$(cat "$bdir/checkpoint/STEPS.txt" 2>/dev/null)"
	ts="$(cat "$bdir/checkpoint/TICKSKIP" 2>/dev/null)"
	if [ -n "$steps" ]; then
		log "PLAYERS:   ${bname:-?}  <- $bdir  ${steps} steps, tickSkip ${ts:-?} ($(awk -v t="${ts:-8}" 'BEGIN{printf "%.0f", 120/t}') Hz)"
	else
		log "PLAYERS:   ${bname:-?}  <- $bdir"
	fi
done
# QUALIFIER: eval runs against a *nexto* config race to QUAL_FOR-before-QUAL_AGAINST
# (default 42/28). Plain eval only - a handicap MODE must never produce a qualifier
# replay, so those runs stay hold-forever even though they force REPLAY=1.
QUAL_ON=0
if [ "$REPLAY" = 1 ] && [ "$NOQUAL" = 0 ] && [ -z "$MODE" ]; then
	case "$CONFIG" in
		*nexto*)
			QUAL_ON=1
			export QUAL_FOR="${QUAL_FOR:-42}" QUAL_AGAINST="${QUAL_AGAINST:-28}" QUAL_TEAM=0
			# The holder deletes the replay a FAILED attempt leaves behind, so the only
			# .replay in the Demos folder is the one that passed. Without this export it
			# cannot find the folder and silently keeps every failure's replay.
			export REPLAY_DIR
			log "QUALIFIER armed: score $QUAL_FOR before Nexto scores $QUAL_AGAINST. A failed"
			log "QUALIFIER: attempt ENDS its match, its replay is DELETED, and a brand new"
			log "QUALIFIER: attempt starts at 0-0. On success the session ends ITSELF with"
			log "QUALIFIER: that replay kept. Leave it running. ('noqual' to disable.)" ;;
	esac
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
if port_held; then
	log "############################################################"
	log "#  RLBot port $RLBOT_PORT IS ALREADY HELD"
	log "#  $PORT_HOLDER"
	log "#"
	log "#  RLBotServer would bind-fail and exit, and because the game connects"
	log "#  first you would see a normal launch followed by a bare teardown."
	log "#  Refusing to start so the cause is visible here instead of buried in"
	log "#  core_play.log."
	log "#"
	case "$PORT_HOLDER" in
		*GigaLearnBot*)
			log "#  That is the VIZ RENDER SERVICE (build-viz is built with"
			log "#  GGL_VIZ_RLBOT=ON, so it binds this port while it runs):"
			log "#    systemctl --user stop pulsar-viz-render.service" ;;
		*bot.py*|*python*)
			log "#  That is a leftover bot.py from a previous session (check BOTH"
			log "#  checkouts - play.sh scans the sibling one too). Kill it by PID." ;;
		*)
			log "#  Identify and stop it:  ss -lptn 'sport = :$RLBOT_PORT'" ;;
	esac
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
	# Qualifier verdicts: the holder ends itself on success (replay already flushed -
	# it StopMatches BEFORE exiting), so a dead holder here is a result, not a crash,
	# whenever the QUALIFIED stamp exists.
	if [ -n "${HOLDER:-}" ] && ! kill -0 "$HOLDER" 2>/dev/null; then
		if [ -f QUALIFIED ]; then
			log "############################################################"
			log "#  QUALIFIED: $(cat QUALIFIED)"
			newest=$(ls -t "$REPLAY_DIR"/*.replay 2>/dev/null | head -1)
			if [ -n "$newest" ] && [ -n "$(find "$newest" -mmin -2 2>/dev/null)" ]; then
				log "#  replay: $newest"
			else
				log "#  WARNING: no fresh .replay in $REPLAY_DIR - check holder.log"
			fi
			log "############################################################"
			kill_all
			exit 0
		elif [ "$QUAL_ON" = 1 ]; then
			log "[qual] holder exited WITHOUT a verdict (crash?) - see holder.log. Tearing down."
			kill_all
			exit 1
		fi
	fi
	sleep 1
done
log "[note] RLBotServer exited. Tearing down. (See core_play.log / holder.log / procs.log.)"
kill "$PROCLOG" 2>/dev/null
kill_all
