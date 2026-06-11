#!/usr/bin/env bash
# Restart wrapper for GigaLearnBot: relaunches it after a crash, stops on a
# deliberate exit. Run it from the build directory, exactly where you would
# run the binary itself:
#
#   ../tools/run_trainer.sh                 # runs ./GigaLearnBot
#   ../tools/run_trainer.sh ./OtherBin ...  # or any custom command
#
# Stops WITHOUT restarting on:
#   - exit 0            (pressing Q to save+quit)
#   - exit 130 / 143    (Ctrl+C / SIGTERM)
#   - MAX_FAST_CRASHES consecutive crashes within FAST_CRASH_SECS of launch
#     (a crash loop means a real bug or bad checkpoint, not a flake)
# Anything else (SIGSEGV=139, SIGABRT=134, uncaught exception) restarts after
# RESTART_DELAY_SECS. Checkpoints auto-resume (newest dir in checkpointFolder,
# saved every tsPerSave=1M steps) and the wandb run ID is stored in the
# checkpoint json, so a restart resumes the same run with at most ~1M steps lost.
#
# To stop training AND the wrapper: press Q (clean save+quit) or Ctrl+C in the
# terminal. Signaling only the wrapper PID (`kill <wrapper>`) does not reach the
# trainer: the wrapper waits for the trainer's natural exit, and bash may even
# swallow a PID-targeted SIGINT entirely if the trainer then dies of something
# else. Kill the process group instead (`kill -- -<wrapper pid>`).
#
# Crash history is appended to restart.log in the current directory.

set -u

CMD=("./GigaLearnBot")
if [ "$#" -gt 0 ]; then CMD=("$@"); fi

LOG_FILE="restart.log"
RESTART_DELAY_SECS=5
FAST_CRASH_SECS=120
MAX_FAST_CRASHES=5

log() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
	echo "[run_trainer] $*" >&2
}

# Let segfaults leave a core to debug (systemd-coredump on fedora captures it
# regardless; this helps on machines without it).
ulimit -c unlimited 2>/dev/null || true

interrupted=0
trap 'interrupted=1' INT TERM

fast_crashes=0
attempt=0

while true; do
	attempt=$((attempt + 1))
	start_ts=$(date +%s)
	log "launch #$attempt: ${CMD[*]}"

	"${CMD[@]}"
	code=$?

	runtime=$(( $(date +%s) - start_ts ))

	# The Q-key thread reads the terminal; a crash mid-read can leave it raw.
	[ -t 0 ] && stty sane 2>/dev/null

	if [ "$interrupted" -eq 1 ] || [ "$code" -eq 130 ] || [ "$code" -eq 143 ]; then
		log "stopped by user (exit $code) after ${runtime}s, not restarting"
		exit "$code"
	fi
	if [ "$code" -eq 0 ]; then
		log "clean exit after ${runtime}s, not restarting"
		exit 0
	fi

	sig=""
	if [ "$code" -gt 128 ]; then
		sig=" (SIG$(kill -l $((code - 128)) 2>/dev/null || echo "?"))"
	fi
	log "CRASH: exit $code$sig after ${runtime}s"
	if [ "$code" -eq 139 ] && command -v coredumpctl >/dev/null 2>&1; then
		log "backtrace: coredumpctl info ${CMD[0]##*/}"
	fi

	if [ "$runtime" -lt "$FAST_CRASH_SECS" ]; then
		fast_crashes=$((fast_crashes + 1))
		if [ "$fast_crashes" -ge "$MAX_FAST_CRASHES" ]; then
			log "$fast_crashes consecutive crashes within ${FAST_CRASH_SECS}s each, giving up"
			exit "$code"
		fi
	else
		fast_crashes=0
	fi

	log "restarting in ${RESTART_DELAY_SECS}s (Ctrl+C to stop)"
	sleep "$RESTART_DELAY_SECS" || exit 130
	if [ "$interrupted" -eq 1 ]; then
		log "stopped by user during restart delay, not restarting"
		exit 130
	fi
done
