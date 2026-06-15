#!/usr/bin/env bash
# Detached restart wrapper for GigaLearnBot.
#
# Run from the build directory:
#
#   ../tools/run_trainer.sh                 # detached start, logs to ../run_logs
#   ../tools/run_trainer.sh --follow        # tail the latest detached run log
#   ../tools/run_trainer.sh --status        # show wrapper/log status
#   ../tools/run_trainer.sh --stop          # stop wrapper + trainer cleanly by signal
#   ../tools/run_trainer.sh --foreground    # old terminal-owned restart loop
#   ../tools/run_trainer.sh -- ./OtherBin   # detached custom command
#
# Detached mode is intentionally safe to launch from tmux/SSH: the wrapper is
# started as a transient systemd --user service when available, stdin is
# /dev/null, and stdout/stderr go to run_logs. tmux should only tail the log; if
# tmux or SSH dies, the training process keeps running and logs stay on disk.
#
# Stops WITHOUT restarting on:
#   - exit 0            (clean trainer exit)
#   - ../tools/run_trainer.sh --stop
#   - foreground Ctrl+C / SIGTERM to the wrapper
#   - MAX_FAST_CRASHES consecutive crashes within FAST_CRASH_SECS of launch
#     (a crash loop means a real bug or bad checkpoint, not a flake)
# Anything else (SIGSEGV=139, SIGABRT=134, uncaught exception) restarts after
# RESTART_DELAY_SECS. Checkpoints auto-resume and the wandb run ID is stored in
# checkpoint json, so a restart resumes the same run with at most one save
# interval lost.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/run_trainer.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_LOG_DIR="$REPO_ROOT/run_logs"
LOG_DIR="${RUN_TRAINER_LOG_DIR:-$DEFAULT_LOG_DIR}"
PID_FILE=""
UNIT_FILE=""
STOP_FILE=""
LOG_FILE=""
MODE="detach"
BACKGROUND_CHILD=0
REDIRECTED_LOG=0
CMD=("./GigaLearnBot")

RESTART_DELAY_SECS="${RESTART_DELAY_SECS:-5}"
FAST_CRASH_SECS="${FAST_CRASH_SECS:-120}"
MAX_FAST_CRASHES="${MAX_FAST_CRASHES:-5}"

usage() {
	sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
	case "$1" in
		--detach)
			MODE="detach"
			shift
			;;
		--foreground)
			MODE="foreground"
			shift
			;;
		--status)
			MODE="status"
			shift
			;;
		--stop)
			MODE="stop"
			shift
			;;
		--follow)
			MODE="follow"
			shift
			;;
		--log-dir)
			[ "$#" -ge 2 ] || { echo "--log-dir requires a path" >&2; exit 2; }
			LOG_DIR="$2"
			shift 2
			;;
		--log-file)
			[ "$#" -ge 2 ] || { echo "--log-file requires a path" >&2; exit 2; }
			LOG_FILE="$2"
			shift 2
			;;
		--pid-file)
			[ "$#" -ge 2 ] || { echo "--pid-file requires a path" >&2; exit 2; }
			PID_FILE="$2"
			shift 2
			;;
		--background-child)
			BACKGROUND_CHILD=1
			shift
			;;
		--redirected-log)
			REDIRECTED_LOG=1
			shift
			;;
		--help|-h)
			usage
			exit 0
			;;
		--)
			shift
			CMD=("$@")
			break
			;;
		-*)
			echo "unknown option: $1" >&2
			usage >&2
			exit 2
			;;
		*)
			CMD=("$@")
			break
			;;
	esac
done

mkdir -p "$LOG_DIR"
PID_FILE="${PID_FILE:-$LOG_DIR/trainer.pid}"
UNIT_FILE="${UNIT_FILE:-$LOG_DIR/trainer.unit}"
STOP_FILE="${STOP_FILE:-$LOG_DIR/trainer.stop}"

is_running() {
	local pid="$1"
	[ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

latest_log_path() {
	if [ -L "$LOG_DIR/latest.log" ] || [ -f "$LOG_DIR/latest.log" ]; then
		printf '%s/latest.log\n' "$LOG_DIR"
	else
		ls -t "$LOG_DIR"/train-*.log 2>/dev/null | head -n 1
	fi
}

read_unit() {
	[ -f "$UNIT_FILE" ] && cat "$UNIT_FILE" 2>/dev/null || true
}

systemd_user_available() {
	[ "${RUN_TRAINER_NO_SYSTEMD:-0}" != "1" ] &&
		command -v systemd-run >/dev/null 2>&1 &&
		command -v systemctl >/dev/null 2>&1 &&
		systemctl --user show-environment >/dev/null 2>&1
}

systemd_unit_active() {
	local unit="$1"
	[ -n "$unit" ] && command -v systemctl >/dev/null 2>&1 &&
		systemctl --user is-active --quiet "$unit" 2>/dev/null
}

case "$MODE" in
	status)
		unit="$(read_unit)"
		if systemd_unit_active "$unit"; then
			echo "running: systemd user unit $unit"
			systemctl --user status "$unit" --no-pager -l 2>/dev/null | sed -n '1,10p' || true
			log_path="$(latest_log_path)"
			[ -n "$log_path" ] && echo "log: $log_path"
			exit 0
		elif [ -n "$unit" ]; then
			rm -f "$UNIT_FILE" 2>/dev/null || true
		fi

		if [ -f "$PID_FILE" ]; then
			pid="$(cat "$PID_FILE" 2>/dev/null || true)"
			if is_running "$pid"; then
				echo "running: pid $pid"
				ps -p "$pid" -o pid=,ppid=,stat=,etime=,command= 2>/dev/null || true
			else
				echo "not running: stale pid $pid"
				rm -f "$PID_FILE" 2>/dev/null || true
			fi
		else
			echo "not running: no pid file at $PID_FILE"
		fi
		log_path="$(latest_log_path)"
		[ -n "$log_path" ] && echo "log: $log_path"
		exit 0
		;;
	stop)
		mkdir -p "$LOG_DIR"
		: > "$STOP_FILE"
		unit="$(read_unit)"
		if systemd_unit_active "$unit"; then
			echo "stopping systemd user unit $unit"
			systemctl --user stop "$unit"
			rm -f "$UNIT_FILE" 2>/dev/null || true
			exit 0
		elif [ -n "$unit" ]; then
			rm -f "$UNIT_FILE" 2>/dev/null || true
		fi

		if [ ! -f "$PID_FILE" ]; then
			echo "no pid file at $PID_FILE"
			exit 1
		fi
		pid="$(cat "$PID_FILE" 2>/dev/null || true)"
		if ! is_running "$pid"; then
			echo "not running: stale pid $pid"
			rm -f "$PID_FILE" 2>/dev/null || true
			exit 0
		fi
		echo "stopping wrapper pid $pid"
		kill -TERM "$pid"
		for _ in 1 2 3 4 5 6 7 8 9 10; do
			if ! is_running "$pid"; then
				echo "stopped"
				exit 0
			fi
			sleep 1
		done
		echo "still running after SIGTERM; inspect with: ps -p $pid -o pid,ppid,stat,etime,command"
		exit 1
		;;
	follow)
		log_path="$(latest_log_path)"
		if [ -z "$log_path" ]; then
			echo "no log found in $LOG_DIR" >&2
			exit 1
		fi
		echo "tailing $log_path"
		exec tail -f "$log_path"
		;;
esac

if [ "$MODE" = "detach" ]; then
	unit="$(read_unit)"
	if systemd_unit_active "$unit"; then
		echo "trainer wrapper is already running: systemd user unit $unit"
		echo "log: $(latest_log_path)"
		exit 0
	elif [ -n "$unit" ]; then
		rm -f "$UNIT_FILE" 2>/dev/null || true
	fi

	if [ -f "$PID_FILE" ]; then
		old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
		if is_running "$old_pid"; then
			echo "trainer wrapper is already running: pid $old_pid"
			echo "log: $(latest_log_path)"
			exit 0
		fi
		rm -f "$PID_FILE" 2>/dev/null || true
	fi

	run_id="$(date '+%Y%m%d-%H%M%S')"
	run_log="$LOG_DIR/train-$run_id.log"
	rm -f "$STOP_FILE" 2>/dev/null || true
	ln -sfn "$(basename "$run_log")" "$LOG_DIR/latest.log" 2>/dev/null || true

	if systemd_user_available; then
		unit="gigalearn-trainer-$run_id.service"
		if systemd-run --user \
			--unit="$unit" \
			--collect \
			--property=WorkingDirectory="$(pwd)" \
			--property=StandardInput=null \
			--property=StandardOutput=append:"$run_log" \
			--property=StandardError=append:"$run_log" \
			--property=KillMode=mixed \
			--property=OOMPolicy=continue \
			"$SCRIPT_PATH" \
			--foreground \
			--background-child \
			--redirected-log \
			--log-file "$run_log" \
			--pid-file "$PID_FILE" \
			-- "${CMD[@]}"; then
			echo "$unit" > "$UNIT_FILE"
			echo "started trainer wrapper: systemd user unit $unit"
			echo "log: $run_log"
			echo "follow: $0 --follow"
			echo "stop:   $0 --stop"
			exit 0
		fi
		echo "systemd-run failed; falling back to setsid + nohup" >&2
	fi

	setsid nohup "$SCRIPT_PATH" \
		--foreground \
		--background-child \
		--redirected-log \
		--log-file "$run_log" \
		--pid-file "$PID_FILE" \
		-- "${CMD[@]}" \
		> "$run_log" 2>&1 < /dev/null &
	wrapper_pid=$!
	echo "$wrapper_pid" > "$PID_FILE"

	echo "started trainer wrapper: pid $wrapper_pid (setsid/nohup fallback)"
	echo "log: $run_log"
	echo "follow: $0 --follow"
	echo "stop:   $0 --stop"
	exit 0
fi

LOG_FILE="${LOG_FILE:-restart.log}"

log() {
	local line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
	if [ "$REDIRECTED_LOG" -eq 1 ]; then
		echo "$line" >&2
	else
		echo "$line" >> "$LOG_FILE"
		echo "[run_trainer] $*" >&2
	fi
}

# Let segfaults leave a core to debug where the platform supports it.
ulimit -c unlimited 2>/dev/null || true

echo "$$" > "$PID_FILE" 2>/dev/null || true

interrupted=0
child_pid=0
handle_signal() {
	interrupted=1
	log "wrapper received stop signal; forwarding SIGTERM to child"
	if [ "$BACKGROUND_CHILD" -eq 1 ] && [ "$child_pid" -gt 0 ] && kill -0 "$child_pid" 2>/dev/null; then
		kill -TERM "$child_pid" 2>/dev/null || true
	fi
}
trap 'handle_signal' INT TERM
trap 'rm -f "$PID_FILE" 2>/dev/null || true' EXIT

fast_crashes=0
attempt=0

while true; do
	attempt=$((attempt + 1))
	start_ts=$(date +%s)
	log "launch #$attempt: ${CMD[*]}"

	if [ "$BACKGROUND_CHILD" -eq 1 ]; then
		"${CMD[@]}" &
		child_pid=$!
		wait "$child_pid"
		code=$?
		child_pid=0
	else
		"${CMD[@]}"
		code=$?
	fi

	runtime=$(( $(date +%s) - start_ts ))

	# The Q-key thread reads the terminal; a crash mid-read can leave it raw.
	[ -t 0 ] && stty sane 2>/dev/null

	stop_requested=0
	if [ -f "$STOP_FILE" ]; then
		stop_requested=1
		rm -f "$STOP_FILE" 2>/dev/null || true
	fi

	if [ "$interrupted" -eq 1 ] || [ "$stop_requested" -eq 1 ]; then
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

	log "restarting in ${RESTART_DELAY_SECS}s"
	sleep "$RESTART_DELAY_SECS" || exit 130
	if [ "$interrupted" -eq 1 ]; then
		log "stopped by user during restart delay, not restarting"
		exit 130
	fi
done
