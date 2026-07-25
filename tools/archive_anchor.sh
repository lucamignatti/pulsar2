#!/usr/bin/env bash
# Permanent spaced checkpoint archival ("anchors").
#
# WHY (2026-07-19): checkpoints rotate (checkpointsToKeep) and even the golden
# best_r* archive rotates by rating - so the run's own history is deleted as it
# goes. When we tried to locate when 1v1 progress slowed, every checkpoint
# between 18.88B and 27.5B was gone: only two branch backups survived. Anchors
# fix that permanently, and double as the fixed opponents for the honest
# match-play Elo battery (research/tools/anchor_battery.py) - the pool Rating
# overstates real progress ~6x, so a FIXED yardstick is the only trustworthy one.
#
# Never deletes anything. Copies are atomic (<ts>.tmp then rename). Safe to run
# repeatedly (idempotent): it only archives when the newest complete checkpoint
# is >= ANCHOR_SPACING steps beyond the newest existing anchor.
#
# Usage:
#   tools/archive_anchor.sh              # archive newest if spacing allows
#   tools/archive_anchor.sh --seed DIR   # force-add DIR as an anchor (seeding)
#   tools/archive_anchor.sh --list       # list anchors
# Env:
#   TRAINERCTL_CKPT_DIR  checkpoint dir (default build/checkpoints_5.0v3)
#   ANCHOR_SPACING       min steps between anchors (default 2000000000 = 2B)
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${TRAINERCTL_CKPT_DIR:-$REPO/build/checkpoints_5.0v3}"
ANCHOR_DIR="${CKPT_DIR}_anchors"
SPACING="${ANCHOR_SPACING:-2000000000}"

# Files load_checkpoint.load_models() needs, plus the stats that carry the rating.
NEEDED=(SHARED_HEAD.lt POLICY.lt CRITIC.lt REACH_PHI.lt REACH_PSI_BALL.lt REACH_PSI_CAR.lt RUNNING_STATS.json)

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

is_complete() {
    local d="$1"
    for f in "${NEEDED[@]}"; do
        [ -f "$d/$f" ] || return 1
    done
    return 0
}

newest_anchor_ts() {
    ls -1 "$ANCHOR_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1
}

# Atomic copy of a checkpoint dir into the anchor store under name <ts>.
archive_dir() {
    local src="$1" ts="$2"
    mkdir -p "$ANCHOR_DIR"
    if [ -d "$ANCHOR_DIR/$ts" ]; then
        log "anchor $ts already present - nothing to do"
        return 0
    fi
    rm -rf "$ANCHOR_DIR/$ts.tmp"
    if ! cp -a "$src" "$ANCHOR_DIR/$ts.tmp" 2>/dev/null; then
        log "WARN: copy of $src failed (rotated mid-copy?)"
        rm -rf "$ANCHOR_DIR/$ts.tmp"
        return 1
    fi
    if ! is_complete "$ANCHOR_DIR/$ts.tmp"; then
        log "WARN: copy of $src incomplete - discarding"
        rm -rf "$ANCHOR_DIR/$ts.tmp"
        return 1
    fi
    mv "$ANCHOR_DIR/$ts.tmp" "$ANCHOR_DIR/$ts" || { log "ERROR: rename failed"; return 1; }
    local sz; sz="$(du -sh "$ANCHOR_DIR/$ts" | cut -f1)"
    log "ARCHIVED anchor $ts ($sz) -> $ANCHOR_DIR/$ts"
    return 0
}

case "${1:-}" in
    --list)
        log "anchors in $ANCHOR_DIR:"
        ls -1 "$ANCHOR_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | while read -r a; do
            r="$(python3 -c "import json;print(round(json.load(open('$ANCHOR_DIR/$a/RUNNING_STATS.json')).get('skill_ratings',{}).get('1v1',float('nan')),1))" 2>/dev/null || echo "?")"
            printf '  %-14s  rating1v1=%s\n' "$a" "$r"
        done
        exit 0
        ;;
    --seed)
        SRC="${2:?--seed needs a checkpoint dir}"
        [ -d "$SRC" ] || { log "ERROR: $SRC not a dir"; exit 1; }
        is_complete "$SRC" || { log "ERROR: $SRC missing required files"; exit 1; }
        archive_dir "$SRC" "$(basename "$SRC")"
        exit $?
        ;;
esac

# --- default: archive the newest complete checkpoint if spacing allows ---
NEWEST_ANCHOR="$(newest_anchor_ts)"
DONE=0
# newest -> older, so a mid-copy rotation falls back gracefully
for ts in $(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -rn | head -5); do
    src="$CKPT_DIR/$ts"
    is_complete "$src" || continue
    if [ -n "$NEWEST_ANCHOR" ]; then
        gap=$(( ts - NEWEST_ANCHOR ))
        if [ "$gap" -lt "$SPACING" ]; then
            log "newest checkpoint $ts is only $gap steps past anchor $NEWEST_ANCHOR (< $SPACING) - skip"
            exit 0
        fi
    fi
    archive_dir "$src" "$ts" && { DONE=1; break; }
done
[ "$DONE" -eq 1 ] || log "no complete checkpoint archived this run"
