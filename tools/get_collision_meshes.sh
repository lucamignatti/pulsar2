#!/usr/bin/env bash
# Provision collision_meshes/ for RocketSim (~160 KB of .cmf files, gitignored).
#
# The meshes are dumped from Rocket League's own assets, so there is no official
# public download. This script fetches them from, in order:
#   1. GGL_MESHES_URL          - a URL to a zip/tar.gz containing collision_meshes/
#   2. GGL_MESHES_SRC          - a local or scp-style path to copy from
#   3. Known sibling checkouts - other clones on this machine that already have them
#
# Usage:
#   tools/get_collision_meshes.sh              # skip if already present
#   tools/get_collision_meshes.sh --force      # re-fetch even if present
#
#   GGL_MESHES_URL=https://.../collision_meshes.zip tools/get_collision_meshes.sh
#   GGL_MESHES_SRC=~/Projects/pulsar2/collision_meshes tools/get_collision_meshes.sh
#   GGL_MESHES_SRC=user@host:~/Projects/pulsar2/collision_meshes tools/get_collision_meshes.sh
#
# If nothing works: dump them yourself with
#   https://github.com/ZealanL/RLArenaCollisionDumper (needs a Rocket League install)
# and place the output at <repo>/collision_meshes/soccar/*.cmf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/collision_meshes"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

have_meshes() {
	compgen -G "$1/soccar/*.cmf" > /dev/null 2>&1
}

if have_meshes "$DEST" && [ "$FORCE" -ne 1 ]; then
	echo "Collision meshes already present at: $DEST"
	echo "Re-run with --force to replace them."
	exit 0
fi

fetch_url() {
	local url="$1"
	local tmp
	tmp="$(mktemp -d)"
	trap 'rm -rf "$tmp"' RETURN

	echo "Downloading collision meshes from: $url"
	local archive="$tmp/meshes_archive"
	if command -v curl >/dev/null 2>&1; then
		curl -fL --retry 3 -o "$archive" "$url"
	elif command -v wget >/dev/null 2>&1; then
		wget -O "$archive" "$url"
	else
		echo "error: need curl or wget" >&2
		return 1
	fi

	mkdir -p "$tmp/unpacked"
	case "$url" in
		*.tar.gz|*.tgz) tar -xzf "$archive" -C "$tmp/unpacked" ;;
		*)              unzip -q "$archive" -d "$tmp/unpacked" ;;
	esac

	# Accept either a collision_meshes/ dir inside the archive or bare soccar/
	local found
	found="$(find "$tmp/unpacked" -type d -name soccar | head -1)"
	if [ -z "$found" ]; then
		echo "error: archive contains no soccar/ mesh directory" >&2
		return 1
	fi

	rm -rf "$DEST"
	mkdir -p "$DEST"
	cp -R "$found" "$DEST/"
	return 0
}

fetch_copy() {
	local src="$1"
	echo "Copying collision meshes from: $src"
	rm -rf "$DEST"
	mkdir -p "$DEST"
	case "$src" in
		*:*) scp -r "$src/soccar" "$DEST/" ;;
		*)   cp -R "${src/#\~/$HOME}/soccar" "$DEST/" ;;
	esac
}

if [ -n "${GGL_MESHES_URL:-}" ]; then
	fetch_url "$GGL_MESHES_URL"
elif [ -n "${GGL_MESHES_SRC:-}" ]; then
	fetch_copy "$GGL_MESHES_SRC"
else
	# Sibling checkouts that commonly have the meshes already
	FOUND=""
	for candidate in \
		"$HOME/Projects/pulsar2/collision_meshes" \
		"$HOME/Projects/GigaLearnCPP-Leak/collision_meshes" \
		"$REPO_ROOT/../pulsar2/collision_meshes" \
		"$REPO_ROOT/../GigaLearnCPP-Leak/collision_meshes"; do
		if have_meshes "$candidate"; then
			FOUND="$candidate"
			break
		fi
	done

	if [ -n "$FOUND" ]; then
		fetch_copy "$FOUND"
	else
		echo "error: no mesh source found." >&2
		echo "Set GGL_MESHES_URL or GGL_MESHES_SRC (see header comments), or dump them" >&2
		echo "with https://github.com/ZealanL/RLArenaCollisionDumper into $DEST/soccar/" >&2
		exit 1
	fi
fi

if have_meshes "$DEST"; then
	echo "Done. Meshes installed at: $DEST ($(ls "$DEST/soccar" | wc -l | tr -d ' ') files)"
else
	echo "error: fetch completed but no .cmf files landed at $DEST/soccar" >&2
	exit 1
fi
