#!/usr/bin/env bash
# Fetch a CUDA (cu128) libtorch and drop it where CMake auto-detects it.
#
# The RTX 5080 (and the rest of the GeForce 50-series) is Blackwell, CUDA compute
# capability sm_120. Only PyTorch/libtorch built for CUDA 12.8 (cu128), version
# 2.7.0 or newer, ships native sm_120 kernels. An older build (cu126/cu124/cu121)
# fails at runtime with "CUDA error: no kernel image is available for execution on
# the device". This script downloads the right cxx11-ABI cu128 libtorch and
# unpacks it into GigaLearnCPP/libtorch/, which GigaLearnCPP/CMakeLists.txt picks
# up automatically (and which .gitignore already excludes).
#
# Usage:
#   tools/setup_cuda_libtorch.sh            # download default version, skip if present
#   tools/setup_cuda_libtorch.sh --force    # re-download even if libtorch/ exists
#
# Override the version/URL:
#   GGL_LIBTORCH_VERSION=2.7.1 tools/setup_cuda_libtorch.sh   # uses the cxx11-abi- infixed name
#   GGL_LIBTORCH_URL=https://.../libtorch-...zip tools/setup_cuda_libtorch.sh
#
# Prerequisite (not handled here): an NVIDIA Linux driver >= 570 with the OPEN
# kernel modules (nvidia-open). The legacy proprietary module does not support
# Blackwell. The libtorch zip bundles its own CUDA runtime libs, so you do NOT
# need a system CUDA toolkit to run -- only the driver.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/GigaLearnCPP/libtorch"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if [ -d "$DEST" ] && [ "$FORCE" -ne 1 ]; then
	echo "libtorch already present at: $DEST"
	echo "Re-run with --force to replace it."
	exit 0
fi

# 2.8.0+ Linux builds are cxx11-ABI by default and use the plain filename.
# For 2.7.x set GGL_LIBTORCH_VERSION=2.7.1 -- the name carries an explicit
# 'cxx11-abi-' infix, handled below.
VERSION="${GGL_LIBTORCH_VERSION:-2.8.0}"
CUDA_TAG="cu128"

if [ -n "${GGL_LIBTORCH_URL:-}" ]; then
	URL="$GGL_LIBTORCH_URL"
else
	case "$VERSION" in
		2.7.*) NAME="libtorch-cxx11-abi-shared-with-deps-${VERSION}%2B${CUDA_TAG}.zip" ;;
		*)     NAME="libtorch-shared-with-deps-${VERSION}%2B${CUDA_TAG}.zip" ;;
	esac
	URL="https://download.pytorch.org/libtorch/${CUDA_TAG}/${NAME}"
fi

echo "Downloading cu128 libtorch ${VERSION}"
echo "  from: $URL"
echo "  into: $DEST"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
ZIP="$TMP_DIR/libtorch.zip"

if command -v curl >/dev/null 2>&1; then
	curl -fL --retry 3 -o "$ZIP" "$URL"
elif command -v wget >/dev/null 2>&1; then
	wget -O "$ZIP" "$URL"
else
	echo "error: need curl or wget to download libtorch" >&2
	exit 1
fi

echo "Unpacking..."
rm -rf "$DEST"
# The zip extracts a top-level 'libtorch/' directory; land it at $DEST.
unzip -q "$ZIP" -d "$TMP_DIR"
mv "$TMP_DIR/libtorch" "$DEST"

echo
echo "Done. libtorch (cu128, $VERSION) installed at:"
echo "  $DEST"
echo
echo "Next (clean rebuild -- the old build/ is wired to the previous libtorch and MUST be discarded):"
echo "  1) rm -rf build"
echo "  2) cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGL_ROCM=OFF"
echo "       (-DGGL_ROCM=OFF skips ROCm stubbing in case /opt/rocm leftovers remain from the AMD card)"
echo "  3) cmake --build build -j"
echo "  4) Verify at startup: the trainer logs 'Using CUDA GPU device...' and"
echo "     'TF32 matmuls (CUDA tensor cores): enabled'."
