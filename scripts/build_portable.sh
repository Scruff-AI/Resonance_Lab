#!/usr/bin/env bash
# Portable build for the Khra'gixx daemon.
#
# Replaces the hardcoded assumptions in build_v5.sh / compile.sh:
#   -arch=sm_89            -> detected from the GPU actually present
#   /mnt/d/Resonance_Engine -> repo-relative
#   /usr/local/cuda-12.6   -> $CUDA_HOME, or whatever nvcc is on PATH
#
# Usage:
#   scripts/build_portable.sh                 # build canonical v5
#   scripts/build_portable.sh observer        # build the observer fork (adds 5560/5561)
#   RESONANCE_ARCH=sm_80 scripts/build_portable.sh    # force an architecture
#
# The physics source is NOT modified by this script. It only changes how it is compiled.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${1:-v5}"
case "$VARIANT" in
  v5)       SRC="cuda/khra_gixx_1024_v5.cu";          OUT="build/khra_gixx_1024_v5" ;;
  observer) SRC="cuda/khra_gixx_1024_v5_observer.cu"; OUT="build/khra_gixx_1024_v5_observer" ;;
  *) echo "Unknown variant '$VARIANT' (expected: v5 | observer)" >&2; exit 2 ;;
esac

[ -f "$SRC" ] || { echo "FATAL: source not found: $SRC" >&2; exit 1; }

# --- CUDA toolkit -----------------------------------------------------------
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    NVCC="$CUDA_HOME/bin/nvcc"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
else
    for d in /usr/local/cuda /usr/local/cuda-*; do
        if [ -x "$d/bin/nvcc" ]; then
            NVCC="$d/bin/nvcc"
            export LD_LIBRARY_PATH="$d/lib64:${LD_LIBRARY_PATH:-}"
            break
        fi
    done
fi
[ -n "${NVCC:-}" ] || { echo "FATAL: nvcc not found. Set CUDA_HOME or put nvcc on PATH." >&2; exit 1; }

# --- GPU architecture -------------------------------------------------------
# On a shared cluster the login node often has no GPU. Set RESONANCE_ARCH in that
# case (A100 = sm_80, H100 = sm_90, RTX 4090 = sm_89), or build inside the job.
if [ -n "${RESONANCE_ARCH:-}" ]; then
    ARCH="$RESONANCE_ARCH"
    ARCH_SRC="RESONANCE_ARCH"
elif command -v nvidia-smi >/dev/null 2>&1 \
     && CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)" \
     && [ -n "$CAPS" ]; then
    # Multiple GPUs: take the lowest capability present so the binary runs on all of them.
    LOWEST="$(echo "$CAPS" | tr -d ' ' | sort -t. -k1,1n -k2,2n | head -1)"
    ARCH="sm_$(echo "$LOWEST" | tr -d '.')"
    ARCH_SRC="nvidia-smi ($(echo "$CAPS" | tr '\n' ' ' | sed 's/ $//'))"
else
    echo "FATAL: no GPU visible and RESONANCE_ARCH not set." >&2
    echo "       Set it explicitly, e.g.  RESONANCE_ARCH=sm_80 $0 $VARIANT" >&2
    exit 1
fi

# --- Dependency check -------------------------------------------------------
MISSING=""
echo 'int main(){return 0;}' > /tmp/_rl_probe.c
"${CC:-cc}" /tmp/_rl_probe.c -lzmq       -o /tmp/_rl_probe.bin 2>/dev/null || MISSING="$MISSING libzmq(-lzmq)"
"${CC:-cc}" /tmp/_rl_probe.c -lnvidia-ml -o /tmp/_rl_probe.bin 2>/dev/null || MISSING="$MISSING NVML(-lnvidia-ml)"
rm -f /tmp/_rl_probe.c /tmp/_rl_probe.bin
if [ -n "$MISSING" ]; then
    echo "FATAL: missing link dependencies:$MISSING" >&2
    echo "       Debian/Ubuntu: apt install libzmq3-dev" >&2
    echo "       NVML ships with the driver; on a cluster try: module load cuda" >&2
    exit 1
fi

mkdir -p build logs

echo "=== Khra'gixx portable build ==="
echo "  repo    : $REPO_ROOT"
echo "  source  : $SRC"
echo "  nvcc    : $NVCC ($("$NVCC" --version | tail -1))"
echo "  arch    : $ARCH   [from $ARCH_SRC]"
echo "  output  : $OUT"
echo

if [ -f "$OUT" ]; then
    BACKUP="$OUT.backup.$(date +%Y%m%d_%H%M%S)"
    mv "$OUT" "$BACKUP"
    echo "Previous binary moved to $BACKUP"
fi

LOG="logs/build_${VARIANT}_$(date +%Y%m%d_%H%M%S).log"
"$NVCC" -O3 -g -lineinfo -arch="$ARCH" \
    -Xcompiler -rdynamic \
    -o "$OUT" "$SRC" \
    -lzmq -lnvidia-ml -l:libstdc++.so.6 2>&1 | tee "$LOG"

if [ ! -f "$OUT" ]; then
    echo "BUILD FAILED — see $LOG" >&2
    exit 1
fi

echo
echo "BUILD OK: $OUT"
ls -la "$OUT"
sha256sum "$OUT"
echo "Build log: $LOG"
