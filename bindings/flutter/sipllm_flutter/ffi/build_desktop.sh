#!/usr/bin/env bash
# build_desktop.sh — build libsipllm_ffi as a native shared library for the host
# (macOS/Linux) so `dart test` and the Flutter desktop embedders can dlopen it.
#
#   ./build_desktop.sh [out_dir]         # default: <plugin>/build/desktop
#   CXX=g++ ARCHFLAGS="-march=native" ./build_desktop.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE_ROOT="$(cd "$HERE/../../../.." && pwd)"
OUT="${1:-$(cd "$HERE/.." && pwd)/build/desktop}"
mkdir -p "$OUT"

if [ ! -f "$ENGINE_ROOT/include/llm/runtime.h" ]; then
  echo "error: SipLLM engine not found at $ENGINE_ROOT" >&2
  exit 1
fi

CXX="${CXX:-clang++}"
ARCHFLAGS="${ARCHFLAGS:-}"
case "$(uname -s)" in
  Darwin) EXT=dylib; LINKFLAGS=(-install_name "@rpath/libsipllm_ffi.dylib") ;;
  *)      EXT=so;    LINKFLAGS=(-Wl,-soname,libsipllm_ffi.so) ;;
esac

# shellcheck disable=SC2206
SRCS=( "$ENGINE_ROOT"/src/*.cpp "$HERE/sipllm_ffi.cpp" )

echo ">> building libsipllm_ffi.$EXT from ${#SRCS[@]} sources"
"$CXX" -std=c++17 -O3 -funroll-loops -fno-math-errno -fPIC -shared \
  -fvisibility=hidden -ftrivial-auto-var-init=zero -Wno-unused-parameter \
  $ARCHFLAGS \
  -I"$ENGINE_ROOT/include" -I"$ENGINE_ROOT" -I"$HERE" \
  "${LINKFLAGS[@]}" "${SRCS[@]}" -lpthread \
  -o "$OUT/libsipllm_ffi.$EXT"

echo ">> built $OUT/libsipllm_ffi.$EXT"
