#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  build.sh  — configure, build, and optionally run tests / benchmarks
#
#  Usage:
#    ./scripts/build.sh              # Release build
#    ./scripts/build.sh debug        # Debug build (ASan + UBSan)
#    ./scripts/build.sh test         # Build + run all tests
#    ./scripts/build.sh bench        # Build + run benchmark
#    ./scripts/build.sh clean        # Remove build directory
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
MODE="${1:-release}"

red()   { echo -e "\033[31m$*\033[0m"; }
green() { echo -e "\033[32m$*\033[0m"; }
blue()  { echo -e "\033[34m$*\033[0m"; }

# ── clean ─────────────────────────────────────────────────────────────────────
if [ "$MODE" = "clean" ]; then
    rm -rf "$BUILD_DIR"
    green "Cleaned build directory."
    exit 0
fi

# ── configure ─────────────────────────────────────────────────────────────────
CMAKE_BUILD_TYPE="Release"
[ "$MODE" = "debug" ] && CMAKE_BUILD_TYPE="Debug"

blue "Configuring ($CMAKE_BUILD_TYPE)..."
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# ── build ─────────────────────────────────────────────────────────────────────
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
blue "Building with $NPROC threads..."
cmake --build "$BUILD_DIR" --parallel "$NPROC"
green "Build complete."

# ── test ──────────────────────────────────────────────────────────────────────
if [ "$MODE" = "test" ] || [ "$MODE" = "debug" ]; then
    blue "Running tests..."
    cd "$BUILD_DIR" && ctest --output-on-failure
    green "All tests passed."
fi

# ── benchmark ─────────────────────────────────────────────────────────────────
if [ "$MODE" = "bench" ]; then
    blue "Running benchmark..."
    "$BUILD_DIR/benchmark"
fi
