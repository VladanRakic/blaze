#!/usr/bin/env bash
#=================================================================================================
# Build and run the Blaze basic-math benchmarks on arm64, with and without the NEON SIMD backend,
# and print a side-by-side speedup comparison.
#
# The benchmark is compiled three times from the same source:
#   * bench_neon    : Blaze NEON SIMD kernels (default)
#   * bench_autovec : -DBLAZE_USE_VECTORIZATION=0 (Blaze scalar paths; compiler may auto-vectorize)
#   * bench_scalar  : -DBLAZE_USE_VECTORIZATION=0 plus -fno-tree-vectorize/-fno-slp (true scalar)
# On aarch64 NEON is part of the baseline ISA, so -O3 auto-vectorizes plain loops; the third build
# disables that to provide an honest scalar reference. Shared-memory parallelization is off in all.
#
# On an aarch64 host the binaries run natively. On other hosts they are cross-compiled with
# aarch64-linux-gnu-g++ and run under qemu-aarch64.
#
# Useful overrides (environment variables):
#   CXX            compiler            (default: g++ native / aarch64-linux-gnu-g++ cross)
#   MARCH          -march value        (default: armv8.2-a)
#   BENCH_MIN_SEC  per-kernel min time (default: 0.05; raise for steadier numbers, e.g. 0.2)
#   QEMU           qemu binary         (default: qemu-aarch64)
#   QEMU_LD_PREFIX sysroot for qemu    (default: /usr/aarch64-linux-gnu)
#=================================================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/benchmarks/build"
SRC="${ROOT}/benchmarks/neon_benchmark.cpp"
mkdir -p "${BUILD_DIR}"

ARCH="$(uname -m)"
if [ "${ARCH}" = "aarch64" ] || [ "${ARCH}" = "arm64" ]; then
   CXX="${CXX:-g++}"
   RUN=""
   echo "==> Native aarch64 host detected (${ARCH})"
else
   CXX="${CXX:-aarch64-linux-gnu-g++}"
   export QEMU_LD_PREFIX="${QEMU_LD_PREFIX:-/usr/aarch64-linux-gnu}"
   RUN="${QEMU:-qemu-aarch64}"
   echo "==> Cross host (${ARCH}); running aarch64 binaries via ${RUN}"
fi

MARCH="${MARCH:-armv8.2-a}"
COMMON="-std=c++14 -O3 -march=${MARCH} -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 -DNDEBUG -I${ROOT}"

# Flags that force the compiler to leave Blaze's scalar fallback loops genuinely scalar. On aarch64
# NEON is part of the baseline ISA, so -O3 auto-vectorizes plain loops unless explicitly told not to.
if ${CXX} --version 2>/dev/null | grep -qi clang; then
   NOVEC="-fno-vectorize -fno-slp-vectorize"
else
   NOVEC="-fno-tree-vectorize -fno-tree-slp-vectorize"
fi

echo "==> Compiler: ${CXX}"
echo "==> Flags:    ${COMMON}"
echo "==> No-vec:   ${NOVEC}"

# 1) Blaze explicit NEON kernels.
echo "==> Building [neon]    Blaze NEON SIMD kernels"
${CXX} ${COMMON} -o "${BUILD_DIR}/bench_neon" "${SRC}"

# 2) Blaze scalar paths, but the compiler is still free to auto-vectorize them.
echo "==> Building [autovec] Blaze scalar paths + compiler auto-vectorization"
${CXX} ${COMMON} -DBLAZE_USE_VECTORIZATION=0 -o "${BUILD_DIR}/bench_autovec" "${SRC}"

# 3) Blaze scalar paths with compiler auto-vectorization disabled -> a true scalar baseline.
echo "==> Building [scalar]  Blaze scalar paths + auto-vectorization disabled"
${CXX} ${COMMON} -DBLAZE_USE_VECTORIZATION=0 ${NOVEC} -o "${BUILD_DIR}/bench_scalar" "${SRC}"

echo "==> Running [neon]"
${RUN} "${BUILD_DIR}/bench_neon"    > "${BUILD_DIR}/results_neon.csv"

echo "==> Running [autovec]"
${RUN} "${BUILD_DIR}/bench_autovec" > "${BUILD_DIR}/results_autovec.csv"

echo "==> Running [scalar]"
${RUN} "${BUILD_DIR}/bench_scalar"  > "${BUILD_DIR}/results_scalar.csv"

echo ""
echo "==> Comparison across all three builds (times in ns/iter; lower is faster)"
echo "      neon_x  = scalar_ns / neon_ns     (Blaze NEON vs true scalar)"
echo "      auto_x  = scalar_ns / autovec_ns  (compiler auto-vec vs true scalar)"
echo "      blaze_x = autovec_ns / neon_ns    (Blaze NEON vs compiler auto-vec)"
echo ""
awk -F, '
   FNR == 1 { fi++; next }                            # new file -> bump file index, skip header
   fi == 1 { key=$2","$3","$4; neon[key]=$5; order[++n]=key; next }
   fi == 2 { key=$2","$3","$4; auto[key]=$5; next }
   fi == 3 { key=$2","$3","$4; scal[key]=$5; next }
   END {
      printf "%-7s %-5s %-7s %12s %12s %12s %9s %9s %9s\n", \
             "type","op","N","neon_ns","autovec_ns","scalar_ns","neon_x","auto_x","blaze_x"
      printf "%s\n", "---------------------------------------------------------------------------------------------"
      for( i=1; i<=n; i++ ) {
         k=order[i]; split( k, p, "," )
         nx = ( neon[k] > 0 ) ? scal[k]/neon[k] : 0
         ax = ( auto[k] > 0 ) ? scal[k]/auto[k] : 0
         bx = ( neon[k] > 0 ) ? auto[k]/neon[k] : 0
         printf "%-7s %-5s %-7s %12.1f %12.1f %12.1f %8.2fx %8.2fx %8.2fx\n", \
                p[1], p[2], p[3], neon[k], auto[k], scal[k], nx, ax, bx
      }
   }' "${BUILD_DIR}/results_neon.csv" "${BUILD_DIR}/results_autovec.csv" "${BUILD_DIR}/results_scalar.csv" \
   | tee "${BUILD_DIR}/comparison.txt"

echo ""
echo "==> Raw CSVs : results_neon.csv , results_autovec.csv , results_scalar.csv  (in ${BUILD_DIR})"
echo "==> Table    : ${BUILD_DIR}/comparison.txt"
