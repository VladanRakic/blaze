#!/usr/bin/env bash
#=================================================================================================
# Build and run the Blaze basic-math benchmarks on arm64, with and without the NEON SIMD backend,
# and print a side-by-side speedup comparison.
#
# The benchmark is compiled twice from the same source:
#   * bench_neon   : NEON vectorization enabled (default)
#   * bench_scalar : -DBLAZE_USE_VECTORIZATION=0 (forces the scalar fallback paths)
# Shared-memory parallelization is disabled in both builds so the only difference is SIMD.
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

echo "==> Compiler: ${CXX}"
echo "==> Flags:    ${COMMON}"

echo "==> Building NEON (vectorized) benchmark"
${CXX} ${COMMON} -o "${BUILD_DIR}/bench_neon" "${SRC}"

echo "==> Building scalar (no-vectorization) benchmark"
${CXX} ${COMMON} -DBLAZE_USE_VECTORIZATION=0 -o "${BUILD_DIR}/bench_scalar" "${SRC}"

echo "==> Running NEON benchmark"
${RUN} "${BUILD_DIR}/bench_neon" > "${BUILD_DIR}/results_neon.csv"

echo "==> Running scalar benchmark"
${RUN} "${BUILD_DIR}/bench_scalar" > "${BUILD_DIR}/results_scalar.csv"

echo ""
echo "==> Comparison  (speedup = scalar_time / neon_time; >1 means NEON is faster)"
echo ""
awk -F, '
   FNR == 1 { next }                                  # skip CSV header in each file
   NR == FNR { key=$2","$3","$4; neon[key]=$5; gneon[key]=$6; order[++n]=key; next }
   { key=$2","$3","$4; scal[key]=$5 }
   END {
      printf "%-7s %-5s %-7s %14s %14s %12s %10s\n", \
             "type","op","N","neon_ns","scalar_ns","neon_Gel/s","speedup"
      printf "%s\n", "-----------------------------------------------------------------------------"
      for( i=1; i<=n; i++ ) {
         k=order[i]; split( k, p, "," )
         sp = ( neon[k] > 0 ) ? scal[k]/neon[k] : 0
         printf "%-7s %-5s %-7s %14.1f %14.1f %12.3f %9.2fx\n", \
                p[1], p[2], p[3], neon[k], scal[k], gneon[k], sp
      }
   }' "${BUILD_DIR}/results_neon.csv" "${BUILD_DIR}/results_scalar.csv" \
   | tee "${BUILD_DIR}/comparison.txt"

echo ""
echo "==> Raw CSVs : ${BUILD_DIR}/results_neon.csv , ${BUILD_DIR}/results_scalar.csv"
echo "==> Table    : ${BUILD_DIR}/comparison.txt"
