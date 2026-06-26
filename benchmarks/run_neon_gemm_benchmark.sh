#!/usr/bin/env bash
#=================================================================================================
# Build and run the Blaze dense matrix-multiply (GEMM) benchmark on arm64, with and without the
# NEON SIMD backend, and print a side-by-side speedup comparison.
#
# Mirrors run_neon_benchmark.sh but for GEMM (C = A * B over square N x N matrices). Two builds
# from the same source:
#   * gemm_neon   : NEON vectorization enabled (default)
#   * gemm_scalar : -DBLAZE_USE_VECTORIZATION=0 (scalar fallback paths)
# Both disable shared-memory parallelization and external BLAS so the only difference is SIMD and
# Blaze's own kernels are measured.
#
# On an aarch64 host the binaries run natively; otherwise they are cross-compiled with
# aarch64-linux-gnu-g++ and run under qemu-aarch64.
#
# NOTE: GEMM is O(N^3). Under QEMU the 1024/2048 cases can take a long time; prefer a real arm64
# host for meaningful numbers, or restrict the sweep, e.g.  GEMM_SIZES="128,256,512".
#
# Useful overrides (environment variables):
#   CXX            compiler            (default: g++ native / aarch64-linux-gnu-g++ cross)
#   MARCH          -march value        (default: armv8.2-a)
#   GEMM_SIZES     comma list of N     (default: 128,256,512,1024,2048)
#   BENCH_MIN_SEC  per-size min time   (default: 0.10)
#   QEMU           qemu binary         (default: qemu-aarch64)
#   QEMU_LD_PREFIX sysroot for qemu    (default: /usr/aarch64-linux-gnu)
#=================================================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/benchmarks/build"
SRC="${ROOT}/benchmarks/neon_gemm_benchmark.cpp"
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
COMMON="-std=c++14 -O3 -march=${MARCH} -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 -DBLAZE_BLAS_MODE=0 -DNDEBUG -I${ROOT}"

echo "==> Compiler: ${CXX}"
echo "==> Flags:    ${COMMON}"
echo "==> Sizes:    ${GEMM_SIZES:-128,256,512,1024,2048}"

echo "==> Building NEON (vectorized) GEMM benchmark"
${CXX} ${COMMON} -o "${BUILD_DIR}/gemm_neon" "${SRC}"

echo "==> Building scalar (no-vectorization) GEMM benchmark"
${CXX} ${COMMON} -DBLAZE_USE_VECTORIZATION=0 -o "${BUILD_DIR}/gemm_scalar" "${SRC}"

echo "==> Running NEON GEMM benchmark"
${RUN} "${BUILD_DIR}/gemm_neon" > "${BUILD_DIR}/gemm_results_neon.csv"

echo "==> Running scalar GEMM benchmark"
${RUN} "${BUILD_DIR}/gemm_scalar" > "${BUILD_DIR}/gemm_results_scalar.csv"

echo ""
echo "==> GEMM comparison  (speedup = scalar_time / neon_time; >1 means NEON is faster)"
echo ""
awk -F, '
   FNR == 1 { next }                                  # skip CSV header in each file
   NR == FNR { key=$2","$4; neon[key]=$5; gneon[key]=$6; order[++n]=key; next }
   { key=$2","$4; scal[key]=$5; gscal[key]=$6 }
   END {
      printf "%-7s %-7s %14s %14s %12s %12s %10s\n", \
             "type","N","neon_ms","scalar_ms","neon_GFLOPs","scal_GFLOPs","speedup"
      printf "%s\n", "------------------------------------------------------------------------------------"
      for( i=1; i<=n; i++ ) {
         k=order[i]; split( k, p, "," )
         sp = ( neon[k] > 0 ) ? scal[k]/neon[k] : 0
         printf "%-7s %-7s %14.3f %14.3f %12.3f %12.3f %9.2fx\n", \
                p[1], p[2], neon[k], scal[k], gneon[k], gscal[k], sp
      }
   }' "${BUILD_DIR}/gemm_results_neon.csv" "${BUILD_DIR}/gemm_results_scalar.csv" \
   | tee "${BUILD_DIR}/gemm_comparison.txt"

echo ""
echo "==> Raw CSVs : ${BUILD_DIR}/gemm_results_neon.csv , ${BUILD_DIR}/gemm_results_scalar.csv"
echo "==> Table    : ${BUILD_DIR}/gemm_comparison.txt"
