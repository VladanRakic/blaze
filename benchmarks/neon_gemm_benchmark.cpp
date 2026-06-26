//=================================================================================================
// Blaze AArch64 NEON micro-benchmark for dense matrix multiplication (GEMM): C = A * B.
//
// Companion to neon_benchmark.cpp, but for the O(N^3) GEMM kernel over square N x N matrices.
// The same binary self-reports whether it was built with the NEON SIMD backend (BLAZE_NEON_MODE)
// so the driver script can build it twice -- vectorized and scalar (-DBLAZE_USE_VECTORIZATION=0) --
// and compare. Blaze's own vectorized kernels are measured: build with -DBLAZE_BLAS_MODE=0 so the
// multiplication is NOT delegated to an external BLAS.
//
// Sizes are square matrix dimensions N (A,B,C are N x N). Default sweep is 128..2048.
//
// Build (cross):
//   aarch64-linux-gnu-g++ -std=c++14 -O3 -march=armv8.2-a \
//       -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 -DBLAZE_BLAS_MODE=0 -DNDEBUG -I.. \
//       -o neon_gemm_benchmark neon_gemm_benchmark.cpp
// Run under QEMU on x86:
//   QEMU_LD_PREFIX=/usr/aarch64-linux-gnu ./neon_gemm_benchmark
//
// Output: CSV on stdout (mode,type,op,N,msec_per_iter,gflops); diagnostics on stderr.
//=================================================================================================

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include <blaze/Math.h>
#include <blaze/system/Vectorization.h>

namespace {

using Clock = std::chrono::steady_clock;

volatile double g_sink = 0.0;

double g_minSec = 0.10;  // min wall-clock time per measurement; override via BENCH_MIN_SEC

template< typename T > const char* typeName();
template<> const char* typeName<double>()  { return "double"; }
template<> const char* typeName<float>()    { return "float";  }
template<> const char* typeName<int32_t>() { return "int32";  }
template<> const char* typeName<int16_t>() { return "int16";  }

const char* modeName() { return BLAZE_NEON_MODE ? "neon" : "scalar"; }

// Per-type magnitude bound so that the GEMM accumulator (sum of N products) stays in range for the
// largest N in the sweep. int8 is intentionally excluded: even unit values overflow an 8-bit
// accumulator at N >= 128, and its multiply is not NEON-vectorized anyway.
template< typename T >
double maxMagnitude()
{
   if( std::is_integral<T>::value ) {
      if( sizeof(T) == 2UL ) return 2.0;     // int16: N*max^2 <= 2048*4 < 32767
      return 100.0;                           // int32: N*max^2 <= 2048*1e4 << 2.1e9
   }
   return 4.0;                                // float / double
}

template< typename M >
void fillData( M& m, uint64_t seed, double maxMag, bool allowNeg )
{
   using ET = typename M::ElementType;
   uint64_t s = seed | 1ULL;
   for( size_t i=0UL; i<m.rows(); ++i ) {
      for( size_t j=0UL; j<m.columns(); ++j ) {
         s ^= s << 13; s ^= s >> 7; s ^= s << 17;             // xorshift64 (deterministic)
         double r = ( double( ( s >> 11 ) & 0xFFFFFF ) / double( 0x1000000 ) ) * ( maxMag - 1.0 ) + 1.0;
         if( allowNeg && ( s & 1ULL ) ) r = -r;
         m(i,j) = static_cast<ET>( r );
      }
   }
}

template< typename Fn >
double timeOp( Fn&& fn )
{
   fn();  // warmup
   size_t iters = 1UL;
   double elapsed = 0.0;
   for(;;) {
      const auto t0 = Clock::now();
      for( size_t i=0UL; i<iters; ++i ) fn();
      elapsed = std::chrono::duration<double>( Clock::now() - t0 ).count();
      if( elapsed >= g_minSec ) break;
      iters <<= 1;
   }
   return elapsed / double( iters );  // seconds per iteration
}

void emit( const char* type, size_t N, double sec )
{
   const double flops  = 2.0 * double( N ) * double( N ) * double( N );  // mul + add per inner step
   const double gflops = ( sec > 0.0 ) ? ( flops / sec / 1e9 ) : 0.0;
   std::printf( "%s,%s,gemm,%zu,%.4f,%.4f\n", modeName(), type, N, sec * 1e3, gflops );
   std::fflush( stdout );
}

template< typename T >
void runType( size_t N )
{
   using Mat = blaze::DynamicMatrix<T>;   // row-major
   const bool fp  = std::is_floating_point<T>::value;
   const double m = maxMagnitude<T>();

   Mat A( N, N ), B( N, N ), C( N, N );
   fillData( A, 0x1111ULL + N, m, fp );
   fillData( B, 0x2222ULL + N, m, fp );

   size_t p = 0UL;
   auto observe = [&]( const Mat& r ){ g_sink += double( r( p % N, ( p / N ) % N ) ); ++p; };

   emit( typeName<T>(), N, timeOp( [&]{ C = A * B; observe( C ); } ) );
}

std::vector<size_t> parseSizes()
{
   std::vector<size_t> sizes;
   if( const char* e = std::getenv( "GEMM_SIZES" ) ) {
      char* buf = strdup( e );
      for( char* tok = std::strtok( buf, ",;: " ); tok; tok = std::strtok( nullptr, ",;: " ) ) {
         const long v = std::atol( tok );
         if( v > 0 ) sizes.push_back( static_cast<size_t>( v ) );
      }
      free( buf );
   }
   if( sizes.empty() ) sizes = { 128UL, 256UL, 512UL, 1024UL, 2048UL };
   return sizes;
}

} // namespace

int main()
{
   if( const char* e = std::getenv( "BENCH_MIN_SEC" ) ) g_minSec = std::atof( e );

   std::fprintf( stderr,
                 "# Blaze NEON GEMM benchmark | mode=%s BLAZE_NEON_MODE=%d BLAZE_FMA_MODE=%d "
                 "BLAZE_BLAS_MODE=%d min_sec=%.3f\n", modeName(), int(BLAZE_NEON_MODE),
                 int(BLAZE_FMA_MODE), int(BLAZE_BLAS_MODE), g_minSec );

   std::printf( "mode,type,op,N,msec_per_iter,gflops\n" );

   const std::vector<size_t> sizes = parseSizes();
   for( size_t N : sizes ) {
      runType<double> ( N );
      runType<float>  ( N );
      runType<int32_t>( N );
      runType<int16_t>( N );
   }

   std::fprintf( stderr, "# done (checksum=%g)\n", double( g_sink ) );
   return 0;
}
