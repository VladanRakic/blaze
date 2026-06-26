//=================================================================================================
// Blaze AArch64 NEON micro-benchmark for basic dense math operations.
//
// Measures throughput of element-wise vector kernels (add, sub, mul, div, fma, axpy, sqrt) and a
// reduction (sum) for several element types and problem sizes. The same binary self-reports
// whether it was built with the NEON SIMD backend enabled (BLAZE_NEON_MODE) so the driver script
// can build it twice -- once vectorized, once scalar (-DBLAZE_USE_VECTORIZATION=0) -- and compare.
//
// "Size" (N) is the vector length, i.e. the number of elements per operand. Square matrices of
// these dimensions would be multiple GB at 16K, so 1D dense vectors are used; the element-wise
// SIMD kernels exercised here are identical to those used for dense matrices.
//
// Build (cross):
//   aarch64-linux-gnu-g++ -std=c++14 -O3 -march=armv8.2-a \
//       -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 -DNDEBUG -I.. \
//       -o neon_benchmark neon_benchmark.cpp
// Run under QEMU on x86:
//   QEMU_LD_PREFIX=/usr/aarch64-linux-gnu ./neon_benchmark
//
// Output: CSV on stdout (mode,type,op,N,nsec_per_iter,gelem_per_s); diagnostics on stderr.
//=================================================================================================

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include <blaze/Math.h>
#include <blaze/system/Vectorization.h>

namespace {

using Clock = std::chrono::steady_clock;

// Volatile sink: every kernel result feeds into this so the optimizer cannot elide the work.
volatile double g_sink = 0.0;

// Minimum wall-clock time (seconds) each kernel is timed for; overridable via BENCH_MIN_SEC.
double g_minSec = 0.05;

template< typename T > const char* typeName();
template<> const char* typeName<double>()  { return "double"; }
template<> const char* typeName<float>()    { return "float";  }
template<> const char* typeName<int32_t>() { return "int32";  }
template<> const char* typeName<int16_t>() { return "int16";  }
template<> const char* typeName<int8_t>()   { return "int8";   }

const char* modeName() { return BLAZE_NEON_MODE ? "neon" : "scalar"; }

// Per-type magnitude bound chosen so that mul / fma / axpy results stay within the type's range.
template< typename T >
double maxMagnitude()
{
   if( std::is_integral<T>::value ) {
      if( sizeof(T) == 1UL ) return 5.0;     // int8 : 5*5 + 5 = 30 < 127
      if( sizeof(T) == 2UL ) return 50.0;    // int16: 50*50 + 50 < 32767
      return 1000.0;                          // int32
   }
   return 1000.0;                             // float / double
}

template< typename V >
void fillData( V& v, uint64_t seed, double maxMag, bool allowNeg )
{
   using ET = typename V::ElementType;
   uint64_t s = seed | 1ULL;
   for( size_t i=0UL; i<v.size(); ++i ) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;                 // xorshift64 (deterministic)
      double r = ( double( ( s >> 11 ) & 0xFFFFFF ) / double( 0x1000000 ) ) * ( maxMag - 1.0 ) + 1.0;
      if( allowNeg && ( s & 1ULL ) ) r = -r;
      v[i] = static_cast<ET>( r );
   }
}

template< typename Fn >
double timeOp( Fn&& fn )
{
   fn();  // warmup (also primes caches)
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

void emit( const char* type, const char* op, size_t N, double sec )
{
   const double gelem = ( sec > 0.0 ) ? ( double( N ) / sec / 1e9 ) : 0.0;
   std::printf( "%s,%s,%s,%zu,%.3f,%.4f\n", modeName(), type, op, N, sec * 1e9, gelem );
   std::fflush( stdout );
}

// Floating-point-only kernels (division, square root) -- enabled via tag dispatch.
template< typename T, typename V >
void runFloatOnly( size_t N, const V& a, const V& b, V& d, std::true_type )
{
   const V ap( blaze::abs( a ) + T( 1 ) );  // strictly positive input for sqrt
   size_t probe = 0UL;
   auto observe = [&]( const V& r ){ g_sink += double( r[probe] ); probe = ( probe + 1UL ) % N; };

   emit( typeName<T>(), "div",  N, timeOp( [&]{ d = a / b;          observe( d ); } ) );
   emit( typeName<T>(), "sqrt", N, timeOp( [&]{ d = blaze::sqrt( ap ); observe( d ); } ) );
}

template< typename T, typename V >
void runFloatOnly( size_t, const V&, const V&, V&, std::false_type ) {}

template< typename T >
void runType( size_t N )
{
   using Vec = blaze::DynamicVector<T>;
   const double mag = maxMagnitude<T>();

   Vec a( N ), b( N ), c( N ), d( N );
   fillData( a, 0x1234ULL + N, mag, true );
   fillData( b, 0x9abcULL + N, mag, true );
   fillData( c, 0x5678ULL + N, mag, true );
   const T s = static_cast<T>( 3 );

   size_t probe = 0UL;
   auto observe = [&]( const Vec& r ){ g_sink += double( r[probe] ); probe = ( probe + 1UL ) % N; };

   emit( typeName<T>(), "add",  N, timeOp( [&]{ d = a + b;     observe( d ); } ) );
   emit( typeName<T>(), "sub",  N, timeOp( [&]{ d = a - b;     observe( d ); } ) );
   emit( typeName<T>(), "mul",  N, timeOp( [&]{ d = a * b;     observe( d ); } ) );
   emit( typeName<T>(), "fma",  N, timeOp( [&]{ d = a * b + c; observe( d ); } ) );
   emit( typeName<T>(), "axpy", N, timeOp( [&]{ d = s * a + b; observe( d ); } ) );
   emit( typeName<T>(), "sum",  N, timeOp( [&]{ g_sink += double( blaze::sum( a ) ); } ) );

   runFloatOnly<T>( N, a, b, d, std::is_floating_point<T>{} );
}

} // namespace

int main()
{
   if( const char* e = std::getenv( "BENCH_MIN_SEC" ) ) g_minSec = std::atof( e );

   std::fprintf( stderr,
                 "# Blaze NEON benchmark | mode=%s BLAZE_NEON_MODE=%d BLAZE_FMA_MODE=%d "
                 "min_sec=%.3f\n", modeName(), int(BLAZE_NEON_MODE), int(BLAZE_FMA_MODE), g_minSec );

   std::printf( "mode,type,op,N,nsec_per_iter,gelem_per_s\n" );

   const size_t sizes[] = { 1024UL, 2048UL, 4096UL, 8192UL, 16384UL };
   for( size_t N : sizes ) {
      runType<double> ( N );
      runType<float>  ( N );
      runType<int32_t>( N );
      runType<int16_t>( N );
      runType<int8_t> ( N );
   }

   std::fprintf( stderr, "# done (checksum=%g)\n", double( g_sink ) );
   return 0;
}
