//=================================================================================================
/*!
//  \file blaze/math/simd/Set.h
//  \brief Header file for the SIMD set functionality
//
//  Copyright (C) 2012-2020 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_MATH_SIMD_SET_H_
#define _BLAZE_MATH_SIMD_SET_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/BasicTypes.h>
#include <blaze/math/simd/NeonDup.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Integral.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/HasSize.h>
#include <blaze/util/typetraits/IsIntegral.h>
#include <blaze/util/typetraits/IsSigned.h>


namespace blaze {

#if BLAZE_NEON_MODE
namespace set_neon_detail {

template< typename T >
BLAZE_ALWAYS_INLINE EnableIf_t< IsSigned_v<T>, SIMDcint8 >
   setc8( complex<T> value )
{
   const int8_t d[16] = { value.real(), value.imag(), value.real(), value.imag(),
                          value.real(), value.imag(), value.real(), value.imag(),
                          value.real(), value.imag(), value.real(), value.imag(),
                          value.real(), value.imag(), value.real(), value.imag() };
   return vld1q_s8( d );
}

template< typename T >
BLAZE_ALWAYS_INLINE EnableIf_t< !IsSigned_v<T>, SIMDcuint8 >
   setc8( complex<T> value )
{
   const uint8_t d[16] = { static_cast<uint8_t>( value.real() ), static_cast<uint8_t>( value.imag() ),
                           static_cast<uint8_t>( value.real() ), static_cast<uint8_t>( value.imag() ),
                           static_cast<uint8_t>( value.real() ), static_cast<uint8_t>( value.imag() ),
                           static_cast<uint8_t>( value.real() ), static_cast<uint8_t>( value.imag() ),
                           static_cast<uint8_t>( value.real() ), static_cast<uint8_t>( value.imag() ),
                           static_cast<uint8_t>( value.real() ), static_cast<uint8_t>( value.imag() ),
                           static_cast<uint8_t>( value.real() ), static_cast<uint8_t>( value.imag() ),
                           static_cast<uint8_t>( value.real() ), static_cast<uint8_t>( value.imag() ) };
   return vld1q_u8( d );
}

template< typename T >
BLAZE_ALWAYS_INLINE EnableIf_t< IsSigned_v<T>, SIMDcint16 >
   setc16( complex<T> value )
{
   const int16_t d[8] = { value.real(), value.imag(), value.real(), value.imag(),
                          value.real(), value.imag(), value.real(), value.imag() };
   return vld1q_s16( d );
}

template< typename T >
BLAZE_ALWAYS_INLINE EnableIf_t< !IsSigned_v<T>, SIMDcuint16 >
   setc16( complex<T> value )
{
   const uint16_t d[8] = { static_cast<uint16_t>( value.real() ), static_cast<uint16_t>( value.imag() ),
                           static_cast<uint16_t>( value.real() ), static_cast<uint16_t>( value.imag() ),
                           static_cast<uint16_t>( value.real() ), static_cast<uint16_t>( value.imag() ),
                           static_cast<uint16_t>( value.real() ), static_cast<uint16_t>( value.imag() ) };
   return vld1q_u16( d );
}

template< typename T >
BLAZE_ALWAYS_INLINE EnableIf_t< IsSigned_v<T>, SIMDcint32 >
   setc32( complex<T> value )
{
   const int32_t d[4] = { value.real(), value.imag(), value.real(), value.imag() };
   return vld1q_s32( d );
}

template< typename T >
BLAZE_ALWAYS_INLINE EnableIf_t< !IsSigned_v<T>, SIMDcuint32 >
   setc32( complex<T> value )
{
   const uint32_t d[4] = { static_cast<uint32_t>( value.real() ), static_cast<uint32_t>( value.imag() ),
                           static_cast<uint32_t>( value.real() ), static_cast<uint32_t>( value.imag() ) };
   return vld1q_u32( d );
}

template< typename T >
BLAZE_ALWAYS_INLINE EnableIf_t< IsSigned_v<T>, SIMDcint64 >
   setc64( complex<T> value )
{
   const int64_t d[2] = { value.real(), value.imag() };
   return vld1q_s64( d );
}

template< typename T >
BLAZE_ALWAYS_INLINE EnableIf_t< !IsSigned_v<T>, SIMDcuint64 >
   setc64( complex<T> value )
{
   const uint64_t d[2] = { static_cast<uint64_t>( value.real() ), static_cast<uint64_t>( value.imag() ) };
   return vld1q_u64( d );
}

} // namespace set_neon_detail
#endif

//=================================================================================================
//
//  8-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Sets all values in the vector to the given 1-byte integral value.
// \ingroup simd
//
// \param value The given 1-byte integral value.
// \return The set vector of 1-byte integral values.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_t< IsIntegral_v<T> && HasSize_v<T,1UL>
                                    , If_t< IsSigned_v<T>, SIMDint8, SIMDuint8 > >
   set( T value ) noexcept
{
#if BLAZE_AVX512F_MODE
   return _mm512_set1_epi8( value );
#elif BLAZE_AVX2_MODE
   return _mm256_set1_epi8( value );
#elif BLAZE_SSE2_MODE
   return _mm_set1_epi8( value );
#elif BLAZE_NEON_MODE
   return neon_dup_detail::dup8( value, BoolConstant< IsSigned_v<T> >{} );
#else
   return value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Sets all values in the vector to the given 1-byte integral complex value.
// \ingroup simd
//
// \param value The given 1-byte integral complex value.
// \return The set vector of 1-byte integral complex values.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_t< IsIntegral_v<T> && HasSize_v<T,1UL>
                                    , If_t< IsSigned_v<T>, SIMDcint8, SIMDcuint8 > >
   set( complex<T> value ) noexcept
{
#if BLAZE_AVX512F_MODE
   return _mm512_set1_epi16( reinterpret_cast<const int16_t&>( value ) );
#elif BLAZE_AVX2_MODE
   return _mm256_set_epi8( value.imag(), value.real(), value.imag(), value.real(),
                           value.imag(), value.real(), value.imag(), value.real(),
                           value.imag(), value.real(), value.imag(), value.real(),
                           value.imag(), value.real(), value.imag(), value.real(),
                           value.imag(), value.real(), value.imag(), value.real(),
                           value.imag(), value.real(), value.imag(), value.real(),
                           value.imag(), value.real(), value.imag(), value.real(),
                           value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_SSE2_MODE
   return _mm_set_epi8( value.imag(), value.real(), value.imag(), value.real(),
                        value.imag(), value.real(), value.imag(), value.real(),
                        value.imag(), value.real(), value.imag(), value.real(),
                        value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_NEON_MODE
   return set_neon_detail::setc8( value );
#else
   return value;
#endif
   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );
}
//*************************************************************************************************




//=================================================================================================
//
//  16-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Sets all values in the vector to the given 2-byte integral value.
// \ingroup simd
//
// \param value The given 2-byte integral value.
// \return The set vector of 2-byte integral values.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_t< IsIntegral_v<T> && HasSize_v<T,2UL>
                                    , If_t< IsSigned_v<T>, SIMDint16, SIMDuint16 > >
   set( T value ) noexcept
{
#if BLAZE_AVX512F_MODE
   return _mm512_set1_epi16( value );
#elif BLAZE_AVX2_MODE
   return _mm256_set1_epi16( value );
#elif BLAZE_SSE2_MODE
   return _mm_set1_epi16( value );
#elif BLAZE_NEON_MODE
   return neon_dup_detail::dup16( value, BoolConstant< IsSigned_v<T> >{} );
#else
   return value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Sets all values in the vector to the given 2-byte integral complex value.
// \ingroup simd
//
// \param value The given 2-byte integral complex value.
// \return The set vector of 2-byte integral complex values.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_t< IsIntegral_v<T> && HasSize_v<T,2UL>
                                    , If_t< IsSigned_v<T>, SIMDcint16, SIMDcuint16 > >
   set( complex<T> value ) noexcept
{
#if BLAZE_AVX512F_MODE
   return _mm512_set1_epi32( reinterpret_cast<const int32_t&>( value ) );
#elif BLAZE_AVX2_MODE
   return _mm256_set_epi16( value.imag(), value.real(), value.imag(), value.real(),
                            value.imag(), value.real(), value.imag(), value.real(),
                            value.imag(), value.real(), value.imag(), value.real(),
                            value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_SSE2_MODE
   return _mm_set_epi16( value.imag(), value.real(), value.imag(), value.real(),
                         value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_NEON_MODE
   return set_neon_detail::setc16( value );
#else
   return value;
#endif
   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );
}
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Sets all values in the vector to the given 4-byte integral value.
// \ingroup simd
//
// \param value The given 4-byte integral value.
// \return The set vector of 4-byte integral values.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_t< IsIntegral_v<T> && HasSize_v<T,4UL>
                                    , If_t< IsSigned_v<T>, SIMDint32, SIMDuint32 > >
   set( T value ) noexcept
{
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
   return _mm512_set1_epi32( value );
#elif BLAZE_AVX2_MODE
   return _mm256_set1_epi32( value );
#elif BLAZE_SSE2_MODE
   return _mm_set1_epi32( value );
#elif BLAZE_NEON_MODE
   return neon_dup_detail::dup32( value, BoolConstant< IsSigned_v<T> >{} );
#else
   return value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Sets all values in the vector to the given 4-byte integral complex value.
// \ingroup simd
//
// \param value The given 4-byte integral complex value.
// \return The set vector of 4-byte integral complex values.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_t< IsIntegral_v<T> && HasSize_v<T,4UL>
                                    , If_t< IsSigned_v<T>, SIMDcint32, SIMDcuint32 > >
   set( complex<T> value ) noexcept
{
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
   return _mm512_set_epi32( value.imag(), value.real(), value.imag(), value.real(),
                            value.imag(), value.real(), value.imag(), value.real(),
                            value.imag(), value.real(), value.imag(), value.real(),
                            value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_AVX2_MODE
   return _mm256_set_epi32( value.imag(), value.real(), value.imag(), value.real(),
                            value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_SSE2_MODE
   return _mm_set_epi32( value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_NEON_MODE
   return set_neon_detail::setc32( value );
#else
   return value;
#endif
   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );
}
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Sets all values in the vector to the given 8-byte integral value.
// \ingroup simd
//
// \param value The given 8-byte integral value.
// \return The set vector of 8-byte integral values.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_t< IsIntegral_v<T> && HasSize_v<T,8UL>
                                    , If_t< IsSigned_v<T>, SIMDint64, SIMDuint64 > >
   set( T value ) noexcept
{
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
   return _mm512_set1_epi64( value );
#elif BLAZE_AVX2_MODE
   return _mm256_set1_epi64x( value );
#elif BLAZE_SSE2_MODE
   return _mm_set1_epi64x( value );
#elif BLAZE_NEON_MODE
   return neon_dup_detail::dup64( value, BoolConstant< IsSigned_v<T> >{} );
#else
   return value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Sets all values in the vector to the given 8-byte integral complex value.
// \ingroup simd
//
// \param value The given 8-byte integral complex value.
// \return The set vector of 8-byte integral complex values.
*/
template< typename T >  // Type of the integral value
BLAZE_ALWAYS_INLINE const EnableIf_t< IsIntegral_v<T> && HasSize_v<T,8UL>
                                    , If_t< IsSigned_v<T>, SIMDcint64, SIMDcuint64 > >
   set( complex<T> value ) noexcept
{
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
   return _mm512_set_epi64( value.imag(), value.real(), value.imag(), value.real(),
                            value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_AVX2_MODE
   return _mm256_set_epi64x( value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_SSE2_MODE
   return _mm_set_epi64x( value.imag(), value.real() );
#elif BLAZE_NEON_MODE
   return set_neon_detail::setc64( value );
#else
   return value;
#endif
   BLAZE_STATIC_ASSERT( sizeof( complex<T> ) == 2UL*sizeof( T ) );
}
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Sets all values in the vector to the given \c float value.
// \ingroup simd
//
// \param value The given \c float value.
// \return The set vector of \c float values.
*/
BLAZE_ALWAYS_INLINE const SIMDfloat set( float value ) noexcept
{
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
   return _mm512_set1_ps( value );
#elif BLAZE_AVX_MODE
   return _mm256_set1_ps( value );
#elif BLAZE_SSE_MODE
   return _mm_set1_ps( value );
#elif BLAZE_NEON_MODE
   return vdupq_n_f32( value );
#else
   return value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Sets all values in the vector to the given \c complex<float> value.
// \ingroup simd
//
// \param value The given \c complex<float> value.
// \return The set vector of \c complex<float> values.
*/
BLAZE_ALWAYS_INLINE const SIMDcfloat set( const complex<float>& value ) noexcept
{
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
   return _mm512_set_ps( value.imag(), value.real(), value.imag(), value.real(),
                         value.imag(), value.real(), value.imag(), value.real(),
                         value.imag(), value.real(), value.imag(), value.real(),
                         value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_AVX_MODE
   return _mm256_set_ps( value.imag(), value.real(), value.imag(), value.real(),
                         value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_SSE_MODE
   return _mm_set_ps( value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_NEON_MODE
   return float32x4_t{ value.real(), value.imag(), value.real(), value.imag() };
#else
   return value;
#endif
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );
}
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Sets all values in the vector to the given \c double value.
// \ingroup simd
//
// \param value The given \c double value.
// \return The set vector of \c double values.
*/
BLAZE_ALWAYS_INLINE const SIMDdouble set( double value ) noexcept
{
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
   return _mm512_set1_pd( value );
#elif BLAZE_AVX_MODE
   return _mm256_set1_pd( value );
#elif BLAZE_SSE2_MODE
   return _mm_set1_pd( value );
#elif BLAZE_NEON_MODE
   return vdupq_n_f64( value );
#else
   return value;
#endif
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Sets all values in the vector to the given \c complex<double> value.
// \ingroup simd
//
// \param value The given \c complex<double> value.
// \return The set vector of \c complex<double> values.
*/
BLAZE_ALWAYS_INLINE const SIMDcdouble set( const complex<double>& value ) noexcept
{
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
   return _mm512_set_pd( value.imag(), value.real(), value.imag(), value.real(),
                         value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_AVX_MODE
   return _mm256_set_pd( value.imag(), value.real(), value.imag(), value.real() );
#elif BLAZE_SSE2_MODE
   return _mm_set_pd( value.imag(), value.real() );
#elif BLAZE_NEON_MODE
   return float64x2_t{ value.real(), value.imag() };
#else
   return value;
#endif
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );
}
//*************************************************************************************************

} // namespace blaze

#endif
