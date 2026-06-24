//=================================================================================================
/*!
//  \file blaze/math/simd/NeonIntegral.h
//  \brief Shared NEON helpers for integral store and all-lanes equality
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

#ifndef _BLAZE_MATH_SIMD_NEONINTEGRAL_H_
#define _BLAZE_MATH_SIMD_NEONINTEGRAL_H_

#include <blaze/system/Vectorization.h>

#if BLAZE_NEON_MODE

namespace blaze {
namespace neon_integral {

template< typename IT >
struct EqualMask;

template<> struct EqualMask<uint8x16_t> { static bool all( uint8x16_t  m ) { return vminvq_u8 ( m ) == 0xFF; } };
template<> struct EqualMask<uint16x8_t> { static bool all( uint16x8_t m ) { return vminvq_u16( m ) == 0xFFFF; } };
template<> struct EqualMask<uint32x4_t> { static bool all( uint32x4_t m ) { return vminvq_u32( m ) == 0xFFFFFFFFU; } };
template<> struct EqualMask<uint64x2_t> { static bool all( uint64x2_t m ) { return vminvq_u32( vreinterpretq_u32_u64( m ) ) == 0xFFFFFFFFU; } };

template< typename IT >
struct Equal;

template<> struct Equal<int8x16_t>  { using mask_type = uint8x16_t; static mask_type cmp( int8x16_t  a, int8x16_t  b ) { return vceqq_s8 ( a, b ); } };
template<> struct Equal<uint8x16_t> { using mask_type = uint8x16_t; static mask_type cmp( uint8x16_t a, uint8x16_t b ) { return vceqq_u8 ( a, b ); } };
template<> struct Equal<int16x8_t>  { using mask_type = uint16x8_t; static mask_type cmp( int16x8_t  a, int16x8_t  b ) { return vceqq_s16( a, b ); } };
template<> struct Equal<uint16x8_t> { using mask_type = uint16x8_t; static mask_type cmp( uint16x8_t a, uint16x8_t b ) { return vceqq_u16( a, b ); } };
template<> struct Equal<int32x4_t>  { using mask_type = uint32x4_t; static mask_type cmp( int32x4_t  a, int32x4_t  b ) { return vceqq_s32( a, b ); } };
template<> struct Equal<uint32x4_t> { using mask_type = uint32x4_t; static mask_type cmp( uint32x4_t a, uint32x4_t b ) { return vceqq_u32( a, b ); } };
template<> struct Equal<int64x2_t>  { using mask_type = uint64x2_t; static mask_type cmp( int64x2_t  a, int64x2_t  b ) { return vceqq_s64( a, b ); } };
template<> struct Equal<uint64x2_t> { using mask_type = uint64x2_t; static mask_type cmp( uint64x2_t a, uint64x2_t b ) { return vceqq_u64( a, b ); } };

template< typename IT >
struct AllEqual
{
   static bool apply( IT a, IT b )
   {
      return EqualMask<typename Equal<IT>::mask_type>::all( Equal<IT>::cmp( a, b ) );
   }
};

template< typename IT >
struct Store;

template<> struct Store<int8x16_t>  { static void apply( void* a, int8x16_t  v ) { vst1q_s8 ( reinterpret_cast<int8_t* >(a), v ); } };
template<> struct Store<uint8x16_t> { static void apply( void* a, uint8x16_t v ) { vst1q_u8 ( reinterpret_cast<uint8_t*>(a), v ); } };
template<> struct Store<int16x8_t>  { static void apply( void* a, int16x8_t  v ) { vst1q_s16( reinterpret_cast<int16_t*>(a), v ); } };
template<> struct Store<uint16x8_t> { static void apply( void* a, uint16x8_t v ) { vst1q_u16( reinterpret_cast<uint16_t*>(a), v ); } };
template<> struct Store<int32x4_t>  { static void apply( void* a, int32x4_t  v ) { vst1q_s32( reinterpret_cast<int32_t*>(a), v ); } };
template<> struct Store<uint32x4_t> { static void apply( void* a, uint32x4_t v ) { vst1q_u32( reinterpret_cast<uint32_t*>(a), v ); } };
template<> struct Store<int64x2_t>  { static void apply( void* a, int64x2_t  v ) { vst1q_s64( reinterpret_cast<int64_t*>(a), v ); } };
template<> struct Store<uint64x2_t> { static void apply( void* a, uint64x2_t v ) { vst1q_u64( reinterpret_cast<uint64_t*>(a), v ); } };

} // namespace neon_integral
} // namespace blaze

#endif

#endif
