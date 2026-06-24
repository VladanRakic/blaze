//=================================================================================================
/*!
//  \file blaze/math/simd/NeonDup.h
//  \brief Shared NEON helpers for vector broadcast and zero initialization
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

#ifndef _BLAZE_MATH_SIMD_NEONDUP_H_
#define _BLAZE_MATH_SIMD_NEONDUP_H_

#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>
#include <blaze/util/IntegralConstant.h>

#if BLAZE_NEON_MODE

namespace blaze {
namespace neon_dup_detail {

BLAZE_ALWAYS_INLINE SIMDint8   dup8 ( int8_t   v, TrueType  ) { return vdupq_n_s8 ( v ); }
BLAZE_ALWAYS_INLINE SIMDuint8  dup8 ( uint8_t  v, FalseType ) { return vdupq_n_u8 ( v ); }
BLAZE_ALWAYS_INLINE SIMDint16  dup16( int16_t  v, TrueType  ) { return vdupq_n_s16( v ); }
BLAZE_ALWAYS_INLINE SIMDuint16 dup16( uint16_t v, FalseType ) { return vdupq_n_u16( v ); }
BLAZE_ALWAYS_INLINE SIMDint32  dup32( int32_t  v, TrueType  ) { return vdupq_n_s32( v ); }
BLAZE_ALWAYS_INLINE SIMDuint32 dup32( uint32_t v, FalseType ) { return vdupq_n_u32( v ); }
BLAZE_ALWAYS_INLINE SIMDint64  dup64( int64_t  v, TrueType  ) { return vdupq_n_s64( v ); }
BLAZE_ALWAYS_INLINE SIMDuint64 dup64( uint64_t v, FalseType ) { return vdupq_n_u64( v ); }

template< typename IT >
struct Zero;

template<> struct Zero<int8x16_t>  { static int8x16_t  apply() { return vdupq_n_s8 ( 0 ); } };
template<> struct Zero<uint8x16_t> { static uint8x16_t apply() { return vdupq_n_u8( 0 ); } };
template<> struct Zero<int16x8_t>  { static int16x8_t  apply() { return vdupq_n_s16( 0 ); } };
template<> struct Zero<uint16x8_t> { static uint16x8_t apply() { return vdupq_n_u16( 0 ); } };
template<> struct Zero<int32x4_t>  { static int32x4_t  apply() { return vdupq_n_s32( 0 ); } };
template<> struct Zero<uint32x4_t> { static uint32x4_t apply() { return vdupq_n_u32( 0 ); } };
template<> struct Zero<int64x2_t>  { static int64x2_t  apply() { return vdupq_n_s64( 0 ); } };
template<> struct Zero<uint64x2_t> { static uint64x2_t apply() { return vdupq_n_u64( 0 ); } };

} // namespace neon_dup_detail
} // namespace blaze

#endif

#endif
