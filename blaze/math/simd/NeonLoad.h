//=================================================================================================
/*!
//  \file blaze/math/simd/NeonLoad.h
//  \brief Shared NEON helpers for integral SIMD loads
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

#ifndef _BLAZE_MATH_SIMD_NEONLOAD_H_
#define _BLAZE_MATH_SIMD_NEONLOAD_H_

#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>
#include <blaze/util/IntegralConstant.h>

#if BLAZE_NEON_MODE

namespace blaze {
namespace load_neon_detail {

BLAZE_ALWAYS_INLINE SIMDint8   load8 ( const void* a, TrueType  ) { return vld1q_s8 ( reinterpret_cast<const int8_t* >(a) ); }
BLAZE_ALWAYS_INLINE SIMDuint8  load8 ( const void* a, FalseType ) { return vld1q_u8 ( reinterpret_cast<const uint8_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDint16  load16( const void* a, TrueType  ) { return vld1q_s16( reinterpret_cast<const int16_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDuint16 load16( const void* a, FalseType ) { return vld1q_u16( reinterpret_cast<const uint16_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDint32  load32( const void* a, TrueType  ) { return vld1q_s32( reinterpret_cast<const int32_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDuint32 load32( const void* a, FalseType ) { return vld1q_u32( reinterpret_cast<const uint32_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDint64  load64( const void* a, TrueType  ) { return vld1q_s64( reinterpret_cast<const int64_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDuint64 load64( const void* a, FalseType ) { return vld1q_u64( reinterpret_cast<const uint64_t*>(a) ); }

BLAZE_ALWAYS_INLINE SIMDcint8   loadc8 ( const void* a, TrueType  ) { return vld1q_s8 ( reinterpret_cast<const int8_t* >(a) ); }
BLAZE_ALWAYS_INLINE SIMDcuint8  loadc8 ( const void* a, FalseType ) { return vld1q_u8 ( reinterpret_cast<const uint8_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDcint16  loadc16( const void* a, TrueType  ) { return vld1q_s16( reinterpret_cast<const int16_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDcuint16 loadc16( const void* a, FalseType ) { return vld1q_u16( reinterpret_cast<const uint16_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDcint32  loadc32( const void* a, TrueType  ) { return vld1q_s32( reinterpret_cast<const int32_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDcuint32 loadc32( const void* a, FalseType ) { return vld1q_u32( reinterpret_cast<const uint32_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDcint64  loadc64( const void* a, TrueType  ) { return vld1q_s64( reinterpret_cast<const int64_t*>(a) ); }
BLAZE_ALWAYS_INLINE SIMDcuint64 loadc64( const void* a, FalseType ) { return vld1q_u64( reinterpret_cast<const uint64_t*>(a) ); }

} // namespace load_neon_detail
} // namespace blaze

#endif

#endif
