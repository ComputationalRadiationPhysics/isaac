/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Lesser Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include <boost/preprocessor.hpp>

#ifndef ISAAC_MAX_DIFFERENCE
    #define ISAAC_MAX_DIFFERENCE 4
#endif

#ifndef ISAAC_MAX_FUNCTORS
    #define ISAAC_MAX_FUNCTORS 3
#endif

#define ISAAC_FUNCTOR_COUNT 5

// ISAAC_FUNCTOR_COUNT ^ ISAAC_MAX_FUNCTORS
#define ISAAC_FUNCTOR_COMPLEX_SUBDEF(Z,I,U) * ISAAC_FUNCTOR_COUNT
#define ISAAC_FUNCTOR_COMPLEX ( ISAAC_FUNCTOR_COUNT BOOST_PP_REPEAT( BOOST_PP_DEC( ISAAC_MAX_FUNCTORS ), ISAAC_FUNCTOR_COMPLEX_SUBDEF, ~) )

#ifndef ISAAC_MAX_SOURCES
    #define ISAAC_MAX_SOURCES 16
#endif


#define ISAAC_MAX_RECEIVE 262144 //256kb
#define ISAAC_Z_NEAR 1.0f
#define ISAAC_Z_FAR 100.0f

#if ISAAC_ALPAKA == 1
    #define ISAAC_HOST_DEVICE_INLINE ALPAKA_FN_ACC
#else
    #define ISAAC_HOST_DEVICE_INLINE __device__ __host__ __forceinline__
#endif

#if ISAAC_ALPAKA == 1
    #define ISAAC_HOST_INLINE ALPAKA_FN_HOST
#else
    #define ISAAC_HOST_INLINE __host__ __forceinline__
#endif

#if ISAAC_ALPAKA == 1
    #define ISAAC_DEVICE_INLINE ALPAKA_FN_ACC_CUDA_ONLY
#else
    #define ISAAC_DEVICE_INLINE __device__ __forceinline__
#endif

#ifdef __CUDACC__
    #define ISAAC_DEVICE __device__ __host__
#else
    #define ISAAC_DEVICE
#endif

#ifdef __CUDACC__
    #define ISAAC_NO_HOST_DEVICE_WARNING _Pragma("hd_warning_disable")
#else
    #define ISAAC_NO_HOST_DEVICE_WARNING
#endif
