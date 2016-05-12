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
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include "isaac_types.hpp"

#define ISAAC_CUDA_CHECK(call)                                                 \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define ISAAC_FOR_EACH_DIM_4(start,end) \
    start.x end \
    start.y end \
    start.z end \
    start.w end

#define ISAAC_FOR_EACH_DIM_3(start,end) \
    start.x end \
    start.y end \
    start.z end

#define ISAAC_FOR_EACH_DIM_2(start,end) \
    start.x end \
    start.y end

#define ISAAC_FOR_EACH_DIM_1(start,end) \
    start.x end

#define ISAAC_FOR_EACH_DIM(dim,start,end) \
    BOOST_PP_CAT( ISAAC_FOR_EACH_DIM_, dim) (start,end)

#define ISAAC_FOR_EACH_DIM_TWICE_4(start,middle,end) \
    start.x middle.x end \
    start.y middle.y end \
    start.z middle.z end \
    start.w middle.w end

#define ISAAC_FOR_EACH_DIM_TWICE_3(start,middle,end) \
    start.x middle.x end \
    start.y middle.y end \
    start.z middle.z end

#define ISAAC_FOR_EACH_DIM_TWICE_2(start,middle,end) \
    start.x middle.x end \
    start.y middle.y end

#define ISAAC_FOR_EACH_DIM_TWICE_1(start,middle,end) \
    start.x middle.x end

#define ISAAC_FOR_EACH_DIM_TWICE(dim,start,middle,end) \
    BOOST_PP_CAT( ISAAC_FOR_EACH_DIM_TWICE_, dim) (start,middle,end)

#define ISAAC_SWITCH_IF_SMALLER(left,right) \
    if (left < right) \
    { \
        auto temp = left; \
        left = right; \
        right = temp; \
    }

#define ISAAC_SET_COLOR( dest, color ) \
    { \
        isaac_uint4 result; \
        result.x = min( isaac_uint( min( isaac_float(1), color.x ) * 255.0f ), 255); \
        result.y = min( isaac_uint( min( isaac_float(1), color.y ) * 255.0f ), 255); \
        result.z = min( isaac_uint( min( isaac_float(1), color.z ) * 255.0f ), 255); \
        result.w = min( isaac_uint( min( isaac_float(1), color.w ) * 255.0f ), 255); \
        dest = (result.w << 24) | (result.z << 16) | (result.y << 8) | (result.x << 0); \
    }

#define ISAAC_START_TIME_MEASUREMENT( unique_name, time_function ) \
    uint64_t BOOST_PP_CAT( __tm_start_, unique_name ) = time_function;
#define ISAAC_STOP_TIME_MEASUREMENT( result, operand, unique_name, time_function ) \
    result operand time_function - BOOST_PP_CAT( __tm_start_, unique_name );

#define ISAAC_SET_IDENTITY(size,matrix) \
    for (isaac_int x = 0; x < size; x++) \
        for (isaac_int y = 0; y < size; y++) \
            (matrix)[x+y*size] = (x==y)?1.0f:0.0f;

#define ISAAC_JSON_ADD_MATRIX(array,matrix,count) \
    for (isaac_int i = 0; i < count; i++) \
        json_array_append_new( array, json_real( (matrix)[i] ) );

#ifdef ISAAC_THREADING
    #define ISAAC_WAIT_VISUALIZATION \
        if (visualizationThread) \
        { \
            pthread_join(visualizationThread,NULL); \
            visualizationThread = 0; \
        }
#else
    #define ISAAC_WAIT_VISUALIZATION {}
#endif
