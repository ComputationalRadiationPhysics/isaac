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

#include "isaac_macros.hpp"
#include "isaac_source.hpp"

namespace isaac
{

#ifndef ISAAC_ALPAKA
    __constant__ isaac_float isaac_inverse_d[16];
    __constant__ isaac_size_struct<3> isaac_size_d[1]; //[1] to access it same for cuda and alpaka
#endif

template <
    typename TSimDim,
    typename TSourceList
>
#ifdef ISAAC_ALPAKA
    struct IsaacFillRectKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
            isaac_float* isaac_inverse_d,
            isaac_size_struct<simdim>* isaac_size_d,
#else
        __global__ void IsaacFillRectKernel(
#endif
            uint32_t* pixels,
            isaac_size2 framebuffer_size,
            isaac_uint2 framebuffer_start,
            TSourceList sources,
            isaac_float step,
            isaac_float4 background_color)
#ifdef ISAAC_ALPAKA
        const
#endif
        {
            #ifdef ISAAC_ALPAKA
                auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                isaac_uint2 pixel =
                {
                    isaac_uint(threadIdx[2]),
                    isaac_uint(threadIdx[1])
                };
            #else
                isaac_uint2 pixel =
                {
                    isaac_uint(threadIdx.x + blockIdx.x * blockDim.x),
                    isaac_uint(threadIdx.y + blockIdx.y * blockDim.y)
                };
            #endif
            pixel = pixel + framebuffer_start;
            if ( ISAAC_FOR_EACH_DIM_TWICE(2, pixel, >= framebuffer_size, || ) 0 )
                return;
            
            isaac_float2 pixel_f =
            {
                isaac_float( pixel.x )/(isaac_float)framebuffer_size.x*isaac_float(2)-isaac_float(1),
                isaac_float( pixel.y )/(isaac_float)framebuffer_size.y*isaac_float(2)-isaac_float(1)
            };
            isaac_float4 start_p = {pixel_f.x*ISAAC_Z_NEAR,pixel_f.y*ISAAC_Z_NEAR,-1.0f*ISAAC_Z_NEAR,1.0f*ISAAC_Z_NEAR}; //znear
            isaac_float4   end_p = {pixel_f.x* ISAAC_Z_FAR,pixel_f.y* ISAAC_Z_FAR, 1.0f* ISAAC_Z_FAR,1.0f* ISAAC_Z_FAR}; //zfar
            isaac_float3 start =
            {
                isaac_inverse_d[ 0] * start_p.x + isaac_inverse_d[ 4] * start_p.y +  isaac_inverse_d[ 8] * start_p.z + isaac_inverse_d[12] * start_p.w,
                isaac_inverse_d[ 1] * start_p.x + isaac_inverse_d[ 5] * start_p.y +  isaac_inverse_d[ 9] * start_p.z + isaac_inverse_d[13] * start_p.w,
                isaac_inverse_d[ 2] * start_p.x + isaac_inverse_d[ 6] * start_p.y +  isaac_inverse_d[10] * start_p.z + isaac_inverse_d[14] * start_p.w
            };
            isaac_float3 end =
            {
                isaac_inverse_d[ 0] *   end_p.x + isaac_inverse_d[ 4] *   end_p.y +  isaac_inverse_d[ 8] *   end_p.z + isaac_inverse_d[12] *   end_p.w,
                isaac_inverse_d[ 1] *   end_p.x + isaac_inverse_d[ 5] *   end_p.y +  isaac_inverse_d[ 9] *   end_p.z + isaac_inverse_d[13] *   end_p.w,
                isaac_inverse_d[ 2] *   end_p.x + isaac_inverse_d[ 6] *   end_p.y +  isaac_inverse_d[10] *   end_p.z + isaac_inverse_d[14] *   end_p.w
            };
            isaac_float max_size = isaac_size_d[0].max_global_size / 2.0f;

            //scale to globale grid size
            start = start * max_size;
              end =   end * max_size;

            //move to local grid
            isaac_size3 move = isaac_size_d[0].global_size / size_t(2) - isaac_size_d[0].position;
            isaac_float3 move_f =
            {
                isaac_float(move.x),
                isaac_float(move.y),
                isaac_float(move.z)
            };
            start = start + move_f;
              end =   end + move_f;

            isaac_float3 vec = end - start;
            isaac_float l = sqrt( vec.x * vec.x + vec.y * vec.y + vec.z * vec.z );
            
            isaac_float3 step_vec = vec / l * step;
            isaac_float3 count_start =  - start / step_vec;
            isaac_float3 local_size_f =
            {
                isaac_float(isaac_size_d[0].local_size.x),
                isaac_float(isaac_size_d[0].local_size.y),
                isaac_float(isaac_size_d[0].local_size.z)
            };
            isaac_float3 count_end = ( local_size_f - start ) / step_vec;

            //count_start shall have the smaller values
            ISAAC_SWITCH_IF_SMALLER( count_end.x, count_start.x )
            ISAAC_SWITCH_IF_SMALLER( count_end.y, count_start.y )
            ISAAC_SWITCH_IF_SMALLER( count_end.z, count_start.z )
            
            //calc intersection of all three super planes and save in [count_start.x ; count_end.x]
            count_start.x = max( max( count_start.x, count_start.y ), count_start.z );
              count_end.x = min( min(   count_end.x,   count_end.y ),   count_end.z );
            if ( count_start.x > count_end.x || count_end.x <= 0.0f )
            {
                ISAAC_SET_COLOR( pixels[pixel.x + pixel.y * framebuffer_size.x], background_color )
                return;
            }
            
            //Starting the main loop
            isaac_int first = floor( count_start.x );
            isaac_int last = ceil( count_end.x );

            isaac_int count = last - first + 1;
            isaac_float4 color = background_color;
            isaac_float3 pos = start + step_vec * isaac_float(first);
            isaac_uint visited = 0;
            for (isaac_int i = 0; i < count; i++)
            {
                isaac_uint3 coord;
                ISAAC_FOR_EACH_DIM_TWICE(3, coord, = (isaac_uint)pos, ; )
                if ( ISAAC_FOR_EACH_DIM_TWICE(3, coord, < isaac_size_d[0].local_size, && ) 1 )
                {
                    isaac_float3 data = boost::fusion::at_c<0>(sources)[coord];
                    isaac_float4 color_add = { data.x, data.y, data.z, isaac_float(0) };
                    color = color + color_add / isaac_float(2);
                    visited++;
                }
                pos = pos + step_vec;
            }
            if (visited)
            {
                color = color / isaac_float(visited);
                color.w = 0.5f;
            }
            else
                color.w = 0.0f;
            ISAAC_SET_COLOR( pixels[pixel.x + pixel.y * framebuffer_size.x], color )
        }
#ifdef ISAAC_ALPAKA
    };
#endif

} //namespace isaac;
