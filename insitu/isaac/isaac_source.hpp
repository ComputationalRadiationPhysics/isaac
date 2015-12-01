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
#include "isaac_fusion_extension.hpp"


namespace isaac
{
    
ISAAC_NO_HOST_DEVICE_WARNING
template <
#if ISAAC_ALPAKA == 1
    typename TDevAcc,
    typename THost,
    typename TStream,
#endif
    typename TFeaDim
>
class IsaacBaseSource
{
    public:
    #if ISAAC_ALPAKA == 1
        using TTexDim = alpaka::dim::DimInt<1>;
    #endif
        ISAAC_NO_HOST_DEVICE_WARNING
        IsaacBaseSource (
            #if ISAAC_ALPAKA == 1
                TDevAcc acc,
                THost host,
                TStream stream,
            #endif
            std::string name,
            size_t transfer_func_size
        ) :
        name(name),
        transfer_func_size(transfer_func_size)
        #if ISAAC_ALPAKA == 1
            ,transfer_func_d( alpaka::mem::buf::alloc<isaac_float4, size_t>(acc, alpaka::Vec<TTexDim, size_t> ( transfer_func_size ) ) )
        #endif
        {
            #if ISAAC_ALPAKA == 0
                ISAAC_CUDA_CHECK(cudaMalloc((isaac_float4**)&transfer_func_d, sizeof(isaac_float4)*transfer_func_size));
            #endif
            //Set transfer function to default
            #if ISAAC_ALPAKA == 1
                alpaka::mem::buf::Buf<THost, isaac_float4, TTexDim, size_t> transfer_func_h_buf ( alpaka::mem::buf::alloc<isaac_float4, size_t>(host, transfer_func_size ) );
                isaac_float4* transfer_func_h = reinterpret_cast<isaac_float4*>(alpaka::mem::view::getPtrNative(transfer_func_h_buf));
            #else
                isaac_float4 transfer_func_h[ transfer_func_size ];
            #endif
            for (size_t i = 0; i < transfer_func_size; i++)
            {
                transfer_func_h[i].x = isaac_float(1);
                transfer_func_h[i].y = isaac_float(1);
                transfer_func_h[i].z = isaac_float(1);
                transfer_func_h[i].w = isaac_float(i) / isaac_float(transfer_func_size-1);
            }
            #if ISAAC_ALPAKA == 1
                alpaka::mem::view::copy(stream, transfer_func_d, transfer_func_h_buf, transfer_func_size );
            #else
                ISAAC_CUDA_CHECK(cudaMemcpy(transfer_func_d, transfer_func_h, sizeof(isaac_float4)*transfer_func_size, cudaMemcpyHostToDevice));
            #endif
        }
        size_t transfer_func_size;
        std::string name;
        static const isaac_uint feature_dim = TFeaDim::value;
        #if ISAAC_ALPAKA == 1
            alpaka::mem::buf::Buf<TDevAcc, isaac_float4, TTexDim, size_t> transfer_func_d;
        #else
            isaac_float4* transfer_func_d;            
        #endif
};

} //namespace isaac;
