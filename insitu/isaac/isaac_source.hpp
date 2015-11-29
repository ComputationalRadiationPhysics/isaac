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

template <
#ifdef ISAAC_ALPAKA
    typename TDevAcc,
#endif
    typename TFeaDim
>
class IsaacBaseSource
{
    #ifdef ISAAC_ALPAKA
        using TTexDim = alpaka::dim::DimInt<1>;
    #endif
    public:
        IsaacBaseSource (
            #ifdef ISAAC_ALPAKA
                TDevAcc acc,
            #endif
            std::string name,
            size_t transfer_func_size
        ) :
        name(name)
        #ifdef ISAAC_ALPAKA
            ,transfer_func_d( alpaka::mem::buf::alloc<isaac_float4, size_t>(acc, alpaka::Vec<TTexDim, size_t> ( transfer_func_size ) ) )
        #endif
        {
            #ifndef ISAAC_ALPAKA
                ISAAC_CUDA_CHECK(cudaMalloc((isaac_float4**)&transfer_func_d, sizeof(isaac_float4)*transfer_func_size));
            #endif
        }
        std::string name;
        static const isaac_uint feature_dim = TFeaDim::value;
        #ifdef ISAAC_ALPAKA
            alpaka::mem::buf::Buf<TDevAcc, isaac_float4, TTexDim, size_t> transfer_func_d;
        #else
            isaac_float4* transfer_func_d;
        #endif
};

} //namespace isaac;
