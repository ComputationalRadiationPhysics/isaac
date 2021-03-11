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

#include "isaac_common_kernel.hpp"

namespace isaac
{
    template<
        typename T_Source
    >
    struct MinMaxKernel
    {
        template<
            typename T_Acc
        >
        ISAAC_DEVICE void operator()(
            T_Acc const & acc,
            const T_Source source,
            const int nr,
            MinMax * const result,
            const isaac_size3 localSize,
            void const * const pointer
        ) const
        {
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            isaac_int3 coord = {
                isaac_int( alpThreadIdx[1] ),
                isaac_int( alpThreadIdx[2] ),
                0
            };

            if( !isInUpperBounds(coord, localSize) )
                return;
            isaac_float min = std::numeric_limits<isaac_float>::max();
            isaac_float max = -std::numeric_limits<isaac_float>::max();
            for( ; coord.z < localSize.z; coord.z++ )
            {
                isaac_float_dim <T_Source::featureDim> data;
                if( T_Source::persistent )
                {
                    data = source[coord];
                }
                else
                {
                    isaac_float_dim <T_Source::featureDim> * ptr = (
                        isaac_float_dim < T_Source::featureDim > *
                    )( pointer );
                    data = ptr[coord.x + ISAAC_GUARD_SIZE
                               + ( coord.y + ISAAC_GUARD_SIZE )
                                 * ( localSize.x + 2 * ISAAC_GUARD_SIZE )
                               + ( coord.z + ISAAC_GUARD_SIZE ) * (
                                   ( localSize.x + 2 * ISAAC_GUARD_SIZE )
                                   * ( localSize.y + 2 * ISAAC_GUARD_SIZE )
                               )];
                };
                isaac_float value = applyFunctorChain(data, nr);
                min = glm::min( min, value );
                max = glm::max( max, value );
            }
            result[coord.x + coord.y * localSize.x].min = min;
            result[coord.x + coord.y * localSize.x].max = max;
        }

    };



    template<
        typename T_ParticleSource
    >
    struct MinMaxParticleKernel
    {
        template<
            typename T_Acc
        >
        ISAAC_DEVICE void operator()(
            T_Acc const & acc,
            const T_ParticleSource particleSource,
            const int nr,
            MinMax * const result,
            const isaac_size3 localSize
        ) const
        {
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            isaac_uint3 coord = {
                isaac_uint( alpThreadIdx[1] ),
                isaac_uint( alpThreadIdx[2] ),
                0
            };
            if( !isInUpperBounds(coord, localSize) )
                return;
            isaac_float min = std::numeric_limits<isaac_float>::max();
            isaac_float max = -std::numeric_limits<isaac_float>::max();
            for( ; coord.z < localSize.z; coord.z++ )
            {
                auto particleIterator = particleSource.getIterator( coord );
                for( int i = 0; i < particleIterator.size; i++ )
                {
                    isaac_float_dim <T_ParticleSource::featureDim> data;

                    data = particleIterator.getAttribute( );

                    isaac_float value = applyFunctorChain(data, nr);
                    min = glm::min( min, value );
                    max = glm::max( max, value );
                }

            }
            result[coord.x + coord.y * localSize.x].min = min;
            result[coord.x + coord.y * localSize.x].max = max;
        }

    };
}