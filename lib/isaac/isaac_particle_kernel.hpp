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
    /**
     * @brief Checks for collision with source particles
     * 
     * Returns color, normal, position of particle
     * 
     * @tparam T_transferSize 
     * @tparam T_offset 
     * @tparam T_Filter 
     */
    template<
        ISAAC_IDX_TYPE T_transferSize,
        int T_offset,
        typename T_Filter
    >
    struct MergeParticleSourceIterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_TransferArray,
            typename T_SourceWeight
        >
        ISAAC_DEVICE_INLINE void operator()(
            const T_NR & nr,
            const T_Source & source,                   //particle source
            const isaac_float3 & start,               //ray start position in local volume
            const isaac_float3 & dir,
            const isaac_uint3 & cellPos,             //cell to test in local volume
            const T_TransferArray & transferArray,     //transfer function
            const T_SourceWeight & sourceWeight,       //weight of this particle source for radius
            const isaac_float3 & particleScale,    //scale of volume to prevent stretched particles
            const isaac_float3 & clippingNormal,     //normal of the intersecting clipping plane
            const bool & isClipped,
            isaac_float4 & out_color,                 //resulting particle color
            isaac_float3 & out_normal,                //resulting particle normal
            isaac_float3 & out_position,              //resulting particle hit position
            bool & out_particleHit,                  //true or false if particle has been hit or not
            isaac_float & out_depth                       //resulting particle depth
        ) const
        {
            const int sourceNumber = T_NR::value + T_offset;
            if( boost::mpl::at_c<
                T_Filter,
                T_NR::value
            >::type::value )
            {
                auto particleIterator = source.getIterator( cellPos );

                // iterate over all particles in current cell
                for( int i = 0; i < particleIterator.size; ++i )
                {
                    // ray sphere intersection
                    isaac_float3 particlePos =
                        ( particleIterator.getPosition( ) + isaac_float3( cellPos ) )
                        * particleScale;
                    isaac_float3 L = particlePos - start;
                    isaac_float radius = particleIterator.getRadius( )
                                         * sourceWeight.value[T_NR::value + T_offset];
                    isaac_float radius2 = radius * radius;
                    isaac_float tca = glm::dot( L, dir );
                    isaac_float d2 = glm::dot( L, L ) - tca * tca;
                    if( d2 <= radius2 )
                    {
                        isaac_float thc = sqrt( radius2 - d2 );
                        isaac_float t0 = tca - thc;
                        isaac_float t1 = tca + thc;

                        // if the ray hits the sphere
                        if( t1 >= 0 && t0 < out_depth )
                        {
                            isaac_float_dim <T_Source::featureDim>
                                data = particleIterator.getAttribute( );

                            isaac_float result = isaac_float( 0 );

                            // apply functorchain
                           result = applyFunctorChain(data, sourceNumber);

                            // apply transferfunction
                            ISAAC_IDX_TYPE lookupValue = ISAAC_IDX_TYPE(
                                glm::round( result * isaac_float( T_transferSize ) )
                            );
                            lookupValue = glm::clamp( lookupValue, ISAAC_IDX_TYPE( 0 ), T_transferSize - 1 );
                            isaac_float4 value = transferArray.pointer[sourceNumber][lookupValue];

                            // check if the alpha value is greater or equal than 0.5
                            if( value.w >= 0.5f )
                            {
                                out_color = value;
                                out_depth = t0;
                                out_particleHit = 1;
                                out_position = particlePos;
                                out_normal = start + t0 * dir - particlePos;
                                if( t0 < 0 && isClipped )
                                {
                                    #if ISAAC_AO_BUG_FIX == 1
                                        out_depth = 0;
                                    #endif
                                        out_normal = -clippingNormal;
                                }
                            }
                        }
                    }
                    particleIterator.next( );
                }
            }
        }
    };


    template<
        typename T_ParticleList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        int T_sourceOffset
    >
    struct ParticleRenderKernel
    {
        template<
            typename T_Acc
        >
        ISAAC_DEVICE void operator()(
            T_Acc const & acc,
            GBuffer gBuffer,
            const T_ParticleList particleSources,   //source simulation particles
            const T_TransferArray transferArray,    //array of pointers to transfer functions
            const T_SourceWeight sourceWeight,      //weights of all sources 
            const isaac_float3 scale,               //isaac set scaling
            const ClippingStruct inputClipping      //clipping planes
        ) const
        {
            //get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            isaac_uint2 pixel = isaac_uint2( alpThreadIdx[2], alpThreadIdx[1] );
            //apply framebuffer offset to pixel
            //stop if pixel position is out of bounds
            pixel = pixel + gBuffer.startOffset;
            if( !isInUpperBounds( pixel, gBuffer.size ) )
                return;

            bool atLeastOne = true;
            forEachWithMplParams(
                particleSources,
                CheckNoSourceIterator< T_Filter >( ),
                atLeastOne
            );

            if( !atLeastOne )
                return;

            Ray ray = pixelToRay( isaac_float2( pixel ), isaac_float2( gBuffer.size ) );

            if( !clipRay(ray, inputClipping ) )
                return;

            ray.endDepth = glm::min(ray.endDepth, gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x]);

            isaac_float4 particleColor;
            isaac_float depth = std::numeric_limits<isaac_float>::max( );
            bool particleHit = false;

            // get the signs of the direction for the raymarch
            isaac_int3 dirSign = glm::sign( ray.dir );

            // calculate current position in scaled object space
            isaac_float3 currentPos = ray.start + ray.dir * ray.startDepth;

            // calculate current local cell coordinates
            isaac_uint3 currentCell = isaac_uint3( glm::clamp( 
                                    isaac_int3( currentPos / scale ), 
                                    isaac_int3( 0 ), 
                                    isaac_int3( SimulationSize.localParticleSize - ISAAC_IDX_TYPE( 1 ) ) 
                                ) );

            isaac_float rayLength = ray.endDepth - ray.startDepth;
            isaac_float testedLength = 0;


            // calculate next intersection with each dimension
            isaac_float3 t = ( ( isaac_float3( currentCell ) + isaac_float3( glm::max( dirSign, 0 ) ) ) 
                    * scale - currentPos ) / ray.dir;

            // calculate delta length to next intersection in the same dimension
            isaac_float3 deltaT = scale / ray.dir * isaac_float3( dirSign );

            isaac_float3 particleHitposition(0);

            // check for 0 to stop infinite looping
            if( ray.dir.x == 0 )
                t.x = std::numeric_limits<isaac_float>::max( );

            if( ray.dir.y == 0 )
                t.y = std::numeric_limits<isaac_float>::max( );

            if( ray.dir.z == 0 )
                t.z = std::numeric_limits<isaac_float>::max( );


            //normal at particle hit position
            isaac_float3 particleNormal;

            // iterate over all cells on the ray path
            // check if the ray leaves the local volume, has a particle hit or exceeds the max ray distance
            while( isInUpperBounds(currentCell, SimulationSize.localParticleSize)
                && particleHit == false
                && testedLength <= rayLength )
            {

                // calculate particle intersections for each particle source
                forEachWithMplParams(
                    particleSources,
                    MergeParticleSourceIterator<
                        T_transferSize,
                        T_sourceOffset,
                        T_Filter
                    >( ),
                    currentPos,
                    ray.dir,
                    currentCell,
                    transferArray,
                    sourceWeight,
                    scale,
                    ray.clippingNormal,
                    ray.isClipped,
                    particleColor,
                    particleNormal,
                    particleHitposition,
                    particleHit,
                    depth
                );


                // adds the deltaT value to the smallest dimension t and increment the cell index in the dimension
                if( t.x < t.y && t.x < t.z )
                {
                    currentCell.x += dirSign.x;
                    testedLength = t.x;
                    t.x += deltaT.x;
                }
                else if( t.y < t.x && t.y < t.z )
                {
                    currentCell.y += dirSign.y;
                    testedLength = t.y;
                    t.y += deltaT.y;
                }
                else
                {
                    currentCell.z += dirSign.z;
                    testedLength = t.z;
                    t.z += deltaT.z;
                }

            }
            // if there was a hit set maximum volume raycast distance to particle hit distance and set particle color
            if( !particleHit )
                return;

            // calculate lighting properties for the last hit particle
            particleNormal = glm::normalize( particleNormal );
            particleColor.a = isaac_float( 1 );
            setColor ( gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], particleColor );
            //save the particle normal in the normal g buffer
            gBuffer.normal[pixel.x + pixel.y * gBuffer.size.x] = particleNormal;
            
            //save the cell depth in our g buffer (depth)
            gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = depth + ray.startDepth;
        }
    };



    template<
        typename T_ParticleList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_WorkDiv,
        typename T_Acc,
        typename T_Stream,
        int T_sourceOffset,
        int T_n
    >
    struct ParticleRenderKernelCaller
    {
        inline static void call(
            T_Stream stream,
            const GBuffer & gBuffer,
            const T_ParticleList & particleSources,
            const T_TransferArray & transferArray,
            const T_SourceWeight & sourceWeight,
            const T_WorkDiv & workdiv,
            const isaac_float3 & scale,
            const ClippingStruct & clipping
        )
        {
            if( sourceWeight.value[T_sourceOffset + boost::mpl::size< T_ParticleList >::type::value - T_n] == isaac_float( 0 ) )
            {
                ParticleRenderKernelCaller<
                    T_ParticleList,
                    T_TransferArray,
                    T_SourceWeight,
                    typename boost::mpl::push_back<
                        T_Filter,
                        boost::mpl::false_
                    >::type,
                    T_transferSize,
                    T_WorkDiv,
                    T_Acc,
                    T_Stream,
                    T_sourceOffset,
                    T_n - 1
                >::call(
                    stream,
                    gBuffer,
                    particleSources,
                    transferArray,
                    sourceWeight,
                    workdiv,
                    scale,
                    clipping
                );
            }
            else
            {
                ParticleRenderKernelCaller<
                    T_ParticleList,
                    T_TransferArray,
                    T_SourceWeight,
                    typename boost::mpl::push_back<
                        T_Filter,
                        boost::mpl::true_
                    >::type,
                    T_transferSize,
                    T_WorkDiv,
                    T_Acc,
                    T_Stream,
                    T_sourceOffset,
                    T_n - 1
                >::call(
                    stream,
                    gBuffer,
                    particleSources,
                    transferArray,
                    sourceWeight,
                    workdiv,
                    scale,
                    clipping
                );
            }
        }
    };

    template<
        typename T_ParticleList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_WorkDiv,
        typename T_Acc,
        typename T_Stream,
        int T_sourceOffset
    >
    struct ParticleRenderKernelCaller<
        T_ParticleList,
        T_TransferArray,
        T_SourceWeight,
        T_Filter,
        T_transferSize,
        T_WorkDiv,
        T_Acc,
        T_Stream,
        T_sourceOffset,
        0 //<-- spezialisation
    >
    {
        inline static void call(
            T_Stream stream,
            const GBuffer & gBuffer,
            const T_ParticleList & particleSources,
            const T_TransferArray & transferArray,
            const T_SourceWeight & sourceWeight,
            const T_WorkDiv & workdiv,
            const isaac_float3 & scale,
            const ClippingStruct & clipping
        )
        {
            ParticleRenderKernel
            <
                T_ParticleList,
                T_TransferArray,
                T_SourceWeight,
                T_Filter,
                T_transferSize,
                T_sourceOffset
            >
            kernel;
            auto const instance
            (
                alpaka::createTaskKernel<T_Acc>
                (
                    workdiv,
                    kernel,
                    gBuffer,
                    particleSources,
                    transferArray,
                    sourceWeight,
                    scale,
                    clipping
                )
            );
            alpaka::enqueue(stream, instance);
        }
    };
}