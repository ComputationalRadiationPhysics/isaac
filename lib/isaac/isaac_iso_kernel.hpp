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
#include "isaac_volume_kernel.hpp"


namespace isaac
{

    template<
        isaac_int T_interpolation,
        isaac_int T_index,
        typename T_NR,
        typename T_Source,
        typename T_PointerArray
    >
    ISAAC_DEVICE_INLINE isaac_float getCompGradient(
        const T_Source & source,
        const isaac_float3 & pos,
        const T_PointerArray & pointerArray,
        const isaac_size3 &  localSize
    )
    {
        isaac_float3 front = { 0, 0, 0 };
        front[T_index] = -1;
        front = front + pos;
        checkCoord< 
            T_Source
        >(
            front,
            localSize
        );

        isaac_float3 back = { 0, 0, 0 };
        back[T_index] = 1;
        back = back + pos;
        checkCoord< 
            T_Source
        >(
            back,
            localSize
        );

        isaac_float d;
        if( T_interpolation )
        {
            d = back[T_index] - front[T_index];
        }
        else
        {
            d = isaac_int( back[T_index] ) - isaac_int( front[T_index] );
        }

        return (
            getValue<
                T_interpolation,
                T_NR
            >(
                source,
                back,
                pointerArray,
                localSize
            ) - getValue<
                T_interpolation,
                T_NR
            >(
                source,
                front,
                pointerArray,
                localSize
            )
        ) / d;
    }

    template<
        isaac_int T_interpolation,
        typename T_NR,
        typename T_Source,
        typename T_PointerArray
    >
    ISAAC_DEVICE_INLINE isaac_float3 getGradient(
        const T_Source & source,
        const isaac_float3 & pos,
        const T_PointerArray & pointerArray,
        const isaac_size3 &  localSize
    )
    {

        isaac_float3 gradient = {
            getCompGradient<
                T_interpolation,
                0,
                T_NR
            >(
                source,
                pos,
                pointerArray,
                localSize
            ),
            getCompGradient<
                T_interpolation,
                1,
                T_NR
            >(
                source,
                pos,
                pointerArray,
                localSize
            ),
            getCompGradient<
                T_interpolation,
                2,
                T_NR
            >(
                source,
                pos,
                pointerArray,
                localSize
            )
        };
        return gradient;
    }

    template<
        ISAAC_IDX_TYPE T_transferSize,
        typename T_Filter,
        isaac_int T_interpolation
    >
    struct IsoCellTraversalSourceIterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_TransferArray,
            typename T_IsoThreshold,
            typename T_PointerArray
        >
        ISAAC_DEVICE_INLINE void operator()(
            const T_NR & nr,
            const T_Source & source,
            const Ray & ray,
            const isaac_float & t0,
            isaac_float * oldValues,
            const isaac_float3 & p1,
            const isaac_float & t1,
            const isaac_size3 & localSize,
            const T_TransferArray & transferArray,
            const T_IsoThreshold & sourceIsoThreshold,
            const T_PointerArray & pointerArray,
            const isaac_float3 & scale,
            const bool & first,
            bool & hit,
            isaac_float4 & hitColor,
            isaac_float3 & hitNormal,
            isaac_float & depth
        ) const
        {
            if( boost::mpl::at_c<
                T_Filter,
                T_NR::value
            >::type::value )
            {
                isaac_float value0 = oldValues[T_NR::value];

                // get value of p1
                isaac_float3 p1Unscaled = p1 / scale;
                checkCoord<T_Source>( p1Unscaled, localSize );
                isaac_float result1 = getValue<
                    T_interpolation,
                    T_NR
                >(
                    source,
                    p1Unscaled,
                    pointerArray,
                    localSize
                );
                ISAAC_IDX_TYPE lookupValue = ISAAC_IDX_TYPE(
                    glm::round( result1 * isaac_float( T_transferSize ) )
                );
                lookupValue = glm::clamp( lookupValue, ISAAC_IDX_TYPE( 0 ), T_transferSize - 1 );
                isaac_float value1 = transferArray.pointer[T_NR::value][lookupValue].a;
                oldValues[T_NR::value] = value1;


                isaac_float isoThreshold = sourceIsoThreshold.value[T_NR::value];
                if( value1 < isoThreshold )
                    return;

                isaac_float testDepth = t0 + (t1 - t0) * ( isoThreshold - value0 ) / ( value1 - value0 );
                testDepth = glm::clamp( testDepth, t0, t1 );

                if( testDepth > depth )
                    return;

                depth = testDepth;
                hit = true;
                isaac_float3 pos = ray.start + ray.dir * depth;
                isaac_float3 posUnscaled = pos / scale;
                checkCoord<T_Source>( posUnscaled, localSize );
                // get color of hit
                isaac_float result = getValue<
                    T_interpolation,
                    T_NR
                >(
                    source,
                    posUnscaled,
                    pointerArray,
                    localSize
                );
                lookupValue = ISAAC_IDX_TYPE(
                    glm::round( result * isaac_float( T_transferSize ) )
                );
                lookupValue = glm::clamp( lookupValue, ISAAC_IDX_TYPE( 0 ), T_transferSize - 1 );
                hitColor = transferArray.pointer[T_NR::value][lookupValue];
                hitColor.a = 1.0f;
                isaac_float3 gradient = 
                getGradient<
                    T_interpolation,
                    T_NR
                >(
                    source,
                    posUnscaled,
                    pointerArray,
                    localSize
                );
                isaac_float gradientLength = glm::length(gradient);
                if( first || gradientLength == isaac_float(0))
                {
                    gradient = ray.clippingNormal;
                    gradientLength = isaac_float( 1 );
                }
                hitNormal = -gradient / gradientLength;
            }
        }
    };

    template<
        typename T_SourceList,
        typename T_TransferArray,
        typename T_IsoThreshold,
        typename T_PointerArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        isaac_int T_interpolation
    >
    struct IsoCellTraversalRenderKernel
    {
        template<
            typename T_Acc
        >
        ISAAC_DEVICE void operator()(
            T_Acc const & acc,
            GBuffer gBuffer,
            const T_SourceList sources,              //source of volumes
            isaac_float stepSize,                       //ray stepSize length
            const T_TransferArray transferArray,     //mapping to simulation memory
            const T_IsoThreshold sourceIsoThreshold,       //weights of sources for blending
            const T_PointerArray pointerArray,
            const isaac_float3 scale,               //isaac set scaling
            const ClippingStruct inputClipping     //clipping planes
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
                sources,
                CheckNoSourceIterator< T_Filter >( ),
                atLeastOne
            );

            if( !atLeastOne )
                return;

            Ray ray = pixelToRay( isaac_float2( pixel ), isaac_float2( gBuffer.size ) );

            if( !clipRay(ray, inputClipping ) )
                return;

            ray.endDepth = glm::min(ray.endDepth, gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x]);

            isaac_float depth = ray.endDepth;

            // get the signs of the direction for the raymarch
            isaac_int3 dirSign = glm::sign( ray.dir );

            // calculate current position in scaled object space
            isaac_float3 currentPos = ray.start + ray.dir * ray.startDepth;

            // calculate current local cell coordinates
            isaac_uint3 currentCell = isaac_uint3( glm::clamp( 
                                    isaac_int3( currentPos / scale ), 
                                    isaac_int3( 0 ), 
                                    isaac_int3( SimulationSize.localSize - ISAAC_IDX_TYPE( 1 ) ) 
                                ) );

            isaac_float testedLength = 0;


            // calculate next intersection with each dimension
            isaac_float3 t = ( ( isaac_float3( currentCell ) + isaac_float3( glm::max( dirSign, 0 ) ) ) 
                    * scale - currentPos ) / ray.dir;

            // calculate delta length to next intersection in the same dimension
            isaac_float3 deltaT = scale / ray.dir * isaac_float3( dirSign );

            isaac_float rayLength = ray.endDepth - ray.startDepth + glm::length( scale );
            // check for 0 to stop infinite looping
            if( ray.dir.x == 0 )
                t.x = std::numeric_limits<isaac_float>::max( );

            if( ray.dir.y == 0 )
                t.y = std::numeric_limits<isaac_float>::max( );

            if( ray.dir.z == 0 )
                t.z = std::numeric_limits<isaac_float>::max( );

            bool hit = false;
            isaac_float3 hitNormal;
            isaac_float4 hitColor;
            // iterate over all cells on the ray path
            // check if the ray leaves the local volume, has a particle hit or exceeds the max ray distance
            isaac_float t0 = ray.startDepth;
            isaac_float oldValues[boost::mpl::size< T_SourceList >::type::value];
            for(int i = 0; i < boost::mpl::size< T_SourceList >::type::value; i++)
                oldValues[i] = 0;
            bool first = true;
            while( hit == false && testedLength <= rayLength )
            {
                isaac_float t1 = ray.startDepth + testedLength;
                isaac_float3 p1 = ray.start + ray.dir * t1;

                // calculate particle intersections for each particle source
                forEachWithMplParams(
                    sources,
                    IsoCellTraversalSourceIterator<
                        T_transferSize,
                        T_Filter,
                        1
                    >( ),
                    ray,
                    t0,
                    oldValues,
                    p1,
                    t1,
                    SimulationSize.localSize,
                    transferArray,
                    sourceIsoThreshold,
                    pointerArray,
                    scale,
                    first,
                    hit,
                    hitColor,
                    hitNormal,
                    depth
                );
                t0 = t1;
                first = false;

                // adds the deltaT value to the smallest dimension t and increment the cell index in the dimension
                if( t.x < t.y && t.x < t.z )
                {
                    testedLength = t.x;
                    t.x += deltaT.x;
                }
                else if( t.y < t.x && t.y < t.z )
                {
                    testedLength = t.y;
                    t.y += deltaT.y;
                }
                else
                {
                    testedLength = t.z;
                    t.z += deltaT.z;
                }
            }

            if( hit )
            {
                setColor ( gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], hitColor );
                gBuffer.normal[pixel.x + pixel.y * gBuffer.size.x] = hitNormal;
                gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = depth;
            }


        }
    };

        template<
        ISAAC_IDX_TYPE T_transferSize,
        typename T_Filter,
        isaac_int T_interpolation
    >
    struct IsoStepSourceIterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_TransferArray,
            typename T_IsoTheshold,
            typename T_PointerArray
        >
        ISAAC_DEVICE_INLINE void operator()(
            const T_NR & nr,
            const T_Source & source,
            const Ray & ray,
            const isaac_float & t,
            const isaac_float3 & pos,
            const isaac_float & stepSize,
            const isaac_size3 & localSize,
            const T_TransferArray & transferArray,
            const T_IsoTheshold & sourceIsoThreshold,
            const T_PointerArray & pointerArray,
            const isaac_float3 & scale,
            const bool & first,
            isaac_float * oldValues,
            bool & hit,
            isaac_float4 & hitColor,
            isaac_float3 & hitNormal,
            isaac_float & depth
        ) const
        {
            if( boost::mpl::at_c<
                T_Filter,
                T_NR::value
            >::type::value )
            {
                isaac_float value = getValue<
                    T_interpolation,
                    T_NR
                >(
                    source,
                    pos,
                    pointerArray,
                    localSize
                );
                ISAAC_IDX_TYPE lookupValue = ISAAC_IDX_TYPE(
                    glm::round( value * isaac_float( T_transferSize ) )
                );
                lookupValue = glm::clamp( lookupValue, ISAAC_IDX_TYPE( 0 ), T_transferSize - 1 );
                value = transferArray.pointer[T_NR::value][lookupValue].a;
                isaac_float prevValue = oldValues[T_NR::value];
                oldValues[T_NR::value] = value;
                isaac_float isoThreshold = sourceIsoThreshold.value[T_NR::value];
                if( value < isoThreshold )
                    return;

                isaac_float testDepth;
                if(first)
                    testDepth = ray.startDepth;
                else
                    testDepth = t + stepSize * ( isoThreshold - prevValue ) / ( value - prevValue );

                if( testDepth > depth )
                    return;

                depth = testDepth;
                hit = true;

                isaac_float3 newPos = ray.start + ray.dir * depth;
                isaac_float3 posUnscaled = newPos / scale;
                checkCoord<T_Source>( posUnscaled, localSize );
                // get color of hit
                isaac_float result = getValue<
                    T_interpolation,
                    T_NR
                >(
                    source,
                    posUnscaled,
                    pointerArray,
                    localSize
                );
                lookupValue = ISAAC_IDX_TYPE(
                    glm::round( result * isaac_float( T_transferSize ) )
                );
                lookupValue = glm::clamp( lookupValue, ISAAC_IDX_TYPE( 0 ), T_transferSize - 1 );
                hitColor = transferArray.pointer[T_NR::value][lookupValue];
                hitColor.a = 1.0f;
                isaac_float3 gradient = 
                getGradient<
                    T_interpolation,
                    T_NR
                >(
                    source,
                    posUnscaled,
                    pointerArray,
                    localSize
                );
                isaac_float gradientLength = glm::length(gradient);
                if( first )
                {
                    gradient = ray.clippingNormal;
                    gradientLength = isaac_float( 1 );
                }
                //gradient *= scale;
                hitNormal = -gradient / gradientLength;
            }
        }
    };

    template<
        typename T_SourceList,
        typename T_TransferArray,
        typename T_IsoTheshold,
        typename T_PointerArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        isaac_int T_interpolation
    >
    struct IsoStepRenderKernel
    {
        template<
            typename T_Acc
        >
        ISAAC_DEVICE void operator()(
            T_Acc const & acc,
            GBuffer gBuffer,
            const T_SourceList sources,              //source of volumes
            isaac_float stepSize,                       //ray stepSize length
            const T_TransferArray transferArray,     //mapping to simulation memory
            const T_IsoTheshold sourceIsoThreshold,       //weights of sources for blending
            const T_PointerArray pointerArray,
            const isaac_float3 scale,               //isaac set scaling
            const ClippingStruct inputClipping     //clipping planes
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

            //set background color
            bool atLeastOne = true;
            forEachWithMplParams(
                sources,
                CheckNoSourceIterator< T_Filter >( ),
                atLeastOne
            );
            if( !atLeastOne )
                return;

            Ray ray = pixelToRay( isaac_float2( pixel ), isaac_float2( gBuffer.size ) );

            if( !clipRay(ray, inputClipping ) )
                return;

            ray.endDepth = glm::min(ray.endDepth, gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x]);
            if( ray.endDepth <= ray.startDepth )
                return;

            //Starting the main loop
            isaac_float min_size = ISAAC_MIN(
                int(
                    SimulationSize.globalSize.x
                ),
                ISAAC_MIN(
                    int(
                        SimulationSize.globalSize.y
                    ),
                    int(
                        SimulationSize.globalSize.z
                    )
                )
            );
            isaac_int startSteps = glm::ceil( ray.startDepth / stepSize );
            isaac_int endSteps = glm::floor( ray.endDepth / stepSize );
            isaac_float3 stepVec =  stepSize * ray.dir / scale;
            //unscale all data for correct memory access
            isaac_float3 startUnscaled = ray.start / scale;

            //move startSteps and endSteps to valid positions in the volume
            isaac_float3 pos = startUnscaled + stepVec * isaac_float( startSteps );
            while( ( !isInLowerBounds( pos, isaac_float3(0) )
                    || !isInUpperBounds( pos, SimulationSize.localSize ) )
                    && startSteps <= endSteps)
            {
                startSteps++;
                pos = startUnscaled + stepVec * isaac_float( startSteps );
            }
            pos = startUnscaled + stepVec * isaac_float( endSteps );
            while( ( !isInLowerBounds( pos, isaac_float3(0) )
                    || !isInUpperBounds( pos, SimulationSize.localSize ) )
                    && startSteps <= endSteps)
            {
                endSteps--;
                pos = startUnscaled + stepVec * isaac_float( endSteps );
            }
            bool hit = false;
            isaac_float depth = ray.endDepth;
            isaac_float4 hitColor = isaac_float4( 0 );
            isaac_float3 hitNormal;
            isaac_float oldValues[boost::mpl::size< T_SourceList >::type::value];
            for(int i = 0; i < boost::mpl::size< T_SourceList >::type::value; i++)
                oldValues[i] = 0;
            //iterate over the volume
            for( isaac_int i = startSteps; i <= endSteps && !hit; i++ )
            {
                pos = startUnscaled + stepVec * isaac_float( i );
                bool first = ray.isClipped && i == startSteps;
                isaac_float t = i * stepSize;
                forEachWithMplParams(
                    sources,
                    IsoStepSourceIterator<
                        T_transferSize,
                        T_Filter,
                        T_interpolation
                    >( ),
                    ray,
                    t,
                    pos,
                    stepSize,
                    SimulationSize.localSize,
                    transferArray,
                    sourceIsoThreshold,
                    pointerArray,
                    scale,
                    first,
                    oldValues,
                    hit,
                    hitColor,
                    hitNormal,
                    depth
                );
            }

            if( hit )
            {   
                setColor ( gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], hitColor );
                gBuffer.normal[pixel.x + pixel.y * gBuffer.size.x] = hitNormal;
                gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = depth;
            }
        }
    };


    template<
        typename T_SourceList,
        typename T_TransferArray,
        typename T_IsoThreshold,
        typename T_PointerArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_WorkDiv,
        typename T_Acc,
        typename T_Stream,
        int T_n
    >
    struct IsoRenderKernelCaller
    {
        inline static void call(
            T_Stream stream,
            const GBuffer & gBuffer,
            const T_SourceList & sources,
            const isaac_float & stepSize,
            const T_TransferArray & transferArray,
            const T_IsoThreshold & sourceIsoThreshold,
            const T_PointerArray & pointerArray,
            const T_WorkDiv & workdiv,
            const isaac_int interpolation,
            const isaac_float3 & scale,
            const ClippingStruct & clipping
        )
        {
            if( sourceIsoThreshold.value[boost::mpl::size< T_SourceList >::type::value
                                   - T_n] == isaac_float( 0 ) )
            {
                IsoRenderKernelCaller<
                    T_SourceList,
                    T_TransferArray,
                    T_IsoThreshold,
                    T_PointerArray,
                    typename boost::mpl::push_back<
                        T_Filter,
                        boost::mpl::false_
                    >::type,
                    T_transferSize,
                    T_WorkDiv,
                    T_Acc,
                    T_Stream,
                    T_n - 1
                >::call(
                    stream,
                    gBuffer,
                    sources,
                    stepSize,
                    transferArray,
                    sourceIsoThreshold,
                    pointerArray,
                    workdiv,
                    interpolation,
                    scale,
                    clipping
                );
            }
            else
            {
                IsoRenderKernelCaller<
                    T_SourceList,
                    T_TransferArray,
                    T_IsoThreshold,
                    T_PointerArray,
                    typename boost::mpl::push_back<
                        T_Filter,
                        boost::mpl::true_
                    >::type,
                    T_transferSize,
                    T_WorkDiv,
                    T_Acc,
                    T_Stream,
                    T_n - 1
                >::call(
                    stream,
                    gBuffer,
                    sources,
                    stepSize,
                    transferArray,
                    sourceIsoThreshold,
                    pointerArray,
                    workdiv,
                    interpolation,
                    scale,
                    clipping
                );
            }
        }
    };

    template<
        typename T_SourceList,
        typename T_TransferArray,
        typename T_IsoThreshold,
        typename T_PointerArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_WorkDiv,
        typename T_Acc,
        typename T_Stream
    >
    struct IsoRenderKernelCaller<
        T_SourceList,
        T_TransferArray,
        T_IsoThreshold,
        T_PointerArray,
        T_Filter,
        T_transferSize,
        T_WorkDiv,
        T_Acc,
        T_Stream,
        0 //<-- spezialisation
    >
    {
        inline static void call(
            T_Stream stream,
            const GBuffer & gBuffer,
            const T_SourceList & sources,
            const isaac_float & stepSize,
            const T_TransferArray & transferArray,
            const T_IsoThreshold & sourceIsoThreshold,
            const T_PointerArray & pointerArray,
            const T_WorkDiv & workdiv,
            const isaac_int interpolation,
            const isaac_float3 & scale,
            const ClippingStruct & clipping
        )
        {
            if( interpolation )
            {
                IsoStepRenderKernel
                <
                    T_SourceList,
                    T_TransferArray,
                    T_IsoThreshold,
                    T_PointerArray,
                    T_Filter,
                    T_transferSize,
                    1
                >
                kernel;
                auto const instance
                (
                    alpaka::createTaskKernel<T_Acc>
                    (
                        workdiv,
                        kernel,
                        gBuffer,
                        sources,
                        stepSize,
                        transferArray,
                        sourceIsoThreshold,
                        pointerArray,
                        scale,
                        clipping
                    )
                );
                alpaka::enqueue(stream, instance);
            }
            else
            {
                IsoStepRenderKernel
                <
                    T_SourceList,
                    T_TransferArray,
                    T_IsoThreshold,
                    T_PointerArray,
                    T_Filter,
                    T_transferSize,
                    0
                >
                kernel;
                auto const instance
                (
                    alpaka::createTaskKernel<T_Acc>
                    (
                        workdiv,
                        kernel,
                        gBuffer,
                        sources,
                        stepSize,
                        transferArray,
                        sourceIsoThreshold,
                        pointerArray,
                        scale,
                        clipping
                    )
                );
                alpaka::enqueue(stream, instance);
            }
        }
    };
}