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
    // iso kernel for the optimized renderer
    template<FilterType T_filterType>
    struct CombinedIsoRenderKernel
    {
        template<typename T_Acc, IndexType T_indexType>
        ISAAC_DEVICE void operator()(
            T_Acc const& acc,
            GBuffer gBuffer,
            Tex3D<isaac_byte4, T_indexType> combinedTexture,
            isaac_float stepSize, // ray stepSize length
            const isaac_float3 scale, // isaac set scaling
            const ClippingStruct inputClipping // clipping planes
        ) const
        {
            // get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            isaac_uint2 pixel = isaac_uint2(alpThreadIdx[2], alpThreadIdx[1]);
            // apply framebuffer offset to pixel
            // stop if pixel position is out of bounds
            pixel = pixel + gBuffer.startOffset;
            if(!isInUpperBounds(pixel, gBuffer.size))
                return;

            Ray ray = pixelToRay(isaac_float2(pixel), isaac_float2(gBuffer.size));

            if(!clipRay(ray, inputClipping))
                return;

            ray.endDepth = glm::min(ray.endDepth, gBuffer.depth[pixel]);
            if(ray.endDepth <= ray.startDepth)
                return;

            // Starting the main loop
            isaac_float min_size = ISAAC_MIN(
                int(SimulationSize.globalSize.x),
                ISAAC_MIN(int(SimulationSize.globalSize.y), int(SimulationSize.globalSize.z)));
            isaac_float stepSizeUnscaled = stepSize * (glm::length(ray.dir) / glm::length(ray.dir / scale));
            isaac_int startSteps = glm::ceil(ray.startDepth / stepSizeUnscaled);
            isaac_int endSteps = glm::floor(ray.endDepth / stepSizeUnscaled);
            isaac_float3 stepVec = stepSizeUnscaled * ray.dir / scale;
            // unscale all data for correct memory access
            isaac_float3 startUnscaled = ray.start / scale;

            bool hit = false;
            isaac_float depth = ray.endDepth;
            isaac_float4 hitColor = isaac_float4(0);
            isaac_float3 hitNormal;

            isaac_float oldValue = 0;

            // iterate over the volume
            for(isaac_int i = startSteps; i <= endSteps && !hit; i++)
            {
                isaac_float3 pos = startUnscaled + stepVec * isaac_float(i);
                bool first = ray.isClipped && i == startSteps;
                isaac_float t = i * stepSizeUnscaled;

                isaac_float value;
                const Sampler<T_filterType, BorderType::CLAMP, true> sampler;
                value = sampler.sample(combinedTexture, pos).a;
                isaac_float tmpValue = oldValue;
                oldValue = value;
                if(value < isaac_float(0.5))
                    continue;

                // only apply interpolation if it was not the first step along the ray
                if(first)
                    depth = ray.startDepth;
                else
                    depth = t + stepSizeUnscaled * (isaac_float(0.5) - tmpValue) / (value - tmpValue);

                hit = true;

                isaac_float3 newPos = ray.start + ray.dir * depth;
                isaac_float3 posUnscaled = newPos / scale;

                hitColor = sampler.sample(combinedTexture, posUnscaled);
                hitColor.a = isaac_float(1);
                // calculate gradient
                isaac_float3 gradient;
                gradient.x = sampler.sample(combinedTexture, posUnscaled + isaac_float3(1, 0, 0)).a
                    - sampler.sample(combinedTexture, posUnscaled - isaac_float3(1, 0, 0)).a;

                gradient.y = sampler.sample(combinedTexture, posUnscaled + isaac_float3(0, 1, 0)).a
                    - sampler.sample(combinedTexture, posUnscaled - isaac_float3(0, 1, 0)).a;

                gradient.z = sampler.sample(combinedTexture, posUnscaled + isaac_float3(0, 0, 1)).a
                    - sampler.sample(combinedTexture, posUnscaled - isaac_float3(0, 0, 1)).a;
                isaac_float gradientLength = glm::length(gradient);
                // normalize gradient
                if(first || gradientLength == isaac_float(0))
                {
                    gradient = ray.clippingNormal;
                    gradientLength = isaac_float(1);
                }
                hitNormal = -gradient / gradientLength;
            }
            if(hit)
            {
                gBuffer.color[pixel] = transformColor(hitColor);
                gBuffer.normal[pixel] = hitNormal;
                gBuffer.depth[pixel] = depth;
            }
        }
    };

// only needed for legacy support
#ifdef ISAAC_RENDERER_LEGACY

    template<
        FilterType T_filterType,
        int T_nr,
        int T_offset,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_Source,
        typename T_TransferArray,
        typename T_PersistentArray>
    class SourceAccessor
    {
    public:
        ISAAC_DEVICE_INLINE SourceAccessor(
            const T_Source& source,
            const T_TransferArray& transferArray,
            const T_PersistentArray& persistentTextureArray,
            const isaac_size3& localSize)
            : source(source)
            , transferArray(transferArray)
            , persistentTextureArray(persistentTextureArray)
            , localSize(localSize)
        {
        }

        ISAAC_DEVICE_INLINE isaac_float4 getValue(const isaac_float3& pos) const
        {
            isaac_float3 coord = pos;
            if(T_Source::persistent)
                checkCoord<T_Source>(coord, localSize);
            return getColorValue<T_filterType, T_nr, T_offset, T_transferSize>(
                source,
                coord,
                transferArray,
                persistentTextureArray,
                localSize);
        }

    private:
        const T_Source& source;
        const T_TransferArray& transferArray;
        const T_PersistentArray& persistentTextureArray;
        const isaac_size3& localSize;
    };

    template<
        FilterType T_filterType,
        int T_nr,
        int T_offset,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_Source,
        typename T_TransferArray,
        typename T_PersistentArray,
        typename T_AdvectionTextureArray>
    class SourceAdvectionBlendedAccessor
    {
    public:
        ISAAC_DEVICE_INLINE SourceAdvectionBlendedAccessor(
            const T_Source& source,
            const T_TransferArray& transferArray,
            const T_PersistentArray& persistentTextureArray,
            const T_AdvectionTextureArray& advectionTextureArray,
            const isaac_size3& localSize)
            : source(source)
            , transferArray(transferArray)
            , persistentTextureArray(persistentTextureArray)
            , advectionTextureArray(advectionTextureArray)
            , localSize(localSize)
        {
        }

        ISAAC_DEVICE_INLINE isaac_float4 getValue(const isaac_float3& pos) const
        {
            isaac_float3 coord = pos;
            if(T_Source::persistent)
                checkCoord<T_Source>(coord, localSize);
            return getAdvectionBlendedColorValue<T_filterType, T_nr, T_offset, T_transferSize>(
                source,
                coord,
                transferArray,
                persistentTextureArray,
                advectionTextureArray,
                localSize);
        }

    private:
        const T_Source& source;
        const T_TransferArray& transferArray;
        const T_PersistentArray& persistentTextureArray;
        const T_AdvectionTextureArray& advectionTextureArray;
        const isaac_size3& localSize;
    };

    template<typename T_Accessor>
    ISAAC_DEVICE_INLINE isaac_float3
    getGradient(const T_Accessor& accessor, const isaac_float3& pos, const isaac_size3& localSize)
    {
        isaac_float3 gradient
            = {accessor.getValue(pos + isaac_float3(1, 0, 0)).a - accessor.getValue(pos - isaac_float3(1, 0, 0)).a,
               accessor.getValue(pos + isaac_float3(0, 1, 0)).a - accessor.getValue(pos - isaac_float3(0, 1, 0)).a,
               accessor.getValue(pos + isaac_float3(0, 0, 1)).a - accessor.getValue(pos - isaac_float3(0, 0, 1)).a};
        return gradient;
    }

    /*
    template<ISAAC_IDX_TYPE T_transferSize, typename T_Filter, isaac_int T_interpolation>
    struct IsoCellTraversalSourceIterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_TransferArray,
            typename T_IsoThreshold,
            typename T_PersistentArray>
        ISAAC_DEVICE_INLINE void operator()(
            const T_NR& nr,
            const T_Source& source,
            const Ray& ray,
            const isaac_float& t0,
            isaac_float* oldValues,
            const isaac_float3& p1,
            const isaac_float& t1,
            const isaac_size3& localSize,
            const T_TransferArray& transferArray,
            const T_IsoThreshold& sourceIsoThreshold,
            const T_PersistentArray& persistentTextureArray,
            const isaac_float3& scale,
            const bool& first,
            bool& hit,
            isaac_float4& hitColor,
            isaac_float3& hitNormal,
            isaac_float& depth) const
        {
            if(boost::mpl::at_c<T_Filter, T_NR::value>::type::value)
            {
                isaac_float value0 = oldValues[T_NR::value];

                // get value of p1
                isaac_float3 p1Unscaled = p1 / scale;
                checkCoord<T_Source>(p1Unscaled, localSize);
                isaac_float result1
                    = getValue<T_interpolation, T_NR::value>(source, p1Unscaled, persistentTextureArray, localSize);
                isaac_float lookupValue = glm::round(result1 * isaac_float(T_transferSize));
                ISAAC_IDX_TYPE lookupIdx = ISAAC_IDX_TYPE(glm::clamp(lookupValue, isaac_float(0),
    isaac_float(T_transferSize - 1))); isaac_float value1 = transferArray.pointer[T_NR::value][lookupIdx].a;
                oldValues[T_NR::value] = value1;


                isaac_float isoThreshold = sourceIsoThreshold.value[T_NR::value];
                if(value1 < isoThreshold)
                    return;

                isaac_float testDepth = t0 + (t1 - t0) * (isoThreshold - value0) / (value1 - value0);
                testDepth = glm::clamp(testDepth, t0, t1);

                if(testDepth > depth)
                    return;

                depth = testDepth;
                hit = true;
                isaac_float3 pos = ray.start + ray.dir * depth;
                isaac_float3 posUnscaled = pos / scale;
                checkCoord<T_Source>(posUnscaled, localSize);
                // get color of hit
                isaac_float result
                    = getValue<T_interpolation, T_NR::value>(source, posUnscaled, persistentTextureArray, localSize);
                lookupValue = glm::round(result * isaac_float(T_transferSize));
                lookupIdx = ISAAC_IDX_TYPE(glm::clamp(lookupValue, isaac_float(0), isaac_float(T_transferSize - 1)));
                hitColor = transferArray.pointer[T_NR::value][lookupIdx];
                hitColor.a = 1.0f;
                isaac_float3 gradient = getGradient<T_interpolation, T_NR::value>(
                    source,
                    posUnscaled,
                    persistentTextureArray,
                    localSize);
                isaac_float gradientLength = glm::length(gradient);
                if(first || gradientLength == isaac_float(0))
                {
                    gradient = ray.clippingNormal;
                    gradientLength = isaac_float(1);
                }
                hitNormal = -gradient / gradientLength;
            }
        }
    };

    template<
        typename T_VolumeSourceList,
        typename T_TransferArray,
        typename T_IsoThreshold,
        typename T_PersistentArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        isaac_int T_interpolation>
    struct IsoCellTraversalRenderKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(
            T_Acc const& acc,
            GBuffer gBuffer,
            const T_VolumeSourceList sources, // source of volumes
            isaac_float stepSize, // ray stepSize length
            const T_TransferArray transferArray, // mapping to simulation memory
            const T_IsoThreshold sourceIsoThreshold, // weights of sources for blending
            const T_PersistentArray persistentTextureArray,
            const isaac_float3 scale, // isaac set scaling
            const ClippingStruct inputClipping // clipping planes
        ) const
        {
            // get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            isaac_uint2 pixel = isaac_uint2(alpThreadIdx[2], alpThreadIdx[1]);
            // apply framebuffer offset to pixel
            // stop if pixel position is out of bounds
            pixel = pixel + gBuffer.startOffset;
            if(!isInUpperBounds(pixel, gBuffer.size))
                return;

            bool atLeastOne = true;
            forEachWithMplParams(sources, CheckNoSourceIterator<T_Filter, 0>(), atLeastOne);

            if(!atLeastOne)
                return;

            Ray ray = pixelToRay(isaac_float2(pixel), isaac_float2(gBuffer.size));

            if(!clipRay(ray, inputClipping))
                return;

            ray.endDepth = glm::min(ray.endDepth, gBuffer.depth[pixel]);

            isaac_float depth = ray.endDepth;

            // get the signs of the direction for the raymarch
            isaac_int3 dirSign = glm::sign(ray.dir);

            // calculate current position in scaled object space
            isaac_float3 currentPos = ray.start + ray.dir * ray.startDepth;

            // calculate current local cell coordinates
            isaac_uint3 currentCell = isaac_uint3(glm::clamp(
                isaac_int3(currentPos / scale),
                isaac_int3(0),
                isaac_int3(SimulationSize.localSize - ISAAC_IDX_TYPE(1))));

            isaac_float testedLength = 0;


            // calculate next intersection with each dimension
            isaac_float3 t
                = ((isaac_float3(currentCell) + isaac_float3(glm::max(dirSign, 0))) * scale - currentPos) / ray.dir;

            // calculate delta length to next intersection in the same dimension
            isaac_float3 deltaT = scale / ray.dir * isaac_float3(dirSign);

            isaac_float rayLength = ray.endDepth - ray.startDepth + glm::length(scale);
            // check for 0 to stop infinite looping
            if(ray.dir.x == 0)
                t.x = std::numeric_limits<isaac_float>::max();

            if(ray.dir.y == 0)
                t.y = std::numeric_limits<isaac_float>::max();

            if(ray.dir.z == 0)
                t.z = std::numeric_limits<isaac_float>::max();

            bool hit = false;
            isaac_float3 hitNormal;
            isaac_float4 hitColor;
            // iterate over all cells on the ray path
            // check if the ray leaves the local volume, has a particle hit or exceeds the max ray distance
            isaac_float t0 = ray.startDepth;
            isaac_float oldValues[boost::mpl::size<T_VolumeSourceList>::type::value];
            for(int i = 0; i < boost::mpl::size<T_VolumeSourceList>::type::value; i++)
                oldValues[i] = 0;
            bool first = true;
            while(hit == false && testedLength <= rayLength)
            {
                isaac_float t1 = ray.startDepth + testedLength;
                isaac_float3 p1 = ray.start + ray.dir * t1;

                // calculate particle intersections for each particle source
                forEachWithMplParams(
                    sources,
                    IsoCellTraversalSourceIterator<T_transferSize, T_Filter, 1>(),
                    ray,
                    t0,
                    oldValues,
                    p1,
                    t1,
                    SimulationSize.localSize,
                    transferArray,
                    sourceIsoThreshold,
                    persistentTextureArray,
                    scale,
                    first,
                    hit,
                    hitColor,
                    hitNormal,
                    depth);
                t0 = t1;
                first = false;

                // adds the deltaT value to the smallest dimension t and increment the cell index in the dimension
                if(t.x < t.y && t.x < t.z)
                {
                    testedLength = t.x;
                    t.x += deltaT.x;
                }
                else if(t.y < t.x && t.y < t.z)
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

            if(hit)
            {
                gBuffer.color[pixel] = transformColor(hitColor);
                gBuffer.normal[pixel] = hitNormal;
                gBuffer.depth[pixel] = depth;
            }
        }
    };
    */


    struct HitInfo
    {
        bool hit = false;
        isaac_float4 color;
        isaac_float3 normal;
        isaac_float depth;
    };

    template<typename T_Accessor, typename T_IsoTheshold>
    ISAAC_DEVICE_INLINE void IsoStepSource(
        const T_Accessor& accessor,
        const Ray& ray,
        const isaac_float& t,
        const isaac_float3& pos,
        const isaac_float& stepSize,
        const isaac_size3& localSize,
        const T_IsoTheshold& sourceIsoThreshold,
        const isaac_float3& scale,
        const bool& first,
        isaac_float* oldValues,
        int index,
        HitInfo& hitInfo)
    {
        isaac_float value = accessor.getValue(pos).a;
        isaac_float prevValue = oldValues[index];
        oldValues[index] = value;
        isaac_float isoThreshold = sourceIsoThreshold.value[index];
        if(value < isoThreshold)
            return;

        isaac_float testDepth;
        if(first)
            testDepth = ray.startDepth;
        else
            testDepth = t + stepSize * (isoThreshold - prevValue) / (value - prevValue);

        if(testDepth > hitInfo.depth)
            return;

        hitInfo.depth = testDepth;
        hitInfo.hit = true;

        isaac_float3 newPos = ray.start + ray.dir * hitInfo.depth;
        isaac_float3 posUnscaled = newPos / scale;
        // get color of hit

        hitInfo.color = accessor.getValue(posUnscaled);
        hitInfo.color.a = 1.0f;
        isaac_float3 gradient = getGradient(accessor, posUnscaled, localSize);
        isaac_float gradientLength = glm::length(gradient);
        if(first)
        {
            gradient = ray.clippingNormal;
            gradientLength = isaac_float(1);
        }
        // gradient *= scale;
        hitInfo.normal = -gradient / gradientLength;
    }

    template<ISAAC_IDX_TYPE T_transferSize, typename T_Filter, FilterType T_filterType, int T_offset = 0>
    struct IsoSourceIterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_TransferArray,
            typename T_IsoTheshold,
            typename T_PersistentArray>
        ISAAC_DEVICE_INLINE void operator()(
            const T_NR& nr,
            const T_Source& source,
            const Ray& ray,
            const isaac_float& t,
            const isaac_float3& pos,
            const isaac_float& stepSize,
            const isaac_size3& localSize,
            const T_TransferArray& transferArray,
            const T_IsoTheshold& sourceIsoThreshold,
            const T_PersistentArray& persistentTextureArray,
            const isaac_float3& scale,
            const bool& first,
            isaac_float* oldValues,
            HitInfo& hitInfo) const
        {
            if(boost::mpl::at_c<T_Filter, T_NR::value + T_offset>::type::value)
            {
                auto sourceAccessor = SourceAccessor<
                    T_filterType,
                    T_NR::value,
                    T_offset,
                    T_transferSize,
                    T_Source,
                    T_TransferArray,
                    T_PersistentArray>(source, transferArray, persistentTextureArray, SimulationSize.localSize);

                IsoStepSource(
                    sourceAccessor,
                    ray,
                    t,
                    pos,
                    stepSize,
                    localSize,
                    sourceIsoThreshold,
                    scale,
                    first,
                    oldValues,
                    T_NR::value + T_offset,
                    hitInfo);
            }
        }
    };

    template<ISAAC_IDX_TYPE T_transferSize, typename T_Filter, FilterType T_filterType, int T_offset = 0>
    struct IsoAdvectionSourceIterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_TransferArray,
            typename T_IsoTheshold,
            typename T_PersistentArray,
            typename T_AdvectionTextureArray>
        ISAAC_DEVICE_INLINE void operator()(
            const T_NR& nr,
            const T_Source& source,
            const Ray& ray,
            const isaac_float& t,
            const isaac_float3& pos,
            const isaac_float& stepSize,
            const isaac_size3& localSize,
            const T_TransferArray& transferArray,
            const T_IsoTheshold& sourceIsoThreshold,
            const T_PersistentArray& persistentTextureArray,
            const T_AdvectionTextureArray& advectionTextureArray,
            const isaac_float3& scale,
            const bool& first,
            isaac_float* oldValues,
            HitInfo& hitInfo) const
        {
            if(boost::mpl::at_c<T_Filter, T_NR::value + T_offset>::type::value)
            {
                auto sourceAccessor = SourceAdvectionBlendedAccessor<
                    T_filterType,
                    T_NR::value,
                    T_offset,
                    T_transferSize,
                    T_Source,
                    T_TransferArray,
                    T_PersistentArray,
                    T_AdvectionTextureArray>(
                    source,
                    transferArray,
                    persistentTextureArray,
                    advectionTextureArray,
                    SimulationSize.localSize);

                IsoStepSource(
                    sourceAccessor,
                    ray,
                    t,
                    pos,
                    stepSize,
                    localSize,
                    sourceIsoThreshold,
                    scale,
                    first,
                    oldValues,
                    T_NR::value + T_offset,
                    hitInfo);
            }
        }
    };

    template<
        typename T_VolumeSourceList,
        typename T_FieldSourceList,
        typename T_TransferArray,
        typename T_IsoTheshold,
        typename T_PersistentArray,
        typename T_AdvectionTextureArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        FilterType T_filterType>
    struct IsoStepRenderKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(
            T_Acc const& acc,
            GBuffer gBuffer,
            const T_VolumeSourceList sources, // source of volumes
            const T_FieldSourceList fieldSources,
            isaac_float stepSize, // ray stepSize length
            const T_TransferArray transferArray, // mapping to simulation memory
            const T_IsoTheshold sourceIsoThreshold, // weights of sources for blending
            const T_PersistentArray persistentTextureArray,
            const T_AdvectionTextureArray advectionTextureArray,
            const isaac_float3 scale, // isaac set scaling
            const ClippingStruct inputClipping // clipping planes
        ) const
        {
            // get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            isaac_uint2 pixel = isaac_uint2(alpThreadIdx[2], alpThreadIdx[1]);
            // apply framebuffer offset to pixel
            // stop if pixel position is out of bounds
            pixel = pixel + gBuffer.startOffset;
            if(!isInUpperBounds(pixel, gBuffer.size))
                return;

            // set background color
            bool atLeastOne = false;
            forEachWithMplParams(sources, CheckNoSourceIterator<T_Filter, 0>(), atLeastOne);
            forEachWithMplParams(
                fieldSources,
                CheckNoSourceIterator<T_Filter, boost::mpl::size<T_VolumeSourceList>::type::value>(),
                atLeastOne);
            if(!atLeastOne)
                return;

            Ray ray = pixelToRay(isaac_float2(pixel), isaac_float2(gBuffer.size));

            if(!clipRay(ray, inputClipping))
                return;

            ray.endDepth = glm::min(ray.endDepth, gBuffer.depth[pixel]);
            if(ray.endDepth <= ray.startDepth)
                return;

            // Starting the main loop
            isaac_float min_size = ISAAC_MIN(
                int(SimulationSize.globalSize.x),
                ISAAC_MIN(int(SimulationSize.globalSize.y), int(SimulationSize.globalSize.z)));
            isaac_float stepSizeUnscaled = stepSize * (glm::length(ray.dir) / glm::length(ray.dir / scale));
            isaac_int startSteps = glm::ceil(ray.startDepth / stepSizeUnscaled);
            isaac_int endSteps = glm::floor(ray.endDepth / stepSizeUnscaled);
            isaac_float3 stepVec = stepSizeUnscaled * ray.dir / scale;
            // unscale all data for correct memory access
            isaac_float3 startUnscaled = ray.start / scale;

            // move startSteps and endSteps to valid positions in the volume
            isaac_float3 pos = startUnscaled + stepVec * isaac_float(startSteps);
            while((!isInLowerBounds(pos, isaac_float3(0)) || !isInUpperBounds(pos, SimulationSize.localSize))
                  && startSteps <= endSteps)
            {
                startSteps++;
                pos = startUnscaled + stepVec * isaac_float(startSteps);
            }
            pos = startUnscaled + stepVec * isaac_float(endSteps);
            while((!isInLowerBounds(pos, isaac_float3(0)) || !isInUpperBounds(pos, SimulationSize.localSize))
                  && startSteps <= endSteps)
            {
                endSteps--;
                pos = startUnscaled + stepVec * isaac_float(endSteps);
            }
            HitInfo hitInfo;
            hitInfo.hit = false;
            hitInfo.depth = ray.endDepth;
            isaac_float oldValues[ZeroCheck<
                boost::mpl::size<T_VolumeSourceList>::type::value
                + boost::mpl::size<T_FieldSourceList>::type::value>::value];
            for(int i = 0; i
                < boost::mpl::size<T_VolumeSourceList>::type::value + boost::mpl::size<T_FieldSourceList>::type::value;
                i++)
                oldValues[i] = 0;
            // iterate over the volume
            for(isaac_int i = startSteps; i <= endSteps && !hitInfo.hit; i++)
            {
                pos = startUnscaled + stepVec * isaac_float(i);
                bool first = ray.isClipped && i == startSteps;
                isaac_float t = i * stepSizeUnscaled;
                forEachWithMplParams(
                    sources,
                    IsoSourceIterator<T_transferSize, T_Filter, T_filterType>(),
                    ray,
                    t,
                    pos,
                    stepSizeUnscaled,
                    SimulationSize.localSize,
                    transferArray,
                    sourceIsoThreshold,
                    persistentTextureArray,
                    scale,
                    first,
                    oldValues,
                    hitInfo);

                forEachWithMplParams(
                    fieldSources,
                    IsoAdvectionSourceIterator<
                        T_transferSize,
                        T_Filter,
                        T_filterType,
                        boost::mpl::size<T_VolumeSourceList>::type::value>(),
                    ray,
                    t,
                    pos,
                    stepSizeUnscaled,
                    SimulationSize.localSize,
                    transferArray,
                    sourceIsoThreshold,
                    persistentTextureArray,
                    advectionTextureArray,
                    scale,
                    first,
                    oldValues,
                    hitInfo);
            }

            if(hitInfo.hit)
            {
                gBuffer.color[pixel] = transformColor(hitInfo.color);
                gBuffer.normal[pixel] = hitInfo.normal;
                gBuffer.depth[pixel] = hitInfo.depth;
            }
        }
    };


    template<
        typename T_VolumeSourceList,
        typename T_FieldSourceList,
        typename T_TransferArray,
        typename T_IsoThreshold,
        typename T_PersistentArray,
        typename T_AdvectionTextureArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_WorkDiv,
        typename T_Acc,
        typename T_Stream,
        int T_n>
    struct IsoRenderKernelCaller
    {
        inline static void call(
            T_Stream stream,
            const GBuffer& gBuffer,
            const T_VolumeSourceList& sources,
            const T_FieldSourceList& fieldSources,
            const isaac_float& stepSize,
            const T_TransferArray& transferArray,
            const T_IsoThreshold& sourceIsoThreshold,
            const T_PersistentArray& persistentTextureArray,
            const T_AdvectionTextureArray& advectionTextureArray,
            const T_WorkDiv& workdiv,
            const isaac_int interpolation,
            const isaac_float3& scale,
            const ClippingStruct& clipping)
        {
            if(sourceIsoThreshold.value
                   [boost::mpl::size<T_VolumeSourceList>::type::value
                    + boost::mpl::size<T_FieldSourceList>::type::value - T_n]
               == isaac_float(0))
            {
                IsoRenderKernelCaller<
                    T_VolumeSourceList,
                    T_FieldSourceList,
                    T_TransferArray,
                    T_IsoThreshold,
                    T_PersistentArray,
                    T_AdvectionTextureArray,
                    typename boost::mpl::push_back<T_Filter, boost::mpl::false_>::type,
                    T_transferSize,
                    T_WorkDiv,
                    T_Acc,
                    T_Stream,
                    T_n - 1>::
                    call(
                        stream,
                        gBuffer,
                        sources,
                        fieldSources,
                        stepSize,
                        transferArray,
                        sourceIsoThreshold,
                        persistentTextureArray,
                        advectionTextureArray,
                        workdiv,
                        interpolation,
                        scale,
                        clipping);
            }
            else
            {
                IsoRenderKernelCaller<
                    T_VolumeSourceList,
                    T_FieldSourceList,
                    T_TransferArray,
                    T_IsoThreshold,
                    T_PersistentArray,
                    T_AdvectionTextureArray,
                    typename boost::mpl::push_back<T_Filter, boost::mpl::true_>::type,
                    T_transferSize,
                    T_WorkDiv,
                    T_Acc,
                    T_Stream,
                    T_n - 1>::
                    call(
                        stream,
                        gBuffer,
                        sources,
                        fieldSources,
                        stepSize,
                        transferArray,
                        sourceIsoThreshold,
                        persistentTextureArray,
                        advectionTextureArray,
                        workdiv,
                        interpolation,
                        scale,
                        clipping);
            }
        }
    };

    template<
        typename T_VolumeSourceList,
        typename T_FieldSourceList,
        typename T_TransferArray,
        typename T_IsoThreshold,
        typename T_PersistentArray,
        typename T_AdvectionTextureArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_WorkDiv,
        typename T_Acc,
        typename T_Stream>
    struct IsoRenderKernelCaller<
        T_VolumeSourceList,
        T_FieldSourceList,
        T_TransferArray,
        T_IsoThreshold,
        T_PersistentArray,
        T_AdvectionTextureArray,
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
            const GBuffer& gBuffer,
            const T_VolumeSourceList& sources,
            const T_FieldSourceList& fieldSources,
            const isaac_float& stepSize,
            const T_TransferArray& transferArray,
            const T_IsoThreshold& sourceIsoThreshold,
            const T_PersistentArray& persistentTextureArray,
            const T_AdvectionTextureArray& advectionTextureArray,
            const T_WorkDiv& workdiv,
            const isaac_int interpolation,
            const isaac_float3& scale,
            const ClippingStruct& clipping)
        {
            if(interpolation)
            {
                IsoStepRenderKernel<
                    T_VolumeSourceList,
                    T_FieldSourceList,
                    T_TransferArray,
                    T_IsoThreshold,
                    T_PersistentArray,
                    T_AdvectionTextureArray,
                    T_Filter,
                    T_transferSize,
                    FilterType::LINEAR>
                    kernel;
                auto const instance = alpaka::createTaskKernel<T_Acc>(
                    workdiv,
                    kernel,
                    gBuffer,
                    sources,
                    fieldSources,
                    stepSize,
                    transferArray,
                    sourceIsoThreshold,
                    persistentTextureArray,
                    advectionTextureArray,
                    scale,
                    clipping);
                alpaka::enqueue(stream, instance);
            }
            else
            {
                IsoStepRenderKernel<
                    T_VolumeSourceList,
                    T_FieldSourceList,
                    T_TransferArray,
                    T_IsoThreshold,
                    T_PersistentArray,
                    T_AdvectionTextureArray,
                    T_Filter,
                    T_transferSize,
                    FilterType::NEAREST>
                    kernel;
                auto const instance = alpaka::createTaskKernel<T_Acc>(
                    workdiv,
                    kernel,
                    gBuffer,
                    sources,
                    fieldSources,
                    stepSize,
                    transferArray,
                    sourceIsoThreshold,
                    persistentTextureArray,
                    advectionTextureArray,
                    scale,
                    clipping);
                alpaka::enqueue(stream, instance);
            }
        }
    };

#endif

} // namespace isaac