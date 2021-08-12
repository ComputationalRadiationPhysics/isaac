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
    template<FilterType T_filterType, int T_sourceCount>
    struct CombinedVolumeRenderKernel
    {
        template<typename T_Acc, IndexType T_indexType>
        ISAAC_DEVICE void operator()(
            T_Acc const& acc,
            GBuffer gBuffer,
            const Tex3D<isaac_byte4, T_indexType> combinedTexture,
            const isaac_float stepSize, // ray stepSize length
            const isaac_float totalWeight,
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
            isaac_float factor = stepSizeUnscaled / min_size * 2.0f * isaac_float(totalWeight);
            isaac_int startSteps = glm::ceil(ray.startDepth / stepSizeUnscaled);
            isaac_int endSteps = glm::floor(ray.endDepth / stepSizeUnscaled);
            isaac_float3 stepVec = stepSizeUnscaled * ray.dir / scale;
            // unscale all data for correct memory access
            isaac_float3 startUnscaled = ray.start / scale;

            isaac_float4 color = isaac_float4(0);
            // iterate over the volume
            for(isaac_int i = startSteps; i <= endSteps; i++)
            {
                isaac_float3 pos = startUnscaled + stepVec * isaac_float(i);
                isaac_float4 value;
                const Sampler<T_filterType, BorderType::CLAMP, true> sampler;
                value = sampler.sample(combinedTexture, pos);
                value *= factor;
                isaac_float weight = isaac_float(1) - color.w;
                color += weight * value;
                if(color.w > isaac_float(0.99))
                {
                    break;
                }
            }
            // Blend solid color and new volume color
            isaac_float4 solidColor = transformColor(gBuffer.color[pixel]);
            color = color + (1 - color.w) * solidColor;
            gBuffer.color[pixel] = transformColor(color);
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
        typename T_PointerArray>
    ISAAC_DEVICE_INLINE isaac_float4 getColorValue(
        const T_Source& source,
        const isaac_float3& pos,
        const T_TransferArray& transferArray,
        const T_PointerArray& persistentArray,
        const isaac_size3& localSize)
    {
        isaac_float_dim<T_Source::featureDim> data;
        isaac_float result = isaac_float(0);
        if(T_Source::persistent)
        {
            if(T_filterType == FilterType::NEAREST)
            {
                isaac_int3 coord = pos;

                data = source[coord];
            }
            else
            {
                isaac_float3 offsetPos = pos - isaac_float(0.5);
                isaac_int3 coord;
                isaac_float_dim<T_Source::featureDim> data8[2][2][2];
                for(int x = 0; x < 2; x++)
                {
                    for(int y = 0; y < 2; y++)
                    {
                        for(int z = 0; z < 2; z++)
                        {
                            coord.x = isaac_int(offsetPos.x) + x;
                            coord.y = isaac_int(offsetPos.y) + y;
                            coord.z = isaac_int(offsetPos.z) + z;
                            if(T_Source::guardSize < 1)
                            {
                                if(isaac_uint(coord.x) >= localSize.x)
                                {
                                    coord.x = isaac_int(offsetPos.x) + 1 - x;
                                }
                                if(isaac_uint(coord.y) >= localSize.y)
                                {
                                    coord.y = isaac_int(offsetPos.y) + 1 - y;
                                }
                                if(isaac_uint(coord.z) >= localSize.z)
                                {
                                    coord.z = isaac_int(offsetPos.z) + 1 - z;
                                }
                            }
                            data8[x][y][z] = source[coord];
                        }
                    }
                }

                data = trilinear(glm::fract(offsetPos), data8);
            }

            result = applyFunctorChain(data, T_nr + T_offset);
        }
        else
        {
            Tex3D<isaac_float> texture = persistentArray.textures[T_nr + T_offset];
            const Sampler<T_filterType, BorderType::CLAMP> sampler;
            result = sampler.sample(texture, pos);
        }
        // return isaac_float4(1, 0, 0, result);
        ISAAC_IDX_TYPE lookupValue = ISAAC_IDX_TYPE(glm::round(result * isaac_float(T_transferSize)));
        lookupValue = glm::clamp(lookupValue, ISAAC_IDX_TYPE(0), T_transferSize - 1);
        return transferArray.pointer[T_nr + T_offset][lookupValue];
    }

    template<
        FilterType T_filterType,
        int T_nr,
        int T_offset,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_Source,
        typename T_TransferArray,
        typename T_PointerArray,
        typename T_AdvectionTextureArray>
    ISAAC_DEVICE_INLINE isaac_float4 getAdvectionBlendedColorValue(
        const T_Source& source,
        const isaac_float3& pos,
        const T_TransferArray& transferArray,
        const T_PointerArray& persistentArray,
        const T_AdvectionTextureArray& advectionTextureArray,
        const isaac_size3& localSize)
    {
        isaac_float4 result = getColorValue<T_filterType, T_nr, T_offset, T_transferSize>(
            source,
            pos,
            transferArray,
            persistentArray,
            localSize);
        const Sampler<T_filterType, BorderType::CLAMP> sampler;
        result.a *= (sampler.sample(advectionTextureArray.textures[T_nr], pos) / isaac_float(255));
        return result;
    }

    /**
     * @brief Clamps coordinates to min/max
     *
     * @tparam T_interpolation
     * @param coord
     * @param localSize
     * @return check_coord clamped coordiantes
     */
    template<typename T_Source>
    ISAAC_HOST_DEVICE_INLINE void checkCoord(isaac_float3& coord, const isaac_size3& localSize)
    {
        coord = glm::clamp(
            coord,
            isaac_float3(-T_Source::guardSize),
            isaac_float3(localSize + T_Source::guardSize - ISAAC_IDX_TYPE(1))
                - std::numeric_limits<isaac_float>::min());
    }

    template<ISAAC_IDX_TYPE T_transferSize, typename T_Filter, FilterType T_filterType, int T_offset = 0>
    struct MergeVolumeSourceIterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_TransferArray,
            typename T_SourceWeight,
            typename T_PointerArray>
        ISAAC_DEVICE_INLINE void operator()(
            const T_NR& nr,
            const T_Source& source,
            const isaac_float3& pos,
            const isaac_size3& localSize,
            const T_TransferArray& transferArray,
            const T_SourceWeight& sourceWeight,
            const T_PointerArray& persistentArray,
            isaac_float4& color) const
        {
            if(boost::mpl::at_c<T_Filter, T_NR::value + T_offset>::type::value)
            {
                isaac_float4 value = getColorValue<T_filterType, T_NR::value, T_offset, T_transferSize>(
                    source,
                    pos,
                    transferArray,
                    persistentArray,
                    localSize);
                value.a *= sourceWeight.value[T_NR::value + T_offset];
                color.r += value.r * value.a;
                color.g += value.g * value.a;
                color.b += value.b * value.a;
                color.a += value.a;
            }
        }
    };

    template<ISAAC_IDX_TYPE T_transferSize, typename T_Filter, FilterType T_filterType, int T_offset = 0>
    struct MergeVolumeAdvectionSourceIterator
    {
        template<
            typename T_NR,
            typename T_Source,
            typename T_TransferArray,
            typename T_SourceWeight,
            typename T_PointerArray,
            typename T_AdvectionTextureArray>
        ISAAC_DEVICE_INLINE void operator()(
            const T_NR& nr,
            const T_Source& source,
            const isaac_float3& pos,
            const isaac_size3& localSize,
            const T_TransferArray& transferArray,
            const T_SourceWeight& sourceWeight,
            const T_PointerArray& persistentArray,
            const T_AdvectionTextureArray& advectionTextureArray,
            isaac_float4& color) const
        {
            if(boost::mpl::at_c<T_Filter, T_NR::value + T_offset>::type::value)
            {
                isaac_float4 value
                    = getAdvectionBlendedColorValue<T_filterType, T_NR::value, T_offset, T_transferSize>(
                        source,
                        pos,
                        transferArray,
                        persistentArray,
                        advectionTextureArray,
                        localSize);
                value.a *= sourceWeight.value[T_NR::value + T_offset];
                color.r += value.r * value.a;
                color.g += value.g * value.a;
                color.b += value.b * value.a;
                color.a += value.a;
            }
        }
    };

    template<
        typename T_VolumeSourceList,
        typename T_FieldSourceList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_PointerArray,
        typename T_AdvectionTextureArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        FilterType T_filterType>
    struct VolumeRenderKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(
            T_Acc const& acc,
            GBuffer gBuffer,
            const T_VolumeSourceList sources, // source of volumes
            const T_FieldSourceList fieldSources,
            isaac_float stepSize, // ray stepSize length
            const T_TransferArray transferArray, // mapping to simulation memory
            const T_SourceWeight sourceWeight, // weights of sources for blending
            const T_PointerArray persistentArray,
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
            isaac_float factor = stepSizeUnscaled / min_size * 2.0f;
            isaac_float4 value = isaac_float4(0);
            isaac_float oma;
            isaac_float4 colorAdd;
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
            isaac_float4 color = isaac_float4(0);
            // iterate over the volume
            for(isaac_int i = startSteps; i <= endSteps; i++)
            {
                pos = startUnscaled + stepVec * isaac_float(i);
                forEachWithMplParams(
                    sources,
                    MergeVolumeSourceIterator<T_transferSize, T_Filter, T_filterType>(),
                    pos,
                    SimulationSize.localSize,
                    transferArray,
                    sourceWeight,
                    persistentArray,
                    value);
                forEachWithMplParams(
                    fieldSources,
                    MergeVolumeAdvectionSourceIterator<
                        T_transferSize,
                        T_Filter,
                        T_filterType,
                        boost::mpl::size<T_VolumeSourceList>::type::value>(),
                    pos,
                    SimulationSize.localSize,
                    transferArray,
                    sourceWeight,
                    persistentArray,
                    advectionTextureArray,
                    value);
                oma = isaac_float(1) - color.w;
                value *= factor;
                colorAdd = oma * value;
                color += colorAdd;
                if(color.w > isaac_float(0.99))
                {
                    break;
                }
            }

#    if ISAAC_SHOWBORDER == 1
            if(color.w <= isaac_float(0.99))
            {
                oma = isaac_float(1) - color.w;
                colorAdd.x = 0;
                colorAdd.y = 0;
                colorAdd.z = 0;
                colorAdd.w = oma * factor * isaac_float(10);
                color += colorAdd;
            }
#    endif

            // Blend solid color and new volume color
            isaac_float4 solidColor = transformColor(gBuffer.color[pixel]);
            color = color + (1 - color.w) * solidColor;
            gBuffer.color[pixel] = transformColor(color);
        }
    };


    template<
        typename T_VolumeSourceList,
        typename T_FieldSourceList,
        typename T_TransferArray,
        typename T_SourceWeight,
        typename T_PointerArray,
        typename T_AdvectionTextureArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_WorkDiv,
        typename T_Acc,
        typename T_Stream,
        int T_n>
    struct VolumeRenderKernelCaller
    {
        inline static void call(
            T_Stream stream,
            const GBuffer& gBuffer,
            const T_VolumeSourceList& sources,
            const T_FieldSourceList& fieldSources,
            const isaac_float& stepSize,
            const T_TransferArray& transferArray,
            const T_SourceWeight& sourceWeight,
            const T_PointerArray& persistentArray,
            const T_AdvectionTextureArray& advectionTextureArray,
            const T_WorkDiv& workdiv,
            const isaac_int interpolation,
            const isaac_float3& scale,
            const ClippingStruct& clipping)
        {
            if(sourceWeight.value
                   [boost::mpl::size<T_VolumeSourceList>::type::value
                    + boost::mpl::size<T_FieldSourceList>::type::value - T_n]
               == isaac_float(0))
            {
                VolumeRenderKernelCaller<
                    T_VolumeSourceList,
                    T_FieldSourceList,
                    T_TransferArray,
                    T_SourceWeight,
                    T_PointerArray,
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
                        sourceWeight,
                        persistentArray,
                        advectionTextureArray,
                        workdiv,
                        interpolation,
                        scale,
                        clipping);
            }
            else
            {
                VolumeRenderKernelCaller<
                    T_VolumeSourceList,
                    T_FieldSourceList,
                    T_TransferArray,
                    T_SourceWeight,
                    T_PointerArray,
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
                        sourceWeight,
                        persistentArray,
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
        typename T_SourceWeight,
        typename T_PointerArray,
        typename T_AdvectionTextureArray,
        typename T_Filter,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_WorkDiv,
        typename T_Acc,
        typename T_Stream>
    struct VolumeRenderKernelCaller<
        T_VolumeSourceList,
        T_FieldSourceList,
        T_TransferArray,
        T_SourceWeight,
        T_PointerArray,
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
            const T_SourceWeight& sourceWeight,
            const T_PointerArray& persistentArray,
            const T_AdvectionTextureArray& advectionTextureArray,
            const T_WorkDiv& workdiv,
            const isaac_int interpolation,
            const isaac_float3& scale,
            const ClippingStruct& clipping)
        {
            if(interpolation)
            {
                VolumeRenderKernel<
                    T_VolumeSourceList,
                    T_FieldSourceList,
                    T_TransferArray,
                    T_SourceWeight,
                    T_PointerArray,
                    T_AdvectionTextureArray,
                    T_Filter,
                    T_transferSize,
                    FilterType::LINEAR>
                    kernel;
                auto const instance(alpaka::createTaskKernel<T_Acc>(
                    workdiv,
                    kernel,
                    gBuffer,
                    sources,
                    fieldSources,
                    stepSize,
                    transferArray,
                    sourceWeight,
                    persistentArray,
                    advectionTextureArray,
                    scale,
                    clipping));
                alpaka::enqueue(stream, instance);
            }
            else
            {
                VolumeRenderKernel<
                    T_VolumeSourceList,
                    T_FieldSourceList,
                    T_TransferArray,
                    T_SourceWeight,
                    T_PointerArray,
                    T_AdvectionTextureArray,
                    T_Filter,
                    T_transferSize,
                    FilterType::NEAREST>
                    kernel;
                auto const instance(alpaka::createTaskKernel<T_Acc>(
                    workdiv,
                    kernel,
                    gBuffer,
                    sources,
                    fieldSources,
                    stepSize,
                    transferArray,
                    sourceWeight,
                    persistentArray,
                    advectionTextureArray,
                    scale,
                    clipping));
                alpaka::enqueue(stream, instance);
            }
        }
    };

#endif
} // namespace isaac