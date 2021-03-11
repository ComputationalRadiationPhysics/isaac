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

#include "isaac_functor_chain.hpp"
#include "isaac_fusion_extension.hpp"
#include "isaac_macros.hpp"

#include <limits>

namespace isaac
{
    // inverse mvp matrix
    ISAAC_CONSTANT isaac_mat4 InverseMVPMatrix;

    // modelview matrix
    ISAAC_CONSTANT isaac_mat4 ModelViewMatrix;

    // projection matrix
    ISAAC_CONSTANT isaac_mat4 ProjectionMatrix;

    // simulation size properties
    ISAAC_CONSTANT SimulationSizeStruct SimulationSize;

    struct Ray
    {
        isaac_float3 dir;
        isaac_float3 start;
        isaac_float3 end;
        isaac_float startDepth;
        isaac_float endDepth;
        bool isClipped;
        isaac_float3 clippingNormal;
    };

    ISAAC_DEVICE_INLINE Ray pixelToRay(const isaac_float2 pixel, const isaac_float2 framebufferSize)
    {
        // relative pixel position in framebuffer [-1.0 ... 1.0]
        // get normalized pixel position in framebuffer
        isaac_float2 viewportPos
            = isaac_float2(pixel) / isaac_float2(framebufferSize) * isaac_float(2) - isaac_float(1);

        // ray start position
        isaac_float4 startPos;
        startPos.x = viewportPos.x;
        startPos.y = viewportPos.y;
        startPos.z = isaac_float(-1);
        startPos.w = isaac_float(1);

        // ray end position
        isaac_float4 endPos;
        endPos.x = viewportPos.x;
        endPos.y = viewportPos.y;
        endPos.z = isaac_float(1);
        endPos.w = isaac_float(1);

        // apply inverse modelview transform to ray start/end and get ray start/end as worldspace
        startPos = InverseMVPMatrix * startPos;
        endPos = InverseMVPMatrix * endPos;

        Ray ray;
        // apply the w-clip
        ray.start = startPos / startPos.w;
        ray.end = endPos / endPos.w;

        isaac_float maxSize = SimulationSize.maxGlobalSizeScaled * isaac_float(0.5);

        // scale to globale grid size
        ray.start = ray.start * maxSize;
        ray.end = ray.end * maxSize;

        // get step vector
        ray.dir = glm::normalize(ray.end - ray.start);
        ray.isClipped = false;
        ray.clippingNormal = isaac_float3(0);
        return ray;
    }

    ISAAC_DEVICE_INLINE bool clipRay(Ray& ray, const ClippingStruct& inputClipping)
    {
        // clipping planes with transformed positions
        ClippingStruct clipping;
        // set values for clipping planes
        // scale position to global size
        for(isaac_uint i = 0; i < inputClipping.count; i++)
        {
            clipping.elem[i].position
                = inputClipping.elem[i].position * isaac_float3(SimulationSize.globalSizeScaled) * isaac_float(0.5);
            clipping.elem[i].normal = inputClipping.elem[i].normal;
        }

        // move to local (scaled) grid
        // get offset of subvolume in global volume
        isaac_float3 position_offset = isaac_float3(
            isaac_int3(SimulationSize.globalSizeScaled) / 2 - isaac_int3(SimulationSize.positionScaled));

        // apply subvolume offset to start and end
        ray.start = ray.start + position_offset;
        ray.end = ray.end + position_offset;

        // apply subvolume offset to position checked clipping plane
        for(isaac_uint i = 0; i < inputClipping.count; i++)
        {
            clipping.elem[i].position = clipping.elem[i].position + position_offset;
        }

        // clip ray on volume bounding box
        isaac_float3 bbIntersectionMin = -ray.start / ray.dir;
        isaac_float3 bbIntersectionMax = (isaac_float3(SimulationSize.localSizeScaled) - ray.start) / ray.dir;

        // bbIntersectionMin shall have the smaller values
        swapIfSmaller(bbIntersectionMax.x, bbIntersectionMin.x);
        swapIfSmaller(bbIntersectionMax.y, bbIntersectionMin.y);
        swapIfSmaller(bbIntersectionMax.z, bbIntersectionMin.z);

        ray.startDepth = glm::max(bbIntersectionMin.x, glm::max(bbIntersectionMin.y, bbIntersectionMin.z));
        ray.endDepth = glm::min(bbIntersectionMax.x, glm::min(bbIntersectionMax.y, bbIntersectionMax.z));

        // clip on the simulation volume edges for each dimension
        for(int i = 0; i < 3; ++i)
        {
            float sign = glm::sign(ray.dir[i]);
            // only clip if it is an outer edge of the simulation volume
            if(bbIntersectionMin[i] == ray.startDepth
               && ((SimulationSize.position[i] == 0 && sign + 1)
                   || (SimulationSize.position[i] + SimulationSize.localSize[i] == SimulationSize.globalSize[i]
                       && sign - 1)))
            {
                ray.isClipped = true;
                ray.clippingNormal[i] = sign;
            }
        }

        // Iterate over clipping planes and adjust ray start and end depth
        for(isaac_uint i = 0; i < inputClipping.count; i++)
        {
            isaac_float d = glm::dot(ray.dir, clipping.elem[i].normal);

            isaac_float intersectionDepth = (glm::dot(clipping.elem[i].position, clipping.elem[i].normal)
                                             - glm::dot(ray.start, clipping.elem[i].normal))
                / d;
            if(d > 0)
            {
                if(ray.endDepth < intersectionDepth)
                {
                    return false;
                }
                if(ray.startDepth <= intersectionDepth)
                {
                    ray.clippingNormal = clipping.elem[i].normal;
                    ray.isClipped = true;
                    ray.startDepth = intersectionDepth;
                }
            }
            else
            {
                if(ray.startDepth > intersectionDepth)
                {
                    return false;
                }
                if(ray.endDepth > intersectionDepth)
                {
                    ray.endDepth = intersectionDepth;
                }
            }
        }
        ray.startDepth = glm::max(ray.startDepth, isaac_float(ISAAC_Z_NEAR));

        // return if the ray doesn't hit the volume
        if(ray.startDepth > ray.endDepth || isinf(ray.startDepth) || isinf(ray.endDepth))
            return false;

        return true;
    }

    struct ClearBufferKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(T_Acc const& acc, GBuffer gBuffer, isaac_float4 bgColor) const
        {
            isaac_uint2 pixel;
            // get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            pixel.x = isaac_uint(alpThreadIdx[2]);
            pixel.y = isaac_uint(alpThreadIdx[1]);

            pixel = pixel + gBuffer.startOffset;

            if(pixel.x >= gBuffer.size.x || pixel.y >= gBuffer.size.y)
                return;

            bgColor.w = 0;
            setColor(gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], bgColor);
            gBuffer.normal[pixel.x + pixel.y * gBuffer.size.x] = isaac_float3(0, 0, 0);
            gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = std::numeric_limits<isaac_float>::max();
            gBuffer.aoStrength[pixel.x + pixel.y * gBuffer.size.x] = 0;
        }
    };


    struct ShadingKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(
            T_Acc const& acc,
            GBuffer gBuffer,
            const AOParams aoProperties,
            isaac_float4 backgroundColor,
            isaac_int rank,
            isaac_uint mode = 0) const
        {
            isaac_uint2 pixel;
            // get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            pixel.x = isaac_uint(alpThreadIdx[2]);
            pixel.y = isaac_uint(alpThreadIdx[1]);

            pixel = pixel + gBuffer.startOffset;


            if(pixel.x >= gBuffer.size.x || pixel.y >= gBuffer.size.y)
                return;

            isaac_float4 color = getColor(gBuffer.color[pixel.x + pixel.y * gBuffer.size.x]);
            isaac_float3 normal = gBuffer.normal[pixel.x + pixel.y * gBuffer.size.x];
            isaac_float aoStrength = isaac_float(1) - gBuffer.aoStrength[pixel.x + pixel.y * gBuffer.size.x];

            // normal blinn-phong shading
            if(mode < 3)
            {
                Ray ray = pixelToRay(isaac_float2(pixel), isaac_float2(gBuffer.size));
                isaac_float3 lightDir = -ray.dir;
                isaac_float lightFactor = glm::abs(glm::dot(normal, lightDir));

                isaac_float3 halfVector = glm::normalize(-ray.dir + lightDir);

                isaac_float specular = glm::dot(normal, halfVector);

                specular = pow(specular, 4);
                specular *= 0.5f;

                // for disabled specular mode
                if(mode == 1)
                {
                    specular = 0;
                }

                isaac_float weight = aoProperties.weight;
                isaac_float aoFactor = ((1.0f - weight) + weight * aoStrength);
                lightFactor += aoFactor;
                lightFactor *= 0.5f;


                isaac_float3 shadedColor = glm::min(color * lightFactor + specular, isaac_float(1));
                setColor(gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], isaac_float4(shadedColor, color.a));

                // render only solid
                if(mode == 2)
                    gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = isaac_float(0);
            }
            // render only volume
            else if(mode == 3)
            {
                backgroundColor.a = color.a;
                setColor(gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], backgroundColor);
            }
            // normal as color for debug
            else if(mode == 4)
            {
                normal = normal * isaac_float(0.5) + isaac_float(0.5);
                setColor(gBuffer.color[pixel.x + pixel.y * gBuffer.size.x], isaac_float4(normal, color.a));
                gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = isaac_float(0);
            }
            // depth as color for debug
            else if(mode == 5)
            {
                isaac_float depth = gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x]
                    / isaac_float(SimulationSize.maxGlobalSizeScaled);
                setColor(
                    gBuffer.color[pixel.x + pixel.y * gBuffer.size.x],
                    isaac_float4(isaac_float3(depth), color.a));
                gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = isaac_float(0);
            }
            // ambient occlusion as color for debug
            else if(mode == 6)
            {
                isaac_float weight = aoProperties.weight;
                isaac_float aoFactor = ((1.0f - weight) + weight * aoStrength);
                setColor(
                    gBuffer.color[pixel.x + pixel.y * gBuffer.size.x],
                    isaac_float4(isaac_float3(aoFactor), color.a));
                gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = isaac_float(0);
            }
            // rank information color coded for debug
            else if(mode == 7)
            {
                const isaac_float3 colorArray[6]
                    = {isaac_float3(1, 0, 0),
                       isaac_float3(0, 1, 0),
                       isaac_float3(0, 0, 1),
                       isaac_float3(0, 1, 1),
                       isaac_float3(1, 1, 0),
                       isaac_float3(1, 0, 1)};
                setColor(
                    gBuffer.color[pixel.x + pixel.y * gBuffer.size.x],
                    isaac_float4(colorArray[rank % 6], color.a));
                gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = isaac_float(0);
            }
            // full buffer rank information color coded for debug
            else if(mode == 8)
            {
                const isaac_float3 colorArray[6]
                    = {isaac_float3(1, 0, 0),
                       isaac_float3(0, 1, 0),
                       isaac_float3(0, 0, 1),
                       isaac_float3(0, 1, 1),
                       isaac_float3(1, 1, 0),
                       isaac_float3(1, 0, 1)};
                setColor(
                    gBuffer.color[pixel.x + pixel.y * gBuffer.size.x],
                    isaac_float4(colorArray[rank % 6], isaac_float(1)));
                gBuffer.depth[pixel.x + pixel.y * gBuffer.size.x] = isaac_float(0);
            }
        }
    };


    template<typename T_Filter>
    struct CheckNoSourceIterator
    {
        template<typename T_NR, typename T_Source, typename T_Result>
        ISAAC_HOST_DEVICE_INLINE void operator()(const T_NR& nr, const T_Source& source, T_Result& result) const
        {
            result |= boost::mpl::at_c<T_Filter, T_NR::value>::type::value;
        }
    };


    template<typename T_Source>
    struct UpdateBufferKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(
            T_Acc const& acc,
            const T_Source source,
            void* const pointer,
            const isaac_int3 localSize) const
        {
            auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            isaac_int3 dest = {isaac_int(alpThreadIdx[1]), isaac_int(alpThreadIdx[2]), 0};
            isaac_int3 coord = dest;
            coord.x -= ISAAC_GUARD_SIZE;
            coord.y -= ISAAC_GUARD_SIZE;
            if(!isInUpperBounds(dest, localSize + isaac_int3(2 * ISAAC_GUARD_SIZE)))
                return;
            isaac_float_dim<T_Source::featureDim>* ptr = (isaac_float_dim<T_Source::featureDim>*) (pointer);
            if(T_Source::hasGuard)
            {
                coord.z = -ISAAC_GUARD_SIZE;
                for(; dest.z < localSize.z + 2 * ISAAC_GUARD_SIZE; dest.z++)
                {
                    ptr[dest.x + dest.y * (localSize.x + 2 * ISAAC_GUARD_SIZE)
                        + dest.z * ((localSize.x + 2 * ISAAC_GUARD_SIZE) * (localSize.y + 2 * ISAAC_GUARD_SIZE))]
                        = source[coord];
                    coord.z++;
                }
            }
            else
            {
                coord.x = glm::clamp(coord.x, 0, localSize.x - 1);
                coord.y = glm::clamp(coord.y, 0, localSize.y - 1);
                coord.z = 0;
                for(; dest.z < ISAAC_GUARD_SIZE; dest.z++)
                {
                    ptr[dest.x + dest.y * (localSize.x + 2 * ISAAC_GUARD_SIZE)
                        + dest.z * ((localSize.x + 2 * ISAAC_GUARD_SIZE) * (localSize.y + 2 * ISAAC_GUARD_SIZE))]
                        = source[coord];
                }
                for(; dest.z < localSize.z + ISAAC_GUARD_SIZE - 1; dest.z++)
                {
                    ptr[dest.x + dest.y * (localSize.x + 2 * ISAAC_GUARD_SIZE)
                        + dest.z * ((localSize.x + 2 * ISAAC_GUARD_SIZE) * (localSize.y + 2 * ISAAC_GUARD_SIZE))]
                        = source[coord];
                    coord.z++;
                }
                for(; dest.z < localSize.z + 2 * ISAAC_GUARD_SIZE; dest.z++)
                {
                    ptr[dest.x + dest.y * (localSize.x + 2 * ISAAC_GUARD_SIZE)
                        + dest.z * ((localSize.x + 2 * ISAAC_GUARD_SIZE) * (localSize.y + 2 * ISAAC_GUARD_SIZE))]
                        = source[coord];
                }
            }
        }
    };
} // namespace isaac