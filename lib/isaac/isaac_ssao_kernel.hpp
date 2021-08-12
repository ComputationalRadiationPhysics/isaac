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
    /*
     * SSAO
     * Kernels for ssao calculation
     */

    // filter kernel
    ISAAC_CONSTANT isaac_float3 SSAOKernelArray[64];

    // vector rotation noise kernel
    ISAAC_CONSTANT isaac_float3 SSAONoiseArray[16];

    /**
     * @brief Calculate SSAO factor
     *
     *
     */
    struct SSAOKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(
            T_Acc const& acc,
            GBuffer gBuffer,
            AOParams aoProperties // properties for ambient occlusion
        ) const
        {
            isaac_uint2 pixel;
            // get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            pixel.x = isaac_uint(alpThreadIdx[2]);
            pixel.y = isaac_uint(alpThreadIdx[1]);

            pixel = pixel + gBuffer.startOffset;
            if(!isInUpperBounds(pixel, gBuffer.size))
                return;


            /*
             * TODO:
             *
             * Old standart ssao by crytech
             *
             * First implemntation failed and the source code is below
             * Possible errors could be mv or proj matrix
             */

            // search radius for depth testing
            isaac_int radius = 10;

            /*
            //isaac_float3 origin = gBuffer.depth[pixel];



            //get the normal value from the gbuffer
            isaac_float3 normal = gNormal[pixel];

            //normalize the normal
            isaac_float len = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
            if(len == 0) {
                gBuffer.aoStrength[pixel] = 0.0f;
                return;
            }

            normal = normal / len;



            isaac_float3 rvec = {0.7f, 0.1f, 0.3f};
            isaac_float3 tangent = rvec - normal * (rvec.x * normal.x + rvec.y * normal.y + rvec.z * normal.z);
            len = sqrt(tangent.x * tangent.x + tangent.y * tangent.y + tangent.z * tangent.z);
            tangent = tangent / len;
            isaac_float3 bitangent = {
                normal.y * tangent.z - normal.z * tangent.y,
                normal.z * tangent.x - normal.x * tangent.z,
                normal.x * tangent.y - normal.y * tangent.y
            };

            isaac_float tbn[9];
            tbn[0] = tangent.x;
            tbn[1] = tangent.y;
            tbn[2] = tangent.z;

            tbn[3] = bitangent.x;
            tbn[4] = bitangent.y;
            tbn[5] = bitangent.z;

            tbn[6] = normal.x;
            tbn[7] = normal.y;
            tbn[8] = normal.z;

            isaac_float occlusion = 0.0f;
            for(int i = 0; i < 1; i++) {
                //sample = tbn * sample_kernel
                isaac_float3 sample = {
                    tbn[0] * SSAOKernel[i].x + tbn[3] * SSAOKernel[i].y + tbn[6] * SSAOKernel[i].z,
                    tbn[1] * SSAOKernel[i].x + tbn[4] * SSAOKernel[i].y + tbn[7] * SSAOKernel[i].z,
                    tbn[2] * SSAOKernel[i].x + tbn[5] * SSAOKernel[i].y + tbn[8] * SSAOKernel[i].z,
                };

                sample = sample * radius + origin;

                isaac_float4 offset = {
                    sample.x,
                    sample.y,
                    sample.z,
                    1.0
                };

                //offset = projection * offset
                offset = isaac_float4({
                    ProjectionMatrix[0] * offset.x + ProjectionMatrix[4] * offset.y + ProjectionMatrix[8 ] * offset.z +
            ProjectionMatrix[12] * offset.w, ProjectionMatrix[1] * offset.x + ProjectionMatrix[5] * offset.y +
            ProjectionMatrix[9 ] * offset.z + ProjectionMatrix[13] * offset.w, ProjectionMatrix[2] * offset.x +
            ProjectionMatrix[6] * offset.y + ProjectionMatrix[10] * offset.z + ProjectionMatrix[14] * offset.w,
                    ProjectionMatrix[3] * offset.x + ProjectionMatrix[7] * offset.y + ProjectionMatrix[11] * offset.z +
            ProjectionMatrix[15] * offset.w
                });

                isaac_float2 offset2d = isaac_float2({offset.x / offset.w, offset.y / offset.w});
                offset2d.x = MAX(MIN(offset2d.x * 0.5 + 0.5, 1.0f), 0.0f);
                offset2d.y = MAX(MIN(offset2d.y * 0.5 + 0.5, 1.0f), 0.0f);

                isaac_uint2 offsetFramePos = {
                    isaac_uint(gBuffer.size.x * offset2d.x) + gBuffer.startOffset.x,
                    isaac_uint(gBuffer.size.y * offset2d.y) + gBuffer.startOffset.y,
                };
                //printf("%f %f -- %u %u\n", offset2d.x, offset2d.y, offsetFramePos.x, offsetFramePos.y);
                isaac_float sampleDepth = gBuffer.depth[offsetFramePos.x + offsetFramePos.y * gBuffer.size.x].z;
                occlusion += (sampleDepth - sample.z ? 1.0f : 0.0f);
            }*/


            /*
             * 1. compare all neighbour (+-2 pixel) depth values with the current one and increase the counter if the
             * neighbour is closer to the camera
             *
             * 2. get average value by dividing the counter by the cell count (7x7=49)       *
             *
             */
            // closer to the camera
            isaac_float occlusion = 0.0f;
            isaac_float refDepth = gBuffer.depth[pixel];
            const Sampler<FilterType::NEAREST, BorderType::CLAMP> sampler;
            for(int i = -3; i <= 3; ++i)
            {
                for(int j = -3; j <= 3; ++j)
                {
                    // get the neighbour depth value
                    isaac_float depthSample
                        = sampler.safeMemoryAccess(gBuffer.depth, isaac_int2(pixel) + isaac_int2(i, j) * radius);

                    if(depthSample < refDepth)
                    {
                        occlusion += 1.0f;
                    }
                }
            }
            isaac_float depth = glm::clamp((occlusion / 42.0f), 0.0f, 1.0f);

            // save the depth value in our ao buffer
            gBuffer.aoStrength[pixel] = depth;
        }
    };

    /**
     * @brief Filter SSAO artifacts and return the color with depth simulation
     *
     *
     */
    struct SSAOFilterKernel
    {
        template<typename T_Acc>
        ISAAC_DEVICE void operator()(
            T_Acc const& acc,
            GBuffer gBuffer,
            AOParams aoProperties // properties for ambient occlusion
        ) const
        {
            isaac_uint2 pixel;
            // get pixel values from thread ids
            auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            pixel.x = isaac_uint(alpThreadIdx[2]);
            pixel.y = isaac_uint(alpThreadIdx[1]);

            // get real pixel coordinate by offset
            pixel = pixel + gBuffer.startOffset;
            if(!isInUpperBounds(pixel, gBuffer.size))
                return;
            /* TODO
             * Normally the depth values are smoothed
             * in this case the smooting filter is not applied for simplicity
             *
             * If the real ssao algorithm is implemented, a real filter will be necessary
             */
            isaac_float depth = gBuffer.aoStrength[pixel];

            isaac_float4 colorValues = transformColor(gBuffer.color[pixel]);

            // read the weight from the global ao settings and merge them with the color value
            isaac_float weight = aoProperties.weight;
            isaac_float aoFactor = ((1.0f - weight) + weight * (1.0f - depth));

            isaac_float4 finalColor
                = {aoFactor * colorValues.x, aoFactor * colorValues.y, aoFactor * colorValues.z, colorValues.w};

            // finally replace the old color value with the new ssao filtered color value
            gBuffer.color[pixel] = transformColor(finalColor);
        }
    };
} // namespace isaac