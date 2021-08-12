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

#include "isaac_types.hpp"

namespace isaac
{
    // Morton code seperation of bits by Fabian "ryg" Giesen
    // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/

    ISAAC_HOST_DEVICE_INLINE uint32_t part1By1(uint32_t x)
    {
        x &= 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
        x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
        x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
        x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
        x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
        return x;
    }

    ISAAC_HOST_DEVICE_INLINE uint32_t part1By2(uint32_t x)
    {
        x &= 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
        x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x ^ (x << 8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x ^ (x << 4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x ^ (x << 2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        return x;
    }

    //--------------------------------------------------------------------------------------------------------------

    uint64_t getTicksUs()
    {
        struct timespec ts;
        if(clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0)
        {
            return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
        }
        return 0;
    }

    ISAAC_HOST_DEVICE_INLINE isaac_float halton(isaac_uint index, isaac_uint base)
    {
        isaac_float result = 0;
        isaac_float f = 1;

        while(index > 0)
        {
            f /= isaac_float(base);
            result += f * (index % base);
            index /= base;
        }
        return result;
    }

    ISAAC_HOST_DEVICE_INLINE isaac_byte4 transformColor(const isaac_float4& floatColor)
    {
        return isaac_byte4(clamp(floatColor, isaac_float(0), isaac_float(1)) * isaac_float(255));
    }

    ISAAC_HOST_DEVICE_INLINE isaac_float4 transformColor(const isaac_byte4& byteColor)
    {
        return isaac_float4(byteColor) / isaac_float(255);
    }

    template<typename T_Type>
    ISAAC_HOST_DEVICE_INLINE void swapIfSmaller(T_Type& left, T_Type& right)
    {
        if(left < right)
        {
            auto temp = left;
            left = right;
            right = temp;
        }
    }

    template<typename T_Type>
    ISAAC_HOST_DEVICE_INLINE T_Type linear(const isaac_float& innerOffset, const T_Type& v1, const T_Type& v2)
    {
        return v1 * (1 - innerOffset) + v2 * innerOffset;
    }

    template<typename T_Type>
    ISAAC_HOST_DEVICE_INLINE T_Type
    bilinear(const isaac_float2& innerOffset, const T_Type& bl, const T_Type& br, const T_Type& tl, const T_Type& tr)
    {
        const T_Type& bottom = linear(innerOffset.x, bl, br);
        const T_Type& top = linear(innerOffset.x, tl, tr);
        return linear(innerOffset.y, bottom, top);
    }

    template<typename T_Type>
    ISAAC_HOST_DEVICE_INLINE T_Type trilinear(
        const isaac_float3& innerOffset,
        const T_Type& fbl,
        const T_Type& fbr,
        const T_Type& ftl,
        const T_Type& ftr,
        const T_Type& bbl,
        const T_Type& bbr,
        const T_Type& btl,
        const T_Type& btr)
    {
        const T_Type& front = bilinear(isaac_float2(innerOffset), fbl, fbr, ftl, ftr);
        const T_Type& back = bilinear(isaac_float2(innerOffset), bbl, bbr, btl, btr);
        return linear(innerOffset.z, front, back);
    }

    template<typename T_Type>
    ISAAC_HOST_DEVICE_INLINE T_Type linear(const isaac_float& innerOffset, const T_Type (&values)[2])
    {
        return values[0] * (1 - innerOffset) + values[1] * innerOffset;
    }

    template<typename T_Type>
    ISAAC_HOST_DEVICE_INLINE T_Type bilinear(const isaac_float2& innerOffset, const T_Type (&values)[2][2])
    {
        T_Type a = linear(innerOffset.y, values[0]);
        T_Type b = linear(innerOffset.y, values[1]);
        return linear(innerOffset.x, a, b);
    }

    template<typename T_Type>
    ISAAC_HOST_DEVICE_INLINE T_Type trilinear(const isaac_float3& innerOffset, const T_Type (&values)[2][2][2])
    {
        T_Type a = bilinear(isaac_float2(innerOffset.y, innerOffset.z), values[0]);
        T_Type b = bilinear(isaac_float2(innerOffset.y, innerOffset.z), values[1]);
        return linear(innerOffset.x, a, b);
    }


    template<int T_n, typename T_Type1, typename T_Type2>
    ISAAC_HOST_DEVICE_INLINE bool isInLowerBounds(
        const isaac_vec_dim<T_n, T_Type1>& vec,
        const isaac_vec_dim<T_n, T_Type2>& lBounds)
    {
        for(int i = 0; i < T_n; ++i)
        {
            if(vec[i] < lBounds[i])
                return false;
        }
        return true;
    }

    template<int T_n, typename T_Type1, typename T_Type2>
    ISAAC_HOST_DEVICE_INLINE bool isInUpperBounds(
        const isaac_vec_dim<T_n, T_Type1>& vec,
        const isaac_vec_dim<T_n, T_Type2>& uBounds)
    {
        for(int i = 0; i < T_n; ++i)
        {
            if(vec[i] >= uBounds[i])
                return false;
        }
        return true;
    }

    void mergeJSON(json_t* result, json_t* candidate)
    {
        const char* cKey;
        const char* rKey;
        json_t* cValue;
        json_t* rValue;
        // metadata merge, old values stay, arrays are merged
        json_t* mCandidate = json_object_get(candidate, "metadata");
        json_t* mResult = json_object_get(result, "metadata");
        void *temp, *temp2;
        if(mCandidate && mResult)
        {
            json_object_foreach_safe(mCandidate, temp, cKey, cValue)
            {
                bool found_array = false;
                json_object_foreach_safe(mResult, temp2, rKey, rValue)
                {
                    if(strcmp(rKey, cKey) == 0)
                    {
                        if(json_is_array(rValue) && json_is_array(cValue))
                        {
                            json_array_extend(rValue, cValue);
                            found_array = true;
                        }
                        break;
                    }
                }
                if(!found_array)
                    json_object_set(mResult, cKey, cValue);
            }
        }
        // general merge, new values stay
        json_object_foreach_safe(candidate, temp, cKey, cValue)
        {
            bool foundMeta = false;
            json_object_foreach_safe(result, temp2, rKey, rValue)
            {
                if(strcmp(rKey, cKey) == 0 && strcmp(rKey, "metadata") == 0)
                {
                    foundMeta = true;
                    break;
                }
            }
            if(!foundMeta)
                json_object_set(result, cKey, cValue);
        }
    }

    ISAAC_HOST_DEVICE_INLINE isaac_float4 getHSVA(isaac_float h, isaac_float s, isaac_float v, isaac_float a)
    {
        isaac_int hi = isaac_int(floor(h / (M_PI / 3)));
        isaac_float f = h / (M_PI / 3) - isaac_float(hi);
        isaac_float p = v * (isaac_float(1) - s);
        isaac_float q = v * (isaac_float(1) - s * f);
        isaac_float t = v * (isaac_float(1) - s * (isaac_float(1) - f));
        isaac_float4 result = {0, 0, 0, a};
        switch(hi)
        {
        case 0:
        case 6:
            result.x = v;
            result.y = t;
            result.z = p;
            break;
        case 1:
            result.x = q;
            result.y = v;
            result.z = p;
            break;
        case 2:
            result.x = p;
            result.y = v;
            result.z = t;
            break;
        case 3:
            result.x = p;
            result.y = q;
            result.z = v;
            break;
        case 4:
            result.x = t;
            result.y = p;
            result.z = v;
            break;
        case 5:
            result.x = v;
            result.y = p;
            result.z = q;
            break;
        }
        return result;
    }

    void setFrustum(
        isaac_double* const projection,
        const isaac_float left,
        const isaac_float right,
        const isaac_float bottom,
        const isaac_float top,
        const isaac_float znear,
        const isaac_float zfar)
    {
        isaac_float znear2 = znear * isaac_float(2);
        isaac_float width = right - left;
        isaac_float height = top - bottom;
        isaac_float zRange = znear - zfar;
        projection[0] = znear2 / width;
        projection[1] = isaac_float(0);
        projection[2] = isaac_float(0);
        projection[3] = isaac_float(0);
        projection[4] = isaac_float(0);
        projection[5] = znear2 / height;
        projection[6] = isaac_float(0);
        projection[7] = isaac_float(0);
        projection[8] = (right + left) / width;
        projection[9] = (top + bottom) / height;
        projection[10] = (zfar + znear) / zRange;
        projection[11] = isaac_float(-1);
        projection[12] = isaac_float(0);
        projection[13] = isaac_float(0);
        projection[14] = (-znear2 * zfar) / -zRange;
        projection[15] = isaac_float(0);
    }

    void setPerspective(
        isaac_double* const projection,
        const isaac_float fovyInDegrees,
        const isaac_float aspectRatio,
        const isaac_float znear,
        const isaac_float zfar)
    {
        isaac_float ymax = znear * tan(fovyInDegrees * M_PI / isaac_float(360));
        isaac_float xmax = ymax * aspectRatio;
        setFrustum(projection, -xmax, xmax, -ymax, ymax, znear, zfar);
    }

    void spSetPerspectiveStereoscopic(
        isaac_double* const projection,
        const isaac_float fovyInDegrees,
        const isaac_float aspectRatio,
        const isaac_float znear,
        const isaac_float zfar,
        const isaac_float z0,
        const isaac_float distance)
    {
        isaac_float t_z0 = znear * tan(fovyInDegrees * M_PI / isaac_float(360));
        isaac_float xmin = -t_z0 + distance / 2.0f * znear / z0;
        isaac_float xmax = t_z0 + distance / 2.0f * znear / z0;
        isaac_float ymin = -t_z0 / aspectRatio;
        isaac_float ymax = t_z0 / aspectRatio;
        setFrustum(projection, xmin, xmax, ymin, ymax, znear, zfar);
        projection[12] += distance;
    }


    void setOrthographic(
        isaac_dmat4& projection,
        const isaac_double right,
        const isaac_double top,
        const isaac_double znear,
        const isaac_double zfar)
    {
        projection = isaac_dmat4(1);
        projection[0][0] = 1.0 / right;
        projection[1][1] = 1.0 / top;
        projection[2][2] = -2.0 / (zfar - znear);
        projection[3][2] = -(zfar + znear) / (zfar - znear);
    }


#if ISAAC_VALGRIND_TWEAKS == 1
    static void* extra_malloc(size_t size)
    {
        /* Get 4 bytes more than requested */
        void* ptr = malloc(size + 4);
        return ptr;
    }

    static void extra_free(void* ptr)
    {
        free(ptr);
    }
#endif


} // namespace isaac
