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

#include <boost/preprocessor.hpp>

#include "isaac_defines.hpp"

#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace isaac
{

using isaac_float = float;
using isaac_double = double;
using isaac_int = int32_t;
using isaac_uint = uint32_t;
using isaac_byte = uint8_t;

#ifndef ISAAC_IDX_TYPE
    #define ISAAC_IDX_TYPE size_t
#endif


template <int T_n, typename T_Type>
using isaac_vec_dim = glm::vec<T_n, T_Type, glm::defaultp>;

template <int T_n, typename T_Type>
using isaac_mat_dim = glm::mat<T_n, T_n, T_Type, glm::defaultp>;

template <int T_n>
using isaac_float_dim = isaac_vec_dim<T_n, isaac_float>;
template <int T_n>
using isaac_double_dim = isaac_vec_dim<T_n, isaac_double>;
template <int T_n>
using isaac_int_dim = isaac_vec_dim<T_n, isaac_int>;
template <int T_n>
using isaac_uint_dim = isaac_vec_dim<T_n, isaac_uint>;
template <int T_n>
using isaac_size_dim = isaac_vec_dim<T_n, ISAAC_IDX_TYPE>;

using isaac_float4 = isaac_vec_dim<4, isaac_float>;
using isaac_float3 = isaac_vec_dim<3, isaac_float>;
using isaac_float2 = isaac_vec_dim<2, isaac_float>;

using isaac_double4 = isaac_vec_dim<4, isaac_double>;
using isaac_double3 = isaac_vec_dim<3, isaac_double>;
using isaac_double2 = isaac_vec_dim<2, isaac_double>;

using isaac_uint4 = isaac_vec_dim<4, isaac_uint>;
using isaac_uint3 = isaac_vec_dim<3, isaac_uint>;
using isaac_uint2 = isaac_vec_dim<2, isaac_uint>;

using isaac_byte4 = isaac_vec_dim<4, isaac_byte>;
using isaac_byte3 = isaac_vec_dim<3, isaac_byte>;
using isaac_byte2 = isaac_vec_dim<2, isaac_byte>;

using isaac_int4 = isaac_vec_dim<4, isaac_int>;
using isaac_int3 = isaac_vec_dim<3, isaac_int>;
using isaac_int2 = isaac_vec_dim<2, isaac_int>;

using isaac_size4 = isaac_vec_dim<4, ISAAC_IDX_TYPE>;
using isaac_size3 = isaac_vec_dim<3, ISAAC_IDX_TYPE>;
using isaac_size2 = isaac_vec_dim<2, ISAAC_IDX_TYPE>;

using isaac_mat4 = isaac_mat_dim<4, isaac_float>;
using isaac_mat3 = isaac_mat_dim<3, isaac_float>;
using isaac_mat2 = isaac_mat_dim<2, isaac_float>;

using isaac_dmat4 = isaac_mat_dim<4, isaac_double>;
using isaac_dmat3 = isaac_mat_dim<3, isaac_double>;
using isaac_dmat2 = isaac_mat_dim<2, isaac_double>;




/**
 * @brief Container for all simulation sizes
 * 
 * @tparam simdim 
 */
struct SimulationSizeStruct
{
    isaac_size3 globalSize;             //size of volume
    ISAAC_IDX_TYPE maxGlobalSize;       //each dimension has a size and this value contains the value of the greatest dimension
    isaac_size3 position;               //local position of subvolume
    isaac_size3 localSize;              //size of local volume grid
    isaac_size3 localParticleSize;      //size of local particle grid 
    isaac_size3 globalSizeScaled;       //scaled version of global size with cells = scale * cells
    ISAAC_IDX_TYPE maxGlobalSizeScaled; //same as globalSizeScaled
    isaac_size3 positionScaled;         //scaled position of local subvolume
    isaac_size3 localSizeScaled;        //same as globalSizeScaled
};

struct GBuffer
{
    isaac_size2 size;
    isaac_uint2 startOffset;
    uint32_t* color;
    isaac_float* depth;
    isaac_float3* normal;
    isaac_float* aoStrength;
};

template< int T_n >
struct TransferDeviceStruct
{
    isaac_float4* pointer[ T_n ];
};

template< int T_n >
struct TransferHostStruct
{
    isaac_float4* pointer[ T_n ];
    std::map< isaac_uint, isaac_float4 > description[ T_n ];
};

struct FunctionsStruct
{
    std::string source;
    isaac_int bytecode[ISAAC_MAX_FUNCTORS];
    isaac_int errorCode;
};

template< int T_n >
struct IsoThresholdStruct
{
    isaac_float value[ T_n ];
};

template< int T_n >
struct SourceWeightStruct
{
    isaac_float value[ T_n ];
};

template< int T_n >
struct PointerArrayStruct
{
    void* pointer[ T_n ];
};

struct MinMax
{
    isaac_float min;
    isaac_float max;
};

template< int T_n>
struct MinMaxArray
{
    isaac_float min[ T_n ];
    isaac_float max[ T_n ];
};

struct ClippingStruct
{
    ISAAC_HOST_DEVICE_INLINE ClippingStruct() :
        count(0)
    {}
    isaac_uint count;
    struct
    {
        isaac_float3 position;
        isaac_float3 normal;
    } elem[ ISAAC_MAX_CLIPPING ];
};

/**
 * @brief Container for ambient occlusion parameters
 * 
 */
struct AOParams {
    ISAAC_HOST_DEVICE_INLINE AOParams() {}

    //weight value (0.0-1.0) for mixing color with depth component (darken) 
    //1.0 = 100% depth component 0.0 = 0% depth component  
    isaac_float weight = 0.5; 

    //true if pseudo ambient occlusion should be visible
    bool isEnabled = false; 
};


typedef enum
{
    META_MERGE = 0,
    META_MASTER = 1
} IsaacVisualizationMetaEnum;

} //namespace isaac;
