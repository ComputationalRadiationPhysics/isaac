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

#ifndef ISAAC_IDX_TYPE
    #define ISAAC_IDX_TYPE size_t
#endif

using isaac_float4 = glm::vec<4, isaac_float, glm::defaultp>;
using isaac_float3 = glm::vec<3, isaac_float, glm::defaultp>;
using isaac_float2 = glm::vec<2, isaac_float, glm::defaultp>;

using isaac_double4 = glm::vec<4, isaac_double, glm::defaultp>;
using isaac_double3 = glm::vec<3, isaac_double, glm::defaultp>;
using isaac_double2 = glm::vec<2, isaac_double, glm::defaultp>;

using isaac_uint4 = glm::vec<4, isaac_uint, glm::defaultp>;
using isaac_uint3 = glm::vec<3, isaac_uint, glm::defaultp>;
using isaac_uint2 = glm::vec<2, isaac_uint, glm::defaultp>;

using isaac_int4 = glm::vec<4, isaac_int, glm::defaultp>;
using isaac_int3 = glm::vec<3, isaac_int, glm::defaultp>;
using isaac_int2 = glm::vec<2, isaac_int, glm::defaultp>;

using isaac_size4 = glm::vec<4, ISAAC_IDX_TYPE, glm::defaultp>;
using isaac_size3 = glm::vec<3, ISAAC_IDX_TYPE, glm::defaultp>;
using isaac_size2 = glm::vec<2, ISAAC_IDX_TYPE, glm::defaultp>;

using isaac_mat4 = glm::mat<4, 4, isaac_float, glm::defaultp>;
using isaac_mat3 = glm::mat<3, 3, isaac_float, glm::defaultp>;
using isaac_mat2 = glm::mat<2, 2, isaac_float, glm::defaultp>;

using isaac_dmat4 = glm::mat<4, 4, isaac_double, glm::defaultp>;
using isaac_dmat3 = glm::mat<3, 3, isaac_double, glm::defaultp>;
using isaac_dmat2 = glm::mat<2, 2, isaac_double, glm::defaultp>;

template <int T>
using isaac_float_dim = glm::vec<T, isaac_float, glm::defaultp>;
template <int T>
using isaac_double_dim = glm::vec<T, isaac_double, glm::defaultp>;
template <int T>
using isaac_int_dim = glm::vec<T, isaac_int, glm::defaultp>;
template <int T>
using isaac_size_dim = glm::vec<T, ISAAC_IDX_TYPE, glm::defaultp>;


/**
 * @brief Container for all simulation sizes
 * 
 * @tparam simdim 
 */
struct isaac_size_struct
{
    isaac_size3 global_size;         //size of volume
    ISAAC_IDX_TYPE max_global_size;                //each dimension has a size and this value contains the value of the greatest dimension
    isaac_size3 position;            //local position of subvolume
    isaac_size3 local_size;          //size of local volume grid
    isaac_size3 local_particle_size; //size of local particle grid 
    isaac_size3 global_size_scaled;  //scaled version of global size with cells = scale * cells
    ISAAC_IDX_TYPE max_global_size_scaled;         //same as global_size_scaled
    isaac_size3 position_scaled;     //scaled position of local subvolume
    isaac_size3 local_size_scaled;   //same as global_size_scaled
};


template< int N >
struct transfer_d_struct
{
    isaac_float4* pointer[ N ];
};

template< int N >
struct transfer_h_struct
{
    isaac_float4* pointer[ N ];
    std::map< isaac_uint, isaac_float4 > description[ N ];
};

struct functions_struct
{
    std::string source;
    isaac_int bytecode[ISAAC_MAX_FUNCTORS];
    isaac_int error_code;
};

template< int N >
struct source_weight_struct
{
    isaac_float value[ N ];
};

template< int N >
struct pointer_array_struct
{
    void* pointer[ N ];
};

struct minmax_struct
{
    isaac_float min;
    isaac_float max;
};

template< int N>
struct minmax_array_struct
{
    isaac_float min[ N ];
    isaac_float max[ N ];
};

struct clipping_struct
{
    ISAAC_HOST_DEVICE_INLINE clipping_struct() :
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
struct ao_struct {
    ISAAC_HOST_DEVICE_INLINE ao_struct() {}

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
