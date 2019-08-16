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

namespace isaac
{

/*
 * TODO
 * cleanup this mess
 * 
 */
typedef float isaac_float;
typedef int32_t isaac_int;
typedef uint32_t isaac_uint;

#define ISAAC_COMPONENTS_SEQ_3 (x)(y)(z)(w)
#define ISAAC_COMPONENTS_SEQ_2 (x)(y)(z)
#define ISAAC_COMPONENTS_SEQ_1 (x)(y)
#define ISAAC_COMPONENTS_SEQ_0 (x)

#ifndef ISAAC_IDX_TYPE
    #define ISAAC_IDX_TYPE size_t
#endif

#ifndef __CUDACC__
    //isaac_float4, isaac_float3, isaac_float2
    #define ISAAC_FLOAT_DEF(Z, I, unused) \
        struct BOOST_PP_CAT(isaac_float, BOOST_PP_INC(I) ) \
        { \
            isaac_float BOOST_PP_SEQ_ENUM( BOOST_PP_CAT( ISAAC_COMPONENTS_SEQ_ , I ) ); \
        };
    BOOST_PP_REPEAT(4, ISAAC_FLOAT_DEF, ~)
    #undef ISAAC_FLOAT_DEF
    //isaac_uint4, isaac_uint3, isaac_uint2
    #define ISAAC_UINT_DEF(Z, I, unused) \
        struct BOOST_PP_CAT(isaac_uint, BOOST_PP_INC(I) ) \
        { \
            isaac_uint BOOST_PP_SEQ_ENUM( BOOST_PP_CAT( ISAAC_COMPONENTS_SEQ_ , I ) ); \
        };
    BOOST_PP_REPEAT(4, ISAAC_UINT_DEF, ~)
    #undef ISAAC_UINT_DEF
    //isaac_int4, isaac_int3, isaac_int2
    #define ISAAC_INT_DEF(Z, I, unused) \
        struct BOOST_PP_CAT(isaac_int, BOOST_PP_INC(I) ) \
        { \
            isaac_int BOOST_PP_SEQ_ENUM( BOOST_PP_CAT( ISAAC_COMPONENTS_SEQ_ , I ) ); \
        };
    BOOST_PP_REPEAT(4, ISAAC_INT_DEF, ~)
    #undef ISAAC_INT_DEF
#else
    //same as above, but we use the builtin cuda variables
    #define ISAAC_CUDA_DEF(Z, I, unused) \
        typedef BOOST_PP_CAT(float, BOOST_PP_INC(I) ) BOOST_PP_CAT(isaac_float, BOOST_PP_INC(I) ); \
        typedef BOOST_PP_CAT(uint, BOOST_PP_INC(I) ) BOOST_PP_CAT(isaac_uint, BOOST_PP_INC(I) ); \
        typedef BOOST_PP_CAT(int, BOOST_PP_INC(I) ) BOOST_PP_CAT(isaac_int, BOOST_PP_INC(I) );
    BOOST_PP_REPEAT(4, ISAAC_CUDA_DEF, ~)
    #undef ISAAC_CUDA_DEF
#endif

//isaac_size4, isaac_size3, isaac_size2, isaac_size1
#define ISAAC_SIZE_DEF(Z, I, unused) \
    struct BOOST_PP_CAT(isaac_size, BOOST_PP_INC(I) ) \
    { \
        ISAAC_IDX_TYPE BOOST_PP_SEQ_ENUM( BOOST_PP_CAT( ISAAC_COMPONENTS_SEQ_ , I ) ); \
    };
BOOST_PP_REPEAT(4, ISAAC_SIZE_DEF, ~)
#undef ISAAC_SIZE_DEF

//Overloading *, /, + and - for isaac_{type}[2,3,4]:
//macro for result.<I> = left.<I> <OP[1]> right.<I>, with I € {x,y,z,w}
#define ISAAC_OVERLOAD_OPERATOR_SUBDEF(Z, I, OP ) \
    result. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_3 ) ) = \
    left. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_3 ) ) BOOST_PP_ARRAY_ELEM( 1, OP ) \
    right. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_3 ) );

#define ISAAC_OVERLOAD_OPERATOR_SUBDEF_ROTHER(Z, I, OP ) \
    result. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_3 ) ) = \
    left. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_3 ) ) BOOST_PP_ARRAY_ELEM( 1, OP ) \
    right;

#define ISAAC_OVERLOAD_OPERATOR_SUBDEF_LOTHER(Z, I, OP ) \
    result. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_3 ) ) = \
    left  BOOST_PP_ARRAY_ELEM( 1, OP ) \
    right. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_3 ) );

#define ISAAC_OVERLOAD_OPERATOR_SUBDEF_UNARY(Z, I, OP ) \
    result. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_3 ) ) = BOOST_PP_ARRAY_ELEM( 1, OP ) \
    left. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_3 ) );

//macro for the any operator for isaac_{type}[2,3,4].
#define ISAAC_OVERLOAD_OPERATOR_DEF(Z, I, OP) \
    ISAAC_HOST_DEVICE_INLINE BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) operator BOOST_PP_ARRAY_ELEM( 1, OP ) ( \
    const BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) & left, \
    const BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) & right) \
    { \
        BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) result; \
        BOOST_PP_REPEAT( BOOST_PP_INC(I), ISAAC_OVERLOAD_OPERATOR_SUBDEF, OP ) \
        return result; \
    } \
    ISAAC_HOST_DEVICE_INLINE BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) operator BOOST_PP_ARRAY_ELEM( 1, OP ) ( \
    const BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) & left, \
    const isaac_float & right) \
    { \
        BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) result; \
        BOOST_PP_REPEAT( BOOST_PP_INC(I), ISAAC_OVERLOAD_OPERATOR_SUBDEF_ROTHER, OP ) \
        return result; \
    } \
    ISAAC_HOST_DEVICE_INLINE BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) operator BOOST_PP_ARRAY_ELEM( 1, OP ) ( \
    const isaac_float & left, \
    const BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) & right ) \
    { \
        BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) result; \
        BOOST_PP_REPEAT( BOOST_PP_INC(I), ISAAC_OVERLOAD_OPERATOR_SUBDEF_LOTHER, OP ) \
        return result; \
    }

#define ISAAC_OVERLOAD_OPERATOR_DEF_UNARY(Z, I, OP) \
    ISAAC_HOST_DEVICE_INLINE BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) operator BOOST_PP_ARRAY_ELEM( 1, OP ) ( \
    const BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) & left ) \
    { \
        BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_INC(I) ) result; \
        BOOST_PP_REPEAT( BOOST_PP_INC(I), ISAAC_OVERLOAD_OPERATOR_SUBDEF_UNARY, OP ) \
        return result; \
    }

#define ISAAC_OVERLOAD_OPERATOR_COUNT 4
#define ISAAC_OVERLOAD_OPERATORS (ISAAC_OVERLOAD_OPERATOR_COUNT, (+, -, *, / ) )

#define ISAAC_OVERLOAD_ITERATOR(Z, I, TYPE) \
    BOOST_PP_REPEAT( 4, ISAAC_OVERLOAD_OPERATOR_DEF, (2, (TYPE, BOOST_PP_ARRAY_ELEM( I, ISAAC_OVERLOAD_OPERATORS ) ) ) ) \

#define ISAAC_OVERLOAD_OPERATOR_CREATE( TYPE ) \
    BOOST_PP_REPEAT( ISAAC_OVERLOAD_OPERATOR_COUNT, ISAAC_OVERLOAD_ITERATOR, TYPE ) \
    BOOST_PP_REPEAT( 4, ISAAC_OVERLOAD_OPERATOR_DEF_UNARY, (2, (TYPE, - ) ) )

ISAAC_OVERLOAD_OPERATOR_CREATE(float)
ISAAC_OVERLOAD_OPERATOR_CREATE(uint)
ISAAC_OVERLOAD_OPERATOR_CREATE(int)
ISAAC_OVERLOAD_OPERATOR_CREATE(size)

#undef ISAAC_OVERLOAD_OPERATOR_SUBDEF_LOTHER
#undef ISAAC_OVERLOAD_OPERATOR_SUBDEF_ROTHER
#undef ISAAC_OVERLOAD_OPERATOR_COUNT
#undef ISAAC_OVERLOAD_OPERATORS
#undef ISAAC_OVERLOAD_ITERATOR
#undef ISAAC_OVERLOAD_OPERATOR_SUBDEF
#undef ISAAC_OVERLOAD_OPERATOR_SUBDEF_UNARY
#undef ISAAC_OVERLOAD_OPERATOR_DEF
#undef ISAAC_OVERLOAD_OPERATOR_DEF_UNARY
#undef ISAAC_OVERLOAD_OPERATOR_CREATE

#define ISAAC_DIM_TYPES ( 3, ( size, float, int ) )
#define ISAAC_DIM_TYPES_DIM ( 3, ( size_dim, float_dim, int_dim ) )

#define ISAAC_DIM_SUBDEF(Z, J, I ) \
    template <> \
    struct BOOST_PP_CAT( isaac_, BOOST_PP_ARRAY_ELEM( I, ISAAC_DIM_TYPES_DIM ) ) \
    < ISAAC_IDX_TYPE( BOOST_PP_INC(J) ) > { \
    BOOST_PP_CAT( isaac_, BOOST_PP_CAT( BOOST_PP_ARRAY_ELEM( I, ISAAC_DIM_TYPES ) , BOOST_PP_INC(J) ) ) value; };

#define ISAAC_DIM_DEF(Z, I, unused) \
    template < ISAAC_IDX_TYPE > \
    struct BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( I, ISAAC_DIM_TYPES_DIM ) ); \
    BOOST_PP_REPEAT( 4, ISAAC_DIM_SUBDEF, I )

BOOST_PP_REPEAT( 3, ISAAC_DIM_DEF, ~ )



#undef ISAAC_DIM_SUBDEF
#undef ISAAC_DIM_DEF
#undef ISAAC_DIM_TYPES
#undef ISAAC_DIM_TYPES_DIM

/**
 * @brief Container for all simulation sizes
 * 
 * @tparam simdim 
 */
template < ISAAC_IDX_TYPE simdim >
struct isaac_size_struct
{
    isaac_size_dim < simdim > global_size;         //size of volume
    ISAAC_IDX_TYPE max_global_size;                //each dimension has a size and this value contains the value of the greatest dimension
    isaac_size_dim < simdim > position;            //local position of subvolume
    isaac_size_dim < simdim > local_size;          //size of local volume grid
    isaac_size_dim < simdim > local_particle_size; //size of local particle grid 
    isaac_size_dim < simdim > global_size_scaled;  //scaled version of global size with cells = scale * cells
    ISAAC_IDX_TYPE max_global_size_scaled;         //same as global_size_scaled
    isaac_size_dim < simdim > position_scaled;     //scaled position of local subvolume
    isaac_size_dim < simdim > local_size_scaled;   //same as global_size_scaled
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



typedef enum
{
    META_MERGE = 0,
    META_MASTER = 1
} IsaacVisualizationMetaEnum;

} //namespace isaac;
