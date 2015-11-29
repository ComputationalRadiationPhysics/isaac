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
 * You should have received a copy of the GNU General Lesser Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include <boost/preprocessor.hpp>

#include "isaac_defines.hpp"

namespace isaac
{

typedef float isaac_float;
typedef int32_t isaac_int;
typedef uint32_t isaac_uint;

#define ISAAC_COMPONENTS_SEQ_2 (x)(y)(z)(w)
#define ISAAC_COMPONENTS_SEQ_1 (x)(y)(z)
#define ISAAC_COMPONENTS_SEQ_0 (x)(y)

#ifndef __CUDACC__
    //isaac_float4, isaac_float3, isaac_float2
    #define ISAAC_FLOAT_DEF(Z, I, unused) \
        struct BOOST_PP_CAT(isaac_float, BOOST_PP_ADD(I,2) ) \
        { \
            isaac_float BOOST_PP_SEQ_ENUM( BOOST_PP_CAT( ISAAC_COMPONENTS_SEQ_ , I ) ); \
        };
    BOOST_PP_REPEAT(3, ISAAC_FLOAT_DEF, ~)
    #undef ISAAC_FLOAT_DEF
    //isaac_uint4, isaac_uint3, isaac_uint2
    #define ISAAC_UINT_DEF(Z, I, unused) \
        struct BOOST_PP_CAT(isaac_uint, BOOST_PP_ADD(I,2) ) \
        { \
            isaac_uint BOOST_PP_SEQ_ENUM( BOOST_PP_CAT( ISAAC_COMPONENTS_SEQ_ , I ) ); \
        };
    BOOST_PP_REPEAT(3, ISAAC_UINT_DEF, ~)
    #undef ISAAC_UINT_DEF
    //isaac_int4, isaac_int3, isaac_int2
    #define ISAAC_INT_DEF(Z, I, unused) \
        struct BOOST_PP_CAT(isaac_int, BOOST_PP_ADD(I,2) ) \
        { \
            isaac_int BOOST_PP_SEQ_ENUM( BOOST_PP_CAT( ISAAC_COMPONENTS_SEQ_ , I ) ); \
        };
    BOOST_PP_REPEAT(3, ISAAC_INT_DEF, ~)
    #undef ISAAC_INT_DEF
#else
    //same as above, but we use the builtin cuda variables
    #define ISAAC_CUDA_DEF(Z, I, unused) \
        typedef BOOST_PP_CAT(float, BOOST_PP_ADD(I,2) ) BOOST_PP_CAT(isaac_float, BOOST_PP_ADD(I,2) ); \
        typedef BOOST_PP_CAT(uint, BOOST_PP_ADD(I,2) ) BOOST_PP_CAT(isaac_uint, BOOST_PP_ADD(I,2) ); \
        typedef BOOST_PP_CAT(int, BOOST_PP_ADD(I,2) ) BOOST_PP_CAT(isaac_int, BOOST_PP_ADD(I,2) );
    BOOST_PP_REPEAT(3, ISAAC_CUDA_DEF, ~)
    #undef ISAAC_CUDA_DEF
#endif

//isaac_size4, isaac_size3, isaac_size2
#define ISAAC_SIZE_DEF(Z, I, unused) \
    struct BOOST_PP_CAT(isaac_size, BOOST_PP_ADD(I,2) ) \
    { \
        size_t BOOST_PP_SEQ_ENUM( BOOST_PP_CAT( ISAAC_COMPONENTS_SEQ_ , I ) ); \
    };
BOOST_PP_REPEAT(3, ISAAC_SIZE_DEF, ~)
#undef ISAAC_SIZE_DEF

//Overloading *, /, + and - for isaac_{type}[2,3,4]:
//macro for result.<I> = left.<I> <OP[1]> right.<I>, with I € {x,y,z,w}
#define ISAAC_OVERLOAD_OPERATOR_SUBDEF(Z, I, OP ) \
    result. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_2 ) ) = \
    left. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_2 ) ) BOOST_PP_ARRAY_ELEM( 1, OP ) \
    right. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_2 ) );

#define ISAAC_OVERLOAD_OPERATOR_SUBDEF_ROTHER(Z, I, OP ) \
    result. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_2 ) ) = \
    left. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_2 ) ) BOOST_PP_ARRAY_ELEM( 1, OP ) \
    right;

#define ISAAC_OVERLOAD_OPERATOR_SUBDEF_LOTHER(Z, I, OP ) \
    result. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_2 ) ) = \
    left  BOOST_PP_ARRAY_ELEM( 1, OP ) \
    right. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_2 ) );

#define ISAAC_OVERLOAD_OPERATOR_SUBDEF_UNARY(Z, I, OP ) \
    result. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_2 ) ) = BOOST_PP_ARRAY_ELEM( 1, OP ) \
    left. BOOST_PP_ARRAY_ELEM( I, BOOST_PP_SEQ_TO_ARRAY( ISAAC_COMPONENTS_SEQ_2 ) );

//macro for the any operator for isaac_{type}[2,3,4]. 
#define ISAAC_OVERLOAD_OPERATOR_DEF(Z, I, OP) \
    ISAAC_HOST_DEVICE inline BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) operator BOOST_PP_ARRAY_ELEM( 1, OP ) ( \
    const BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) & left, \
    const BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) & right) \
    { \
        BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) result; \
        BOOST_PP_REPEAT( BOOST_PP_ADD( I, 2 ), ISAAC_OVERLOAD_OPERATOR_SUBDEF, OP ) \
        return result; \
    } \
    ISAAC_HOST_DEVICE inline BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) operator BOOST_PP_ARRAY_ELEM( 1, OP ) ( \
    const BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) & left, \
    const isaac_float & right) \
    { \
        BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) result; \
        BOOST_PP_REPEAT( BOOST_PP_ADD( I, 2 ), ISAAC_OVERLOAD_OPERATOR_SUBDEF_ROTHER, OP ) \
        return result; \
    } \
    ISAAC_HOST_DEVICE inline BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) operator BOOST_PP_ARRAY_ELEM( 1, OP ) ( \
    const isaac_float & left, \
    const BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) & right ) \
    { \
        BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) result; \
        BOOST_PP_REPEAT( BOOST_PP_ADD( I, 2 ), ISAAC_OVERLOAD_OPERATOR_SUBDEF_LOTHER, OP ) \
        return result; \
    }

#define ISAAC_OVERLOAD_OPERATOR_DEF_UNARY(Z, I, OP) \
    ISAAC_HOST_DEVICE inline BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) operator BOOST_PP_ARRAY_ELEM( 1, OP ) ( \
    const BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) & left ) \
    { \
        BOOST_PP_CAT( BOOST_PP_CAT( isaac_ , BOOST_PP_ARRAY_ELEM( 0, OP ) ), BOOST_PP_ADD(I,2) ) result; \
        BOOST_PP_REPEAT( BOOST_PP_ADD( I, 2 ), ISAAC_OVERLOAD_OPERATOR_SUBDEF_UNARY, OP ) \
        return result; \
    }

#define ISAAC_OVERLOAD_OPERATOR_COUNT 4
#define ISAAC_OVERLOAD_OPERATORS (ISAAC_OVERLOAD_OPERATOR_COUNT, (+, -, *, / ) )

#define ISAAC_OVERLOAD_ITERATOR(Z, I, TYPE) \
    BOOST_PP_REPEAT( 3, ISAAC_OVERLOAD_OPERATOR_DEF, (2, (TYPE, BOOST_PP_ARRAY_ELEM( I, ISAAC_OVERLOAD_OPERATORS ) ) ) ) \

#define ISAAC_OVERLOAD_OPERATOR_CREATE( TYPE ) \
    BOOST_PP_REPEAT( ISAAC_OVERLOAD_OPERATOR_COUNT, ISAAC_OVERLOAD_ITERATOR, TYPE ) \
    BOOST_PP_REPEAT( 3, ISAAC_OVERLOAD_OPERATOR_DEF_UNARY, (2, (TYPE, - ) ) )

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

template < size_t >
struct isaac_size_dim;

template <>
struct isaac_size_dim < size_t(2) > : isaac_size2 {};

template <>
struct isaac_size_dim < size_t(3) > : isaac_size3 {};

template < size_t simdim >
struct isaac_size_struct
{
    isaac_size_dim < simdim > global_size;
    size_t max_global_size;
    isaac_size_dim < simdim > position;
    isaac_size_dim < simdim > local_size;
};

typedef enum
{
    META_MERGE = 0,
    META_MASTER = 1
} IsaacVisualizationMetaEnum;

} //namespace isaac;
