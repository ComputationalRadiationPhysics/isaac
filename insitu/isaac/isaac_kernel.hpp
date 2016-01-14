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

#include "isaac_macros.hpp"
#include "isaac_fusion_extension.hpp"
#include "isaac_functors.hpp"

#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/mpl/size.hpp>

#include <float.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wsign-compare"

namespace isaac
{

namespace fus = boost::fusion;
namespace mpl = boost::mpl;

#if ISAAC_ALPAKA == 1
    #define ISAAC_FUNCTOR_PARAM ,isaac_float4 const * const
#else
    #define ISAAC_FUNCTOR_PARAM
#endif            
typedef isaac_float (*isaac_functor_chain_pointer_4)(isaac_float_dim <4>, isaac_int ISAAC_FUNCTOR_PARAM );
typedef isaac_float (*isaac_functor_chain_pointer_3)(isaac_float_dim <3>, isaac_int ISAAC_FUNCTOR_PARAM );
typedef isaac_float (*isaac_functor_chain_pointer_2)(isaac_float_dim <2>, isaac_int ISAAC_FUNCTOR_PARAM );
typedef isaac_float (*isaac_functor_chain_pointer_1)(isaac_float_dim <1>, isaac_int ISAAC_FUNCTOR_PARAM );
typedef isaac_float (*isaac_functor_chain_pointer_N)(void*              , isaac_int ISAAC_FUNCTOR_PARAM );
#undef ISAAC_FUNCTOR_PARAM

#if ISAAC_ALPAKA == 0
    __constant__ isaac_float isaac_inverse_d[16];
    __constant__ isaac_size_struct<3> isaac_size_d[1]; //[1] to access it for cuda and alpaka the same way
    __constant__ isaac_float4 isaac_parameter_d[ ISAAC_MAX_SOURCES*ISAAC_MAX_FUNCTORS ];
    __constant__ isaac_functor_chain_pointer_N isaac_function_chain_d[ ISAAC_MAX_SOURCES ];
#endif


template
<
    typename TFunctorVector,
    int TFeatureDim,
    int NR
>
struct FillFunctorChainPointerKernelStruct
{
    ISAAC_DEVICE static isaac_functor_chain_pointer_N call( isaac_int const * const bytecode )
    {
        #define ISAAC_SUB_CALL(Z, I, U) \
            if (bytecode[ISAAC_MAX_FUNCTORS-NR] == I) \
                return FillFunctorChainPointerKernelStruct \
                < \
                    typename mpl::push_back< TFunctorVector, typename boost::mpl::at_c<IsaacFunctorPool,I>::type >::type, \
                    TFeatureDim, \
                    NR - 1 \
                > ::call( bytecode );
        BOOST_PP_REPEAT( ISAAC_FUNCTOR_COUNT, ISAAC_SUB_CALL, ~)
        #undef ISAAC_SUB_CALL
        return NULL; //Should never be reached anyway
    }
};

template
<
    typename TFunctorVector,
    int TFeatureDim
>
ISAAC_DEVICE isaac_float applyFunctorChain (
    isaac_float_dim < TFeatureDim > const value,
    isaac_int const src_id
#if ISAAC_ALPAKA == 1
    ,isaac_float4 const * const isaac_parameter_d
#endif    
)
{
    #define  ISAAC_LEFT_DEF(Z,I,U) mpl::at_c< TFunctorVector, ISAAC_MAX_FUNCTORS - I - 1 >::type::call(
    #define ISAAC_RIGHT_DEF(Z,I,U) , isaac_parameter_d[ src_id * ISAAC_MAX_FUNCTORS + I ] )
    #define  ISAAC_LEFT BOOST_PP_REPEAT( ISAAC_MAX_FUNCTORS, ISAAC_LEFT_DEF, ~)
    #define ISAAC_RIGHT BOOST_PP_REPEAT( ISAAC_MAX_FUNCTORS, ISAAC_RIGHT_DEF, ~)
    // expands to: funcN( ... func1( func0( data, p[0] ), p[1] ) ... p[N] );
    return ISAAC_LEFT value ISAAC_RIGHT .value.x;
    #undef ISAAC_LEFT_DEF
    #undef ISAAC_LEFT
    #undef ISAAC_RIGHT_DEF
    #undef ISAAC_RIGHT
}


template
<
    typename TFunctorVector,
    int TFeatureDim
>
struct FillFunctorChainPointerKernelStruct
<
    TFunctorVector,
    TFeatureDim,
    0 //<- Specialization
>
{
    ISAAC_DEVICE static isaac_functor_chain_pointer_N call( isaac_int const * const bytecode)
    {
        return reinterpret_cast<isaac_functor_chain_pointer_N>(applyFunctorChain<TFunctorVector,TFeatureDim>);
    }
};


#if ISAAC_ALPAKA == 1
    struct fillFunctorChainPointerKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
#else
        __global__ void fillFunctorChainPointerKernel(
#endif
            isaac_functor_chain_pointer_N * const functor_chain_d)
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            isaac_int bytecode[ISAAC_MAX_FUNCTORS];
            for (int i = 0; i < ISAAC_MAX_FUNCTORS; i++)
                bytecode[i] = 0;
            for (int i = 0; i < ISAAC_FUNCTOR_COMPLEX; i++)
            {
                functor_chain_d[i*4+0] = FillFunctorChainPointerKernelStruct<mpl::vector<>,1,ISAAC_MAX_FUNCTORS>::call( bytecode );
                functor_chain_d[i*4+1] = FillFunctorChainPointerKernelStruct<mpl::vector<>,2,ISAAC_MAX_FUNCTORS>::call( bytecode );
                functor_chain_d[i*4+2] = FillFunctorChainPointerKernelStruct<mpl::vector<>,3,ISAAC_MAX_FUNCTORS>::call( bytecode );
                functor_chain_d[i*4+3] = FillFunctorChainPointerKernelStruct<mpl::vector<>,4,ISAAC_MAX_FUNCTORS>::call( bytecode );
                for (int j = ISAAC_MAX_FUNCTORS - 1; j >= 0; j--)
                    if ( bytecode[j] < ISAAC_FUNCTOR_COUNT-1 )
                    {
                        bytecode[j]++;
                        break;
                    }
                    else
                        bytecode[j] = 0;
            }
        }
#if ISAAC_ALPAKA == 1
    };
#endif

template <
    isaac_int TInterpolation,
    typename NR,
    typename TSource,
    typename TPos,
    typename TPointerArray,
    typename TLocalSize,
    typename TScale
>
ISAAC_HOST_DEVICE_INLINE isaac_float get_value (
    const TSource& source,
    const TPos& pos,
    const TPointerArray& pointerArray,
    const TLocalSize& local_size,
    const TScale& scale
#if ISAAC_ALPAKA == 1
    ,isaac_float4 const * const isaac_parameter_d
    ,isaac_functor_chain_pointer_N const * const isaac_function_chain_d
#endif
)
{
    isaac_float_dim < TSource::feature_dim > data;
    isaac_float_dim < TSource::feature_dim >* ptr = (isaac_float_dim < TSource::feature_dim >*)(pointerArray.pointer[ NR::value ] );
    if (TInterpolation == 0)
    {
        isaac_int3 coord =
        {
            isaac_int(pos.x),
            isaac_int(pos.y),
            isaac_int(pos.z)
        };
        if (TSource::persistent)
            data = source[coord];
        else
            data = ptr[coord.x + ISAAC_GUARD_SIZE + (coord.y + ISAAC_GUARD_SIZE) * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + (coord.z + ISAAC_GUARD_SIZE) * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )];
    }
    else
    {
        isaac_int3 coord;
        isaac_float_dim < TSource::feature_dim > data8[2][2][2];
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                for (int z = 0; z < 2; z++)
                {
                    coord.x = isaac_int(x?ceil(pos.x):floor(pos.x));
                    coord.y = isaac_int(y?ceil(pos.y):floor(pos.y));
                    coord.z = isaac_int(z?ceil(pos.z):floor(pos.z));
                    if (!TSource::has_guard && TSource::persistent)
                    {
                        if ( isaac_uint(coord.x) >= local_size.value.x )
                            coord.x = isaac_int(x?floor(pos.x):ceil(pos.x));
                        if ( isaac_uint(coord.y) >= local_size.value.y )
                            coord.y = isaac_int(y?floor(pos.y):ceil(pos.y));
                        if ( isaac_uint(coord.z) >= local_size.value.z )
                            coord.z = isaac_int(z?floor(pos.z):ceil(pos.z));
                    }
                    if (TSource::persistent)
                        data8[x][y][z] = source[coord];
                    else
                        data8[x][y][z] = ptr[coord.x + ISAAC_GUARD_SIZE + (coord.y + ISAAC_GUARD_SIZE) * (local_size.value.x + 2 * ISAAC_GUARD_SIZE) + (coord.z + ISAAC_GUARD_SIZE) * ( (local_size.value.x + 2 * ISAAC_GUARD_SIZE) * (local_size.value.y + 2 * ISAAC_GUARD_SIZE) )];
                }
        isaac_float_dim < 3 > pos_in_cube =
        {
            pos.x - floor(pos.x),
            pos.y - floor(pos.y),
            pos.z - floor(pos.z)
        };
        isaac_float_dim < TSource::feature_dim > data4[2][2];
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                data4[x][y].value =
                    data8[x][y][0].value * (isaac_float(1) - pos_in_cube.value.z) +
                    data8[x][y][1].value * (                 pos_in_cube.value.z);
        isaac_float_dim < TSource::feature_dim > data2[2];
        for (int x = 0; x < 2; x++)
            data2[x].value =
                data4[x][0].value * (isaac_float(1) - pos_in_cube.value.y) +
                data4[x][1].value * (                 pos_in_cube.value.y);
        data.value =
            data2[0].value * (isaac_float(1) - pos_in_cube.value.x) +
            data2[1].value * (                 pos_in_cube.value.x);
    }
    isaac_float result = isaac_float(0);

    #if ISAAC_ALPAKA == 1
        #define ISAAC_PARAMETER_PARAM ,isaac_parameter_d
    #else
        #define ISAAC_PARAMETER_PARAM
    #endif            

    #if ISAAC_ALPAKA == 1 || defined(__CUDA_ARCH__)
        if (TSource::feature_dim == 1)
            result = reinterpret_cast<isaac_functor_chain_pointer_1>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< isaac_float_dim<1>* >(&data)), NR::value ISAAC_PARAMETER_PARAM );
        if (TSource::feature_dim == 2)
            result = reinterpret_cast<isaac_functor_chain_pointer_2>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< isaac_float_dim<2>* >(&data)), NR::value ISAAC_PARAMETER_PARAM );
        if (TSource::feature_dim == 3)
            result = reinterpret_cast<isaac_functor_chain_pointer_3>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< isaac_float_dim<3>* >(&data)), NR::value ISAAC_PARAMETER_PARAM );
        if (TSource::feature_dim == 4)
            result = reinterpret_cast<isaac_functor_chain_pointer_4>(isaac_function_chain_d[ NR::value ])( *(reinterpret_cast< isaac_float_dim<4>* >(&data)), NR::value ISAAC_PARAMETER_PARAM );
    #endif
    #undef ISAAC_PARAMETER_PARAM
    return result;
}

template < typename TLocalSize >
ISAAC_HOST_DEVICE_INLINE void check_coord( isaac_float3& coord, const TLocalSize local_size)
{
    if (coord.x < isaac_float(0))
        coord.x = isaac_float(0);
    if (coord.y < isaac_float(0))
        coord.y = isaac_float(0);
    if (coord.z < isaac_float(0))
        coord.z = isaac_float(0);
    if ( coord.x >= isaac_float(local_size.value.x) )
        coord.x = isaac_float(local_size.value.x)-isaac_float(1);
    if ( coord.y >= isaac_float(local_size.value.y) )
        coord.y = isaac_float(local_size.value.y)-isaac_float(1);
    if ( coord.z >= isaac_float(local_size.value.z) )
        coord.z = isaac_float(local_size.value.z)-isaac_float(1);
}

template <
    size_t Ttransfer_size,
    typename TFilter,
    isaac_int TInterpolation,
    isaac_int TIsoSurface
>
struct merge_source_iterator
{
    template
    <
        typename NR,
        typename TSource,
        typename TColor,
        typename TPos,
        typename TLocalSize,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TFeedback,
        typename TStep,
        typename TStepLength,
        typename TScale
#if ISAAC_ALPAKA == 1
        ,typename TParameter
#endif
    >
    ISAAC_HOST_DEVICE_INLINE  void operator()(
        const NR& nr,
        const TSource& source,
        TColor& color,
        const TPos& pos,
        const TLocalSize& local_size,
        const TTransferArray& transferArray,
        const TSourceWeight& sourceWeight,
        const TPointerArray& pointerArray,
        TFeedback& feedback,
        const TStep& step,
        const TStepLength& stepLength,
        const TScale& scale
#if ISAAC_ALPAKA == 1
        ,const TParameter isaac_parameter_d
        ,isaac_functor_chain_pointer_N const * const isaac_function_chain_d
#endif
    ) const
    {
        #if ISAAC_ALPAKA == 1
            #define ISAAC_FUNCTION_CHAIN_PARAM ,isaac_parameter_d, isaac_function_chain_d
        #else
            #define ISAAC_FUNCTION_CHAIN_PARAM
        #endif            
        
        if ( mpl::at_c< TFilter, NR::value >::type::value )
        {
            isaac_float result = get_value< TInterpolation, NR >( source, pos, pointerArray, local_size, scale ISAAC_FUNCTION_CHAIN_PARAM );
            isaac_int lookup_value = isaac_int( round(result * isaac_float( Ttransfer_size ) ) );
            if (lookup_value < 0 )
                lookup_value = 0;
            if (lookup_value >= Ttransfer_size )
                lookup_value = Ttransfer_size - 1;
            isaac_float4 value = transferArray.pointer[ NR::value ][ lookup_value ];
            if (TIsoSurface)
            {
                if (value.w >= isaac_float(0.5))
                {
                    isaac_float3  left = {-1, 0, 0};
                    left = left + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( left, local_size);
                    isaac_float3 right = { 1, 0, 0};
                    right = right + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( right, local_size );
                    isaac_float d1;
                    if (TInterpolation)
                        d1 = right.x - left.x;
                    else
                        d1 = isaac_int(right.x) - isaac_int(left.x);
                    
                    isaac_float3    up = { 0,-1, 0};
                    up = up + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( up, local_size );
                    isaac_float3  down = { 0, 1, 0};
                    down = down + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( down, local_size );
                    isaac_float d2;
                    if (TInterpolation)
                        d2 = down.y - up.y;
                    else
                        d2 = isaac_int(down.y) - isaac_int(up.y);

                    isaac_float3 front = { 0, 0,-1};
                    front = front + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( front, local_size );
                    isaac_float3  back = { 0, 0, 1};
                    back = back + pos;
                    if (!TSource::has_guard && TSource::persistent)
                        check_coord( back, local_size );
                    isaac_float d3;
                    if (TInterpolation)
                        d3 = back.z - front.z;
                    else
                        d3 = isaac_int(back.z) - isaac_int(front.z);
                    
                    isaac_float3 gradient=
                    {
                        (get_value< TInterpolation, NR >( source, right, pointerArray, local_size, scale ISAAC_FUNCTION_CHAIN_PARAM ) -
                         get_value< TInterpolation, NR >( source,  left, pointerArray, local_size, scale ISAAC_FUNCTION_CHAIN_PARAM )) / d1,
                        (get_value< TInterpolation, NR >( source,  down, pointerArray, local_size, scale ISAAC_FUNCTION_CHAIN_PARAM ) -
                         get_value< TInterpolation, NR >( source,    up, pointerArray, local_size, scale ISAAC_FUNCTION_CHAIN_PARAM )) / d2,
                        (get_value< TInterpolation, NR >( source,  back, pointerArray, local_size, scale ISAAC_FUNCTION_CHAIN_PARAM ) -
                         get_value< TInterpolation, NR >( source, front, pointerArray, local_size, scale ISAAC_FUNCTION_CHAIN_PARAM )) / d3
                    };
                    isaac_float l = sqrt(
                        gradient.x * gradient.x +
                        gradient.y * gradient.y +
                        gradient.z * gradient.z
                    );
                    if (l == isaac_float(0))
                        color = value;
                    else
                    {
                        gradient = gradient / l;
                        isaac_float3 light = step / stepLength;
                        isaac_float ac = fabs(
                            gradient.x * light.x +
                            gradient.y * light.y +
                            gradient.z * light.z);
                        color.x = value.x * ac + ac * ac * ac * ac;
                        color.y = value.y * ac + ac * ac * ac * ac;
                        color.z = value.z * ac + ac * ac * ac * ac;
                    }
                    color.w = isaac_float(1);
                    feedback = 1;
                }
            }
            else
            {
                value.w *= sourceWeight.value[ NR::value ];
                color.x = color.x + value.x * value.w;
                color.y = color.y + value.y * value.w;
                color.z = color.z + value.z * value.w;
                color.w = color.w + value.w;
            }
        }
    }
    #undef ISAAC_FUNCTION_CHAIN_PARAM
};

template <
    typename TFilter
>
struct check_no_source_iterator
{
    template
    <
        typename NR,
        typename TSource,
        typename TResult
    >
    ISAAC_HOST_DEVICE_INLINE  void operator()(
        const NR& nr,
        const TSource& source,
        TResult& result
    ) const
    {
        result |= mpl::at_c< TFilter, NR::value >::type::value;
    }
};


template <
    typename TSimDim,
    typename TSourceList,
    typename TTransferArray,
    typename TSourceWeight,
    typename TPointerArray,
    typename TFilter,
    size_t Ttransfer_size,
    isaac_int TInterpolation,
    isaac_int TIsoSurface,
    typename TScale
>
#if ISAAC_ALPAKA == 1
    struct isaacFillRectKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
            isaac_float const * const isaac_inverse_d,
            isaac_size_struct<TSimDim::value> const * const isaac_size_d,
            isaac_float4 const * const isaac_parameter_d,
            isaac_functor_chain_pointer_N const * const isaac_function_chain_d,
#else
        __global__ void isaacFillRectKernel(
#endif
            uint32_t * const pixels,
            const isaac_size2 framebuffer_size,
            const isaac_uint2 framebuffer_start,
            const TSourceList sources,
            isaac_float step,
            const isaac_float4 background_color,
            const TTransferArray transferArray,
            const TSourceWeight sourceWeight,
            const TPointerArray pointerArray,
            const TScale scale)
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            #if ISAAC_ALPAKA == 1
                auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                isaac_uint2 pixel =
                {
                    isaac_uint(threadIdx[1]),
                    isaac_uint(threadIdx[2])
                };
            #else
                isaac_uint2 pixel =
                {
                    isaac_uint(threadIdx.x + blockIdx.x * blockDim.x),
                    isaac_uint(threadIdx.y + blockIdx.y * blockDim.y)
                };
            #endif
            pixel = pixel + framebuffer_start;
            if ( ISAAC_FOR_EACH_DIM_TWICE(2, pixel, >= framebuffer_size, || ) 0 )
                return;

            bool at_least_one;
            isaac_for_each_with_mpl_params( sources, check_no_source_iterator<TFilter>(), at_least_one );
            if (!at_least_one)
            {
                ISAAC_SET_COLOR( pixels[pixel.x + pixel.y * framebuffer_size.x], background_color )
                return;
            }
            
            isaac_float2 pixel_f =
            {
                isaac_float( pixel.x )/(isaac_float)framebuffer_size.x*isaac_float(2)-isaac_float(1),
                isaac_float( pixel.y )/(isaac_float)framebuffer_size.y*isaac_float(2)-isaac_float(1)
            };
            isaac_float4 start_p = {pixel_f.x*ISAAC_Z_NEAR,pixel_f.y*ISAAC_Z_NEAR,-1.0f*ISAAC_Z_NEAR,1.0f*ISAAC_Z_NEAR}; //znear
            isaac_float4   end_p = {pixel_f.x* ISAAC_Z_FAR,pixel_f.y* ISAAC_Z_FAR, 1.0f* ISAAC_Z_FAR,1.0f* ISAAC_Z_FAR}; //zfar
            isaac_float3 start =
            {
                isaac_inverse_d[ 0] * start_p.x + isaac_inverse_d[ 4] * start_p.y +  isaac_inverse_d[ 8] * start_p.z + isaac_inverse_d[12] * start_p.w,
                isaac_inverse_d[ 1] * start_p.x + isaac_inverse_d[ 5] * start_p.y +  isaac_inverse_d[ 9] * start_p.z + isaac_inverse_d[13] * start_p.w,
                isaac_inverse_d[ 2] * start_p.x + isaac_inverse_d[ 6] * start_p.y +  isaac_inverse_d[10] * start_p.z + isaac_inverse_d[14] * start_p.w
            };
            isaac_float3 end =
            {
                isaac_inverse_d[ 0] *   end_p.x + isaac_inverse_d[ 4] *   end_p.y +  isaac_inverse_d[ 8] *   end_p.z + isaac_inverse_d[12] *   end_p.w,
                isaac_inverse_d[ 1] *   end_p.x + isaac_inverse_d[ 5] *   end_p.y +  isaac_inverse_d[ 9] *   end_p.z + isaac_inverse_d[13] *   end_p.w,
                isaac_inverse_d[ 2] *   end_p.x + isaac_inverse_d[ 6] *   end_p.y +  isaac_inverse_d[10] *   end_p.z + isaac_inverse_d[14] *   end_p.w
            };
            isaac_float max_size = isaac_size_d[0].max_global_size_scaled / 2.0f;

            //scale to globale grid size
            start = start * max_size;
              end =   end * max_size;

            //move to local (scaled) grid
            isaac_int3 move =
            {
                isaac_int(isaac_size_d[0].global_size_scaled.value.x) / isaac_int(2) - isaac_int(isaac_size_d[0].position_scaled.value.x),
                isaac_int(isaac_size_d[0].global_size_scaled.value.y) / isaac_int(2) - isaac_int(isaac_size_d[0].position_scaled.value.y),
                isaac_int(isaac_size_d[0].global_size_scaled.value.z) / isaac_int(2) - isaac_int(isaac_size_d[0].position_scaled.value.z)
            };
            isaac_float3 move_f =
            {
                isaac_float(move.x),
                isaac_float(move.y),
                isaac_float(move.z)
            };
            start = start + move_f;
              end =   end + move_f;

            isaac_float3 vec = end - start;
            isaac_float l_scaled = sqrt( vec.x * vec.x + vec.y * vec.y + vec.z * vec.z );

            start.x = start.x / scale.x;
            start.y = start.y / scale.y;
            start.z = start.z / scale.z;
              end.x =   end.x / scale.x;
              end.y =   end.y / scale.y;
              end.z =   end.z / scale.z;

            vec = end - start;
            isaac_float l = sqrt( vec.x * vec.x + vec.y * vec.y + vec.z * vec.z );
            
            step *= l/l_scaled;
            
            isaac_float3 step_vec = vec / l * step;
            isaac_float3 count_start =  - start / step_vec;
            isaac_float3 local_size_f =
            {
                isaac_float(isaac_size_d[0].local_size.value.x),
                isaac_float(isaac_size_d[0].local_size.value.y),
                isaac_float(isaac_size_d[0].local_size.value.z)
            };
            isaac_float3 count_end = ( local_size_f - start ) / step_vec;

            //count_start shall have the smaller values
            ISAAC_SWITCH_IF_SMALLER( count_end.x, count_start.x )
            ISAAC_SWITCH_IF_SMALLER( count_end.y, count_start.y )
            ISAAC_SWITCH_IF_SMALLER( count_end.z, count_start.z )
            
            //calc intersection of all three super planes and save in [count_start.x ; count_end.x]
            count_start.x = max( max( count_start.x, count_start.y ), count_start.z );
              count_end.x = min( min(   count_end.x,   count_end.y ),   count_end.z );
            if ( count_start.x > count_end.x)
            {
                ISAAC_SET_COLOR( pixels[pixel.x + pixel.y * framebuffer_size.x], background_color )
                return;
            }
            
            isaac_int first = isaac_int( floor(count_start.x) );
            isaac_int last = isaac_int( ceil(count_end.x) );


            //Moving last and first until their points are valid
            isaac_float3 pos = start + step_vec * isaac_float(last);
            isaac_int3 coord = { isaac_int(floor(pos.x)), isaac_int(floor(pos.y)), isaac_int(floor(pos.z)) };
            while ( (ISAAC_FOR_EACH_DIM_TWICE(3, coord, >= isaac_size_d[0].local_size.value, || )
                     ISAAC_FOR_EACH_DIM      (3, coord, < 0 || ) 0 ) && first <= last)
            {
                last--;
                pos = start + step_vec * isaac_float(last);
                coord = { isaac_int(floor(pos.x)), isaac_int(floor(pos.y)), isaac_int(floor(pos.z)) };
            }
            pos = start + step_vec * isaac_float(first);
            coord = { isaac_int(floor(pos.x)), isaac_int(floor(pos.y)), isaac_int(floor(pos.z)) };
            while ( (ISAAC_FOR_EACH_DIM_TWICE(3, coord, >= isaac_size_d[0].local_size.value, || )
                     ISAAC_FOR_EACH_DIM      (3, coord, < 0 || ) 0 ) && first <= last)
            {
                first++;
                pos = start + step_vec * isaac_float(first);
                coord = { isaac_int(floor(pos.x)), isaac_int(floor(pos.y)), isaac_int(floor(pos.z)) };
            }

            //Starting the main loop
            isaac_float4 color = background_color;
            isaac_float min_size = min(
                int(isaac_size_d[0].global_size.value.x), min (
                int(isaac_size_d[0].global_size.value.y),
                int(isaac_size_d[0].global_size.value.z) ) );
            isaac_float factor = step / /*isaac_size_d[0].max_global_size*/ min_size;
            for (isaac_int i = first; i <= last; i++)
            {
                pos = start + step_vec * isaac_float(i);
                isaac_float4 value = {0, 0, 0, 0};
                isaac_int result = 0;
                isaac_for_each_with_mpl_params
                (
                    sources,
                    merge_source_iterator
                    <
                        Ttransfer_size,
                        TFilter,
                        TInterpolation,
                        TIsoSurface
                    >(),
                    value,
                    pos,
                    isaac_size_d[0].local_size,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    result,
                    step_vec,
                    step,
                    scale
#if ISAAC_ALPAKA == 1
                    ,isaac_parameter_d
                    ,isaac_function_chain_d
#endif
                );
                /*if ( mpl::size< TSourceList >::type::value > 1)
                    value = value / isaac_float( mpl::size< TSourceList >::type::value );*/
                if (TIsoSurface)
                {
                    if (result)
                    {
                        color = value;
                        break;
                    }
                }
                else
                {
                    isaac_float oma = isaac_float(1) - color.w;
                    value = value * factor;
                    isaac_float4 color_add =
                    {
                        oma * value.x, // * value.w does merge_source_iterator
                        oma * value.y, // * value.w does merge_source_iterator
                        oma * value.z, // * value.w does merge_source_iterator
                        oma * value.w
                    };
                    color = color + color_add;
                    if (color.w > isaac_float(0.99))
                        break;
                }
            }
            ISAAC_SET_COLOR( pixels[pixel.x + pixel.y * framebuffer_size.x], color )
        }
#if ISAAC_ALPAKA == 1
    };
#endif

template <
    typename TSimDim,
    typename TSourceList,
    typename TTransferArray,
    typename TSourceWeight,
    typename TPointerArray,
    typename TFilter,
    typename TFramebuffer,
    size_t TTransfer_size,
    typename TScale,
#if ISAAC_ALPAKA == 1
    typename TAccDim,
    typename TAcc,
    typename TStream,
    typename TInverse,
    typename TSize,
    typename TParameter,
    typename TFunctionChain,
#endif
    int N
>
struct IsaacFillRectKernelStruct
{
    inline static void call(
#if ISAAC_ALPAKA == 1
        const TInverse inverse_d,
        const TSize size_d,
        const TParameter parameter_d,
        TStream stream,
        const TFunctionChain function_chain_d,
#endif
        TFramebuffer framebuffer,
        const isaac_size2& framebuffer_size,
        const isaac_uint2& framebuffer_start,
        const TSourceList& sources,
        const isaac_float& step,
        const isaac_float4& background_color,
        const TTransferArray& transferArray,
        const TSourceWeight& sourceWeight,
        const TPointerArray& pointerArray,
        IceTInt const * const readback_viewport,
        const isaac_int interpolation,
        const isaac_int iso_surface,
        const TScale& scale
    )
    {
        if (sourceWeight.value[ mpl::size< TSourceList >::type::value - N] == isaac_float(0) )
            IsaacFillRectKernelStruct
            <
                TSimDim,
                TSourceList,
                TTransferArray,
                TSourceWeight,
                TPointerArray,
                typename mpl::push_back< TFilter, mpl::false_ >::type,
                TFramebuffer,
                TTransfer_size,
                TScale,
#if ISAAC_ALPAKA == 1
                TAccDim,
                TAcc,
                TStream,
                TInverse,
                TSize,
                TParameter,
                TFunctionChain,
#endif
                N - 1
            >
            ::call(
#if ISAAC_ALPAKA == 1
                inverse_d,
                size_d,
                parameter_d,
                stream,
                function_chain_d,
#endif
                framebuffer,
                framebuffer_size,
                framebuffer_start,
                sources,
                step,
                background_color,
                transferArray,
                sourceWeight,
                pointerArray,
                readback_viewport,
                interpolation,
                iso_surface,
                scale
            );
    else
            IsaacFillRectKernelStruct
            <
                TSimDim,
                TSourceList,
                TTransferArray,
                TSourceWeight,
                TPointerArray,
                typename mpl::push_back< TFilter, mpl::true_ >::type,
                TFramebuffer,
                TTransfer_size,
                TScale,
#if ISAAC_ALPAKA == 1
                TAccDim,
                TAcc,
                TStream,
                TInverse,
                TSize,
                TParameter,
                TFunctionChain,
#endif
                N - 1
            >
            ::call(
#if ISAAC_ALPAKA == 1
                inverse_d,
                size_d,
                parameter_d,
                stream,
                function_chain_d,
#endif
                framebuffer,
                framebuffer_size,
                framebuffer_start,
                sources,
                step,
                background_color,
                transferArray,
                sourceWeight,
                pointerArray,
                readback_viewport,
                interpolation,
                iso_surface,
                scale
            );
    }
};

template <
    typename TSimDim,
    typename TSourceList,
    typename TTransferArray,
    typename TSourceWeight,
    typename TPointerArray,
    typename TFilter,
    typename TFramebuffer,
    size_t TTransfer_size,
    typename TScale
#if ISAAC_ALPAKA == 1
    ,typename TAccDim
    ,typename TAcc
    ,typename TStream
    ,typename TInverse
    ,typename TSize
    ,typename TParameter
    ,typename TFunctionChain
#endif
>
struct IsaacFillRectKernelStruct
<
    TSimDim,
    TSourceList,
    TTransferArray,
    TSourceWeight,
    TPointerArray,
    TFilter,
    TFramebuffer,
    TTransfer_size,
    TScale,
#if ISAAC_ALPAKA == 1
    TAccDim,
    TAcc,
    TStream,
    TInverse,
    TSize,
    TParameter,
    TFunctionChain,
#endif
    0 //<-- spezialisation
>
{
    inline static void call(
#if ISAAC_ALPAKA == 1
        const TInverse inverse_d,
        const TSize size_d,
        const TParameter parameter_d,
        TStream stream,
        const TFunctionChain function_chain_d,
#endif
        TFramebuffer framebuffer,
        const isaac_size2& framebuffer_size,
        const isaac_uint2& framebuffer_start,
        const TSourceList& sources,
        const isaac_float& step,
        const isaac_float4& background_color,
        const TTransferArray& transferArray,
        const TSourceWeight& sourceWeight,
        const TPointerArray& pointerArray,
        IceTInt const * const readback_viewport,
        const isaac_int interpolation,
        const isaac_int iso_surface,
        const TScale& scale
    )
    {
        isaac_size2 grid_size=
        {
            size_t((readback_viewport[2]+15)/16),
            size_t((readback_viewport[3]+15)/16)
        };
        isaac_size2 block_size=
        {
            size_t(16),
            size_t(16)
        };
        #if ISAAC_ALPAKA == 1
            if ( mpl::not_<boost::is_same<TAcc, alpaka::acc::AccGpuCudaRt<TAccDim, size_t> > >::value )
            {
                grid_size.x = size_t(readback_viewport[2]);
                grid_size.y = size_t(readback_viewport[3]);
                block_size.x = size_t(1);
                block_size.y = size_t(1);                    
            }
            const alpaka::Vec<TAccDim, size_t> threads (size_t(1), size_t(1), size_t(1));
            const alpaka::Vec<TAccDim, size_t> blocks  (size_t(1), block_size.x, block_size.y);
            const alpaka::Vec<TAccDim, size_t> grid    (size_t(1), grid_size.x, grid_size.y);
            auto const workdiv(alpaka::workdiv::WorkDivMembers<TAccDim, size_t>(grid,blocks,threads));
            #define ISAAC_KERNEL_START \
            { \
                isaacFillRectKernel \
                < \
                    TSimDim, \
                    TSourceList, \
                    TTransferArray, \
                    TSourceWeight, \
                    TPointerArray, \
                    TFilter, \
                    TTransfer_size,
            #define ISAAC_KERNEL_END \
                    ,TScale \
                > \
                kernel; \
                auto const instance \
                ( \
                    alpaka::exec::create<TAcc> \
                    ( \
                        workdiv, \
                        kernel, \
                        alpaka::mem::view::getPtrNative(inverse_d), \
                        alpaka::mem::view::getPtrNative(size_d), \
                        alpaka::mem::view::getPtrNative(parameter_d), \
                        alpaka::mem::view::getPtrNative(function_chain_d), \
                        alpaka::mem::view::getPtrNative(framebuffer), \
                        framebuffer_size, \
                        framebuffer_start, \
                        sources, \
                        step, \
                        background_color, \
                        transferArray, \
                        sourceWeight, \
                        pointerArray, \
                        scale \
                    ) \
                ); \
                alpaka::stream::enqueue(stream, instance); \
            }
        #else
            dim3 block (block_size.x, block_size.y);
            dim3 grid  (grid_size.x, grid_size.y);
            #define ISAAC_KERNEL_START \
                isaacFillRectKernel \
                < \
                    TSimDim, \
                    TSourceList, \
                    TTransferArray, \
                    TSourceWeight, \
                    TPointerArray, \
                    TFilter, \
                    TTransfer_size,
            #define ISAAC_KERNEL_END \
                > \
                <<<grid, block>>> \
                ( \
                    framebuffer, \
                    framebuffer_size, \
                    framebuffer_start, \
                    sources, \
                    step, \
                    background_color, \
                    transferArray, \
                    sourceWeight, \
                    pointerArray, \
                    scale \
                );            
            
        #endif
        if (interpolation)
        {
            if (iso_surface)
                ISAAC_KERNEL_START
                    1,
                    1
                ISAAC_KERNEL_END
            else
                ISAAC_KERNEL_START
                    1,
                    0
                ISAAC_KERNEL_END
        }
        else
        {
            if (iso_surface)
                ISAAC_KERNEL_START
                    0,
                    1
                ISAAC_KERNEL_END
            else
                ISAAC_KERNEL_START
                    0,
                    0
                ISAAC_KERNEL_END
        }
        #undef ISAAC_KERNEL_START
        #undef ISAAC_KERNEL_END
    }
};

template <int N>
struct dest_array_struct
{
    isaac_int nr[N];
};

template
<
    int count,
    typename TDest
>
#if ISAAC_ALPAKA == 1
    struct updateFunctorChainPointerKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
#else
        __global__ void updateFunctorChainPointerKernel(
#endif
            isaac_functor_chain_pointer_N * const functor_chain_choose_d,
            isaac_functor_chain_pointer_N const * const functor_chain_d,
            TDest dest)
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            for (int i = 0; i < count; i++)
                functor_chain_choose_d[i] = functor_chain_d[dest.nr[i]];
        }
#if ISAAC_ALPAKA == 1
    };
#endif

template
<
    typename TSource
>
#if ISAAC_ALPAKA == 1
    struct updateBufferKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
#else
        __global__ void updateBufferKernel(
#endif
            const TSource source,
            void * const pointer,
            const isaac_int3 local_size)
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            #if ISAAC_ALPAKA == 1
                auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                isaac_int3 dest =
                {
                    isaac_int(threadIdx[1]),
                    isaac_int(threadIdx[2]),
                    0
                };
            #else
                isaac_int3 dest =
                {
                    isaac_int(threadIdx.x + blockIdx.x * blockDim.x),
                    isaac_int(threadIdx.y + blockIdx.y * blockDim.y),
                    0
                };
            #endif
            isaac_int3 coord = dest;
            coord.x -= ISAAC_GUARD_SIZE;
            coord.y -= ISAAC_GUARD_SIZE;
            if ( ISAAC_FOR_EACH_DIM_TWICE(2, dest, >= local_size, + 2 * ISAAC_GUARD_SIZE || ) 0 )
                return;
            isaac_float_dim < TSource::feature_dim >* ptr = (isaac_float_dim < TSource::feature_dim >*)(pointer);
            if (TSource::has_guard)
            {
                coord.z = -ISAAC_GUARD_SIZE;
                for (;dest.z < local_size.z + 2 * ISAAC_GUARD_SIZE; dest.z++)
                {
                    ptr[dest.x + dest.y * (local_size.x + 2 * ISAAC_GUARD_SIZE) + dest.z * ( (local_size.x + 2 * ISAAC_GUARD_SIZE) * (local_size.y + 2 * ISAAC_GUARD_SIZE) )] = source[coord];
                    coord.z++;
                }
            }
            else
            {
                if (coord.x < 0)
                    coord.x = 0;
                if (coord.x >= local_size.x)
                    coord.x = local_size.x-1;
                if (coord.y < 0)
                    coord.y = 0;
                if (coord.y >= local_size.y)
                    coord.y = local_size.y-1;
                coord.z = 0;
                for (; dest.z < ISAAC_GUARD_SIZE; dest.z++)
                    ptr[dest.x + dest.y * (local_size.x + 2 * ISAAC_GUARD_SIZE) + dest.z * ( (local_size.x + 2 * ISAAC_GUARD_SIZE) * (local_size.y + 2 * ISAAC_GUARD_SIZE) )] = source[coord];
                for (;dest.z < local_size.z + ISAAC_GUARD_SIZE - 1; dest.z++)
                {
                    ptr[dest.x + dest.y * (local_size.x + 2 * ISAAC_GUARD_SIZE) + dest.z * ( (local_size.x + 2 * ISAAC_GUARD_SIZE) * (local_size.y + 2 * ISAAC_GUARD_SIZE) )] = source[coord];
                    coord.z++;
                }
                for (;dest.z < local_size.z + 2 * ISAAC_GUARD_SIZE; dest.z++)
                    ptr[dest.x + dest.y * (local_size.x + 2 * ISAAC_GUARD_SIZE) + dest.z * ( (local_size.x + 2 * ISAAC_GUARD_SIZE) * (local_size.y + 2 * ISAAC_GUARD_SIZE) )] = source[coord];
            }
        }
#if ISAAC_ALPAKA == 1
    };
#endif

template
<
    typename TSource
>
#if ISAAC_ALPAKA == 1
    struct minMaxKernel
    {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator()(
            TAcc__ const &acc,
            isaac_functor_chain_pointer_N const * const isaac_function_chain_d,
            isaac_float4 const * const isaac_parameter_d,
#else
        __global__ void minMaxKernel(
#endif
            const TSource source,
            const int nr,
            minmax_struct * const result,
            const isaac_int3 local_size,
            void const * const pointer)
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            #if ISAAC_ALPAKA == 1
                auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                isaac_int3 coord =
                {
                    isaac_int(threadIdx[1]),
                    isaac_int(threadIdx[2]),
                    0
                };
            #else
                isaac_int3 coord =
                {
                    isaac_int(threadIdx.x + blockIdx.x * blockDim.x),
                    isaac_int(threadIdx.y + blockIdx.y * blockDim.y),
                    0
                };
            #endif
            if ( ISAAC_FOR_EACH_DIM_TWICE(2, coord, >= local_size, || ) 0 )
                return;
            isaac_float min =  FLT_MAX;
            isaac_float max = -FLT_MAX;
            for (;coord.z < local_size.z; coord.z++)
            {
                isaac_float_dim < TSource::feature_dim > data;
                if (TSource::persistent)
                    data = source[coord];
                else
                {
                    isaac_float_dim < TSource::feature_dim >* ptr = (isaac_float_dim < TSource::feature_dim >*)(pointer);
                    data = ptr[coord.x + ISAAC_GUARD_SIZE + (coord.y + ISAAC_GUARD_SIZE) * (local_size.x + 2 * ISAAC_GUARD_SIZE) + (coord.z + ISAAC_GUARD_SIZE) * ( (local_size.x + 2 * ISAAC_GUARD_SIZE) * (local_size.y + 2 * ISAAC_GUARD_SIZE) )];
                };
                isaac_float value = isaac_float(0);
                #if ISAAC_ALPAKA == 1
                    #define ISAAC_PARAMETER_PARAM ,isaac_parameter_d
                #else
                    #define ISAAC_PARAMETER_PARAM
                #endif            
                #if ISAAC_ALPAKA == 1 || defined(__CUDA_ARCH__)
                    if (TSource::feature_dim == 1)
                        value = reinterpret_cast<isaac_functor_chain_pointer_1>(isaac_function_chain_d[ nr ])( *(reinterpret_cast< isaac_float_dim<1>* >(&data)), nr ISAAC_PARAMETER_PARAM );
                    if (TSource::feature_dim == 2)
                        value = reinterpret_cast<isaac_functor_chain_pointer_2>(isaac_function_chain_d[ nr ])( *(reinterpret_cast< isaac_float_dim<2>* >(&data)), nr ISAAC_PARAMETER_PARAM );
                    if (TSource::feature_dim == 3)
                        value = reinterpret_cast<isaac_functor_chain_pointer_3>(isaac_function_chain_d[ nr ])( *(reinterpret_cast< isaac_float_dim<3>* >(&data)), nr ISAAC_PARAMETER_PARAM );
                    if (TSource::feature_dim == 4)
                        value = reinterpret_cast<isaac_functor_chain_pointer_4>(isaac_function_chain_d[ nr ])( *(reinterpret_cast< isaac_float_dim<4>* >(&data)), nr ISAAC_PARAMETER_PARAM );
                #endif
                #undef ISAAC_PARAMETER_PARAM
                if (value > max)
                    max = value;
                if (value < min)
                    min = value;
            }
            result[coord.x +  coord.y * local_size.x].min = min;
            result[coord.x +  coord.y * local_size.x].max = max;
        }
#if ISAAC_ALPAKA == 1
    };
#endif

} //namespace isaac;

#pragma GCC diagnostic pop
