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
#include <limits>

#include <float.h>


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wsign-compare"

namespace isaac
{

    namespace fus = boost::fusion;
    namespace mpl = boost::mpl;

    typedef isaac_float( * isaac_functor_chain_pointer_4 )(
        isaac_float_dim< 4 >,
        isaac_int
    );

    typedef isaac_float( * isaac_functor_chain_pointer_3 )(
        isaac_float_dim< 3 >,
        isaac_int
    );

    typedef isaac_float( * isaac_functor_chain_pointer_2 )(
        isaac_float_dim< 2 >,
        isaac_int
    );

    typedef isaac_float( * isaac_functor_chain_pointer_1 )(
        isaac_float_dim< 1 >,
        isaac_int
    );

    typedef isaac_float( * isaac_functor_chain_pointer_N )(
        void *,
        isaac_int
    );

    //inverse mvp matrix
    ISAAC_CONSTANT isaac_float
    isaac_inverse_d[16];

    //modelview matrix
    ISAAC_CONSTANT 
    isaac_float isaac_modelview_d[16];

    //projection matrix
    ISAAC_CONSTANT 
    isaac_float isaac_projection_d[16];

    //simulation size properties
    ISAAC_CONSTANT isaac_size_struct< 3 >
    isaac_size_d[1]; //[1] to access it for cuda and alpaka the same way

    ISAAC_CONSTANT isaac_float4
    isaac_parameter_d[ ISAAC_MAX_SOURCES*ISAAC_MAX_FUNCTORS ];

    ISAAC_CONSTANT isaac_functor_chain_pointer_N
    isaac_function_chain_d[ ISAAC_MAX_SOURCES ];


    /* 
     * SSAO
     * Kernels for ssao calculation
     */

    //filter kernel
    ISAAC_CONSTANT 
    isaac_float3 ssao_kernel_d[64];

    //vector rotation noise kernel
    ISAAC_CONSTANT 
    isaac_float3 ssao_noise_d[16];



    template<
        typename T
    >
    ISAAC_DEVICE_INLINE


    int sgn( T val )
    {
        return ( T( 0 ) < val ) - ( val < T( 0 ) );
    }


    template<
        typename TFunctorVector,
        int TFeatureDim,
        int NR
    >
    struct FillFunctorChainPointerKernelStruct
    {
        ISAAC_DEVICE static isaac_functor_chain_pointer_N
        call( isaac_int const * const bytecode )
        {
#define ISAAC_SUB_CALL( Z, I, U ) \
            if (bytecode[ISAAC_MAX_FUNCTORS-NR] == I) \
                return FillFunctorChainPointerKernelStruct \
                < \
                    typename mpl::push_back< TFunctorVector, typename boost::mpl::at_c<IsaacFunctorPool,I>::type >::type, \
                    TFeatureDim, \
                    NR - 1 \
                > ::call( bytecode );
            BOOST_PP_REPEAT(
                ISAAC_FUNCTOR_COUNT,
                ISAAC_SUB_CALL,
                ~
            )
#undef ISAAC_SUB_CALL
            return NULL; //Should never be reached anyway
        }
    };


    template<
        typename TFunctorVector,
        int TFeatureDim
    >
    ISAAC_DEVICE isaac_float applyFunctorChain(
        isaac_float_dim <TFeatureDim> const value,
        isaac_int const src_id
    )
    {
#define  ISAAC_LEFT_DEF( Z, I, U ) mpl::at_c< TFunctorVector, ISAAC_MAX_FUNCTORS - I - 1 >::type::call(
#define ISAAC_RIGHT_DEF( Z, I, U ) , isaac_parameter_d[ src_id * ISAAC_MAX_FUNCTORS + I ] )
#define  ISAAC_LEFT BOOST_PP_REPEAT( ISAAC_MAX_FUNCTORS, ISAAC_LEFT_DEF, ~)
#define ISAAC_RIGHT BOOST_PP_REPEAT( ISAAC_MAX_FUNCTORS, ISAAC_RIGHT_DEF, ~)
        // expands to: funcN( ... func1( func0( data, p[0] ), p[1] ) ... p[N] );
        return ISAAC_LEFT
        value
        ISAAC_RIGHT.value
            .x;
#undef ISAAC_LEFT_DEF
#undef ISAAC_LEFT
#undef ISAAC_RIGHT_DEF
#undef ISAAC_RIGHT
    }


    template<
        typename TFunctorVector,
        int TFeatureDim
    >
    struct FillFunctorChainPointerKernelStruct<
        TFunctorVector,
        TFeatureDim,
        0 //<- Specialization
    >
    {
        ISAAC_DEVICE static isaac_functor_chain_pointer_N
        call( isaac_int const * const bytecode )
        {
            return reinterpret_cast<isaac_functor_chain_pointer_N> ( applyFunctorChain<
                TFunctorVector,
                TFeatureDim
            > );
        }
    };


#if ISAAC_ALPAKA == 1

    struct fillFunctorChainPointerKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
#else


            __global__ void fillFunctorChainPointerKernel(
#endif
            isaac_functor_chain_pointer_N * const functor_chain_d
        )
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            isaac_int bytecode[ISAAC_MAX_FUNCTORS];
            for( int i = 0; i < ISAAC_MAX_FUNCTORS; i++ )
            {
                bytecode[i] = 0;
            }
            for( int i = 0; i < ISAAC_FUNCTOR_COMPLEX;
            i++ )
            {
                functor_chain_d[i * 4 + 0] =
                    FillFunctorChainPointerKernelStruct<
                        mpl::vector< >,
                        1,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                functor_chain_d[i * 4 + 1] =
                    FillFunctorChainPointerKernelStruct<
                        mpl::vector< >,
                        2,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                functor_chain_d[i * 4 + 2] =
                    FillFunctorChainPointerKernelStruct<
                        mpl::vector< >,
                        3,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                functor_chain_d[i * 4 + 3] =
                    FillFunctorChainPointerKernelStruct<
                        mpl::vector< >,
                        4,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                for( int j = ISAAC_MAX_FUNCTORS - 1; j >= 0; j-- )
                {
                    if( bytecode[j] < ISAAC_FUNCTOR_COUNT - 1 )
                    {
                        bytecode[j]++;
                        break;
                    }
                    else
                    {
                        bytecode[j] = 0;
                    }
                }
            }
        }


#if ISAAC_ALPAKA == 1
    };

#endif

    template<
        isaac_int TInterpolation,
        typename NR,
        typename TSource,
        typename TPos,
        typename TPointerArray,
        typename TLocalSize,
        typename TScale
    >
    ISAAC_HOST_DEVICE_INLINE isaac_float


    get_value(
        const TSource & source,
        const TPos & pos,
        const TPointerArray & pointerArray,
        const TLocalSize & local_size,
        const TScale & scale
    )
    {
        isaac_float_dim <TSource::feature_dim> data;
        isaac_float_dim <TSource::feature_dim> * ptr = (
            isaac_float_dim < TSource::feature_dim > *
        )( pointerArray.pointer[NR::value] );
        if( TInterpolation == 0 )
        {
            isaac_int3 coord = {
                isaac_int( pos.x ),
                isaac_int( pos.y ),
                isaac_int( pos.z )
            };
            if( TSource::persistent )
            {
                data = source[coord];
            }
            else
            {
                data = ptr[coord.x + ISAAC_GUARD_SIZE
                           + ( coord.y + ISAAC_GUARD_SIZE ) * (
                               local_size.value
                                   .x + 2 * ISAAC_GUARD_SIZE
                           ) + ( coord.z + ISAAC_GUARD_SIZE ) * (
                    (
                        local_size.value
                            .x + 2 * ISAAC_GUARD_SIZE
                    ) * (
                        local_size.value
                            .y + 2 * ISAAC_GUARD_SIZE
                    )
                )];
            }
        }
        else
        {
            isaac_int3 coord;
            isaac_float_dim <TSource::feature_dim> data8[2][2][2];
            for( int x = 0; x < 2; x++ )
            {
                for( int y = 0; y < 2; y++ )
                {
                    for( int z = 0; z < 2; z++ )
                    {
                        coord.x =
                            isaac_int( x ? ceil( pos.x ) : floor( pos.x ) );
                        coord.y =
                            isaac_int( y ? ceil( pos.y ) : floor( pos.y ) );
                        coord.z =
                            isaac_int( z ? ceil( pos.z ) : floor( pos.z ) );
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            if( isaac_uint( coord.x ) >= local_size.value
                                .x )
                            {
                                coord.x = isaac_int(
                                    x ? floor( pos.x ) : ceil( pos.x )
                                );
                            }
                            if( isaac_uint( coord.y ) >= local_size.value
                                .y )
                            {
                                coord.y = isaac_int(
                                    y ? floor( pos.y ) : ceil( pos.y )
                                );
                            }
                            if( isaac_uint( coord.z ) >= local_size.value
                                .z )
                            {
                                coord.z = isaac_int(
                                    z ? floor( pos.z ) : ceil( pos.z )
                                );
                            }
                        }
                        if( TSource::persistent )
                        {
                            data8[x][y][z] = source[coord];
                        }
                        else
                        {
                            data8[x][y][z] = ptr[coord.x + ISAAC_GUARD_SIZE +
                                                 ( coord.y + ISAAC_GUARD_SIZE )
                                                 * (
                                                     local_size.value
                                                         .x
                                                     + 2 * ISAAC_GUARD_SIZE
                                                 ) +
                                                 ( coord.z + ISAAC_GUARD_SIZE )
                                                 * (
                                                     (
                                                         local_size.value
                                                             .x
                                                         + 2 * ISAAC_GUARD_SIZE
                                                     ) * (
                                                         local_size.value
                                                             .y
                                                         + 2 * ISAAC_GUARD_SIZE
                                                     )
                                                 )];
                        }
                    }
                }
            }
            //Against annoying double->float casting warning with gcc5
#if __CUDACC_VER_MAJOR__ > 7
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif
            isaac_float_dim< 3 > pos_in_cube = {
                pos.x - floor( pos.x ),
                pos.y - floor( pos.y ),
                pos.z - floor( pos.z )
            };
#if __CUDACC_VER_MAJOR__ > 7
#pragma GCC diagnostic pop
#endif
            isaac_float_dim <TSource::feature_dim> data4[2][2];
            for( int x = 0; x < 2; x++ )
            {
                for( int y = 0; y < 2; y++ )
                {
                    data4[x][y].value = data8[x][y][0].value * (
                        isaac_float( 1 ) - pos_in_cube.value
                            .z
                    ) + data8[x][y][1].value * (
                        pos_in_cube.value
                            .z
                    );
                }
            }
            isaac_float_dim <TSource::feature_dim> data2[2];
            for( int x = 0; x < 2; x++ )
            {
                data2[x].value = data4[x][0].value * (
                    isaac_float( 1 ) - pos_in_cube.value
                        .y
                ) + data4[x][1].value * (
                    pos_in_cube.value
                        .y
                );
            }
            data.value = data2[0].value * (
                isaac_float( 1 ) - pos_in_cube.value
                    .x
            ) + data2[1].value * (
                pos_in_cube.value
                    .x
            );
        }
        isaac_float result = isaac_float( 0 );

#if ISAAC_ALPAKA == 1 || defined(__CUDA_ARCH__)
        if( TSource::feature_dim == 1 )
        {
            result =
                reinterpret_cast<isaac_functor_chain_pointer_1> ( isaac_function_chain_d[NR::value] )(
                    *( reinterpret_cast< isaac_float_dim< 1 > * > ( &data ) ),
                    NR::value
                );
        }
        if( TSource::feature_dim == 2 )
        {
            result =
                reinterpret_cast<isaac_functor_chain_pointer_2> ( isaac_function_chain_d[NR::value] )(
                    *( reinterpret_cast< isaac_float_dim< 2 > * > ( &data ) ),
                    NR::value
                );
        }
        if( TSource::feature_dim == 3 )
        {
            result =
                reinterpret_cast<isaac_functor_chain_pointer_3> ( isaac_function_chain_d[NR::value] )(
                    *( reinterpret_cast< isaac_float_dim< 3 > * > ( &data ) ),
                    NR::value
                );
        }
        if( TSource::feature_dim == 4 )
        {
            result =
                reinterpret_cast<isaac_functor_chain_pointer_4> ( isaac_function_chain_d[NR::value] )(
                    *( reinterpret_cast< isaac_float_dim< 4 > * > ( &data ) ),
                    NR::value
                );
        }
#endif
        return result;
    }

    /**
     * @brief Clamps coordinates to min/max
     * 
     * @tparam TInterpolation 
     * @tparam TLocalSize 
     * @param coord 
     * @param local_size 
     * @return ISAAC_HOST_DEVICE_INLINE check_coord clamped coordiantes
     */
    template<
        bool TInterpolation,
        typename TLocalSize
    >
    ISAAC_HOST_DEVICE_INLINE void
    check_coord(
        isaac_float3 & coord,
        const TLocalSize local_size
    )
    {
        constexpr auto extra_border = static_cast<decltype( local_size.value
            .x )>(TInterpolation);
        if( coord.x < isaac_float( 0 ) )
        {
            coord.x = isaac_float( 0 );
        }
        if( coord.y < isaac_float( 0 ) )
        {
            coord.y = isaac_float( 0 );
        }
        if( coord.z < isaac_float( 0 ) )
        {
            coord.z = isaac_float( 0 );
        }
        if( coord.x >= isaac_float(
            local_size.value
                .x - extra_border
        ) )
        {
            coord.x = isaac_float(
                local_size.value
                    .x - extra_border
            ) - FLT_MIN;
        }
        if( coord.y >= isaac_float(
            local_size.value
                .y - extra_border
        ) )
        {
            coord.y = isaac_float(
                local_size.value
                    .y - extra_border
            ) - FLT_MIN;
        }
        if( coord.z >= isaac_float(
            local_size.value
                .z - extra_border
        ) )
        {
            coord.z = isaac_float(
                local_size.value
                    .z - extra_border
            ) - FLT_MIN;
        }
    }

    /**
     * @brief Clamps coordinates to min/max +- Guard margin
     * 
     * @tparam TInterpolation 
     * @tparam TLocalSize 
     * @param coord 
     * @param local_size 
     * @return ISAAC_HOST_DEVICE_INLINE check_coord_with_guard clamped coordinate
     */
    template<
        bool TInterpolation,
        typename TLocalSize
    >
    ISAAC_HOST_DEVICE_INLINE void
    check_coord_with_guard(
        isaac_float3 & coord,
        const TLocalSize local_size
    )
    {
        constexpr auto extra_border = static_cast<decltype( local_size.value
            .x )>(TInterpolation);
        if( coord.x < isaac_float( -ISAAC_GUARD_SIZE ) )
        {
            coord.x = isaac_float( -ISAAC_GUARD_SIZE );
        }
        if( coord.y < isaac_float( -ISAAC_GUARD_SIZE ) )
        {
            coord.y = isaac_float( -ISAAC_GUARD_SIZE );
        }
        if( coord.z < isaac_float( -ISAAC_GUARD_SIZE ) )
        {
            coord.z = isaac_float( -ISAAC_GUARD_SIZE );
        }
        if( coord.x >= isaac_float(
            local_size.value
                .x + ISAAC_GUARD_SIZE - extra_border
        ) )
        {
            coord.x = isaac_float(
                local_size.value
                    .x + ISAAC_GUARD_SIZE - extra_border
            ) - FLT_MIN;
        }
        if( coord.y >= isaac_float(
            local_size.value
                .y + ISAAC_GUARD_SIZE - extra_border
        ) )
        {
            coord.y = isaac_float(
                local_size.value
                    .y + ISAAC_GUARD_SIZE - extra_border
            ) - FLT_MIN;
        }
        if( coord.z >= isaac_float(
            local_size.value
                .z + ISAAC_GUARD_SIZE - extra_border
        ) )
        {
            coord.z = isaac_float(
                local_size.value
                    .z + ISAAC_GUARD_SIZE - extra_border
            ) - FLT_MIN;
        }
    }

    /**
     * @brief Checks for collision with source particles
     * 
     * Returns color, normal, position of particle
     * 
     * @tparam Ttransfer_size 
     * @tparam TOffset 
     * @tparam TFilter 
     */
    template<
        ISAAC_IDX_TYPE Ttransfer_size,
        int TOffset,
        typename TFilter
    >
    struct merge_particle_iterator
    {
        template<
            typename NR,
            typename TSource,
            typename TColor,
            typename TNormal,
            typename TPosition,
            typename TStart,
            typename TDir,
            typename TLightDir,
            typename TCellPos,
            typename TTransferArray,
            typename TSourceWeight,
            typename TFeedback,
            typename TParticleScale,
            typename TClippingNormal,
            typename TClipped
        >
        ISAAC_HOST_DEVICE_INLINE void operator()(
            const NR & nr,
            const TSource & source,                     //particle source
            TColor & color,                             //particle color
            TNormal & normal,                           //particle normal
            TPosition & position,                       //particle hit position
            const TStart & start,
            const TDir & dir,
            const TLightDir & light_dir,                //direction of incoming light
            const TCellPos & cell_pos,
            const TTransferArray & transferArray,
            const TSourceWeight & sourceWeight,
            TFeedback & feedback,                       //true or false if particle has been hit or not
            const TParticleScale & particle_scale,
            const TClippingNormal & clipping_normal,
            const TClipped & is_clipped
        ) const
        {
            const int sourceNumber = NR::value + TOffset;
            if( mpl::at_c<
                TFilter,
                sourceNumber
            >::type::value )
            {
                auto particle_iterator = source.getIterator( cell_pos );
                isaac_float3 cell_pos_f = {
                    isaac_float( cell_pos.x ),
                    isaac_float( cell_pos.y ),
                    isaac_float( cell_pos.z )
                };
                // iterate over all particles in current cell
                for( int i = 0; i < particle_iterator.size; i++ )
                {
                    // ray sphere intersection
                    isaac_float3 particle_pos =
                        ( particle_iterator.getPosition( ) + cell_pos_f )
                        * particle_scale;
                    isaac_float3 L = particle_pos - start;
                    isaac_float radius = particle_iterator.getRadius( )
                                         * sourceWeight.value[NR::value
                                                              + TOffset];
                    isaac_float radius2 = radius * radius;
                    isaac_float tca = L.x * dir.x + L.y * dir.y + L.z * dir.z;
                    isaac_float
                        d2 = ( L.x * L.x + L.y * L.y + L.z * L.z ) - tca * tca;
                    if( d2 > radius2 )
                    {
                        particle_iterator.next( );
                        continue;
                    }
                    isaac_float thc = sqrt( radius2 - d2 );
                    isaac_float t0 = tca - thc;
                    isaac_float t1 = tca + thc;

                    // if the ray hits the sphere
                    if( t1 >= 0 && t0 < color.w )
                    {
                        isaac_float_dim <TSource::feature_dim>
                            data = particle_iterator.getAttribute( );

                        isaac_float result = isaac_float( 0 );

                        // apply functorchain
#if ISAAC_ALPAKA == 1 || defined(__CUDA_ARCH__)
                        if( TSource::feature_dim == 1 )
                        {
                            result =
                                reinterpret_cast<isaac_functor_chain_pointer_1> ( isaac_function_chain_d[sourceNumber] )(
                                    *( reinterpret_cast< isaac_float_dim< 1 > * > ( &data ) ),
                                    sourceNumber
                                );
                        }
                        if( TSource::feature_dim == 2 )
                        {
                            result =
                                reinterpret_cast<isaac_functor_chain_pointer_2> ( isaac_function_chain_d[sourceNumber] )(
                                    *( reinterpret_cast< isaac_float_dim< 2 > * > ( &data ) ),
                                    sourceNumber
                                );
                        }
                        if( TSource::feature_dim == 3 )
                        {
                            result =
                                reinterpret_cast<isaac_functor_chain_pointer_3> ( isaac_function_chain_d[sourceNumber] )(
                                    *( reinterpret_cast< isaac_float_dim< 3 > * > ( &data ) ),
                                    sourceNumber
                                );
                        }
                        if( TSource::feature_dim == 4 )
                        {
                            result =
                                reinterpret_cast<isaac_functor_chain_pointer_4> ( isaac_function_chain_d[sourceNumber] )(
                                    *( reinterpret_cast< isaac_float_dim< 4 > * > ( &data ) ),
                                    sourceNumber
                                );
                        }
#endif

                        // apply transferfunction
                        isaac_int lookup_value = isaac_int(
                            round( result * isaac_float( Ttransfer_size ) )
                        );
                        if( lookup_value < 0 )
                        {
                            lookup_value = 0;
                        }
                        if( lookup_value >= Ttransfer_size )
                        {
                            lookup_value = Ttransfer_size - 1;
                        }
                        isaac_float4 value = transferArray.pointer[NR::value
                                                                   + TOffset][lookup_value];

                        // check if the alpha value is greater or equal than 0.5
                        if( value.w >= 0.5f )
                        {
                            color = value;
                            color.w = t0;
                            feedback = 1;
                            position = particle_pos;
                            normal = start + t0 * dir - particle_pos;
                            if( t0 < 0 && is_clipped )
                            {
                                #if ISAAC_AO_BUG_FIX == 1
                                   color.w = 0;
                                #endif
                                normal = -clipping_normal;
                            }
                        }
                    }
                    particle_iterator.next( );
                }
            }
        }
    };


    template<
        ISAAC_IDX_TYPE Ttransfer_size,
        typename TFilter,
        isaac_int TInterpolation,
        isaac_int TIsoSurface
    >
    struct merge_source_iterator
    {
        template<
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
            typename TScale,
            typename TFirst,
            typename TStartNormal
        >
        ISAAC_HOST_DEVICE_INLINE void operator()(
            const NR & nr,
            const TSource & source,
            TColor & color,
            const TPos & pos,
            const TLocalSize & local_size,
            const TTransferArray & transferArray,
            const TSourceWeight & sourceWeight,
            const TPointerArray & pointerArray,
            TFeedback & feedback,
            const TStep & step,
            const TStepLength & stepLength,
            const TScale & scale,
            const TFirst & first,
            const TStartNormal & start_normal
        ) const
        {
            if( mpl::at_c<
                TFilter,
                NR::value
            >::type::value )
            {
                isaac_float result = get_value<
                    TInterpolation,
                    NR
                >(
                    source,
                    pos,
                    pointerArray,
                    local_size,
                    scale
                );
                isaac_int lookup_value = isaac_int(
                    round( result * isaac_float( Ttransfer_size ) )
                );
                if( lookup_value < 0 )
                {
                    lookup_value = 0;
                }
                if( lookup_value >= Ttransfer_size )
                {
                    lookup_value = Ttransfer_size - 1;
                }
                isaac_float4
                    value = transferArray.pointer[NR::value][lookup_value];
                if( TIsoSurface )
                {
                    if( value.w >= isaac_float( 0.5 ) )
                    {
                        isaac_float3 left = {
                            -1,
                            0,
                            0
                        };
                        left = left + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                left,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                left,
                                local_size
                            );
                        }
                        isaac_float3 right = {
                            1,
                            0,
                            0
                        };
                        right = right + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                right,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                right,
                                local_size
                            );
                        }
                        isaac_float d1;
                        if( TInterpolation )
                        {
                            d1 = right.x - left.x;
                        }
                        else
                        {
                            d1 = isaac_int( right.x ) - isaac_int( left.x );
                        }

                        isaac_float3 up = {
                            0,
                            -1,
                            0
                        };
                        up = up + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                up,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                up,
                                local_size
                            );
                        }
                        isaac_float3 down = {
                            0,
                            1,
                            0
                        };
                        down = down + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                down,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                down,
                                local_size
                            );
                        }
                        isaac_float d2;
                        if( TInterpolation )
                        {
                            d2 = down.y - up.y;
                        }
                        else
                        {
                            d2 = isaac_int( down.y ) - isaac_int( up.y );
                        }

                        isaac_float3 front = {
                            0,
                            0,
                            -1
                        };
                        front = front + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                front,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                front,
                                local_size
                            );
                        }
                        isaac_float3 back = {
                            0,
                            0,
                            1
                        };
                        back = back + pos;
                        if( !TSource::has_guard && TSource::persistent )
                        {
                            check_coord< TInterpolation >(
                                back,
                                local_size
                            );
                        }
                        else
                        {
                            check_coord_with_guard< TInterpolation >(
                                back,
                                local_size
                            );
                        }
                        isaac_float d3;
                        if( TInterpolation )
                        {
                            d3 = back.z - front.z;
                        }
                        else
                        {
                            d3 = isaac_int( back.z ) - isaac_int( front.z );
                        }

                        isaac_float3 gradient = {
                            (
                                get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    right,
                                    pointerArray,
                                    local_size,
                                    scale
                                ) - get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    left,
                                    pointerArray,
                                    local_size,
                                    scale
                                )
                            ) / d1,
                            (
                                get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    down,
                                    pointerArray,
                                    local_size,
                                    scale
                                ) - get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    up,
                                    pointerArray,
                                    local_size,
                                    scale
                                )
                            ) / d2,
                            (
                                get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    back,
                                    pointerArray,
                                    local_size,
                                    scale
                                ) - get_value<
                                    TInterpolation,
                                    NR
                                >(
                                    source,
                                    front,
                                    pointerArray,
                                    local_size,
                                    scale
                                )
                            ) / d3
                        };
                        if( first )
                        {
                            gradient.x = start_normal.x;
                            gradient.y = start_normal.y;
                            gradient.z = start_normal.z;
                        }
                        isaac_float l = sqrt(
                            gradient.x * gradient.x + gradient.y * gradient.y
                            + gradient.z * gradient.z
                        );
                        if( l == isaac_float( 0 ) )
                        {
                            color = value;
                        }
                        else
                        {
                            gradient = gradient / l;
                            isaac_float3 light = step / stepLength;
                            isaac_float ac = fabs(
                                gradient.x * light.x + gradient.y * light.y
                                + gradient.z * light.z
                            );
#if ISAAC_SPECULAR == 1
                            color.x = value.x * ac + ac * ac * ac * ac;
                            color.y = value.y * ac + ac * ac * ac * ac;
                            color.z = value.z * ac + ac * ac * ac * ac;
#else
                            color.x = value.x * ac;
                            color.y = value.y * ac;
                            color.z = value.z * ac;
#endif
                        }
                        color.w = isaac_float( 1 );
                        feedback = 1;
                    }
                }
                else
                {
                    value.w *= sourceWeight.value[NR::value];
                    color.x = color.x + value.x * value.w;
                    color.y = color.y + value.y * value.w;
                    color.z = color.z + value.z * value.w;
                    color.w = color.w + value.w;
                }
            }
        }
    };

    template<
        typename TFilter
    >
    struct check_no_source_iterator
    {
        template<
            typename NR,
            typename TSource,
            typename TResult
        >
        ISAAC_HOST_DEVICE_INLINE void operator()(
            const NR & nr,
            const TSource & source,
            TResult & result
        ) const
        {
            result |= mpl::at_c<
                TFilter,
                NR::value
            >::type::value;
        }
    };

    constexpr auto maxFloat = std::numeric_limits< isaac_float >::max( );


    template<
        typename TSimDim,
        typename TParticleList,
        typename TSourceList,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TFilter,
        ISAAC_IDX_TYPE Ttransfer_size,
        isaac_int TInterpolation,
        isaac_int TIsoSurface,
        typename TScale
    >
#if ISAAC_ALPAKA == 1
    struct isaacRenderKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
#else
            __global__ void isaacRenderKernel(
#endif
            uint32_t * const pixels,                //ptr to output pixels
            isaac_float3 * const gDepth,            //depth buffer
            isaac_float3 * const gNormal,           //normal buffer
            const isaac_size2 framebuffer_size,     //size of framebuffer
            const isaac_uint2 framebuffer_start,    //framebuffer offset
            const TParticleList particle_sources,   //source simulation particles
            const TSourceList sources,              //source of volumes
            isaac_float step,                       //ray step length
            const isaac_float4 background_color,    //color of render background
            const TTransferArray transferArray,     //mapping to simulation memory
            const TSourceWeight sourceWeight,       //weights of sources for blending
            const TPointerArray pointerArray,
            const TScale scale,                     //isaac set scaling
            const clipping_struct input_clipping,   //clipping planes
            const ao_struct ambientOcclusion        //ambient occlusion params
        )
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            isaac_uint2 pixel[ISAAC_VECTOR_ELEM];
            bool finish[ISAAC_VECTOR_ELEM];

            //get pixel values from thread ids
#if ISAAC_ALPAKA == 1
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            ISAAC_ELEM_ITERATE ( e )
            {
                pixel[e].x = isaac_uint( alpThreadIdx[2] )
                             * isaac_uint( ISAAC_VECTOR_ELEM ) + e;
                pixel[e].y = isaac_uint( alpThreadIdx[1] );
#else
                ISAAC_ELEM_ITERATE ( e )
                {
                    pixel[e].x = isaac_uint( threadIdx.x + blockIdx.x * blockDim.x )
                                 * isaac_uint( ISAAC_VECTOR_ELEM ) + e;
                    pixel[e].y = isaac_uint( threadIdx.y + blockIdx.y * blockDim.y );
#endif
                //apply framebuffer offset to pixel
                //stop if pixel position is out of bounds
                finish[e] = false;
                pixel[e] = pixel[e] + framebuffer_start;
                if( ISAAC_FOR_EACH_DIM_TWICE ( 2,
                        pixel[e],
                        >= framebuffer_size,
                        || ) 0 )
                    finish[e] = true;

            }
            ISAAC_ELEM_ALL_TRUE_RETURN ( finish )
            bool at_least_one[ISAAC_VECTOR_ELEM];
            isaac_float4 color[ISAAC_VECTOR_ELEM];

            //gNormalBuffer default value
            isaac_float3 default_normal = {0.0, 0.0, 0.0};
            isaac_float3 default_depth = {1.0, 1.0, 1.0};

            //set background color
            ISAAC_ELEM_ITERATE ( e )
            {
                color[e] = background_color;
                at_least_one[e] = true;
                isaac_for_each_with_mpl_params(
                    sources,
                    check_no_source_iterator< TFilter >( ),
                    at_least_one[e]
                );
                if( !at_least_one[e] )
                {
                    if( !finish[e] ) {
                        //if no source is found set all values to their defaults
                        //color = background, ...
                        ISAAC_SET_COLOR ( 
                            pixels[pixel[e].x + pixel[e].y * framebuffer_size.x], 
                            color[e] 
                        )
                        gNormal[pixel[e].x + pixel[e].y * framebuffer_size.x] = default_normal;
                        gDepth[pixel[e].x + pixel[e].y * framebuffer_size.x] = default_depth;
                    }
                    finish[e] = true;
                }
            }
            ISAAC_ELEM_ALL_TRUE_RETURN ( finish )

            isaac_float2 pixel_f[ISAAC_VECTOR_ELEM];        //relative pixel position in framebuffer [0.0 ... 1.0]
            isaac_float4 start_p[ISAAC_VECTOR_ELEM];        //ray start position
            isaac_float4 end_p[ISAAC_VECTOR_ELEM];          //ray end position
            isaac_float3 start[ISAAC_VECTOR_ELEM];          //ray start position (world space)
            isaac_float3 end[ISAAC_VECTOR_ELEM];            //ray end position (world space)
            isaac_int3 move[ISAAC_VECTOR_ELEM];             //offset of subvolume
            isaac_float3 move_f[ISAAC_VECTOR_ELEM];         //offset of subvolume as float
            clipping_struct clipping[ISAAC_VECTOR_ELEM];    //clipping planes with transformed positions
            isaac_float3 vec[ISAAC_VECTOR_ELEM];            //temp storage vector
            isaac_float l_scaled[ISAAC_VECTOR_ELEM];        //scaled length of ray
            isaac_float l[ISAAC_VECTOR_ELEM];               //isaac scaled length of ray
            isaac_float3 step_vec[ISAAC_VECTOR_ELEM];       //ray direction vector scaled by step length
            isaac_float3 count_start[ISAAC_VECTOR_ELEM];    //start index for ray
            isaac_float3 local_size_f[ISAAC_VECTOR_ELEM];   //subvolume size as float
            isaac_float3 count_end[ISAAC_VECTOR_ELEM];      //end index for ray
            isaac_float3 start_normal[ISAAC_VECTOR_ELEM];
            bool global_front[ISAAC_VECTOR_ELEM];

            ISAAC_ELEM_ITERATE ( e )
            {
                
                global_front[e] = false;

                //get normalized pixel position in framebuffer
                pixel_f[e].x = isaac_float( pixel[e].x )
                               / ( isaac_float ) framebuffer_size.x
                               * isaac_float( 2 ) - isaac_float( 1 );
                pixel_f[e].y = isaac_float( pixel[e].y )
                               / ( isaac_float ) framebuffer_size.y
                               * isaac_float( 2 ) - isaac_float( 1 );
                
                //get ray start/end position
                start_p[e].x = pixel_f[e].x * ISAAC_Z_NEAR;
                start_p[e].y = pixel_f[e].y * ISAAC_Z_NEAR;
                start_p[e].z = -1.0f * ISAAC_Z_NEAR;
                start_p[e].w = 1.0f * ISAAC_Z_NEAR;

                end_p[e].x = pixel_f[e].x * ISAAC_Z_FAR;
                end_p[e].y = pixel_f[e].y * ISAAC_Z_FAR;
                end_p[e].z = 1.0f * ISAAC_Z_FAR;
                end_p[e].w = 1.0f * ISAAC_Z_FAR;

                //apply inverse modelview transform to ray start/end and get ray start/end as worldspace
                start[e].x = isaac_inverse_d[0] * start_p[e].x
                             + isaac_inverse_d[4] * start_p[e].y
                             + isaac_inverse_d[8] * start_p[e].z
                             + isaac_inverse_d[12] * start_p[e].w;
                start[e].y = isaac_inverse_d[1] * start_p[e].x
                             + isaac_inverse_d[5] * start_p[e].y
                             + isaac_inverse_d[9] * start_p[e].z
                             + isaac_inverse_d[13] * start_p[e].w;
                start[e].z = isaac_inverse_d[2] * start_p[e].x
                             + isaac_inverse_d[6] * start_p[e].y
                             + isaac_inverse_d[10] * start_p[e].z
                             + isaac_inverse_d[14] * start_p[e].w;

                end[e].x = isaac_inverse_d[0] * end_p[e].x
                           + isaac_inverse_d[4] * end_p[e].y
                           + isaac_inverse_d[8] * end_p[e].z
                           + isaac_inverse_d[12] * end_p[e].w;
                end[e].y = isaac_inverse_d[1] * end_p[e].x
                           + isaac_inverse_d[5] * end_p[e].y
                           + isaac_inverse_d[9] * end_p[e].z
                           + isaac_inverse_d[13] * end_p[e].w;
                end[e].z = isaac_inverse_d[2] * end_p[e].x
                           + isaac_inverse_d[6] * end_p[e].y
                           + isaac_inverse_d[10] * end_p[e].z
                           + isaac_inverse_d[14] * end_p[e].w;
                isaac_float
                    max_size = isaac_size_d[0].max_global_size_scaled / 2.0f;

                //scale to globale grid size
                start[e] = start[e] * max_size;
                end[e] = end[e] * max_size;

                //set values for clipping planes
                //scale position to global size
                for( isaac_int i = 0; i < input_clipping.count; i++ )
                {
                    clipping[e].elem[i].position =
                        input_clipping.elem[i].position * max_size;
                    clipping[e].elem[i].normal = input_clipping.elem[i].normal;
                }

                //move to local (scaled) grid
                //get offset of subvolume in global volume
                move[e].x = isaac_int(
                    isaac_size_d[0].global_size_scaled
                        .value
                        .x
                ) / isaac_int( 2 ) - isaac_int(
                    isaac_size_d[0].position_scaled
                        .value
                        .x
                );
                move[e].y = isaac_int(
                    isaac_size_d[0].global_size_scaled
                        .value
                        .y
                ) / isaac_int( 2 ) - isaac_int(
                    isaac_size_d[0].position_scaled
                        .value
                        .y
                );
                move[e].z = isaac_int(
                    isaac_size_d[0].global_size_scaled
                        .value
                        .z
                ) / isaac_int( 2 ) - isaac_int(
                    isaac_size_d[0].position_scaled
                        .value
                        .z
                );

                move_f[e].x = isaac_float( move[e].x );
                move_f[e].y = isaac_float( move[e].y );
                move_f[e].z = isaac_float( move[e].z );

                //apply subvolume offset to start and end
                start[e] = start[e] + move_f[e];
                end[e] = end[e] + move_f[e];

                //apply subvolume offset to position checked clipping plane
                for( isaac_int i = 0; i < input_clipping.count; i++ )
                {
                    clipping[e].elem[i].position =
                        clipping[e].elem[i].position + move_f[e];
                }

                //get ray length
                vec[e] = end[e] - start[e];
                l_scaled[e] = sqrt(
                    vec[e].x * vec[e].x + vec[e].y * vec[e].y
                    + vec[e].z * vec[e].z
                );

                //apply isaac scaling to start, end and position tested by clipping plane
                start[e].x = start[e].x / scale.x;
                start[e].y = start[e].y / scale.y;
                start[e].z = start[e].z / scale.z;
                end[e].x = end[e].x / scale.x;
                end[e].y = end[e].y / scale.y;
                end[e].z = end[e].z / scale.z;
                for( isaac_int i = 0; i < input_clipping.count; i++ )
                {
                    clipping[e].elem[i].position
                        .x = clipping[e].elem[i].position
                                 .x / scale.x;
                    clipping[e].elem[i].position
                        .y = clipping[e].elem[i].position
                                 .y / scale.y;
                    clipping[e].elem[i].position
                        .z = clipping[e].elem[i].position
                                 .z / scale.z;
                }

                //get ray length (scaled by isaac scaling)
                vec[e] = end[e] - start[e];
                l[e] = sqrt(
                    vec[e].x * vec[e].x + vec[e].y * vec[e].y
                    + vec[e].z * vec[e].z
                );

                //get step vector
                step_vec[e] = vec[e] / l[e] * step;

                //start index for ray
                count_start[e] = -start[e] / step_vec[e];

                //get subvolume size as float
                local_size_f[e].x = isaac_float(
                    isaac_size_d[0].local_size
                        .value
                        .x
                );
                local_size_f[e].y = isaac_float(
                    isaac_size_d[0].local_size
                        .value
                        .y
                );
                local_size_f[e].z = isaac_float(
                    isaac_size_d[0].local_size
                        .value
                        .z
                );

                //end index for ray
                count_end[e] = ( local_size_f[e] - start[e] ) / step_vec[e];

                //count_start shall have the smaller values
                ISAAC_SWITCH_IF_SMALLER ( count_end[e].x,
                    count_start[e].x )
                ISAAC_SWITCH_IF_SMALLER ( count_end[e].y,
                    count_start[e].y )
                ISAAC_SWITCH_IF_SMALLER ( count_end[e].z,
                    count_start[e].z )

                //calc intersection of all three super planes and save in [count_start.x ; count_end.x]
                float max_start = ISAAC_MAX(
                    ISAAC_MAX(
                        count_start[e].x,
                        count_start[e].y
                    ),
                    count_start[e].z
                );
                if( ceil( count_start[e].x ) == ceil( max_start ) )
                {
                    if( step_vec[e].x > 0.0f )
                    {
                        if( isaac_size_d[0].position
                                .value
                                .x == 0 )
                        {
                            global_front[e] = true;
                            start_normal[e] = {
                                1.0f,
                                0,
                                0
                            };
                        }
                    }
                    else
                    {
                        if( isaac_size_d[0].position
                                .value
                                .x == isaac_size_d[0].global_size
                                          .value
                                          .x - isaac_size_d[0].local_size
                                          .value
                                          .x )
                        {
                            global_front[e] = true;
                            start_normal[e] = {
                                -1.0f,
                                0,
                                0
                            };
                        }
                    }
                }
                if( ceil( count_start[e].y ) == ceil( max_start ) )
                {
                    if( step_vec[e].y > 0.0f )
                    {
                        if( isaac_size_d[0].position
                                .value
                                .y == 0 )
                        {
                            global_front[e] = true;
                            start_normal[e] = {
                                0,
                                1.0f,
                                0
                            };
                        }
                    }
                    else
                    {
                        if( isaac_size_d[0].position
                                .value
                                .y == isaac_size_d[0].global_size
                                          .value
                                          .y - isaac_size_d[0].local_size
                                          .value
                                          .y )
                        {
                            global_front[e] = true;
                            start_normal[e] = {
                                0,
                                -1.0f,
                                0
                            };
                        }
                    }
                }
                if( ceil( count_start[e].z ) == ceil( max_start ) )
                {
                    if( step_vec[e].z > 0.0f )
                    {
                        if( isaac_size_d[0].position
                                .value
                                .z == 0 )
                        {
                            global_front[e] = true;
                            start_normal[e] = {
                                0,
                                0,
                                1.0f
                            };
                        }
                    }
                    else
                    {
                        if( isaac_size_d[0].position
                                .value
                                .z == isaac_size_d[0].global_size
                                          .value
                                          .z - isaac_size_d[0].local_size
                                          .value
                                          .z )
                        {
                            global_front[e] = true;
                            start_normal[e] = {
                                0,
                                0,
                                -1.0f
                            };
                        }
                    }
                }
                count_start[e].x = max_start;
                count_end[e].x = ISAAC_MIN(
                    ISAAC_MIN(
                        count_end[e].x,
                        count_end[e].y
                    ),
                    count_end[e].z
                );
                if( count_start[e].x > count_end[e].x )
                {
                    if ( !finish[e] ) {
                        //TODO understand the superplanes stuff...
                        ISAAC_SET_COLOR ( pixels[pixel[e].x + pixel[e].y * framebuffer_size.x], color[e] )

                        //this function aborts drawing and therfore wont set any normal or depth values
                        //defaults will be applied for clean images
                        gNormal[pixel[e].x + pixel[e].y * framebuffer_size.x] = default_normal;
                        gDepth[pixel[e].x + pixel[e].y * framebuffer_size.x] = default_depth;

                        finish[e] = true;
                    }
                }
            }
            ISAAC_ELEM_ALL_TRUE_RETURN ( finish )

            isaac_int first[ISAAC_VECTOR_ELEM];                 //start index of ray
            isaac_int last[ISAAC_VECTOR_ELEM];                  //end index of ray
            isaac_float first_f[ISAAC_VECTOR_ELEM];             //start index as float
            isaac_float last_f[ISAAC_VECTOR_ELEM];              //end index as float
            isaac_float3 pos[ISAAC_VECTOR_ELEM];
            isaac_int3 coord[ISAAC_VECTOR_ELEM];
            isaac_float d[ISAAC_VECTOR_ELEM];
            isaac_float intersection_step[ISAAC_VECTOR_ELEM];
            isaac_float3 clipping_normal[ISAAC_VECTOR_ELEM];
            bool is_clipped[ISAAC_VECTOR_ELEM];

            ISAAC_ELEM_ITERATE ( e )
            {
                //set start and end index of ray
                first[e] = isaac_int( ceil( count_start[e].x ) );
                last[e] = isaac_int( floor( count_end[e].x ) );

                first_f[e] = count_start[e].x;
                last_f[e] = count_end[e].x;

                //Moving last and first until their points are valid
                pos[e] = start[e] + step_vec[e] * isaac_float( last[e] );
                coord[e].x = isaac_int( floor( pos[e].x ) );
                coord[e].y = isaac_int( floor( pos[e].y ) );
                coord[e].z = isaac_int( floor( pos[e].z ) );
                while( (
                    ISAAC_FOR_EACH_DIM_TWICE ( 3,
                        coord[e],
                        >= isaac_size_d[0].local_size.value,
                        || )
                    ISAAC_FOR_EACH_DIM ( 3,
                        coord[e],
                        < 0 || ) 0 )
                    && first[e] <= last[e] )
                {
                    last[e]--;
                    pos[e] = start[e] + step_vec[e] * isaac_float( last[e] );
                    coord[e].x = isaac_int( floor( pos[e].x ) );
                    coord[e].y = isaac_int( floor( pos[e].y ) );
                    coord[e].z = isaac_int( floor( pos[e].z ) );
                }
                pos[e] = start[e] + step_vec[e] * isaac_float( first[e] );
                coord[e].x = isaac_int( floor( pos[e].x ) );
                coord[e].y = isaac_int( floor( pos[e].y ) );
                coord[e].z = isaac_int( floor( pos[e].z ) );
                while( (
                    ISAAC_FOR_EACH_DIM_TWICE ( 3,
                        coord[e],
                        >= isaac_size_d[0].local_size.value,
                        || )
                    ISAAC_FOR_EACH_DIM ( 3,
                        coord[e],
                        < 0 || ) 0 )
                    && first[e] <= last[e] )
                {
                    first[e]++;
                    pos[e] = start[e] + step_vec[e] * isaac_float( first[e] );
                    coord[e].x = isaac_int( floor( pos[e].x ) );
                    coord[e].y = isaac_int( floor( pos[e].y ) );
                    coord[e].z = isaac_int( floor( pos[e].z ) );
                }
                first[e] = ISAAC_MAX(
                    first[e],
                    0
                );
                first_f[e] = ISAAC_MAX(
                    first_f[e],
                    0.0f
                );
                is_clipped[e] = false;
                //Extra clipping
                for( isaac_int i = 0; i < input_clipping.count; i++ )
                {
                    d[e] = step_vec[e].x * clipping[e].elem[i].normal
                        .x + step_vec[e].y * clipping[e].elem[i].normal
                        .y + step_vec[e].z * clipping[e].elem[i].normal
                        .z;
                    intersection_step[e] = (
                                               clipping[e].elem[i].position
                                                   .x
                                               * clipping[e].elem[i].normal
                                                   .x +
                                               clipping[e].elem[i].position
                                                   .y
                                               * clipping[e].elem[i].normal
                                                   .y +
                                               clipping[e].elem[i].position
                                                   .z
                                               * clipping[e].elem[i].normal
                                                   .z - start[e].x
                                                        * clipping[e].elem[i].normal
                                                            .x - start[e].y
                                                                 * clipping[e].elem[i].normal
                                                                     .y
                                               - start[e].z
                                                 * clipping[e].elem[i].normal
                                                     .z
                                           ) / d[e];
                    if( d[e] > 0 )
                    {
                        if( last_f[e] < intersection_step[e] )
                        {
                            if( !finish[e] )
                            ISAAC_SET_COLOR ( pixels[pixel[e].x + pixel[e].y
                                                                  * framebuffer_size.x],
                                color[e] )
                            finish[e] = true;
                        }
                        if( first_f[e] <= intersection_step[e] )
                        {
                            first[e] = ceil( intersection_step[e] );
                            first_f[e] = intersection_step[e];
                            clipping_normal[e] = clipping[e].elem[i].normal;
                            is_clipped[e] = true;
                            global_front[e] = true;
                            start_normal[e].x = clipping[e].elem[i].normal
                                .x;
                            start_normal[e].y = clipping[e].elem[i].normal
                                .y;
                            start_normal[e].z = clipping[e].elem[i].normal
                                .z;
                        }
                    }
                    else
                    {
                        if( first_f[e] > intersection_step[e] )
                        {
                            if( !finish[e] )
                            ISAAC_SET_COLOR ( pixels[pixel[e].x + pixel[e].y
                                                                  * framebuffer_size.x],
                                color[e] )
                            finish[e] = true;
                        }
                        if( last_f[e] > intersection_step[e] )
                        {
                            last[e] = floor( intersection_step[e] );
                            last_f[e] = intersection_step[e];
                        }
                    }
                }

            }
            ISAAC_ELEM_ALL_TRUE_RETURN ( finish )


            isaac_float4 particle_color[ISAAC_VECTOR_ELEM];          //color at particle hist position
            isaac_float3 particle_normal[ISAAC_VECTOR_ELEM];         //normal at particle hit position
            isaac_float3 particle_hitposition[ISAAC_VECTOR_ELEM];    //hit position of particle
            
            isaac_int result_particle[ISAAC_VECTOR_ELEM];
            isaac_float3 local_start[ISAAC_VECTOR_ELEM];
            isaac_float3 light_dir[ISAAC_VECTOR_ELEM];
            isaac_float3 normalized_dir[ISAAC_VECTOR_ELEM];

            isaac_int3 dir_sign[ISAAC_VECTOR_ELEM];
            isaac_float3 current_pos[ISAAC_VECTOR_ELEM];
            isaac_uint3 current_cell[ISAAC_VECTOR_ELEM];
            isaac_float ray_length[ISAAC_VECTOR_ELEM];
            isaac_float march_length[ISAAC_VECTOR_ELEM];
            isaac_float3 t[ISAAC_VECTOR_ELEM];
            isaac_float3 delta_t[ISAAC_VECTOR_ELEM];


            isaac_float3 particle_scale = {
                isaac_size_d[0].local_size_scaled
                    .value
                    .x / isaac_float(
                    isaac_size_d[0].local_particle_size
                        .value
                        .x
                ),
                isaac_size_d[0].local_size_scaled
                    .value
                    .y / isaac_float(
                    isaac_size_d[0].local_particle_size
                        .value
                        .y
                ),
                isaac_size_d[0].local_size_scaled
                    .value
                    .z / isaac_float(
                    isaac_size_d[0].local_particle_size
                        .value
                        .z
                )
            };
            ISAAC_ELEM_ITERATE ( e )
            {
                // set distance check in alpha channel on scaled max distance
                particle_color[e].w =
                    ( last_f[e] - first_f[e] ) * step * l_scaled[e] / l[e];
                local_start[e] =
                    ( start[e] + step_vec[e] * first_f[e] ) * scale;
                result_particle[e] = 0;
                normalized_dir[e] = step_vec[e] * scale / step;
                normalized_dir[e] = normalized_dir[e] / sqrt(
                    normalized_dir[e].x * normalized_dir[e].x
                    + normalized_dir[e].y * normalized_dir[e].y
                    + normalized_dir[e].z * normalized_dir[e].z
                );
                // light direction is camera direction
                light_dir[e] = -normalized_dir[e];

                /* RAYMARCH */

                // get the signs of the direction for the raymarch
                dir_sign[e].x = sgn( normalized_dir[e].x );
                dir_sign[e].y = sgn( normalized_dir[e].y );
                dir_sign[e].z = sgn( normalized_dir[e].z );

                //TODO: alternative for constant 0.001f
                // calculate current position in scaled object space
                current_pos[e] = (
                                     start[e] + step_vec[e] * (
                                         ISAAC_MAX(
                                             first_f[e],
                                             0.0f
                                         ) + 0.001f * particle_scale
                                     )
                                 ) * scale;

                // calculate current local cell coordinates
                current_cell[e].x = int( current_pos[e].x / particle_scale.x );
                current_cell[e].y = int( current_pos[e].y / particle_scale.y );
                current_cell[e].z = int( current_pos[e].z / particle_scale.z );

                ray_length[e] =
                    ( last_f[e] - first_f[e] ) * step * l_scaled[e] / l[e];
                march_length[e] = 0;


                // calculate next intersection with each dimension
                t[e].x = (
                             (
                                 current_cell[e].x + ISAAC_MAX(
                                     dir_sign[e].x,
                                     0
                                 )
                             ) * particle_scale.x - current_pos[e].x
                         ) / normalized_dir[e].x;
                t[e].y = (
                             (
                                 current_cell[e].y + ISAAC_MAX(
                                     dir_sign[e].y,
                                     0
                                 )
                             ) * particle_scale.y - current_pos[e].y
                         ) / normalized_dir[e].y;
                t[e].z = (
                             (
                                 current_cell[e].z + ISAAC_MAX(
                                     dir_sign[e].z,
                                     0
                                 )
                             ) * particle_scale.z - current_pos[e].z
                         ) / normalized_dir[e].z;

                // calculate delta length to next intersection in the same dimension
                delta_t[e].x =
                    particle_scale.x / normalized_dir[e].x * dir_sign[e].x;
                delta_t[e].y =
                    particle_scale.y / normalized_dir[e].y * dir_sign[e].y;
                delta_t[e].z =
                    particle_scale.z / normalized_dir[e].z * dir_sign[e].z;

                particle_hitposition[e] = {
                    0.0,
                    0.0,
                    0.0
                };

                // check for 0 to stop infinite looping
                if( normalized_dir[e].x == 0 )
                {
                    t[e].x = maxFloat;
                }
                if( normalized_dir[e].y == 0 )
                {
                    t[e].y = maxFloat;
                }
                if( normalized_dir[e].z == 0 )
                {
                    t[e].z = maxFloat;
                }


                // check if the ray leaves the local volume, has a particle hit or exceeds the max ray distance
                while( current_cell[e].x < isaac_size_d[0].local_particle_size
                    .value
                    .x && current_cell[e].y
                          < isaac_size_d[0].local_particle_size
                              .value
                              .y && current_cell[e].z
                                    < isaac_size_d[0].local_particle_size
                                        .value
                                        .z && result_particle[e] == false
                       && march_length[e] <= ray_length[e] )
                {

                    // calculate particle intersections for each particle source
                    isaac_for_each_with_mpl_params(
                        particle_sources,
                        merge_particle_iterator<
                            Ttransfer_size,
                            mpl::size< TSourceList >::type::value,
                            TFilter
                        >( ),
                        particle_color[e],
                        particle_normal[e],
                        particle_hitposition[e],
                        local_start[e],
                        normalized_dir[e],
                        light_dir[e],
                        current_cell[e],
                        transferArray,
                        sourceWeight,
                        result_particle[e],
                        particle_scale,
                        clipping_normal[e],
                        is_clipped[e]
                    );


                    // adds the delta t value to the smallest dimension t and increment the cell index in the dimension
                    if( t[e].x < t[e].y && t[e].x < t[e].z )
                    {
                        current_cell[e].x += dir_sign[e].x;
                        march_length[e] = t[e].x;
                        t[e].x += delta_t[e].x;
                    }
                    else if( t[e].y < t[e].x && t[e].y < t[e].z )
                    {
                        current_cell[e].y += dir_sign[e].y;
                        march_length[e] = t[e].y;
                        t[e].y += delta_t[e].y;
                    }
                    else
                    {
                        current_cell[e].z += dir_sign[e].z;
                        march_length[e] = t[e].z;
                        t[e].z += delta_t[e].z;
                    }

                }
                // if there was a hit set maximum volume raycast distance to particle hit distance and set particle color
                if( result_particle[e] )
                {
                    last[e] = ISAAC_MIN(
                        last[e],
                        int(
                            ceil(
                                first_f[e] + particle_color[e].w
                                             / ( step * l_scaled[e] / l[e] )
                            )
                        )
                    );

                    // calculate lighting properties for the last hit particle
                    particle_normal[e] = particle_normal[e] / sqrt(
                        particle_normal[e].x * particle_normal[e].x
                        + particle_normal[e].y * particle_normal[e].y
                        + particle_normal[e].z * particle_normal[e].z
                    );

                    isaac_float light_factor =
                        particle_normal[e].x * light_dir[e].x
                        + particle_normal[e].y * light_dir[e].y
                        + particle_normal[e].z * light_dir[e].z;
                    isaac_float3
                        half_vector = -normalized_dir[e] + light_dir[e];
                    half_vector = half_vector / sqrt(
                        half_vector.x * half_vector.x
                        + half_vector.y * half_vector.y
                        + half_vector.z * half_vector.z
                    );
                    isaac_float specular = particle_normal[e].x * half_vector.x
                                           + particle_normal[e].y
                                             * half_vector.y
                                           + particle_normal[e].z
                                             * half_vector.z;
                    specular = pow(
                        specular,
                        10
                    );
                    specular *= 0.5f;
                    light_factor = light_factor * 0.5f + 0.5f;

                    particle_color[e].x = ISAAC_MIN(
                        particle_color[e].x * light_factor + specular,
                        1.0f
                    );
                    particle_color[e].y = ISAAC_MIN(
                        particle_color[e].y * light_factor + specular,
                        1.0f
                    );
                    particle_color[e].z = ISAAC_MIN(
                        particle_color[e].z * light_factor + specular,
                        1.0f
                    );
                }
            }


            isaac_float min_size[ISAAC_VECTOR_ELEM];
            isaac_float factor[ISAAC_VECTOR_ELEM];
            isaac_float4 value[ISAAC_VECTOR_ELEM];
            isaac_int result[ISAAC_VECTOR_ELEM];
            isaac_float oma[ISAAC_VECTOR_ELEM];
            isaac_float4 color_add[ISAAC_VECTOR_ELEM];
            isaac_float ao_blend = 0.0f; //indicates how strong particle ao should be when gas is overlapping

            ISAAC_ELEM_ITERATE ( e )
            {

                //Starting the main loop
                min_size[e] = ISAAC_MIN(
                    int(
                        isaac_size_d[0].global_size
                            .value
                            .x
                    ),
                    ISAAC_MIN(
                        int(
                            isaac_size_d[0].global_size
                                .value
                                .y
                        ),
                        int(
                            isaac_size_d[0].global_size
                                .value
                                .z
                        )
                    )
                );
                factor[e] = step / min_size[e] * isaac_float( 2 );
                value[e].x = 0;
                value[e].y = 0;
                value[e].z = 0;
                value[e].w = 0;
                result[e] = 0;

                for( isaac_int i = first[e]; i <= last[e]; i++ )
                {
                    pos[e] = start[e] + step_vec[e] * isaac_float( i );
                    value[e].x = 0;
                    value[e].y = 0;
                    value[e].z = 0;
                    value[e].w = 0;
                    result[e] = 0;
                    bool firstRound = ( global_front[e] && i == first[e] );
                    isaac_for_each_with_mpl_params(
                        sources,
                        merge_source_iterator<
                            Ttransfer_size,
                            TFilter,
                            TInterpolation,
                            TIsoSurface
                        >( ),
                        value[e],
                        pos[e],
                        isaac_size_d[0].local_size,
                        transferArray,
                        sourceWeight,
                        pointerArray,
                        result[e],
                        step_vec[e],
                        step,
                        scale,
                        firstRound,
                        start_normal[e]
                    );
                    if( TIsoSurface )
                    {
                        if( result[e] )
                        {
                            color[e] = value[e];
                            break;
                        }
                    }
                    else
                    {
                        oma[e] = isaac_float( 1 ) - color[e].w;
                        value[e] = value[e] * factor[e];
                        color_add[e].x = oma[e]
                                         * value[e].x; // * value.w does merge_source_iterator
                        color_add[e].y = oma[e]
                                         * value[e].y; // * value.w does merge_source_iterator
                        color_add[e].z = oma[e]
                                         * value[e].z; // * value.w does merge_source_iterator
                        color_add[e].w = oma[e] * value[e].w;
                        color[e] = color[e] + color_add[e];
                        if( color[e].w > isaac_float( 0.99 ) )
                        {
                            break;
                        }
                    }
                }
                if( result_particle[e] && !result[e] )
                {
                    //extracting real particle depth and override block march length
                    march_length[e] = particle_color[e].w;
                    ao_blend = (1 - color[e].w);

                    particle_color[e].w = 1;
                    
                    color[e] =
                        color[e] + particle_color[e] * ( 1 - color[e].w );
                    
                }

#if ISAAC_SHOWBORDER == 1
                if ( color[e].w <= isaac_float ( 0.99 ) ) {
                    oma[e] = isaac_float ( 1 ) - color[e].w;
                    color_add[e].x = 0;
                    color_add[e].y = 0;
                    color_add[e].z = 0;
                    color_add[e].w = oma[e] * factor[e] * isaac_float ( 10 );
                    color[e] = color[e] + color_add[e];
                }
#endif


                if( !finish[e] )
                {
                    ISAAC_SET_COLOR ( pixels[pixel[e].x + pixel[e].y * framebuffer_size.x], color[e] )
                    //save the particle normal in the normal g buffer
                    gNormal[pixel[e].x + pixel[e].y * framebuffer_size.x] = particle_normal[e];
                    
                    //save the cell depth in our g buffer (depth)
                    //march_length takes the old particle_color w component 
                    //the w component stores the particle depth and will be replaced later by new alpha values and 
                    //is therefore stored in march_length
                    //LINE 2044
                    isaac_float3 depth_value = {
                        0.0f,
                        ao_blend,
                        march_length[e]
                    };                

                    gDepth[pixel[e].x + pixel[e].y * framebuffer_size.x] = depth_value;
                    
                }
            }
        }
#if ISAAC_ALPAKA == 1
    };
#endif

    /**
     * @brief Calculate SSAO factor
     * 
     * Requires AO Buffer     (dim 1)
     *          Depth Buffer  (dim 1)
     *          Normal Buffer (dim 3)
     * 
     */
#if ISAAC_ALPAKA == 1
    struct isaacSSAOKernel {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator() (
            TAcc__ const &acc,
#else
    __global__ void isaacSSAOKernel (
#endif
            isaac_float * const gAOBuffer,       //ao buffer
            isaac_float3 * const gDepth,         //depth buffer (will be used as y=blending of particles and volume, z=depth of pixels)
            isaac_float3 * const gNormal,        //normal buffer
            const isaac_size2 framebuffer_size,  //size of framebuffer
            const isaac_uint2 framebuffer_start, //framebuffer offset
            ao_struct ao_properties              //properties for ambient occlusion
            )
#if ISAAC_ALPAKA == 1
        const
#endif
        {

            isaac_uint2 pixel;
            //get pixel values from thread ids
#if ISAAC_ALPAKA == 1
                    auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads> ( acc );
                    pixel.x = isaac_uint ( alpThreadIdx[2] );
                    pixel.y = isaac_uint ( alpThreadIdx[1] );
#else
                    pixel.x = isaac_uint ( threadIdx.x + blockIdx.x * blockDim.x );
                    pixel.y = isaac_uint ( threadIdx.y + blockIdx.y * blockDim.y );
#endif

            pixel = pixel + framebuffer_start;    

            


            /*
            * TODO
            * 
            * Old standart ssao by crytech 
            * 
            * First implemntation failed and the source code is below
            * Possible errors could be mv or proj matrix
            */

            //search radius for depth testing
            isaac_int radius = 10;

            /*
            //isaac_float3 origin = gDepth[pixel.x + pixel.y * framebuffer_size.x];

            

            //get the normal value from the gbuffer
            isaac_float3 normal = gNormal[pixel.x + pixel.y * framebuffer_size.x];

            //normalize the normal
            isaac_float len = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
            if(len == 0) {
                gAOBuffer[pixel.x + pixel.y * framebuffer_size.x] = 0.0f;
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
                    tbn[0] * ssao_kernel_d[i].x + tbn[3] * ssao_kernel_d[i].y + tbn[6] * ssao_kernel_d[i].z,
                    tbn[1] * ssao_kernel_d[i].x + tbn[4] * ssao_kernel_d[i].y + tbn[7] * ssao_kernel_d[i].z,
                    tbn[2] * ssao_kernel_d[i].x + tbn[5] * ssao_kernel_d[i].y + tbn[8] * ssao_kernel_d[i].z,
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
                    isaac_projection_d[0] * offset.x + isaac_projection_d[4] * offset.y + isaac_projection_d[8 ] * offset.z + isaac_projection_d[12] * offset.w,
                    isaac_projection_d[1] * offset.x + isaac_projection_d[5] * offset.y + isaac_projection_d[9 ] * offset.z + isaac_projection_d[13] * offset.w,
                    isaac_projection_d[2] * offset.x + isaac_projection_d[6] * offset.y + isaac_projection_d[10] * offset.z + isaac_projection_d[14] * offset.w,
                    isaac_projection_d[3] * offset.x + isaac_projection_d[7] * offset.y + isaac_projection_d[11] * offset.z + isaac_projection_d[15] * offset.w
                });

                isaac_float2 offset2d = isaac_float2({offset.x / offset.w, offset.y / offset.w});
                offset2d.x = MAX(MIN(offset2d.x * 0.5 + 0.5, 1.0f), 0.0f);
                offset2d.y = MAX(MIN(offset2d.y * 0.5 + 0.5, 1.0f), 0.0f);

                isaac_uint2 offsetFramePos = {
                    isaac_uint(framebuffer_size.x * offset2d.x) + framebuffer_start.x,
                    isaac_uint(framebuffer_size.y * offset2d.y) + framebuffer_start.y,
                };
                //printf("%f %f -- %u %u\n", offset2d.x, offset2d.y, offsetFramePos.x, offsetFramePos.y);
                isaac_float sampleDepth = gDepth[offsetFramePos.x + offsetFramePos.y * framebuffer_size.x].z; 
                occlusion += (sampleDepth - sample.z ? 1.0f : 0.0f);
            }*/
            

            /* 
            * 1. compare all neighbour (+-2 pixel) depth values with the current one and increase the counter if the neighbour is
            *    closer to the camera
            * 
            * 2. get average value by dividing the counter by the cell count (7x7=49)       * 
            *
            */
            //closer to the camera
            isaac_float occlusion = 0.0f;
            isaac_float ref_depth = gDepth[pixel.x + pixel.y * framebuffer_size.x].z;
            for(int i = -3; i <= 3; i++) {
                for(int j = -3; j <= 3; j++) {
                    //avoid out of bounds by simple min max
                    isaac_int x = ISAAC_MAX(ISAAC_MIN(pixel.x + i * radius, framebuffer_start.x + framebuffer_size.x), framebuffer_start.x);
                    isaac_int y = ISAAC_MAX(ISAAC_MIN(pixel.y + j * radius, framebuffer_start.y + framebuffer_size.y), framebuffer_start.y);

                    //get the neighbour depth value
                    isaac_float depth_sample = gDepth[x + y * framebuffer_size.x].z;

                    // only increase the counter if the neighbour depth is closer to the camera
                    // use <= because we will discard pixels with a depth/ao value 0.0 (for background pixels and image merging), 
                    // but planes will have pixels with depth/ao with 0 because of neighbor pixels
                    if(depth_sample <= ref_depth) {
                        occlusion += 1.0f;
                    }
                }
            }
            isaac_float depth = (occlusion / 49.0f);

            //save the depth value in our ao buffer
            gAOBuffer[pixel.x + pixel.y * framebuffer_size.x] = depth;
        }
#if ISAAC_ALPAKA == 1
    };
#endif

    /**
     * @brief Filter SSAO artifacts and return the color with depth simulation
     * 
     * Requires Color Buffer      (dim 4)
     * Requires AO Values Buffer  (dim 1)
     * 
     */
#if ISAAC_ALPAKA == 1
    struct isaacSSAOFilterKernel {
        template <typename TAcc__>
        ALPAKA_FN_ACC void operator() (
            TAcc__ const &acc,
#else
    __global__ void isaacSSAOFilterKernel (
#endif
            uint32_t * const gColor,             //ptr to output pixels
            isaac_float * const gAOBuffer,       //ambient occlusion values from ssao kernel
            isaac_float3 * const gDepthBuffer,   //depth and blending values
            const isaac_size2 framebuffer_size,  //size of framebuffer
            const isaac_uint2 framebuffer_start, //framebuffer offset
            ao_struct ao_properties              //properties for ambient occlusion
            )
#if ISAAC_ALPAKA == 1
        const
#endif
        {

            isaac_uint2 pixel;
            //get pixel values from thread ids
#if ISAAC_ALPAKA == 1
                    auto alpThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads> ( acc );
                    pixel.x = isaac_uint ( alpThreadIdx[2] );
                    pixel.y = isaac_uint ( alpThreadIdx[1] );
#else
                    pixel.x = isaac_uint ( threadIdx.x + blockIdx.x * blockDim.x );
                    pixel.y = isaac_uint ( threadIdx.y + blockIdx.y * blockDim.y );
#endif

            //get real pixel coordinate by offset
            pixel = pixel + framebuffer_start;

            /* TODO
            * Normally the depth values are smoothed
            * in this case the smooting filter is not applied for simplicity
            * 
            * If the real ssao algorithm is implemented, a real filter will be necessary
            */
            isaac_float depth = gAOBuffer[pixel.x + pixel.y * framebuffer_size.x];
            
            //convert uint32 back to 4x 1 Byte color values
            uint32_t color = gColor[pixel.x + pixel.y * framebuffer_size.x];
            isaac_float4 color_values = {
                ((color >>  0) & 0xff) / 255.0f,
                ((color >>  8) & 0xff) / 255.0f,
                ((color >> 16) & 0xff) / 255.0f,
                ((color >> 24) & 0xff) / 255.0f
            };        

            //read the weight from the global ao settings and merge them with the color value
            isaac_float weight = ao_properties.weight;
            isaac_float ao_factor = ((1.0f - weight) + weight * (1.0f - depth));
            isaac_float particle_blend = gDepthBuffer[pixel.x + pixel.y * framebuffer_size.x].y;
            
            isaac_float4 final_color = { 
                particle_blend * ao_factor * color_values.x + (1.0f - particle_blend) * color_values.x,
                particle_blend * ao_factor * color_values.y + (1.0f - particle_blend) * color_values.y,
                particle_blend * ao_factor * color_values.z + (1.0f - particle_blend) * color_values.z,
                1.0f  * color_values.w
            };
        
            //if the depth value is 0 the ssao kernel found a background value and the color
            //merging is therefore removed
            if(depth == 0.0f) { 
                final_color = { 0, 0, 0, 1.0 };
            }

            //finally replace the old color value with the new ssao filtered color value
            ISAAC_SET_COLOR(gColor[pixel.x + pixel.y * framebuffer_size.x], final_color);
            
        }
#if ISAAC_ALPAKA == 1
    };
#endif


    template<
        typename TSimDim,
        typename TParticleList,
        typename TSourceList,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TFilter,
        typename TFramebuffer, 
        typename TFramebufferDepth,
        typename TFramebufferNormal,
        ISAAC_IDX_TYPE TTransfer_size,
        typename TScale,
#if ISAAC_ALPAKA == 1
        typename TAccDim,
        typename TAcc,
        typename TStream,
        typename TFunctionChain,
#endif
        int N
    >
    struct IsaacRenderKernelCaller
    {
        inline static void call(
#if ISAAC_ALPAKA == 1
            TStream stream,
#endif
            TFramebuffer framebuffer,
            TFramebufferDepth depthBuffer,
            TFramebufferNormal normalBuffer,
            const isaac_size2 & framebuffer_size,
            const isaac_uint2 & framebuffer_start,
            const TParticleList & particle_sources,
            const TSourceList & sources,
            const isaac_float & step,
            const isaac_float4 & background_color,
            const TTransferArray & transferArray,
            const TSourceWeight & sourceWeight,
            const TPointerArray & pointerArray,
            IceTInt const * const readback_viewport,
            const isaac_int interpolation,
            const isaac_int iso_surface,
            const TScale & scale,
            const clipping_struct & clipping,
            const ao_struct & ambientOcclusion
        )
        {
            if( sourceWeight.value[mpl::size< TSourceList >::type::value
                                   + mpl::size< TParticleList >::type::value
                                   - N] == isaac_float( 0 ) )
            {
                IsaacRenderKernelCaller<
                    TSimDim,
                    TParticleList,
                    TSourceList,
                    TTransferArray,
                    TSourceWeight,
                    TPointerArray,
                    typename mpl::push_back<
                        TFilter,
                        mpl::false_
                    >::type,
                    TFramebuffer,
                    TFramebufferDepth,
                    TFramebufferNormal,
                    TTransfer_size,
                    TScale,
#if ISAAC_ALPAKA == 1
                    TAccDim,
                    TAcc,
                    TStream,
                    TFunctionChain,
#endif
N - 1
                >::call(
#if ISAAC_ALPAKA == 1
                    stream,
#endif
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebuffer_size,
                    framebuffer_start,
                    particle_sources,
                    sources,
                    step,
                    background_color,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    readback_viewport,
                    interpolation,
                    iso_surface,
                    scale,
                    clipping,
                    ambientOcclusion
                );
            }
            else
            {
                IsaacRenderKernelCaller<
                    TSimDim,
                    TParticleList,
                    TSourceList,
                    TTransferArray,
                    TSourceWeight,
                    TPointerArray,
                    typename mpl::push_back<
                        TFilter,
                        mpl::true_
                    >::type,
                    TFramebuffer,
                    TFramebufferDepth,
                    TFramebufferNormal,
                    TTransfer_size,
                    TScale,
#if ISAAC_ALPAKA == 1
                    TAccDim,
                    TAcc,
                    TStream,
                    TFunctionChain,
#endif
N - 1
                >::call(
#if ISAAC_ALPAKA == 1
                    stream,
#endif
                    framebuffer,
                    depthBuffer,
                    normalBuffer,
                    framebuffer_size,
                    framebuffer_start,
                    particle_sources,
                    sources,
                    step,
                    background_color,
                    transferArray,
                    sourceWeight,
                    pointerArray,
                    readback_viewport,
                    interpolation,
                    iso_surface,
                    scale,
                    clipping,
                    ambientOcclusion
                );
            }
        }
    };

    template<
        typename TSimDim,
        typename TParticleList,
        typename TSourceList,
        typename TTransferArray,
        typename TSourceWeight,
        typename TPointerArray,
        typename TFilter,
        typename TFramebuffer, 
        typename TFramebufferDepth,
        typename TFramebufferNormal,
        ISAAC_IDX_TYPE TTransfer_size,
        typename TScale
#if ISAAC_ALPAKA == 1
        ,
        typename TAccDim,
        typename TAcc,
        typename TStream,
        typename TFunctionChain
#endif
    >
    struct IsaacRenderKernelCaller<
        TSimDim,
        TParticleList,
        TSourceList,
        TTransferArray,
        TSourceWeight,
        TPointerArray,
        TFilter,
        TFramebuffer,
        TFramebufferDepth,
        TFramebufferNormal,
        TTransfer_size,
        TScale,
#if ISAAC_ALPAKA == 1
        TAccDim,
        TAcc,
        TStream,
        TFunctionChain,
#endif
        0 //<-- spezialisation
    >
    {
        inline static void call(
#if ISAAC_ALPAKA == 1
            TStream stream,
#endif
            TFramebuffer framebuffer,
            TFramebufferDepth depthBuffer,
            TFramebufferNormal normalBuffer,
            const isaac_size2 & framebuffer_size,
            const isaac_uint2 & framebuffer_start,
            const TParticleList & particle_sources,
            const TSourceList & sources,
            const isaac_float & step,
            const isaac_float4 & background_color,
            const TTransferArray & transferArray,
            const TSourceWeight & sourceWeight,
            const TPointerArray & pointerArray,
            IceTInt const * const readback_viewport,
            const isaac_int interpolation,
            const isaac_int iso_surface,
            const TScale & scale,
            const clipping_struct & clipping,
            const ao_struct & ambientOcclusion
        )
        {
            isaac_size2 block_size = {
                ISAAC_IDX_TYPE( 8 ),
                ISAAC_IDX_TYPE( 16 )
            };
            isaac_size2 grid_size = {
                ISAAC_IDX_TYPE(
                    ( readback_viewport[2] + block_size.x - 1 ) / block_size.x
                    + ISAAC_VECTOR_ELEM - 1
                ) / ISAAC_IDX_TYPE( ISAAC_VECTOR_ELEM ),
                ISAAC_IDX_TYPE(
                    ( readback_viewport[3] + block_size.y - 1 ) / block_size.y
                )
            };
#if ISAAC_ALPAKA == 1
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
            if ( mpl::not_<boost::is_same<TAcc, alpaka::AccGpuCudaRt<TAccDim, ISAAC_IDX_TYPE> > >::value )
#endif
            {
                grid_size.x = ISAAC_IDX_TYPE(
                    readback_viewport[2] + ISAAC_VECTOR_ELEM - 1
                ) / ISAAC_IDX_TYPE( ISAAC_VECTOR_ELEM );
                grid_size.y = ISAAC_IDX_TYPE( readback_viewport[3] );
                block_size.x = ISAAC_IDX_TYPE( 1 );
                block_size.y = ISAAC_IDX_TYPE( 1 );
            }
            const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> threads(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( ISAAC_VECTOR_ELEM )
            );
            const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> blocks(
                ISAAC_IDX_TYPE( 1 ),
                block_size.y,
                block_size.x
            );
            const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> grid(
                ISAAC_IDX_TYPE( 1 ),
                grid_size.y,
                grid_size.x
            );
            auto const workdiv(
                alpaka::WorkDivMembers<
                    TAccDim,
                    ISAAC_IDX_TYPE
                >(
                    grid,
                    blocks,
                    threads
                )
            );
#define ISAAC_KERNEL_START \
            { \
                isaacRenderKernel \
                < \
                    TSimDim, \
                    TParticleList, \
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
                    alpaka::createTaskKernel<TAcc> \
                    ( \
                        workdiv, \
                        kernel, \
                        alpaka::getPtrNative(framebuffer), \
                        alpaka::getPtrNative(depthBuffer), \
                        alpaka::getPtrNative(normalBuffer), \
                        framebuffer_size, \
                        framebuffer_start, \
                        particle_sources, \
                        sources, \
                        step, \
                        background_color, \
                        transferArray, \
                        sourceWeight, \
                        pointerArray, \
                        scale, \
                        clipping, \
                        ambientOcclusion \
                    ) \
                ); \
                alpaka::enqueue(stream, instance); \
            }
#else
            dim3 block(
                block_size.x,
                block_size.y
            );
            dim3 grid(
                grid_size.x,
                grid_size.y
            );
#define ISAAC_KERNEL_START \
                isaacRenderKernel \
                < \
                    TSimDim, \
                    TParticleList, \
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
                    depthBuffer, \
                    normalBuffer, \
                    framebuffer_size, \
                    framebuffer_start, \
                    particle_sources, \
                    sources, \
                    step, \
                    background_color, \
                    transferArray, \
                    sourceWeight, \
                    pointerArray, \
                    scale, \
                    clipping, \
                    ambientOcclusion \
                );

#endif
            if( interpolation )
            {
                if( iso_surface )
                ISAAC_KERNEL_START 1,
                        1 ISAAC_KERNEL_END
                else
                ISAAC_KERNEL_START 1,
                        0 ISAAC_KERNEL_END
            }
            else
            {
                if( iso_surface )
                ISAAC_KERNEL_START 0,
                        1 ISAAC_KERNEL_END
                else
                ISAAC_KERNEL_START 0,
                        0 ISAAC_KERNEL_END
            }
#undef ISAAC_KERNEL_START
#undef ISAAC_KERNEL_END
        }
    };

    template <
        typename TAOBuffer,
        typename TDepthBuffer,
        typename TNormalBuffer
#if ISAAC_ALPAKA == 1
        ,typename TAccDim
        ,typename TAcc
        ,typename TStream
#endif
        >
    struct IsaacSSAOKernelCaller {
        
        inline static void call (
#if ISAAC_ALPAKA == 1
            TStream stream,
#endif
            TAOBuffer aoBuffer,
            TDepthBuffer depthBuffer,
            TNormalBuffer normalBuffer,
            const isaac_size2& framebuffer_size,
            const isaac_uint2& framebuffer_start,
            IceTInt const * const readback_viewport,
            ao_struct ao_properties
        )
        {
            isaac_size2 block_size= {
                ISAAC_IDX_TYPE ( 8 ),
                ISAAC_IDX_TYPE ( 16 )
            };
            isaac_size2 grid_size= {
                ISAAC_IDX_TYPE ( ( readback_viewport[2]+block_size.x-1 ) /block_size.x + ISAAC_VECTOR_ELEM - 1 ) /ISAAC_IDX_TYPE ( ISAAC_VECTOR_ELEM ),
                ISAAC_IDX_TYPE ( ( readback_viewport[3]+block_size.y-1 ) /block_size.y )
            };
#if ISAAC_ALPAKA == 1
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
            if ( mpl::not_<boost::is_same<TAcc, alpaka::AccGpuCudaRt<TAccDim, ISAAC_IDX_TYPE> > >::value )
#endif
            {
                grid_size.x = ISAAC_IDX_TYPE ( readback_viewport[2] + ISAAC_VECTOR_ELEM - 1 ) /ISAAC_IDX_TYPE ( ISAAC_VECTOR_ELEM );
                grid_size.y = ISAAC_IDX_TYPE ( readback_viewport[3] );
                block_size.x = ISAAC_IDX_TYPE ( 1 );
                block_size.y = ISAAC_IDX_TYPE ( 1 );
            }
            const alpaka::Vec<TAccDim, ISAAC_IDX_TYPE> threads ( ISAAC_IDX_TYPE ( 1 ), ISAAC_IDX_TYPE ( 1 ), ISAAC_IDX_TYPE ( ISAAC_VECTOR_ELEM ) );
            const alpaka::Vec<TAccDim, ISAAC_IDX_TYPE> blocks ( ISAAC_IDX_TYPE ( 1 ), block_size.y, block_size.x );
            const alpaka::Vec<TAccDim, ISAAC_IDX_TYPE> grid ( ISAAC_IDX_TYPE ( 1 ), grid_size.y, grid_size.x );
            auto const workdiv ( alpaka::WorkDivMembers<TAccDim, ISAAC_IDX_TYPE> ( grid,blocks,threads ) );

            {
                isaacSSAOKernel kernel;
                auto const instance
                (
                    alpaka::createTaskKernel<TAcc>
                    (
                        workdiv,
                        kernel,
                        alpaka::getPtrNative(aoBuffer),
                        alpaka::getPtrNative(depthBuffer),
                        alpaka::getPtrNative(normalBuffer),
                        framebuffer_size,
                        framebuffer_start,
                        ao_properties
                    )
                );
                alpaka::enqueue(stream, instance);
            }
#else
            dim3 block ( block_size.x, block_size.y );
            dim3 grid ( grid_size.x, grid_size.y );
            isaacSSAOKernel<<<grid, block>>>
            (
                aoBuffer,
                depthBuffer,
                normalBuffer,
                framebuffer_size,
                framebuffer_start,
                ao_properties
            );
#endif
        }
    };


    template <
        typename TFramebuffer,
        typename TFramebufferAO,
        typename TFramebufferDepth
#if ISAAC_ALPAKA == 1
        ,typename TAccDim
        ,typename TAcc
        ,typename TStream
#endif
        >
    struct IsaacSSAOFilterKernelCaller {
        
        inline static void call (
#if ISAAC_ALPAKA == 1
            TStream stream,
#endif
            TFramebuffer framebuffer,
            TFramebufferAO aobuffer,
            TFramebufferDepth depthbuffer,
            const isaac_size2& framebuffer_size,
            const isaac_uint2& framebuffer_start,
            IceTInt const * const readback_viewport,
            ao_struct ao_properties
        )
        {
            isaac_size2 block_size= {
                ISAAC_IDX_TYPE ( 8 ),
                ISAAC_IDX_TYPE ( 16 )
            };
            isaac_size2 grid_size= {
                ISAAC_IDX_TYPE ( ( readback_viewport[2]+block_size.x-1 ) /block_size.x + ISAAC_VECTOR_ELEM - 1 ) /ISAAC_IDX_TYPE ( ISAAC_VECTOR_ELEM ),
                ISAAC_IDX_TYPE ( ( readback_viewport[3]+block_size.y-1 ) /block_size.y )
            };
#if ISAAC_ALPAKA == 1
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
            if ( mpl::not_<boost::is_same<TAcc, alpaka::AccGpuCudaRt<TAccDim, ISAAC_IDX_TYPE> > >::value )
#endif
            {
                grid_size.x = ISAAC_IDX_TYPE ( readback_viewport[2] + ISAAC_VECTOR_ELEM - 1 ) /ISAAC_IDX_TYPE ( ISAAC_VECTOR_ELEM );
                grid_size.y = ISAAC_IDX_TYPE ( readback_viewport[3] );
                block_size.x = ISAAC_IDX_TYPE ( 1 );
                block_size.y = ISAAC_IDX_TYPE ( 1 );
            }
            const alpaka::Vec<TAccDim, ISAAC_IDX_TYPE> threads ( ISAAC_IDX_TYPE ( 1 ), ISAAC_IDX_TYPE ( 1 ), ISAAC_IDX_TYPE ( ISAAC_VECTOR_ELEM ) );
            const alpaka::Vec<TAccDim, ISAAC_IDX_TYPE> blocks ( ISAAC_IDX_TYPE ( 1 ), block_size.y, block_size.x );
            const alpaka::Vec<TAccDim, ISAAC_IDX_TYPE> grid ( ISAAC_IDX_TYPE ( 1 ), grid_size.y, grid_size.x );
            auto const workdiv ( alpaka::WorkDivMembers<TAccDim, ISAAC_IDX_TYPE> ( grid,blocks,threads ) );

            {
                isaacSSAOFilterKernel kernel;
                auto const instance
                (
                    alpaka::createTaskKernel<TAcc>
                    (
                        workdiv,
                        kernel,
                        alpaka::getPtrNative(framebuffer),
                        alpaka::getPtrNative(aobuffer),
                        alpaka::getPtrNative(depthbuffer),
                        framebuffer_size,
                        framebuffer_start,
                        ao_properties
                    )
                );
                alpaka::enqueue(stream, instance);
            }
#else
            dim3 block ( block_size.x, block_size.y );
            dim3 grid ( grid_size.x, grid_size.y );
            isaacSSAOFilterKernel<<<grid, block>>>
            (
                framebuffer,
                aobuffer,
                depthbuffer,
                framebuffer_size,
                framebuffer_start,
                ao_properties
            );
#endif
        }
    };



    template<
        int count,
        typename TDest
    >
#if ISAAC_ALPAKA == 1
    struct updateFunctorChainPointerKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
#else
            __global__ void updateFunctorChainPointerKernel(
#endif
            isaac_functor_chain_pointer_N * const functor_chain_choose_d,
            isaac_functor_chain_pointer_N const * const functor_chain_d,
            TDest dest
        )
#if ISAAC_ALPAKA == 1
        const
#endif
        {
            for( int i = 0; i < count; i++ )
            {
                functor_chain_choose_d[i] = functor_chain_d[dest.nr[i]];
            }
        }


#if ISAAC_ALPAKA == 1
    };

#endif


    template<
        typename TSource
    >
#if ISAAC_ALPAKA == 1
    struct updateBufferKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
#else
            __global__ void updateBufferKernel(
#endif
            const TSource source,
            void * const pointer,
            const isaac_int3 local_size
        )
#if ISAAC_ALPAKA == 1
        const
#endif
        {
#if ISAAC_ALPAKA == 1
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            isaac_int3 dest = {
                isaac_int( alpThreadIdx[1] ),
                isaac_int( alpThreadIdx[2] ),
                0
            };
#else
            isaac_int3 dest = {
                isaac_int( threadIdx.x + blockIdx.x * blockDim.x ),
                isaac_int( threadIdx.y + blockIdx.y * blockDim.y ),
                0
            };
#endif
            isaac_int3 coord = dest;
            coord.x -= ISAAC_GUARD_SIZE;
            coord.y -= ISAAC_GUARD_SIZE;
            if( ISAAC_FOR_EACH_DIM_TWICE( 2,
                    dest,
                    >= local_size,
                    + 2 * ISAAC_GUARD_SIZE || ) 0 )
                return;
            isaac_float_dim <TSource::feature_dim> * ptr =
                ( isaac_float_dim < TSource::feature_dim > * )( pointer );
            if( TSource::has_guard )
            {
                coord.z = -ISAAC_GUARD_SIZE;
                for( ; dest.z < local_size.z + 2 * ISAAC_GUARD_SIZE; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                            * ( local_size.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                    coord.z++;
                }
            }
            else
            {
                if( coord.x < 0 )
                {
                    coord.x = 0;
                }
                if( coord.x >= local_size.x )
                {
                    coord.x = local_size.x - 1;
                }
                if( coord.y < 0 )
                {
                    coord.y = 0;
                }
                if( coord.y >= local_size.y )
                {
                    coord.y = local_size.y - 1;
                }
                coord.z = 0;
                for( ; dest.z < ISAAC_GUARD_SIZE; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                            * ( local_size.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                }
                for( ; dest.z < local_size.z + ISAAC_GUARD_SIZE - 1; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                            * ( local_size.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                    coord.z++;
                }
                for( ; dest.z < local_size.z + 2 * ISAAC_GUARD_SIZE; dest.z++ )
                {
                    ptr[dest.x
                        + dest.y * ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                        + dest.z * (
                            ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                            * ( local_size.y + 2 * ISAAC_GUARD_SIZE )
                        )] = source[coord];
                }
            }
        }


#if ISAAC_ALPAKA == 1
    };

#endif


    template<
        int N
    >
    struct dest_array_struct
    {
        isaac_int nr[N];
    };


    template<
        typename TSource
    >
#if ISAAC_ALPAKA == 1
    struct minMaxKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
#else
            __global__ void minMaxKernel(
#endif
            const TSource source,
            const int nr,
            minmax_struct * const result,
            const isaac_int3 local_size,
            void const * const pointer
        )
#if ISAAC_ALPAKA == 1
        const
#endif
        {
#if ISAAC_ALPAKA == 1
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            isaac_int3 coord = {
                isaac_int( alpThreadIdx[1] ),
                isaac_int( alpThreadIdx[2] ),
                0
            };
#else
            isaac_int3 coord = {
                isaac_int( threadIdx.x + blockIdx.x * blockDim.x ),
                isaac_int( threadIdx.y + blockIdx.y * blockDim.y ),
                0
            };
#endif
            if( ISAAC_FOR_EACH_DIM_TWICE ( 2,
                    coord,
                    >= local_size,
                    || ) 0 )
                return;
            isaac_float min = FLT_MAX;
            isaac_float max = -FLT_MAX;
            for( ; coord.z < local_size.z; coord.z++ )
            {
                isaac_float_dim <TSource::feature_dim> data;
                if( TSource::persistent )
                {
                    data = source[coord];
                }
                else
                {
                    isaac_float_dim <TSource::feature_dim> * ptr = (
                        isaac_float_dim < TSource::feature_dim > *
                    )( pointer );
                    data = ptr[coord.x + ISAAC_GUARD_SIZE
                               + ( coord.y + ISAAC_GUARD_SIZE )
                                 * ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                               + ( coord.z + ISAAC_GUARD_SIZE ) * (
                                   ( local_size.x + 2 * ISAAC_GUARD_SIZE )
                                   * ( local_size.y + 2 * ISAAC_GUARD_SIZE )
                               )];
                };
                isaac_float value = isaac_float( 0 );
#if ISAAC_ALPAKA == 1 || defined(__CUDA_ARCH__)
                if( TSource::feature_dim == 1 )
                {
                    value =
                        reinterpret_cast<isaac_functor_chain_pointer_1> ( isaac_function_chain_d[nr] )(
                            *( reinterpret_cast< isaac_float_dim< 1 > * > ( &data ) ),
                            nr
                        );
                }
                if( TSource::feature_dim == 2 )
                {
                    value =
                        reinterpret_cast<isaac_functor_chain_pointer_2> ( isaac_function_chain_d[nr] )(
                            *( reinterpret_cast< isaac_float_dim< 2 > * > ( &data ) ),
                            nr
                        );
                }
                if( TSource::feature_dim == 3 )
                {
                    value =
                        reinterpret_cast<isaac_functor_chain_pointer_3> ( isaac_function_chain_d[nr] )(
                            *( reinterpret_cast< isaac_float_dim< 3 > * > ( &data ) ),
                            nr
                        );
                }
                if( TSource::feature_dim == 4 )
                {
                    value =
                        reinterpret_cast<isaac_functor_chain_pointer_4> ( isaac_function_chain_d[nr] )(
                            *( reinterpret_cast< isaac_float_dim< 4 > * > ( &data ) ),
                            nr
                        );
                }
#endif
                if( value > max )
                {
                    max = value;
                }
                if( value < min )
                {
                    min = value;
                }
            }
            result[coord.x + coord.y * local_size.x].min = min;
            result[coord.x + coord.y * local_size.x].max = max;
        }


#if ISAAC_ALPAKA == 1
    };

#endif


    template<
        typename TParticleSource
    >
#if ISAAC_ALPAKA == 1
    struct minMaxPartikelKernel
    {
        template<
            typename TAcc__
        >
        ALPAKA_FN_ACC void operator()(
            TAcc__ const & acc,
#else
            __global__ void minMaxPartikelKernel(
#endif
            const TParticleSource particle_source,
            const int nr,
            minmax_struct * const result,
            const isaac_int3 local_size
        )
#if ISAAC_ALPAKA == 1
        const
#endif
        {
#if ISAAC_ALPAKA == 1
            auto alpThreadIdx = alpaka::getIdx<
                alpaka::Grid,
                alpaka::Threads
            >( acc );
            isaac_uint3 coord = {
                isaac_uint( alpThreadIdx[1] ),
                isaac_uint( alpThreadIdx[2] ),
                0
            };
#else
            isaac_uint3 coord = {
                isaac_uint( threadIdx.x + blockIdx.x * blockDim.x ),
                isaac_uint( threadIdx.y + blockIdx.y * blockDim.y ),
                0
            };
#endif
            if( ISAAC_FOR_EACH_DIM_TWICE ( 2,
                    coord,
                    >= local_size,
                    || ) 0 )
                return;
            isaac_float min = FLT_MAX;
            isaac_float max = -FLT_MAX;
            for( ; coord.z < local_size.z; coord.z++ )
            {
                auto particle_iterator = particle_source.getIterator( coord );
                for( int i = 0; i < particle_iterator.size; i++ )
                {
                    isaac_float_dim <TParticleSource::feature_dim> data;

                    data = particle_iterator.getAttribute( );

                    isaac_float value = isaac_float( 0 );
#if ISAAC_ALPAKA == 1 || defined(__CUDA_ARCH__)
                    if( TParticleSource::feature_dim == 1 )
                    {
                        value =
                            reinterpret_cast<isaac_functor_chain_pointer_1> ( isaac_function_chain_d[nr] )(
                                *( reinterpret_cast< isaac_float_dim< 1 > * > ( &data ) ),
                                nr
                            );
                    }
                    if( TParticleSource::feature_dim == 2 )
                    {
                        value =
                            reinterpret_cast<isaac_functor_chain_pointer_2> ( isaac_function_chain_d[nr] )(
                                *( reinterpret_cast< isaac_float_dim< 2 > * > ( &data ) ),
                                nr
                            );
                    }
                    if( TParticleSource::feature_dim == 3 )
                    {
                        value =
                            reinterpret_cast<isaac_functor_chain_pointer_3> ( isaac_function_chain_d[nr] )(
                                *( reinterpret_cast< isaac_float_dim< 3 > * > ( &data ) ),
                                nr
                            );
                    }
                    if( TParticleSource::feature_dim == 4 )
                    {
                        value =
                            reinterpret_cast<isaac_functor_chain_pointer_4> ( isaac_function_chain_d[nr] )(
                                *( reinterpret_cast< isaac_float_dim< 4 > * > ( &data ) ),
                                nr
                            );
                    }
#endif
                    if( value > max )
                    {
                        max = value;
                    }
                    if( value < min )
                    {
                        min = value;
                    }
                }

            }
            result[coord.x + coord.y * local_size.x].min = min;
            result[coord.x + coord.y * local_size.x].max = max;
        }


#if ISAAC_ALPAKA == 1
    };

#endif

} //namespace isaac;

#pragma GCC diagnostic pop
