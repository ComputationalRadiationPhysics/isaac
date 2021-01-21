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

/* Hack for a bug, which occurs only in CUDA 7.0
 * __CUDACC_VER_MAJOR__ is first defined in CUDA 7.5, so this checks for
 * CUDA Version < 7.5 */
#if !defined(__CUDACC_VER_MAJOR__) && !defined(BOOST_RESULT_OF_USE_TR1)
#define BOOST_RESULT_OF_USE_TR1
#endif

#include <iostream>
#include <random>
#include <string>
#include <string.h>
#include <jansson.h>
#include <pthread.h>
#include <list>
#include <vector>
#include <stdexcept>
#include <memory>
#include <mpi.h>
//Against annoying C++11 warning in mpi.h
#include <IceT.h>


#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wall"


#include <IceTMPI.h>


#pragma GCC diagnostic pop


#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <map>


#if ISAAC_ALPAKA == 1


#include <alpaka/alpaka.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/not.hpp>


#else


#include <boost/mpl/int.hpp>


#endif


#include "isaac/isaac_kernel.hpp"
#include "isaac/isaac_communicator.hpp"
#include "isaac/isaac_helper.hpp"
#include "isaac/isaac_controllers.hpp"
#include "isaac/isaac_compositors.hpp"
#include "isaac/isaac_compositors.hpp"
#include "isaac/isaac_version.hpp"


namespace isaac
{

    template<
#if ISAAC_ALPAKA == 1
    typename THost,
    typename TAcc,
    typename TStream,
    typename TAccDim,
#endif
        typename TSimDim,
        typename TParticleList,
        typename TSourceList,
        typename TDomainSize,
        ISAAC_IDX_TYPE TTransfer_size,
        typename TScale,
        typename TController,
        typename TCompositor
    >
    class IsaacVisualization
    {
    public:

        IsaacCommunicator * getCommunicator( )
        {
            return communicator;
        }


#if ISAAC_ALPAKA == 1
        using TDevAcc = alpaka::Dev< TAcc >;
        using TFraDim = alpaka::DimInt< 1 >;
        using TTexDim = alpaka::DimInt< 1 >;
#endif

        struct source_2_json_iterator
        {
            template<
                typename TSource,
                typename TJsonRoot
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const TSource & s,
                TJsonRoot & jsonRoot
            ) const
            {
                json_t * content = json_object( );
                json_array_append_new(
                    jsonRoot,
                    content
                );
                json_object_set_new(
                    content,
                    "name",
                    json_string( TSource::getName( ).c_str( ) )
                );
                json_object_set_new(
                    content,
                    "feature dimension",
                    json_integer( s.feature_dim )
                );
            }
        };

        struct particle_source_2_json_iterator
        {
            template<
                typename TSource,
                typename TJsonRoot
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const TSource & s,
                TJsonRoot & jsonRoot
            ) const
            {
                json_t * content = json_object( );
                json_array_append_new(
                    jsonRoot,
                    content
                );
                json_object_set_new(
                    content,
                    "name",
                    json_string( TSource::getName( ).c_str( ) )
                );
                json_object_set_new(
                    content,
                    "feature dimension",
                    json_integer( 3 )
                );
            }
        };

        struct functor_2_json_iterator
        {
            template<
                typename TFunctor,
                typename TJsonRoot
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const TFunctor & f,
                TJsonRoot & jsonRoot
            ) const
            {
                json_t * content = json_object( );
                json_array_append_new(
                    jsonRoot,
                    content
                );
                json_object_set_new(
                    content,
                    "name",
                    json_string( TFunctor::getName( ).c_str( ) )
                );
                json_object_set_new(
                    content,
                    "description",
                    json_string( TFunctor::getDescription( ).c_str( ) )
                );
                json_object_set_new(
                    content,
                    "uses parameter",
                    json_boolean( TFunctor::uses_parameter )
                );
            }
        };

        struct parse_functor_iterator
        {
            template<
                typename TFunctor,
                typename TName,
                typename TValue,
                typename TFound
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                TFunctor & f,
                const TName & name,
                TValue & value,
                TFound & found
            ) const
            {
                if( !found && name == TFunctor::getName( ) )
                {
                    value = I;
                    found = true;
                }
            }
        };

        struct update_functor_chain_iterator
        {
            template<
                typename TSource,
                typename TFunctions,
                typename TOffset,
                typename TDest
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const TSource & source,
                const TFunctions & functions,
                const TOffset & offset,
                TDest & dest
            ) const
            {
                isaac_int chain_nr = 0;
                for( int i = 0; i < ISAAC_MAX_FUNCTORS; i++ )
                {
                    chain_nr *= ISAAC_FUNCTOR_COUNT;
                    chain_nr += functions[I + offset].bytecode[i];
                }
                dest.nr[I + offset] = chain_nr * 4 + TSource::feature_dim - 1;
            }
        };

        struct allocate_pointer_array_iterator
        {
            template<
                typename TSource,
                typename TArray,
                typename TLocalSize
#if ISAAC_ALPAKA == 1
            ,
            typename TVector,
            typename TDevAcc__
#endif
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const TSource & source,
                TArray & pointer_array,
                const TLocalSize & local_size
#if ISAAC_ALPAKA == 1
            ,
            TVector & alpaka_vector,
            const TDevAcc__ & acc
#endif
            ) const
            {
                if( TSource::persistent )
                {
                    pointer_array.pointer[I] = NULL;
                }
                else
                {
#if ISAAC_ALPAKA == 1
                    alpaka_vector.push_back(
                        alpaka::Buf<
                            TDevAcc,
                            isaac_float,
                            TFraDim,
                            ISAAC_IDX_TYPE
                        >(
                            alpaka::allocBuf<
                                isaac_float,
                                ISAAC_IDX_TYPE
                            >(
                                acc,
                                alpaka::Vec<
                                    TFraDim,
                                    ISAAC_IDX_TYPE
                                >(
                                    ISAAC_IDX_TYPE(
                                        TSource::feature_dim * (
                                            local_size[0] + 2 * ISAAC_GUARD_SIZE
                                        ) * (
                                            local_size[1] + 2 * ISAAC_GUARD_SIZE
                                        ) * (
                                            local_size[2] + 2 * ISAAC_GUARD_SIZE
                                        )
                                    )
                                )
                            )
                        )
                    );
                    pointer_array.pointer[I] =
                        alpaka::getPtrNative( alpaka_vector.back( ) );
#else
                    ISAAC_CUDA_CHECK ( cudaMalloc(
                        ( void ** ) &( pointer_array.pointer[I] ),
                        sizeof( isaac_float_dim < TSource::feature_dim > )
                        * ( local_size[0] + 2 * ISAAC_GUARD_SIZE )
                        * ( local_size[1] + 2 * ISAAC_GUARD_SIZE )
                        * ( local_size[2] + 2 * ISAAC_GUARD_SIZE )
                    ) );
#endif
                }
            }
        };


        struct update_particle_source_iterator
        {
            /** Update iterator for particle sources
             *
             * Iterator for updating the particle sources with a boolean if the
             * Particle Source is enabled and further user defined information
             * in the pointer
             *
             * @tparam TParticleSource is the particle source type
             * @tparam TWeight is weight type
             * @tparam TOffset is the offset type
             * @tparam TPointer is the pointer type
             * @param I is the index of the current source
             * @param particle_source is the current particle source
             * @param weight is the array with all source weights
             * @param weightArrayOffset is the offset in the array to the particle sources
             * @param pointer is the pointer to the user defined additional information
             *
             */
            template<
                typename TParticleSource,
                typename TWeight,
                typename TOffset,
                typename TPointer
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                TParticleSource & particle_source,
                const TWeight & weight,
                const TOffset & weightArrayOffset,
                const TPointer & pointer
            ) const
            {
                bool enabled =
                    weight.value[I + weightArrayOffset] != isaac_float( 0 );
                particle_source.update(
                    enabled,
                    pointer
                );
            }

        };

        struct update_pointer_array_iterator
        {
            template<
                typename TSource,
                typename TArray,
                typename TLocalSize,
                typename TWeight,
                typename TPointer
#if ISAAC_ALPAKA == 1
            ,
            typename TStream__
#endif
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                TSource & source,
                TArray & pointer_array,
                const TLocalSize & local_size,
                const TWeight & weight,
                const TPointer & pointer
#if ISAAC_ALPAKA == 1
            ,
            TStream__ & stream
#endif
            ) const
            {
                bool enabled = weight.value[I] != isaac_float( 0 );
                source.update(
                    enabled,
                    pointer
                );
                if( !TSource::persistent && enabled )
                {
                    isaac_size2 grid_size = {
                        ISAAC_IDX_TYPE(
                            ( local_size[0] + ISAAC_GUARD_SIZE * 2 + 15 ) / 16
                        ),
                        ISAAC_IDX_TYPE(
                            ( local_size[1] + ISAAC_GUARD_SIZE * 2 + 15 ) / 16
                        ),
                    };
                    isaac_size2 block_size = {
                        ISAAC_IDX_TYPE( 16 ),
                        ISAAC_IDX_TYPE( 16 ),
                    };
                    isaac_int3 local_size_array = {
                        isaac_int( local_size[0] ),
                        isaac_int( local_size[1] ),
                        isaac_int( local_size[2] )
                    };
#if ISAAC_ALPAKA == 1
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
                    if ( mpl::not_<boost::is_same<TAcc, alpaka::AccGpuCudaRt<TAccDim, ISAAC_IDX_TYPE> > >::value )
#endif
                    {
                        grid_size.x = ISAAC_IDX_TYPE(
                            local_size[0] + ISAAC_GUARD_SIZE * 2
                        );
                        grid_size.y = ISAAC_IDX_TYPE(
                            local_size[0] + ISAAC_GUARD_SIZE * 2
                        );
                        block_size.x = ISAAC_IDX_TYPE( 1 );
                        block_size.y = ISAAC_IDX_TYPE( 1 );
                    }
                    const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> threads(
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 )
                    );
                    const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> blocks(
                        ISAAC_IDX_TYPE( 1 ),
                        block_size.x,
                        block_size.y
                    );
                    const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> grid(
                        ISAAC_IDX_TYPE( 1 ),
                        grid_size.x,
                        grid_size.y
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
                    updateBufferKernel< TSource > kernel;
                    auto const instance(
                        alpaka::createTaskKernel< TAcc >(
                            workdiv,
                            kernel,
                            source,
                            pointer_array.pointer[I],
                            local_size_array
                        )
                    );
                    alpaka::enqueue(
                        stream,
                        instance
                    );
                    alpaka::wait( stream );
#else
                    dim3 block(
                        block_size.x,
                        block_size.y
                    );
                    dim3 grid(
                        grid_size.x,
                        grid_size.y
                    );
                    updateBufferKernel <<< grid, block >>> (
                        source,
                        pointer_array.pointer[I],
                        local_size_array
                    );
                    ISAAC_CUDA_CHECK ( cudaDeviceSynchronize( ) );
#endif
                }
            }
        };

        struct calc_minmax_iterator
        {
            template<
                typename TSource,
                typename TArray,
                typename TMinmax,
                typename TLocalMinmax,
                typename TLocalSize
#if ISAAC_ALPAKA == 1
            ,
            typename TStream__,
            typename THost__
#endif
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const TSource & source,
                TArray & pointer_array,
                TMinmax & minmax,
                TLocalMinmax & local_minmax,
                TLocalSize & local_size
#if ISAAC_ALPAKA == 1
            ,
            TStream__ & stream,
            const THost__ & host
#endif
            ) const
            {
                isaac_size2 grid_size = {
                    ISAAC_IDX_TYPE( ( local_size[0] + 15 ) / 16 ),
                    ISAAC_IDX_TYPE( ( local_size[1] + 15 ) / 16 ),
                };
                isaac_size2 block_size = {
                    ISAAC_IDX_TYPE( 16 ),
                    ISAAC_IDX_TYPE( 16 ),
                };
                isaac_int3 local_size_array = {
                    isaac_int( local_size[0] ),
                    isaac_int( local_size[1] ),
                    isaac_int( local_size[2] )
                };
                minmax_struct
                    local_minmax_array_h[local_size_array.x * local_size_array.y
                ];
                if( local_size[0] != 0 && local_size[1] != 0 )
                {
#if ISAAC_ALPAKA == 1
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
                    if ( mpl::not_<boost::is_same<TAcc, alpaka::AccGpuCudaRt<TAccDim, ISAAC_IDX_TYPE> > >::value )
#endif
                    {
                        grid_size.x = ISAAC_IDX_TYPE( local_size[0] );
                        grid_size.y = ISAAC_IDX_TYPE( local_size[0] );
                        block_size.x = ISAAC_IDX_TYPE( 1 );
                        block_size.y = ISAAC_IDX_TYPE( 1 );
                    }
                    const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> threads(
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 )
                    );
                    const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> blocks(
                        ISAAC_IDX_TYPE( 1 ),
                        block_size.x,
                        block_size.y
                    );
                    const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> grid(
                        ISAAC_IDX_TYPE( 1 ),
                        grid_size.x,
                        grid_size.y
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
                    minMaxKernel< TSource > kernel;
                    auto const instance(
                        alpaka::createTaskKernel< TAcc >(
                            workdiv,
                            kernel,
                            source,
                            I,
                            alpaka::getPtrNative( local_minmax ),
                            local_size_array,
                            pointer_array.pointer[I]
                        )
                    );
                    alpaka::enqueue(
                        stream,
                        instance
                    );
                    alpaka::wait( stream );
                    alpaka::ViewPlainPtr <THost, minmax_struct, TFraDim, ISAAC_IDX_TYPE>
                        minmax_buffer(
                        local_minmax_array_h,
                        host,
                        alpaka::Vec<
                            TFraDim,
                            ISAAC_IDX_TYPE
                        >(
                            ISAAC_IDX_TYPE(
                                local_size_array.x * local_size_array.y
                            )
                        )
                    );
                    alpaka::memcpy(
                        stream,
                        minmax_buffer,
                        local_minmax,
                        alpaka::Vec<
                            TFraDim,
                            ISAAC_IDX_TYPE
                        >(
                            ISAAC_IDX_TYPE(
                                local_size_array.x * local_size_array.y
                            )
                        )
                    );
#else
                    dim3 block(
                        block_size.x,
                        block_size.y
                    );
                    dim3 grid(
                        grid_size.x,
                        grid_size.y
                    );
                    minMaxKernel <<< grid, block >>> (
                        source,
                        I,
                        local_minmax,
                        local_size_array,
                        pointer_array.pointer[I]
                    );
                    ISAAC_CUDA_CHECK ( cudaMemcpy(
                        local_minmax_array_h,
                        local_minmax,
                        sizeof( minmax_struct ) * local_size_array.x
                        * local_size_array.y,
                        cudaMemcpyDeviceToHost
                    ) );
#endif
                }
                minmax.min[I] = FLT_MAX;
                minmax.max[I] = -FLT_MAX;
                for( int i = 0;
                    i < local_size_array.x * local_size_array.y;
                    i++ )
                {
                    if( local_minmax_array_h[i].min < minmax.min[I] )
                    {
                        minmax.min[I] = local_minmax_array_h[i].min;
                    }
                    if( local_minmax_array_h[i].max > minmax.max[I] )
                    {
                        minmax.max[I] = local_minmax_array_h[i].max;
                    }
                }
            }
        };

        // calculate minmax for particles
        template<
            int TOffset
        >
        struct calc_particle_minmax_iterator
        {
            template<
                typename TParticleSource,
                typename TMinmax,
                typename TLocalMinmax,
                typename TLocalSize
#if ISAAC_ALPAKA == 1
            ,
            typename TStream__,
            typename THost__
#endif
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const TParticleSource & particle_source,
                TMinmax & minmax,
                TLocalMinmax & local_minmax,
                TLocalSize & local_size
#if ISAAC_ALPAKA == 1
            ,
            TStream__ & stream,
            const THost__ & host
#endif
            ) const
            {
                // iterate over all cells and the particle lists

                isaac_size2 grid_size = {
                    ISAAC_IDX_TYPE( ( local_size[0] + 15 ) / 16 ),
                    ISAAC_IDX_TYPE( ( local_size[1] + 15 ) / 16 ),
                };
                isaac_size2 block_size = {
                    ISAAC_IDX_TYPE( 16 ),
                    ISAAC_IDX_TYPE( 16 ),
                };
                isaac_int3 local_size_array = {
                    isaac_int( local_size[0] ),
                    isaac_int( local_size[1] ),
                    isaac_int( local_size[2] )
                };
                minmax_struct
                    local_minmax_array_h[local_size_array.x * local_size_array.y
                ];
                if( local_size[0] != 0 && local_size[1] != 0 )
                {
#if ISAAC_ALPAKA == 1
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
                    if ( mpl::not_<boost::is_same<TAcc, alpaka::AccGpuCudaRt<TAccDim, ISAAC_IDX_TYPE> > >::value )
#endif
                    {
                        grid_size.x = ISAAC_IDX_TYPE( local_size[0] );
                        grid_size.y = ISAAC_IDX_TYPE( local_size[0] );
                        block_size.x = ISAAC_IDX_TYPE( 1 );
                        block_size.y = ISAAC_IDX_TYPE( 1 );
                    }
                    const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> threads(
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 )
                    );
                    const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> blocks(
                        ISAAC_IDX_TYPE( 1 ),
                        block_size.x,
                        block_size.y
                    );
                    const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> grid(
                        ISAAC_IDX_TYPE( 1 ),
                        grid_size.x,
                        grid_size.y
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
                    minMaxPartikelKernel< TParticleSource > kernel;
                    auto const instance(
                        alpaka::createTaskKernel< TAcc >(
                            workdiv,
                            kernel,
                            particle_source,
                            I + TOffset,
                            alpaka::getPtrNative( local_minmax ),
                            local_size_array
                        )
                    );
                    alpaka::enqueue(
                        stream,
                        instance
                    );
                    alpaka::wait( stream );
                    alpaka::ViewPlainPtr <THost, minmax_struct, TFraDim, ISAAC_IDX_TYPE>
                        minmax_buffer(
                        local_minmax_array_h,
                        host,
                        alpaka::Vec<
                            TFraDim,
                            ISAAC_IDX_TYPE
                        >(
                            ISAAC_IDX_TYPE(
                                local_size_array.x * local_size_array.y
                            )
                        )
                    );
                    alpaka::memcpy(
                        stream,
                        minmax_buffer,
                        local_minmax,
                        alpaka::Vec<
                            TFraDim,
                            ISAAC_IDX_TYPE
                        >(
                            ISAAC_IDX_TYPE(
                                local_size_array.x * local_size_array.y
                            )
                        )
                    );
#else
                    dim3 block(
                        block_size.x,
                        block_size.y
                    );
                    dim3 grid(
                        grid_size.x,
                        grid_size.y
                    );
                    minMaxPartikelKernel <<< grid, block >>> (
                        particle_source,
                        I + TOffset,
                        local_minmax,
                        local_size_array
                    );
                    ISAAC_CUDA_CHECK ( cudaMemcpy(
                        local_minmax_array_h,
                        local_minmax,
                        sizeof( minmax_struct ) * local_size_array.x
                        * local_size_array.y,
                        cudaMemcpyDeviceToHost
                    ) );
#endif
                }
                minmax.min[I + TOffset] = FLT_MAX;
                minmax.max[I + TOffset] = -FLT_MAX;
                // find the min and max
                for( int i = 0;
                    i < local_size_array.x * local_size_array.y;
                    i++ )
                {
                    if( local_minmax_array_h[i].min < minmax.min[I + TOffset] )
                    {
                        minmax.min[I + TOffset] = local_minmax_array_h[i].min;
                    }
                    if( local_minmax_array_h[i].max > minmax.max[I + TOffset] )
                    {
                        minmax.max[I + TOffset] = local_minmax_array_h[i].max;
                    }
                }
            }
        };


        IsaacVisualization(
#if ISAAC_ALPAKA == 1
        THost host,
        TDevAcc acc,
        TStream stream,
#endif
            const std::string name,
            const isaac_int master,
            const std::string server_url,
            const isaac_uint server_port,
            const isaac_size2 framebuffer_size,
            const TDomainSize global_size,
            const TDomainSize local_size,
            const TDomainSize local_particle_size,
            const TDomainSize position,
            TParticleList & particle_sources,
            TSourceList & sources,
            TScale scale

        ) :
#if ISAAC_ALPAKA == 1
        host( host ),
        acc( acc ),
        stream( stream ),
#endif
            global_size( global_size ),
            local_size( local_size ),
            local_particle_size( local_particle_size ),
            position( position ),
            name( name ),
            master( master ),
            server_url( server_url ),
            server_port( server_port ),
            framebuffer_size( framebuffer_size ),
            compbuffer_size( TCompositor::getCompositedbufferSize( framebuffer_size ) ),
            compositor( framebuffer_size ),
            metaNr( 0 ),
            visualizationThread( 0 ),
            kernel_time( 0 ),
            merge_time( 0 ),
            video_send_time( 0 ),
            copy_time( 0 ),
            sorting_time( 0 ),
            buffer_time( 0 ),
            interpolation( false ),
            iso_surface( false ),
            step( isaac_float( ISAAC_DEFAULT_STEP ) ),
            framebuffer_prod(
                ISAAC_IDX_TYPE( framebuffer_size.x )
                * ISAAC_IDX_TYPE( framebuffer_size.y )
            ),
            particle_sources( particle_sources ),
            sources( sources ),
            scale( scale ),
            icet_bounding_box( true )
#if ISAAC_ALPAKA == 1
        ,
        framebuffer(
            alpaka::allocBuf<
                uint32_t,
                ISAAC_IDX_TYPE
            >(
                acc,
                framebuffer_prod
            )
        ),
        framebufferAO(
            alpaka::allocBuf<
                isaac_float,
                ISAAC_IDX_TYPE
            >(
                acc,
                framebuffer_prod
            )
        ),
        framebufferDepth(
            alpaka::allocBuf<
                isaac_float3,
                ISAAC_IDX_TYPE
            >(
                acc,
                framebuffer_prod
            )
        ),
        framebufferNormal(
            alpaka::allocBuf<
                isaac_float3,
                ISAAC_IDX_TYPE
            >(
                acc,
                framebuffer_prod
            )
        ),
        functor_chain_d(
            alpaka::allocBuf<
                isaac_functor_chain_pointer_N,
                ISAAC_IDX_TYPE
            >(
                acc,
                ISAAC_IDX_TYPE( ISAAC_FUNCTOR_COMPLEX * 4 ) ) )
    ,

    functor_chain_choose_d ( alpaka::allocBuf<
        isaac_functor_chain_pointer_N,
        ISAAC_IDX_TYPE
    >(
        acc,
        ISAAC_IDX_TYPE ( ( boost
        ::mpl::size< TSourceList >::type::value

    + boost::mpl::size< TParticleList >::type::value ) ) ) )
    ,

    local_minmax_array_d ( alpaka::allocBuf<
        minmax_struct,
        ISAAC_IDX_TYPE
    >(
        acc,
        ISAAC_IDX_TYPE ( local_size[0]

    * local_size[1] ) ) )
    ,

    local_particle_minmax_array_d ( alpaka::allocBuf<
        minmax_struct,
        ISAAC_IDX_TYPE
    >(
        acc,
        ISAAC_IDX_TYPE ( local_particle_size[0]

    * local_particle_size[1] ) ) )
    {
#else
        {
            ISAAC_CUDA_CHECK ( cudaMalloc(
                ( uint32_t * * ) & framebuffer,
                sizeof( uint32_t ) * framebuffer_prod
            ) );
            ISAAC_CUDA_CHECK ( cudaMalloc(
                ( isaac_float * * ) & framebufferAO,
                sizeof( isaac_float ) * framebuffer_prod
            ) );
            ISAAC_CUDA_CHECK ( cudaMalloc(
                ( isaac_float3 * * ) & framebufferDepth,
                sizeof( isaac_float3 ) * framebuffer_prod
            ) );
            ISAAC_CUDA_CHECK ( cudaMalloc(
                ( isaac_float3 * * ) & framebufferNormal,
                sizeof( isaac_float3 ) * framebuffer_prod
            ) );
            ISAAC_CUDA_CHECK ( cudaMalloc(
                ( isaac_functor_chain_pointer_N ** ) &functor_chain_d,
                sizeof( isaac_functor_chain_pointer_N ) * ISAAC_FUNCTOR_COMPLEX
                * 4
            ) );
            ISAAC_CUDA_CHECK ( cudaMalloc(
                ( isaac_functor_chain_pointer_N ** ) &functor_chain_choose_d,
                sizeof( isaac_functor_chain_pointer_N ) * (
                    boost::mpl::size< TSourceList >::type::value
                    + boost::mpl::size< TParticleList >::type::value
                )
            ) );
            ISAAC_CUDA_CHECK ( cudaMalloc(
                ( minmax_struct ** ) &local_minmax_array_d,
                sizeof( minmax_struct ) * local_size[0] * local_size[1]
            ) );
            ISAAC_CUDA_CHECK ( cudaMalloc(
                ( minmax_struct ** ) &local_particle_minmax_array_d,
                sizeof( minmax_struct ) * local_particle_size[0]
                * local_particle_size[1]
            ) );
#endif
#if ISAAC_VALGRIND_TWEAKS == 1
            //Jansson has some optimizations for 2 and 4 byte aligned
            //memory. However valgrind complains then sometimes about reads
            //in not allocated memory. Valgrind is right, but nevertheless
            //reads will never crash and be much faster. But for
            //debugging reasons let's alloc 4 extra bytes for valgrind:
            json_set_alloc_funcs(
                extra_malloc,
                extra_free
            );
#endif
            json_object_seed( 0 );
            for( int i = 0; i < 3; i++ )
            {
                global_size_scaled.push_back(
                    isaac_int(
                        ( isaac_float ) global_size[i]
                        * ( isaac_float ) scale[i]
                    )
                );
                local_size_scaled.push_back(
                    isaac_int(
                        ( isaac_float ) local_size[i] * ( isaac_float ) scale[i]
                    )
                );
                position_scaled.push_back(
                    isaac_int(
                        ( isaac_float ) position[i] * ( isaac_float ) scale[i]
                    )
                );
            }

            background_color[0] = 0;
            background_color[1] = 0;
            background_color[2] = 0;
            background_color[3] = 1;

            //INIT
            MPI_Comm_dup(
                MPI_COMM_WORLD,
                &mpi_world
            );
            MPI_Comm_rank(
                mpi_world,
                &rank
            );
            MPI_Comm_size(
                mpi_world,
                &numProc
            );
            if( rank == master )
            {
                this->communicator = new IsaacCommunicator(
                    server_url,
                    server_port
                );
            }
            else
            {
                this->communicator = NULL;
            }
            recreateJSON( );
            controller.updateProjection(
                projection,
                framebuffer_size,
                NULL,
                true
            );
            look_at[0] = 0.0f;
            look_at[1] = 0.0f;
            look_at[2] = 0.0f;
            ISAAC_SET_IDENTITY ( 3,
                rotation )
            distance = -4.5f;
            updateModelview( );

            //Create functor chain pointer lookup table
#if ISAAC_ALPAKA == 1
            const alpaka::Vec<
                TAccDim,
                ISAAC_IDX_TYPE
            > threads(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            const alpaka::Vec<
                TAccDim,
                ISAAC_IDX_TYPE
            > blocks(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            const alpaka::Vec<
                TAccDim,
                ISAAC_IDX_TYPE
            > grid(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
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
            fillFunctorChainPointerKernel kernel;
            auto const instance(
                alpaka::createTaskKernel< TAcc >(
                    workdiv,
                    kernel,
                    alpaka::getPtrNative( functor_chain_d )
                )
            );
            alpaka::enqueue(
                stream,
                instance
            );
            alpaka::wait( stream );
#else
            dim3 grid( 1 );
            dim3 block( 1 );
            fillFunctorChainPointerKernel <<< grid, block >>> ( functor_chain_d );
            ISAAC_CUDA_CHECK ( cudaDeviceSynchronize( ) );
#endif
            //Init functions:
            for( int i = 0; i < (
                boost::mpl::size< TSourceList >::type::value
                + boost::mpl::size< TParticleList >::type::value
            ); i++ )
            {
                functions[i].source = std::string( "idem" );
            }
            updateFunctions( );

            //non persistent buffer memory
            isaac_for_each_params(
                sources,
                allocate_pointer_array_iterator( ),
                pointer_array,
                local_size
#if ISAAC_ALPAKA == 1
            ,
            pointer_array_alpaka,
            acc
#endif
            );

            //Transfer func memory:
            for( int i = 0; i < (
                boost::mpl::size< TSourceList >::type::value
                + boost::mpl::size< TParticleList >::type::value
            ); i++ )
            {
                source_weight.value[i] = ISAAC_DEFAULT_WEIGHT;
#if ISAAC_ALPAKA == 1
                transfer_d_buf.push_back(
                    alpaka::Buf<
                        TDevAcc,
                        isaac_float4,
                        TTexDim,
                        ISAAC_IDX_TYPE
                    >(
                        alpaka::allocBuf<
                            isaac_float4,
                            ISAAC_IDX_TYPE
                        >(
                            acc,
                            alpaka::Vec<
                                TTexDim,
                                ISAAC_IDX_TYPE
                            >( TTransfer_size )
                        )
                    )
                );
                transfer_h_buf.push_back(
                    alpaka::Buf<
                        THost,
                        isaac_float4,
                        TTexDim,
                        ISAAC_IDX_TYPE
                    >(
                        alpaka::allocBuf<
                            isaac_float4,
                            ISAAC_IDX_TYPE
                        >(
                            host,
                            alpaka::Vec<
                                TTexDim,
                                ISAAC_IDX_TYPE
                            >( TTransfer_size )
                        )
                    )
                );
                transfer_d.pointer[i] =
                    alpaka::getPtrNative( transfer_d_buf[i] );
                transfer_h.pointer[i] =
                    alpaka::getPtrNative( transfer_h_buf[i] );
#else
                ISAAC_CUDA_CHECK ( cudaMalloc(
                    ( isaac_float4 * * ) & ( transfer_d.pointer[i] ),
                    sizeof( isaac_float4 ) * TTransfer_size
                ) );
                transfer_h.pointer[i] = ( isaac_float4 * )
                malloc( sizeof( isaac_float4 ) * TTransfer_size );
#endif
                //Init volume transfer func with a alpha ramp from 0 -> 1
                if( i < boost::mpl::size< TSourceList >::type::value )
                {
                    transfer_h.description[i].insert(
                        std::pair<
                            isaac_uint,
                            isaac_float4
                        >(
                            0,
                            getHSVA(
                                isaac_float( 2 * i ) * M_PI / isaac_float(
                                    (
                                        boost::mpl::size< TSourceList >::type::value
                                        + boost::mpl::size< TParticleList >::type::value
                                    )
                                ),
                                1,
                                1,
                                0
                            )
                        )
                    );
                    transfer_h.description[i].insert(
                        std::pair<
                            isaac_uint,
                            isaac_float4
                        >(
                            TTransfer_size,
                            getHSVA(
                                isaac_float( 2 * i ) * M_PI / isaac_float(
                                    (
                                        boost::mpl::size< TSourceList >::type::value
                                        + boost::mpl::size< TParticleList >::type::value
                                    )
                                ),
                                1,
                                1,
                                1
                            )
                        )
                    );
                }
                    //Init particle transfer func with constant alpha = 1
                else
                {
                    transfer_h.description[i].insert(
                        std::pair<
                            isaac_uint,
                            isaac_float4
                        >(
                            0,
                            getHSVA(
                                isaac_float( 2 * i ) * M_PI / isaac_float(
                                    (
                                        boost::mpl::size< TSourceList >::type::value
                                        + boost::mpl::size< TParticleList >::type::value
                                    )
                                ),
                                1,
                                1,
                                1
                            )
                        )
                    );
                    transfer_h.description[i].insert(
                        std::pair<
                            isaac_uint,
                            isaac_float4
                        >(
                            TTransfer_size,
                            getHSVA(
                                isaac_float( 2 * i ) * M_PI / isaac_float(
                                    (
                                        boost::mpl::size< TSourceList >::type::value
                                        + boost::mpl::size< TParticleList >::type::value
                                    )
                                ),
                                1,
                                1,
                                1
                            )
                        )
                    );
                }

            }
            updateTransfer( );

            max_size = ISAAC_MAX(
                uint32_t( global_size[0] ),
                uint32_t( global_size[1] )
            );
            if( TSimDim::value > 2 )
            {
                max_size = ISAAC_MAX(
                    uint32_t( global_size[2] ),
                    uint32_t( max_size )
                );
            }
            max_size_scaled = ISAAC_MAX(
                uint32_t( global_size_scaled[0] ),
                uint32_t( global_size_scaled[1] )
            );
            if( TSimDim::value > 2 )
            {
                max_size_scaled = ISAAC_MAX(
                    uint32_t( global_size_scaled[2] ),
                    uint32_t( max_size_scaled )
                );
            }

            //ICET:
            IceTCommunicator icetComm;
            icetComm = icetCreateMPICommunicator( mpi_world );
            for( int pass = 0; pass < TController::pass_count; pass++ )
            {
                icetContext[pass] = icetCreateContext( icetComm );
                icetResetTiles( );
                icetAddTile(
                    0,
                    0,
                    framebuffer_size.x,
                    framebuffer_size.y,
                    master
                );
                //icetStrategy(ICET_STRATEGY_DIRECT);
                icetStrategy( ICET_STRATEGY_SEQUENTIAL );
                //icetStrategy(ICET_STRATEGY_REDUCE);

                //icetSingleImageStrategy( ICET_SINGLE_IMAGE_STRATEGY_AUTOMATIC );
                icetSingleImageStrategy( ICET_SINGLE_IMAGE_STRATEGY_BSWAP );
                //icetSingleImageStrategy( ICET_SINGLE_IMAGE_STRATEGY_RADIXK );
                //icetSingleImageStrategy( ICET_SINGLE_IMAGE_STRATEGY_TREE );

                /*IceTBoolean supports;
                icetGetBooleanv( ICET_STRATEGY_SUPPORTS_ORDERING, &supports );
                if (supports)
                    printf("yes\n");
                else
                    printf("no\n");*/

                icetSetColorFormat( ICET_IMAGE_COLOR_RGBA_UBYTE );
                icetSetDepthFormat( ICET_IMAGE_DEPTH_NONE );
                icetCompositeMode( ICET_COMPOSITE_MODE_BLEND );
                icetEnable( ICET_ORDERED_COMPOSITE );
                icetPhysicalRenderSize(
                    framebuffer_size.x,
                    framebuffer_size.y
                );
                icetDrawCallback( drawCallBack );
            }
            icetDestroyMPICommunicator( icetComm );
            updateBounding( );

            //JSON
            if( rank == master )
            {
                json_object_set_new(
                    json_root,
                    "type",
                    json_string( "register" )
                );
                json_object_set_new(
                    json_root,
                    "name",
                    json_string( name.c_str( ) )
                );
                json_object_set_new(
                    json_root,
                    "nodes",
                    json_integer( numProc )
                );
                json_object_set_new(
                    json_root,
                    "framebuffer width",
                    json_integer( compbuffer_size.x )
                );
                json_object_set_new(
                    json_root,
                    "framebuffer height",
                    json_integer( compbuffer_size.y )
                );

                json_object_set_new(
                    json_root,
                    "max functors",
                    json_integer( ISAAC_MAX_FUNCTORS )
                );
                json_t * json_functors_array = json_array( );
                json_object_set_new(
                    json_root,
                    "functors",
                    json_functors_array
                );
                IsaacFunctorPool functors;
                isaac_for_each_params(
                    functors,
                    functor_2_json_iterator( ),
                    json_functors_array
                );

                json_t * matrix;
                json_object_set_new(
                    json_root,
                    "projection count",
                    json_integer( TController::pass_count )
                );
                json_object_set_new(
                    json_root,
                    "projection",
                    matrix = json_array( )
                );
                ISAAC_JSON_ADD_MATRIX ( matrix,
                    projection,
                    16 * TController::pass_count )
                json_object_set_new(
                    json_root,
                    "position",
                    matrix = json_array( )
                );
                ISAAC_JSON_ADD_MATRIX ( matrix,
                    look_at,
                    3 )
                json_object_set_new(
                    json_root,
                    "rotation",
                    matrix = json_array( )
                );
                ISAAC_JSON_ADD_MATRIX ( matrix,
                    rotation,
                    9 )
                json_object_set_new(
                    json_root,
                    "distance",
                    json_real( distance )
                );

                json_t * json_sources_array = json_array( );
                json_object_set_new(
                    json_root,
                    "sources",
                    json_sources_array
                );

                isaac_for_each_params(
                    sources,
                    source_2_json_iterator( ),
                    json_sources_array
                );
                isaac_for_each_params(
                    particle_sources,
                    source_2_json_iterator( ),
                    json_sources_array
                );

                json_object_set_new(
                    json_root,
                    "interpolation",
                    json_boolean( interpolation )
                );
                json_object_set_new(
                    json_root,
                    "iso surface",
                    json_boolean( iso_surface )
                );
                json_object_set_new(
                    json_root,
                    "step",
                    json_real( step )
                );

                json_object_set_new(
                    json_root,
                    "dimension",
                    json_integer( TSimDim::value )
                );
                json_object_set_new(
                    json_root,
                    "width",
                    json_integer( global_size_scaled[0] )
                );
                if( TSimDim::value > 1 )
                {
                    json_object_set_new(
                        json_root,
                        "height",
                        json_integer( global_size_scaled[1] )
                    );
                }
                if( TSimDim::value > 2 )
                {
                    json_object_set_new(
                        json_root,
                        "depth",
                        json_integer( global_size_scaled[2] )
                    );
                }
                json_t * json_version_array = json_array( );
                json_array_append_new(
                    json_version_array,
                    json_integer( ISAAC_PROTOCOL_VERSION_MAJOR )
                );
                json_array_append_new(
                    json_version_array,
                    json_integer( ISAAC_PROTOCOL_VERSION_MINOR )
                );
                json_object_set_new(
                    json_root,
                    "protocol",
                    json_version_array
                );

                //send inital ambientOcclusion settings
                json_object_set_new(
                    json_root, 
                    "ao isEnabled", 
                    json_boolean(ambientOcclusion.isEnabled)
                );
                json_object_set_new(
                    json_root, 
                    "ao weight", 
                    json_real(ambientOcclusion.weight)
                ); 
            }

#if ISAAC_ALPAKA == 1
            //allocate ssao kernel (16x16 matrix)
            alpaka::Buf<
                THost, 
                isaac_float3, 
                TFraDim, 
                ISAAC_IDX_TYPE
            > ssao_kernel_h_buf (
                alpaka::allocBuf<
                    isaac_float3, 
                    ISAAC_IDX_TYPE
                > (
                    host, 
                    ISAAC_IDX_TYPE(64)
                )
            );
            isaac_float3* ssao_kernel_h = reinterpret_cast<isaac_float3*> ( alpaka::getPtrNative ( ssao_kernel_h_buf ) );

            //allocate ssao noise kernel (4x4 matrix)
            alpaka::Buf<
                THost, 
                isaac_float3, 
                TFraDim, 
                ISAAC_IDX_TYPE
            > ssao_noise_h_buf (
                alpaka::allocBuf<
                    isaac_float3, 
                    ISAAC_IDX_TYPE
                > (
                    host, 
                    ISAAC_IDX_TYPE(16)
                )
            );
            isaac_float3* ssao_noise_h = reinterpret_cast<isaac_float3*> ( alpaka::getPtrNative ( ssao_noise_h_buf ) );
    
#else

            isaac_float3 ssao_kernel_h[64];
            isaac_float3 ssao_noise_h[16];
    
#endif
            
            std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
            std::default_random_engine generator;

            //set ssao_kernel values
            for (unsigned int i = 0; i < 64; i++ ) {
                isaac_float3 sample({
                    randomFloats(generator) * 2.0f - 1.0f,
                    randomFloats(generator) * 2.0f - 1.0f,
                    randomFloats(generator)
                });
                isaac_float sample_length = sqrt(sample.x * sample.x + sample.y * sample.y + sample.z * sample.z);
                sample = sample / sample_length;
                sample = sample * randomFloats(generator);
                isaac_float scale = (isaac_float)i / 64.0;
                //lerp
                scale = 0.1f + (scale * scale) * (1.0f - 0.1f);
                ssao_kernel_h[i] = sample;
            }

            //set ssao_noise values
            for(unsigned int i = 0; i < 16; i++) {
                isaac_float3 noise({
                    randomFloats(generator) * 2.0f - 1.0f,
                    randomFloats(generator) * 2.0f - 1.0f,
                    0.0f
                });
                ssao_noise_h[i] = noise;
            }

            //move ssao kernel to device
#if ISAAC_ALPAKA == 1
            //copy ssao kernel to constant memory
            alpaka::Vec<alpaka::DimInt<1u>, ISAAC_IDX_TYPE> const 
                ssao_kernel_d_extent(ISAAC_IDX_TYPE(64));

            auto ssao_kernel_d_view (
                alpaka::createStaticDevMemView(
                    &ssao_kernel_d[0u], 
                    acc, 
                    ssao_kernel_d_extent
                )
            );
            alpaka::memcpy(
                stream, 
                ssao_kernel_d_view, 
                ssao_kernel_h_buf, 
                ISAAC_IDX_TYPE(64)
            );

            //copy ssao noise kernel to constant memory
            alpaka::Vec<alpaka::DimInt<1u>, ISAAC_IDX_TYPE> const 
                ssao_noise_d_extent(ISAAC_IDX_TYPE(16));

            auto ssao_noise_d_view ( 
                alpaka::createStaticDevMemView(
                    &ssao_noise_d[0u], 
                    acc, 
                    ssao_noise_d_extent
                )
            );
            alpaka::memcpy ( 
                stream, 
                ssao_noise_d_view, 
                ssao_noise_h_buf, 
                ISAAC_IDX_TYPE(16)
            );
    
#else

            ISAAC_CUDA_CHECK ( cudaMemcpyToSymbol ( ssao_kernel_d, ssao_kernel_h, 64 * sizeof ( isaac_float3 ) ) );
            ISAAC_CUDA_CHECK ( cudaMemcpyToSymbol ( ssao_noise_d, ssao_noise_h, 16 * sizeof ( isaac_float3 ) ) );

#endif

        }


        void setJpegQuality( isaac_uint jpeg_quality )
        {
            ISAAC_WAIT_VISUALIZATION
            if( communicator )
            {
                communicator->setJpegQuality( jpeg_quality );
            }
        }


        bool editClipping(
            isaac_uint nr,
            isaac_float px,
            isaac_float py,
            isaac_float pz,
            isaac_float nx,
            isaac_float ny,
            isaac_float nz
        )
        {
            ISAAC_WAIT_VISUALIZATION
            if( nr >= ISAAC_MAX_CLIPPING )
            {
                return false;
            }
            isaac_float nx_s = nx * scale[0];
            isaac_float ny_s = ny * scale[1];
            isaac_float nz_s = nz * scale[2];
            isaac_float l = sqrt( nx_s * nx_s + ny_s * ny_s + nz_s * nz_s );
            if( l == 0.0f )
            {
                return false;
            }
            nx_s /= l;
            ny_s /= l;
            nz_s /= l;
            clipping.elem[nr].position
                .x = px;
            clipping.elem[nr].position
                .y = py;
            clipping.elem[nr].position
                .z = pz;
            clipping.elem[nr].normal
                .x = nx_s;
            clipping.elem[nr].normal
                .y = ny_s;
            clipping.elem[nr].normal
                .z = nz_s;
            clipping_saved_normals[nr].x = nx;
            clipping_saved_normals[nr].y = ny;
            clipping_saved_normals[nr].z = nz;
            return true;
        }


        void addClipping(
            isaac_float px,
            isaac_float py,
            isaac_float pz,
            isaac_float nx,
            isaac_float ny,
            isaac_float nz
        )
        {
            if( editClipping(
                clipping.count,
                px,
                py,
                pz,
                nx,
                ny,
                nz
            ) )
            {
                clipping.count++;
            }
        }


        void removeClipping( isaac_uint nr )
        {
            ISAAC_WAIT_VISUALIZATION
            if( nr >= clipping.count )
            {
                return;
            }
            clipping.count--;
            for( isaac_uint i = nr; i < clipping.count; i++ )
            {
                clipping.elem[i] = clipping.elem[i + 1];
                clipping_saved_normals[i] = clipping_saved_normals[i + 1];
            }
        }


        void updateBounding( )
        {
            ISAAC_WAIT_VISUALIZATION
            for( int pass = 0; pass < TController::pass_count; pass++ )
            {
                icetSetContext( icetContext[pass] );
                if( icet_bounding_box )
                {
                    isaac_float f_l_width = ( isaac_float ) local_size_scaled[0]
                                            / ( isaac_float ) max_size_scaled
                                            * 2.0f;
                    isaac_float f_l_height =
                        ( isaac_float ) local_size_scaled[1]
                        / ( isaac_float ) max_size_scaled * 2.0f;
                    isaac_float f_l_depth = 0.0f;
                    if( TSimDim::value > 2 )
                    {
                        f_l_depth = ( isaac_float ) local_size_scaled[2]
                                    / ( isaac_float ) max_size_scaled * 2.0f;
                    }
                    isaac_float f_x = ( isaac_float ) position_scaled[0]
                                      / ( isaac_float ) max_size_scaled * 2.0f
                                      - ( isaac_float ) global_size_scaled[0]
                                        / ( isaac_float ) max_size_scaled;
                    isaac_float f_y = ( isaac_float ) position_scaled[1]
                                      / ( isaac_float ) max_size_scaled * 2.0f
                                      - ( isaac_float ) global_size_scaled[1]
                                        / ( isaac_float ) max_size_scaled;
                    isaac_float f_z = 0.0f;
                    if( TSimDim::value > 2 )
                    {
                        f_z = ( isaac_float ) position_scaled[2]
                              / ( isaac_float ) max_size_scaled
                              * isaac_float( 2 )
                              - ( isaac_float ) global_size_scaled[2]
                                / ( isaac_float ) max_size_scaled;
                    }
                    icetBoundingBoxf(
                        f_x,
                        f_x + f_l_width,
                        f_y,
                        f_y + f_l_height,
                        f_z,
                        f_z + f_l_depth
                    );
                }
                else
                {
                    icetBoundingVertices(
                        0,
                        0,
                        0,
                        0,
                        NULL
                    );
                }
            }
        }


        void updatePosition( const TDomainSize position )
        {
            ISAAC_WAIT_VISUALIZATION
            this->position = position;
            for( int i = 0; i < 3; i++ )
            {
                position_scaled[i] = isaac_int(
                    ( isaac_float ) position[i] * ( isaac_float ) scale[i]
                );
            }
        }


        void updateLocalSize( const TDomainSize local_size )
        {
            ISAAC_WAIT_VISUALIZATION
            this->local_size = local_size;
            for( int i = 0; i < 3; i++ )
            {
                local_size_scaled[i] = isaac_int(
                    ( isaac_float ) local_size[i] * ( isaac_float ) scale[i]
                );
            }
        }


        void updateLocalParticleSize( const TDomainSize local_particle_size )
        {
            ISAAC_WAIT_VISUALIZATION
            this->local_particle_size = local_particle_size;
        }


        void updateFunctions( )
        {
            ISAAC_WAIT_VISUALIZATION
            IsaacFunctorPool functors;
            isaac_float4 isaac_parameter_h[(
                                               boost::mpl::size< TSourceList >::type::value
                                               + boost::mpl::size< TParticleList >::type::value
                                           ) * ISAAC_MAX_FUNCTORS];
            for( int i = 0; i < (
                boost::mpl::size< TSourceList >::type::value
                + boost::mpl::size< TParticleList >::type::value
            ); i++ )
            {
                functions[i].error_code = 0;
                //Going from | to |...
                std::string source = functions[i].source;
                size_t pos = 0;
                bool again = true;
                int elem = 0;
                while( again && (
                    ( pos = source.find( "|" ) ) != std::string::npos
                    || ( ( again = false ) == false )
                ) )
                {
                    if( elem >= ISAAC_MAX_FUNCTORS )
                    {
                        functions[i].error_code = 1;
                        break;
                    }
                    std::string token = source.substr(
                        0,
                        pos
                    );
                    source.erase(
                        0,
                        pos + 1
                    );
                    //ignore " " in token
                    token.erase(
                        remove_if(
                            token.begin( ),
                            token.end( ),
                            isspace
                        ),
                        token.end( )
                    );
                    //search "(" and parse parameters
                    size_t t_begin = token.find( "(" );
                    if( t_begin == std::string::npos )
                    {
                        memset(
                            &(
                                isaac_parameter_h[i * ISAAC_MAX_FUNCTORS + elem]
                            ),
                            0,
                            sizeof( isaac_float4 )
                        );
                    }
                    else
                    {
                        size_t t_end = token.find( ")" );
                        if( t_end == std::string::npos )
                        {
                            functions[i].error_code = -1;
                            break;
                        }
                        if( t_end - t_begin == 1 )
                        { //()
                            memset(
                                &(
                                    isaac_parameter_h[i * ISAAC_MAX_FUNCTORS
                                                      + elem]
                                ),
                                0,
                                sizeof( isaac_float4 )
                            );
                        }
                        else
                        {
                            std::string parameters = token.substr(
                                t_begin + 1,
                                t_end - t_begin - 1
                            );
                            size_t p_pos = 0;
                            bool p_again = true;
                            int p_elem = 0;
                            isaac_float * parameter_array = ( float * ) &(
                                isaac_parameter_h[i * ISAAC_MAX_FUNCTORS + elem]
                            );
                            while( p_again && (
                                ( p_pos = parameters.find( "," ) )
                                != std::string::npos
                                || ( ( p_again = false ) == false )
                            ) )
                            {
                                if( p_elem >= 4 )
                                {
                                    functions[i].error_code = 2;
                                    break;
                                }
                                std::string par = parameters.substr(
                                    0,
                                    p_pos
                                );
                                parameters.erase(
                                    0,
                                    p_pos + 1
                                );
                                try
                                {
                                    parameter_array[p_elem] = std::stof( par );
                                } catch( const std::invalid_argument & ia )
                                {
                                    std::cerr << "Invalid argument: "
                                              << ia.what( ) << '\n';
                                    functions[i].error_code = -2;
                                    p_elem++;
                                    break;
                                } catch( const std::out_of_range & oor )
                                {
                                    std::cerr << "Out of range: " << oor.what( )
                                              << '\n';
                                    functions[i].error_code = -2;
                                    p_elem++;
                                    break;
                                }

                                p_elem++;
                            }
                            for( ; p_elem < 4; p_elem++ )
                            {
                                parameter_array[p_elem] = parameter_array[p_elem
                                                                          - 1];    //last one repeated
                            }
                        }
                    }
                    //parse token itself
                    if( t_begin != std::string::npos )
                    {
                        token = token.substr(
                            0,
                            t_begin
                        );
                    }
                    bool found = false;
                    isaac_for_each_params(
                        functors,
                        parse_functor_iterator( ),
                        token,
                        functions[i].bytecode[elem],
                        found
                    );
                    if( !found )
                    {
                        functions[i].error_code = -2;
                        break;
                    }

                    elem++;
                }
                for( ; elem < ISAAC_MAX_FUNCTORS; elem++ )
                {
                    functions[i].bytecode[elem] = 0; //last one idem
                    memset(
                        &(
                            isaac_parameter_h[i * ISAAC_MAX_FUNCTORS + elem]
                        ),
                        0,
                        sizeof( isaac_float4 )
                    );
                }
            }

            //Calculate functor chain nr per source
            dest_array_struct<
                (
                    boost::mpl::size< TSourceList >::type::value
                    + boost::mpl::size< TParticleList >::type::value
                )
            > dest;
            int zero = 0;
            isaac_for_each_params(
                sources,
                update_functor_chain_iterator( ),
                functions,
                zero,
                dest
            );
            isaac_for_each_params(
                particle_sources,
                update_functor_chain_iterator( ),
                functions,
                boost::mpl::size< TSourceList >::type::value,
                dest
            );
#if ISAAC_ALPAKA == 1
            alpaka::ViewPlainPtr <THost, isaac_float4, TFraDim, ISAAC_IDX_TYPE>
                parameter_buffer(
                isaac_parameter_h,
                host,
                alpaka::Vec<
                    TFraDim,
                    ISAAC_IDX_TYPE
                >(
                    ISAAC_IDX_TYPE(
                        ISAAC_MAX_FUNCTORS * (
                            boost::mpl::size< TSourceList >::type::value
                            + boost::mpl::size< TParticleList >::type::value
                        )
                    )
                )
            );

            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                parameter_d_extent( ISAAC_IDX_TYPE( 16 ) );
            auto parameter_d_view(
                alpaka::createStaticDevMemView ( & isaac_parameter_d[0u],
                acc,
                parameter_d_extent
            ) );
            alpaka::memcpy(
                stream,
                parameter_d_view,
                parameter_buffer,
                alpaka::Vec<
                    TFraDim,
                    ISAAC_IDX_TYPE
                >(
                    ISAAC_IDX_TYPE(
                        ISAAC_MAX_FUNCTORS * (
                            boost::mpl::size< TSourceList >::type::value
                            + boost::mpl::size< TParticleList >::type::value
                        )
                    )
                )
            );

            const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> threads(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> blocks(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            const alpaka::Vec <TAccDim, ISAAC_IDX_TYPE> grid(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
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
            updateFunctorChainPointerKernel<
                (
                    boost::mpl::size< TSourceList >::type::value
                    + boost::mpl::size< TParticleList >::type::value
                ),
                dest_array_struct<
                    (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    )
                >
            > kernel;
            auto const instance(
                alpaka::createTaskKernel< TAcc >(
                    workdiv,
                    kernel,
                    alpaka::getPtrNative( functor_chain_choose_d ),
                    alpaka::getPtrNative( functor_chain_d ),
                    dest
                )
            );
            alpaka::enqueue(
                stream,
                instance
            );
            alpaka::wait( stream );

            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                function_chain_d_extent( ISAAC_IDX_TYPE( ISAAC_MAX_SOURCES ) );
            auto function_chain_d_view(
                alpaka::createStaticDevMemView ( & isaac_function_chain_d[0u],
                acc,
                function_chain_d_extent
            ) );
            alpaka::memcpy(
                stream,
                function_chain_d_view,
                functor_chain_choose_d,
                ISAAC_IDX_TYPE(
                    (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    )
                )
            );
#else
            ISAAC_CUDA_CHECK ( cudaMemcpyToSymbol(
                isaac_parameter_d,
                isaac_parameter_h,
                sizeof( isaac_parameter_h )
            ) );
            dim3 grid( 1 );
            dim3 block( 1 );
            updateFunctorChainPointerKernel<
                (
                    boost::mpl::size< TSourceList >::type::value
                    + boost::mpl::size< TParticleList >::type::value
                )
            > <<< grid, block >>> (
                functor_chain_choose_d,
                functor_chain_d,
                dest
            );
            ISAAC_CUDA_CHECK ( cudaDeviceSynchronize( ) );
            isaac_functor_chain_pointer_N * constant_ptr;
            ISAAC_CUDA_CHECK ( cudaGetSymbolAddress(
                ( void ** ) &constant_ptr,
                isaac_function_chain_d
            ) );
            ISAAC_CUDA_CHECK ( cudaMemcpy(
                constant_ptr,
                functor_chain_choose_d,
                (
                    boost::mpl::size< TSourceList >::type::value
                    + boost::mpl::size< TParticleList >::type::value
                ) * sizeof( isaac_functor_chain_pointer_N ),
                cudaMemcpyDeviceToDevice
            ) );
#endif
        }


        void updateTransfer( )
        {
            ISAAC_WAIT_VISUALIZATION
            for( int i = 0; i < (
                boost::mpl::size< TSourceList >::type::value
                + boost::mpl::size< TParticleList >::type::value
            ); i++ )
            {
                auto next = transfer_h.description[i].begin( );
                auto before = next;
                for( next++; next != transfer_h.description[i].end( ); next++ )
                {
                    isaac_uint width = next->first - before->first;
                    for( ISAAC_IDX_TYPE j = 0;
                        j < ISAAC_IDX_TYPE( width )
                        && ISAAC_IDX_TYPE( j + before->first )
                           < ISAAC_IDX_TYPE( TTransfer_size );
                        j++ )
                    {
                        transfer_h.pointer[i][before->first + j] = (
                                                                       before->second
                                                                       * isaac_float(
                                                                           width
                                                                           - j
                                                                       ) +
                                                                       next->second
                                                                       * isaac_float( j )
                                                                   )
                                                                   / isaac_float( width );
                    }
                    before = next;
                }
#if ISAAC_ALPAKA == 1
                alpaka::memcpy(
                    stream,
                    transfer_d_buf[i],
                    transfer_h_buf[i],
                    TTransfer_size
                );
#else
                ISAAC_CUDA_CHECK ( cudaMemcpy(
                    transfer_d.pointer[i],
                    transfer_h.pointer[i],
                    sizeof( isaac_float4 ) * TTransfer_size,
                    cudaMemcpyHostToDevice
                ) );
#endif
            }
        }


        json_t * getJsonMetaRoot( )
        {
            ISAAC_WAIT_VISUALIZATION
            return json_meta_root;
        }


        int init( CommunicatorSetting communicatorBehaviour = ReturnAtError )
        {
            int failed = 0;
            if( communicator && (
                communicator->serverConnect( communicatorBehaviour ) < 0
            ) )
            {
                failed = 1;
            }
            MPI_Bcast(
                &failed,
                1,
                MPI_INT,
                master,
                mpi_world
            );
            if( failed )
            {
                return -1;
            }
            if( rank == master )
            {
                json_init_root = json_root;
                communicator->serverSendRegister( &json_init_root );
                recreateJSON( );
            }
            return 0;
        }


        json_t * doVisualization(
            const IsaacVisualizationMetaEnum metaTargets = META_MASTER,
            void * pointer = NULL,
            bool redraw = true
        )
        {
            if( redraw )
            {
                ISAAC_START_TIME_MEASUREMENT ( buffer,
                    getTicksUs( ) )
                isaac_for_each_params(
                    sources,
                    update_pointer_array_iterator( ),
                    pointer_array,
                    local_size,
                    source_weight,
                    pointer
#if ISAAC_ALPAKA == 1
                ,
                stream
#endif
                );
                isaac_for_each_params(
                    particle_sources,
                    update_particle_source_iterator( ),
                    source_weight,
                    boost::mpl::size< TSourceList >::type::value,
                    pointer
                );
                ISAAC_STOP_TIME_MEASUREMENT ( buffer_time,
                    +=,
                    buffer,
                    getTicksUs( ) )

            }
            ISAAC_WAIT_VISUALIZATION

            myself = this;

            send_distance = false;
            send_look_at = false;
            send_projection = false;
            send_rotation = false;
            send_transfer = false;
            send_interpolation = false;
            send_step = false;
            send_iso_surface = false;
            send_functions = false;
            send_weight = false;
            send_minmax = false;
            send_background_color = false;
            send_clipping = false;
            send_controller = false;
            send_init_json = false;
            send_ao = false;

            //Handle messages
            json_t * message;
            char message_buffer[ISAAC_MAX_RECEIVE] = "{}";
            //Master merges all messages and broadcasts it.

            if( rank == master )
            {
                message = json_object( );
                bool add_modelview = false;
                while( json_t * last = communicator->getLastMessage( ) )
                {
                    json_t * js;
                    size_t index;
                    json_t * value;
                    //search for update requests
                    if( js = json_object_get(
                        last,
                        "request"
                    ) )
                    {
                        const char * target = json_string_value( js );
                        if( strcmp(
                            target,
                            "rotation"
                        ) == 0 )
                        {
                            send_rotation = true;
                        }
                        if( strcmp(
                            target,
                            "position"
                        ) == 0 )
                        {
                            send_look_at = true;
                        }
                        if( strcmp(
                            target,
                            "distance"
                        ) == 0 )
                        {
                            send_distance = true;
                        }
                        if( strcmp(
                            target,
                            "projection"
                        ) == 0 )
                        {
                            send_projection = true;
                        }
                        if( strcmp(
                            target,
                            "transfer"
                        ) == 0 )
                        {
                            send_transfer = true;
                        }
                        if( strcmp(
                            target,
                            "interpolation"
                        ) == 0 )
                        {
                            send_interpolation = true;
                        }
                        if( strcmp(
                            target,
                            "step"
                        ) == 0 )
                        {
                            send_step = true;
                        }
                        if( strcmp(
                            target,
                            "iso surface"
                        ) == 0 )
                        {
                            send_iso_surface = true;
                        }
                        if( strcmp(
                            target,
                            "functions"
                        ) == 0 )
                        {
                            send_functions = true;
                        }
                        if( strcmp(
                            target,
                            "weight"
                        ) == 0 )
                        {
                            send_weight = true;
                        }
                        if( strcmp(
                            target,
                            "background color"
                        ) == 0 )
                        {
                            send_background_color = true;
                        }
                        if( strcmp(
                            target,
                            "clipping"
                        ) == 0 )
                        {
                            send_clipping = true;
                        }
                        if( strcmp(
                            target,
                            "controller"
                        ) == 0 )
                        {
                            send_controller = true;
                        }
                        if( strcmp(
                            target,
                            "init"
                        ) == 0 )
                        {
                            send_init_json = true;
                        }
                        if(strcmp(target, "ao") == 0) {
                            send_ao = true;
                        }
                    }
                    //Search for scene changes
                    if( json_array_size(
                        js = json_object_get(
                            last,
                            "rotation absolute"
                        )
                    ) == 9 )
                    {
                        add_modelview = true;
                        send_rotation = true;
                        json_array_foreach(
                            js,
                            index,
                            value
                        )
                        rotation[index] = json_number_value( value );
                        json_object_del(
                            last,
                            "rotation absolute"
                        );
                    }
                    if( json_array_size(
                        js = json_object_get(
                            last,
                            "rotation relative"
                        )
                    ) == 9 )
                    {
                        add_modelview = true;
                        send_rotation = true;
                        IceTDouble relative[9];
                        IceTDouble new_rotation[9];
                        json_array_foreach(
                            js,
                            index,
                            value
                        )
                        relative[index] = json_number_value( value );
                        for( isaac_int x = 0; x < 3; x++ )
                        {
                            for( isaac_int y = 0; y < 3; y++ )
                            {
                                new_rotation[y + x * 3] =
                                    relative[y + 0 * 3] * rotation[0 + x * 3]
                                    + relative[y + 1 * 3] * rotation[1 + x * 3]
                                    + relative[y + 2 * 3] * rotation[2 + x * 3];
                            }
                        }
                        memcpy(
                            rotation,
                            new_rotation,
                            9 * sizeof( IceTDouble )
                        );
                        json_object_del(
                            last,
                            "rotation relative"
                        );
                    }
                    if( json_array_size(
                        js = json_object_get(
                            last,
                            "rotation axis"
                        )
                    ) == 4 )
                    {
                        IceTDouble relative[9];
                        IceTDouble x = json_number_value(
                            json_array_get(
                                js,
                                0
                            )
                        );
                        IceTDouble y = json_number_value(
                            json_array_get(
                                js,
                                1
                            )
                        );
                        IceTDouble z = json_number_value(
                            json_array_get(
                                js,
                                2
                            )
                        );
                        IceTDouble rad = json_number_value(
                            json_array_get(
                                js,
                                3
                            )
                        );
                        IceTDouble s = sin( rad * M_PI / 180.0 );
                        IceTDouble c = cos( rad * M_PI / 180.0 );
                        IceTDouble l = sqrt( x * x + y * y + z * z );
                        if( l != 0.0 )
                        {
                            add_modelview = true;
                            send_rotation = true;
                            x /= l;
                            y /= l;
                            z /= l;
                            relative[0] = c + x * x * ( 1 - c );
                            relative[3] = x * y * ( 1 - c ) - z * s;
                            relative[6] = x * z * ( 1 - c ) + y * s;
                            relative[1] = y * x * ( 1 - c ) + z * s;
                            relative[4] = c + y * y * ( 1 - c );
                            relative[7] = y * z * ( 1 - c ) - x * s;
                            relative[2] = z * x * ( 1 - c ) - y * s;
                            relative[5] = z * y * ( 1 - c ) + x * s;
                            relative[8] = c + z * z * ( 1 - c );
                            IceTDouble new_rotation[9];
                            for( isaac_int x = 0; x < 3; x++ )
                            {
                                for( isaac_int y = 0; y < 3; y++ )
                                {
                                    new_rotation[y + x * 3] =
                                        relative[y + 0 * 3]
                                        * rotation[0 + x * 3]
                                        + relative[y + 1 * 3]
                                          * rotation[1 + x * 3]
                                        + relative[y + 2 * 3]
                                          * rotation[2 + x * 3];
                                }
                            }
                            memcpy(
                                rotation,
                                new_rotation,
                                9 * sizeof( IceTDouble )
                            );
                        }
                        json_object_del(
                            last,
                            "rotation axis"
                        );
                    }
                    if( json_array_size(
                        js = json_object_get(
                            last,
                            "position absolute"
                        )
                    ) == 3 )
                    {
                        add_modelview = true;
                        send_look_at = true;
                        json_array_foreach(
                            js,
                            index,
                            value
                        )
                        look_at[index] = json_number_value( value );
                        json_object_del(
                            last,
                            "position absolute"
                        );
                    }
                    if( json_array_size(
                        js = json_object_get(
                            last,
                            "position relative"
                        )
                    ) == 3 )
                    {
                        add_modelview = true;
                        send_look_at = true;
                        IceTDouble add[3];
                        json_array_foreach(
                            js,
                            index,
                            value
                        )
                        add[index] = json_number_value( value );
                        IceTDouble add_p[3] = {
                            rotation[0] * add[0] + rotation[1] * add[1]
                            + rotation[2] * add[2],
                            rotation[3] * add[0] + rotation[4] * add[1]
                            + rotation[5] * add[2],
                            rotation[6] * add[0] + rotation[7] * add[1]
                            + rotation[8] * add[2]
                        };
                        look_at[0] += add_p[0];
                        look_at[1] += add_p[1];
                        look_at[2] += add_p[2];
                        json_object_del(
                            last,
                            "position relative"
                        );
                    }
                    if( js = json_object_get(
                        last,
                        "distance absolute"
                    ) )
                    {
                        add_modelview = true;
                        send_distance = true;
                        distance = json_number_value( js );
                        json_object_del(
                            last,
                            "distance absolute"
                        );
                    }
                    if( js = json_object_get(
                        last,
                        "distance relative"
                    ) )
                    {
                        add_modelview = true;
                        send_distance = true;
                        distance += json_number_value( js );
                        json_object_del(
                            last,
                            "distance relative"
                        );
                    }
                    //Giving the Controller the chance to grep for controller specific messages:
                    if( controller.updateProjection(
                        projection,
                        framebuffer_size,
                        last
                    ) )
                    {
                        redraw = true;
                        send_projection = true;
                        json_t * matrix;
                        json_object_set_new(
                            message,
                            "projection",
                            matrix = json_array( )
                        );
                        ISAAC_JSON_ADD_MATRIX ( matrix,
                            projection,
                            16 * TController::pass_count )
                    }
                    mergeJSON(
                        message,
                        last
                    );
                    json_decref( last );
                }
                if( add_modelview )
                {
                    redraw = true;
                    updateModelview( );
                    json_t * matrix;
                    json_object_set_new(
                        message,
                        "modelview",
                        matrix = json_array( )
                    );
                    ISAAC_JSON_ADD_MATRIX ( matrix,
                        modelview,
                        16 )
                }
                char * buffer = json_dumps(
                    message,
                    0
                );
                strncpy(
                    message_buffer,
                    buffer,
                    ISAAC_MAX_RECEIVE - 1
                );
                message_buffer[ISAAC_MAX_RECEIVE - 1] = 0;
                free( buffer );
                int l = strlen( message_buffer );
                MPI_Bcast(
                    &l,
                    1,
                    MPI_INT,
                    master,
                    mpi_world
                );
                MPI_Bcast(
                    message_buffer,
                    l,
                    MPI_CHAR,
                    master,
                    mpi_world
                );
            }
            else
            { //The others just get the message
                int l;
                MPI_Bcast(
                    &l,
                    1,
                    MPI_INT,
                    master,
                    mpi_world
                );
                MPI_Bcast(
                    message_buffer,
                    l,
                    MPI_CHAR,
                    master,
                    mpi_world
                );
                message_buffer[l] = 0;
                message = json_loads(
                    message_buffer,
                    0,
                    NULL
                );
            }

            json_t * js;
            size_t index;
            json_t * value;

            //search for requests for all ranks
            if( js = json_object_get(
                message,
                "request"
            ) )
            {
                const char * target = json_string_value( js );
                if( strcmp(
                    target,
                    "redraw"
                ) == 0 )
                {
                    redraw = true;
                }
                if( strcmp(
                    target,
                    "minmax"
                ) == 0 )
                {
                    send_minmax = true;
                }
            }

            //Scene set?
            if( json_array_size(
                js = json_object_get(
                    message,
                    "projection"
                )
            ) == 16 * TController::pass_count )
            {
                redraw = true;
                send_projection = true;
                json_array_foreach(
                    js,
                    index,
                    value
                )
                projection[index] = json_number_value( value );
            }
            if( rank != master && json_array_size(
                js = json_object_get(
                    message,
                    "modelview"
                )
            ) == 16 )
            {
                redraw = true;
                json_array_foreach(
                    js,
                    index,
                    value
                )
                modelview[index] = json_number_value( value );
            }
            if( json_array_size(
                js = json_object_get(
                    message,
                    "transfer points"
                )
            ) )
            {
                redraw = true;
                json_array_foreach(
                    js,
                    index,
                    value
                )
                {
                    transfer_h.description[index].clear( );
                    size_t index_2;
                    json_t * element;
                    json_array_foreach(
                        value,
                        index_2,
                        element
                    )
                    {
                        transfer_h.description[index].insert(
                            std::pair<
                                isaac_uint,
                                isaac_float4
                            >(
                                isaac_uint(
                                    json_number_value(
                                        json_object_get(
                                            element,
                                            "value"
                                        )
                                    )
                                ),
                                {
                                    isaac_float(
                                        json_number_value(
                                            json_object_get(
                                                element,
                                                "r"
                                            )
                                        )
                                    ),
                                    isaac_float(
                                        json_number_value(
                                            json_object_get(
                                                element,
                                                "g"
                                            )
                                        )
                                    ),
                                    isaac_float(
                                        json_number_value(
                                            json_object_get(
                                                element,
                                                "b"
                                            )
                                        )
                                    ),
                                    isaac_float(
                                        json_number_value(
                                            json_object_get(
                                                element,
                                                "a"
                                            )
                                        )
                                    )
                                }
                            )
                        );
                    }
                }
                updateTransfer( );
                send_transfer = true;
            }
            if( js = json_object_get(
                message,
                "interpolation"
            ) )
            {
                redraw = true;
                interpolation = json_boolean_value( js );
                send_interpolation = true;
            }
            if( js = json_object_get(
                message,
                "step"
            ) )
            {
                redraw = true;
                step = json_number_value( js );
                if( step < isaac_float( 0.01 ) )
                {
                    step = isaac_float( 0.01 );
                }
                send_step = true;
            }
            if( js = json_object_get(
                message,
                "iso surface"
            ) )
            {
                redraw = true;
                iso_surface = json_boolean_value( js );
                send_iso_surface = true;
            }
            if( json_array_size(
                js = json_object_get(
                    message,
                    "functions"
                )
            ) )
            {
                redraw = true;
                json_array_foreach(
                    js,
                    index,
                    value
                )
                functions[index].source =
                    std::string( json_string_value( value ) );
                updateFunctions( );
                send_functions = true;
            }
            if( json_array_size(
                js = json_object_get(
                    message,
                    "weight"
                )
            ) )
            {
                redraw = true;
                json_array_foreach(
                    js,
                    index,
                    value
                )
                source_weight.value[index] = json_number_value( value );
                send_weight = true;
            }
            if( js = json_object_get(
                message,
                "bounding box"
            ) )
            {
                redraw = true;
                icet_bounding_box = !icet_bounding_box;
                updateBounding( );
            }
            if( json_array_size(
                js = json_object_get(
                    message,
                    "background color"
                )
            ) == 3 )
            {
                redraw = true;
                json_array_foreach(
                    js,
                    index,
                    value
                )
                background_color[index] = json_number_value( value );
                for( int pass = 0; pass < TController::pass_count; pass++ )
                {
                    icetSetContext( icetContext[pass] );
                    if( background_color[0] == 0.0f
                        && background_color[1] == 0.0f
                        && background_color[2] == 0.0f )
                    {
                        icetDisable( ICET_CORRECT_COLORED_BACKGROUND );
                    }
                    else
                    {
                        icetEnable( ICET_CORRECT_COLORED_BACKGROUND );
                    }
                }
                send_background_color = true;
            }
            if( js = json_object_get(
                message,
                "clipping add"
            ) )
            {
                redraw = true;
                send_clipping = true;
                json_t * position = json_object_get(
                    js,
                    "position"
                );
                json_t * normal = json_object_get(
                    js,
                    "normal"
                );
                addClipping(
                    json_number_value(
                        json_array_get(
                            position,
                            0
                        )
                    ),
                    json_number_value(
                        json_array_get(
                            position,
                            1
                        )
                    ),
                    json_number_value(
                        json_array_get(
                            position,
                            2
                        )
                    ),
                    json_number_value(
                        json_array_get(
                            normal,
                            0
                        )
                    ),
                    json_number_value(
                        json_array_get(
                            normal,
                            1
                        )
                    ),
                    json_number_value(
                        json_array_get(
                            normal,
                            2
                        )
                    )
                );
            }
            if( js = json_object_get(
                message,
                "clipping remove"
            ) )
            {
                redraw = true;
                send_clipping = true;
                removeClipping( json_integer_value( js ) );
            }
            if( js = json_object_get(
                message,
                "clipping edit"
            ) )
            {
                redraw = true;
                send_clipping = true;
                json_t * nr = json_object_get(
                    js,
                    "nr"
                );
                json_t * position = json_object_get(
                    js,
                    "position"
                );
                json_t * normal = json_object_get(
                    js,
                    "normal"
                );
                editClipping(
                    json_integer_value( nr ),
                    json_number_value(
                        json_array_get(
                            position,
                            0
                        )
                    ),
                    json_number_value(
                        json_array_get(
                            position,
                            1
                        )
                    ),
                    json_number_value(
                        json_array_get(
                            position,
                            2
                        )
                    ),
                    json_number_value(
                        json_array_get(
                            normal,
                            0
                        )
                    ),
                    json_number_value(
                        json_array_get(
                            normal,
                            1
                        )
                    ),
                    json_number_value(
                        json_array_get(
                            normal,
                            2
                        )
                    )
                );
            }

            if ( js = json_object_get ( message, "ao" ) ) {
                redraw = true;
                json_t * isEnabled = json_object_get(js, "isEnabled");
                json_t * weight = json_object_get(js, "weight");
                
                myself->ambientOcclusion.isEnabled = json_boolean_value ( isEnabled );
                myself->ambientOcclusion.weight = (isaac_float)json_number_value ( weight );

                send_ao = true;
            }

            json_t * metadata = json_object_get(
                message,
                "metadata"
            );
            if( metadata )
            {
                json_incref( metadata );
            }
            json_decref( message );
            thr_metaTargets = metaTargets;

            if( send_minmax )
            {
                isaac_for_each_params(
                    sources,
                    calc_minmax_iterator( ),
                    pointer_array,
                    minmax_array,
                    local_minmax_array_d,
                    local_size
#if ISAAC_ALPAKA == 1
                ,
                stream,
                host
#endif
                );
                isaac_for_each_params(
                    particle_sources,
                    calc_particle_minmax_iterator< boost::mpl::size< TSourceList >::type::value >( ),
                    minmax_array,
                    local_particle_minmax_array_d,
                    local_particle_size
#if ISAAC_ALPAKA == 1
                ,
                stream,
                host
#endif
                );

                if( rank == master )
                {
                    MPI_Reduce(
                        MPI_IN_PLACE,
                        minmax_array.min,
                        (
                            boost::mpl::size< TSourceList >::type::value
                            + boost::mpl::size< TParticleList >::type::value
                        ),
                        MPI_FLOAT,
                        MPI_MIN,
                        master,
                        mpi_world
                    );
                    MPI_Reduce(
                        MPI_IN_PLACE,
                        minmax_array.max,
                        (
                            boost::mpl::size< TSourceList >::type::value
                            + boost::mpl::size< TParticleList >::type::value
                        ),
                        MPI_FLOAT,
                        MPI_MAX,
                        master,
                        mpi_world
                    );
                }
                else
                {
                    MPI_Reduce(
                        minmax_array.min,
                        NULL,
                        (
                            boost::mpl::size< TSourceList >::type::value
                            + boost::mpl::size< TParticleList >::type::value
                        ),
                        MPI_FLOAT,
                        MPI_MIN,
                        master,
                        mpi_world
                    );
                    MPI_Reduce(
                        minmax_array.max,
                        NULL,
                        (
                            boost::mpl::size< TSourceList >::type::value
                            + boost::mpl::size< TParticleList >::type::value
                        ),
                        MPI_FLOAT,
                        MPI_MAX,
                        master,
                        mpi_world
                    );
                }
            }

            for( int pass = 0; pass < TController::pass_count; pass++ )
            {
                image[pass].opaque_internals = NULL;
            }

            if( redraw )
            {
                for( int pass = 0; pass < TController::pass_count; pass++ )
                {
                    icetSetContext( icetContext[pass] );
                    //Calc order
                    ISAAC_START_TIME_MEASUREMENT ( sorting,
                        getTicksUs( ) )
                    //Every rank calculates it's distance to the camera
                    IceTDouble point[4] = {
                        (
                            IceTDouble( position_scaled[0] ) + (
                                                                   IceTDouble( local_size_scaled[0] )
                                                                   - IceTDouble( global_size_scaled[0] )
                                                               ) / 2.0
                        ) / IceTDouble( max_size_scaled / 2 ),
                        (
                            IceTDouble( position_scaled[1] ) + (
                                                                   IceTDouble( local_size_scaled[1] )
                                                                   - IceTDouble( global_size_scaled[1] )
                                                               ) / 2.0
                        ) / IceTDouble( max_size_scaled / 2 ),
                        (
                            IceTDouble( position_scaled[2] ) + (
                                                                   IceTDouble( local_size_scaled[2] )
                                                                   - IceTDouble( global_size_scaled[2] )
                                                               ) / 2.0
                        ) / IceTDouble( max_size_scaled / 2 ),
                        1.0
                    };
                    IceTDouble result[4];
                    mulMatrixVector(
                        result,
                        modelview,
                        point
                    );
                    float point_distance = sqrt(
                        result[0] * result[0] + result[1] * result[1]
                        + result[2] * result[2]
                    );
                    //Allgather of the distances
                    float receive_buffer[numProc];
                    MPI_Allgather(
                        &point_distance,
                        1,
                        MPI_FLOAT,
                        receive_buffer,
                        1,
                        MPI_FLOAT,
                        mpi_world
                    );
                    //Putting to a std::multimap of {rank, distance}
                    std::multimap< float, isaac_int, std::less< float > >
                        distance_map;
                    for( isaac_int i = 0; i < numProc; i++ )
                    {
                        distance_map.insert(
                            std::pair<
                                float,
                                isaac_int
                            >(
                                receive_buffer[i],
                                i
                            )
                        );
                    }
                    //Putting in an array for IceT
                    IceTInt icet_order_array[numProc];
                    {
                        isaac_int i = 0;
                        for( auto it = distance_map.begin( );
                            it != distance_map.end( );
                            it++ )
                        {
                            icet_order_array[i] = it->second;
                            i++;
                        }
                    }
                    icetCompositeOrder( icet_order_array );
                    ISAAC_STOP_TIME_MEASUREMENT ( sorting_time,
                        +=,
                        sorting,
                        getTicksUs( ) )

                    //Drawing
                    ISAAC_START_TIME_MEASUREMENT ( merge,
                        getTicksUs( ) )
                    image[pass] = icetDrawFrame(
                        &( projection[pass * 16] ),
                        modelview,
                        background_color
                    );
                    ISAAC_STOP_TIME_MEASUREMENT ( merge_time,
                        +=,
                        merge,
                        getTicksUs( ) )
                }
            }
            else
            {
                usleep( 10000 );
            }

            //Message merging
            char * buffer = json_dumps(
                json_root,
                0
            );
            strcpy(
                message_buffer,
                buffer
            );
            free( buffer );
            if( metaTargets == META_MERGE )
            {
                if( rank == master )
                {
                    char receive_buffer[numProc][ISAAC_MAX_RECEIVE];
                    MPI_Gather(
                        message_buffer,
                        ISAAC_MAX_RECEIVE,
                        MPI_CHAR,
                        receive_buffer,
                        ISAAC_MAX_RECEIVE,
                        MPI_CHAR,
                        master,
                        mpi_world
                    );
                    for( isaac_int i = 0; i < numProc; i++ )
                    {
                        if( i == master )
                        {
                            continue;
                        }
                        json_t * js = json_loads(
                            receive_buffer[i],
                            0,
                            NULL
                        );
                        mergeJSON(
                            json_root,
                            js
                        );
                        json_decref( js );
                    }
                }
                else
                {
                    MPI_Gather(
                        message_buffer,
                        ISAAC_MAX_RECEIVE,
                        MPI_CHAR,
                        NULL,
                        0,
                        MPI_CHAR,
                        master,
                        mpi_world
                    );
                }
            }

#ifdef ISAAC_THREADING
            pthread_create ( &visualizationThread,NULL,visualizationFunction,NULL );
#else
            visualizationFunction( NULL );
#endif
            return metadata;
        }


        ~IsaacVisualization( )
        {
            ISAAC_WAIT_VISUALIZATION
            json_decref( json_root );
            if( rank == master )
            {
                json_root = json_object( );
                json_object_set_new(
                    json_root,
                    "type",
                    json_string( "exit" )
                );
                char * buffer = json_dumps(
                    json_root,
                    0
                );
                communicator->serverSend(
                    buffer,
                    true,
                    true
                );
                free( buffer );
                json_decref( json_root );
            }
            for( int pass = 0; pass < TController::pass_count; pass++ )
            {
                icetDestroyContext( icetContext[pass] );
            }
#if ISAAC_ALPAKA == 0
            for( int i = 0; i < (
                boost::mpl::size< TSourceList >::type::value
                + boost::mpl::size< TParticleList >::type::value
            ); i++ )
            {
                if( pointer_array.pointer[i] )
                {
                    ISAAC_CUDA_CHECK ( cudaFree( pointer_array.pointer[i] ) );
                }
                ISAAC_CUDA_CHECK ( cudaFree( transfer_d.pointer[i] ) );
                free( transfer_h.pointer[i] );
            }
            ISAAC_CUDA_CHECK ( cudaFree( framebuffer ) );
            ISAAC_CUDA_CHECK ( cudaFree( functor_chain_d ) );
            ISAAC_CUDA_CHECK ( cudaFree( functor_chain_choose_d ) );
            ISAAC_CUDA_CHECK ( cudaFree( local_minmax_array_d ) );
            ISAAC_CUDA_CHECK ( cudaFree( local_particle_minmax_array_d ) );
#endif
            delete communicator;
            json_decref( json_init_root );
        }


        uint64_t getTicksUs( )
        {
            struct timespec ts;
            if( clock_gettime(
                CLOCK_MONOTONIC_RAW,
                &ts
            ) == 0 )
            {
                return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
            }
            return 0;
        }


        uint64_t kernel_time;
        uint64_t merge_time;
        uint64_t video_send_time;
        uint64_t copy_time;
        uint64_t sorting_time;
        uint64_t buffer_time;
    private:
        static void drawCallBack(
            const IceTDouble * projection_matrix,
            const IceTDouble * modelview_matrix,
            const IceTFloat * background_color,
            const IceTInt * readback_viewport,
            IceTImage result
        )
        {
            //allocate memory for inverse mvp, mv and p matrix and simulation size properties
#if ISAAC_ALPAKA == 1

            //inverse mvp
            alpaka::Buf <THost, isaac_float, TFraDim, ISAAC_IDX_TYPE>
                inverse_h_buf(
                alpaka::allocBuf<
                    isaac_float,
                    ISAAC_IDX_TYPE
                >(
                    myself->host,
                    ISAAC_IDX_TYPE( 16 )
                )
            );
            isaac_float * inverse_h =
                reinterpret_cast<float *> ( alpaka::getPtrNative( inverse_h_buf ) );


            //model-view matrix
            alpaka::Buf <THost, isaac_float, TFraDim, ISAAC_IDX_TYPE>
                modelview_h_buf(
                alpaka::allocBuf<
                    isaac_float,
                    ISAAC_IDX_TYPE
                >(
                    myself->host,
                    ISAAC_IDX_TYPE( 16 )
                )
            );
            isaac_float * modelview_h =
                reinterpret_cast<float *> ( alpaka::getPtrNative( modelview_h_buf ) );


            //projection matrix
            alpaka::Buf <THost, isaac_float, TFraDim, ISAAC_IDX_TYPE>
                projection_h_buf(
                alpaka::allocBuf<
                    isaac_float,
                    ISAAC_IDX_TYPE
                >(
                    myself->host,
                    ISAAC_IDX_TYPE( 16 )
                )
            );
            isaac_float * projection_h =
                reinterpret_cast<float *> ( alpaka::getPtrNative( modelview_h_buf ) );


            //sim size values
            alpaka::Buf <THost, isaac_size_struct< TSimDim::value >, TFraDim, ISAAC_IDX_TYPE>
                size_h_buf(
                alpaka::allocBuf < isaac_size_struct < TSimDim::value > ,
                ISAAC_IDX_TYPE > ( myself->host, ISAAC_IDX_TYPE( 1 ) )
            );
            isaac_size_struct < TSimDim::value > *size_h =
                reinterpret_cast<isaac_size_struct< TSimDim::value > *> ( alpaka::getPtrNative( size_h_buf ) );

#else
            isaac_float inverse_h[16];
            isaac_float modelview_h[16];
            isaac_float projection_h[16];
            isaac_size_struct < TSimDim::value > size_h[1];
#endif

            //calculate inverse mvp matrix for render kernel
            IceTDouble inverse[16];
            calcInverse(
                inverse,
                projection_matrix,
                modelview_matrix
            );
            for( int i = 0; i < 16; i++ )
            {
                //set values for inverse, mv and p matrix
                inverse_h[i] = static_cast<float> ( inverse[i] );
                modelview_h[i] = static_cast<float>(modelview_matrix[i]);
                projection_h[i] = static_cast<float>(projection_matrix[i]);
            }

            //set global simulation size
            size_h[0].global_size
                .value
                .x = myself->global_size[0];
            size_h[0].global_size
                .value
                .y = myself->global_size[1];
            if( TSimDim::value > 2 )
            {
                size_h[0].global_size
                    .value
                    .z = myself->global_size[2];
            }
            size_h[0].position
                .value
                .x = myself->position[0];
            size_h[0].position
                .value
                .y = myself->position[1];
            if( TSimDim::value > 2 )
            {
                size_h[0].position
                    .value
                    .z = myself->position[2];
            }

            //set subvolume size
            size_h[0].local_size
                .value
                .x = myself->local_size[0];
            size_h[0].local_size
                .value
                .y = myself->local_size[1];
            if( TSimDim::value > 2 )
            {
                size_h[0].local_size
                    .value
                    .z = myself->local_size[2];
            }
            size_h[0].local_particle_size
                .value
                .x = myself->local_particle_size[0];
            size_h[0].local_particle_size
                .value
                .y = myself->local_particle_size[1];
            if( TSimDim::value > 2 )
            {
                size_h[0].local_particle_size
                    .value
                    .z = myself->local_particle_size[2];
            }
            
            //get maximum size from biggest dimesnion MAX(x-dim, y-dim, z-dim)
            size_h[0].max_global_size = static_cast<float> ( ISAAC_MAX(
                ISAAC_MAX(
                    uint32_t( myself->global_size[0] ),
                    uint32_t( myself->global_size[1] )
                ),
                uint32_t( myself->global_size[2] )
            ) );

            //set global size with cellcount scaled
            size_h[0].global_size_scaled
                .value
                .x = myself->global_size_scaled[0];
            size_h[0].global_size_scaled
                .value
                .y = myself->global_size_scaled[1];
            if( TSimDim::value > 2 )
            {
                size_h[0].global_size_scaled
                    .value
                    .z = myself->global_size_scaled[2];
            }

            //set position in subvolume (adjusted to cellcount scale)
            size_h[0].position_scaled
                .value
                .x = myself->position_scaled[0];
            size_h[0].position_scaled
                .value
                .y = myself->position_scaled[1];
            if( TSimDim::value > 2 )
            {
                size_h[0].position_scaled
                    .value
                    .z = myself->position_scaled[2];
            }
            size_h[0].local_size_scaled
                .value
                .x = myself->local_size_scaled[0];
            size_h[0].local_size_scaled
                .value
                .y = myself->local_size_scaled[1];
            if( TSimDim::value > 2 )
            {
                size_h[0].local_size_scaled
                    .value
                    .z = myself->local_size_scaled[2];
            }

            //get maximum size from biggest dimesnion after scaling MAX(x-dim, y-dim, z-dim)
            size_h[0].max_global_size_scaled = static_cast<float> ( ISAAC_MAX(
                ISAAC_MAX(
                    uint32_t( myself->global_size_scaled[0] ),
                    uint32_t( myself->global_size_scaled[1] )
                ),
                uint32_t( myself->global_size_scaled[2] )
            ) );

            //set volume scale parameters
            isaac_float3 isaac_scale = {
                myself->scale[0],
                myself->scale[1],
                myself->scale[2]
            };

            //copy matrices and simulation size properties to constant memory
#if ISAAC_ALPAKA == 1

            //inverse matrix
            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                inverse_d_extent( ISAAC_IDX_TYPE( 16 ) );
            //get view
            auto inverse_d_view(
                alpaka::createStaticDevMemView ( & isaac_inverse_d[0u],
                myself -> acc, inverse_d_extent ) );
            //copy to constant memory
            alpaka::memcpy(
                myself->stream,
                inverse_d_view,
                inverse_h_buf,
                ISAAC_IDX_TYPE( 16 )
            );

            //modelview matrix
            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                modelview_d_extent( ISAAC_IDX_TYPE( 16 ) );
            //get view
            auto modelview_d_view(
                alpaka::createStaticDevMemView ( & isaac_modelview_d[0u],
                myself -> acc, modelview_d_extent ) );
            //copy to constant memory 
            alpaka::memcpy(
                myself->stream,
                modelview_d_view,
                modelview_h_buf,
                ISAAC_IDX_TYPE( 16 )
            );


            //projection matrix
            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                projection_d_extent( ISAAC_IDX_TYPE( 16 ) );
            //get view
            auto projection_d_view(
                alpaka::createStaticDevMemView ( & isaac_projection_d[0u],
                myself -> acc, projection_d_extent ) );
            //copy to constant memory
            alpaka::memcpy(
                myself->stream,
                projection_d_view,
                projection_h_buf,
                ISAAC_IDX_TYPE( 16 )
            );

            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                size_d_extent( ISAAC_IDX_TYPE( 1 ) );
            auto size_d_view(
                alpaka::createStaticDevMemView ( & isaac_size_d[0u],
                myself -> acc, size_d_extent ) );
            alpaka::memcpy(
                myself->stream,
                size_d_view,
                size_h_buf,
                ISAAC_IDX_TYPE( 1 )
            );

#else

            //copy inverse matrix to constant memory
            ISAAC_CUDA_CHECK ( cudaMemcpyToSymbol(
                isaac_inverse_d,
                inverse_h,
                16 * sizeof( float )
            ) );

            //copy modelview matrix to constant memory
            ISAAC_CUDA_CHECK ( cudaMemcpyToSymbol(
                isaac_modelview_d,
                modelview_h,
                16 * sizeof( float )
            ) );

            //copy projection matrix to constant memory
            ISAAC_CUDA_CHECK ( cudaMemcpyToSymbol(
                isaac_projection_d,
                projection_h,
                16 * sizeof( float )
            ) );

            //copy simulation size values to constant memory
            ISAAC_CUDA_CHECK ( cudaMemcpyToSymbol(
                isaac_size_d,
                size_h,
                sizeof( isaac_size_struct< TSimDim::value > )
            ) );

#endif

            //get pixel pointer from image as unsigned byte
            IceTUByte * pixels = icetImageGetColorub( result );

            //start time for performance measurment
            ISAAC_START_TIME_MEASUREMENT ( kernel,
                myself->getTicksUs( ) )

            //set color values for background color
            isaac_float4 bg_color = {
                isaac_float( background_color[3] ),
                isaac_float( background_color[2] ),
                isaac_float( background_color[1] ),
                isaac_float( background_color[0] )
            };

            //set framebuffer offset calculated from icet
            isaac_uint2 framebuffer_start = {
                isaac_uint( readback_viewport[0] ),
                isaac_uint( readback_viewport[1] )
            };

            //call render kernel
#if ISAAC_ALPAKA == 1
            IsaacRenderKernelCaller<
                TSimDim,
                TParticleList,
                TSourceList,
                transfer_d_struct<
                    (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    )
                >,
                source_weight_struct<
                    (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    )
                >,
                pointer_array_struct<
                    (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    )
                >,
                mpl::vector< >,
                alpaka::Buf< 
                    TDevAcc, 
                    uint32_t, 
                    TFraDim, 
                    ISAAC_IDX_TYPE
                >, 
                alpaka::Buf<
                    TDevAcc, 
                    isaac_float3, 
                    TFraDim, 
                    ISAAC_IDX_TYPE
                >,
                alpaka::Buf<
                    TDevAcc, 
                    isaac_float3, 
                    TFraDim, 
                    ISAAC_IDX_TYPE
                >,
                TTransfer_size, 
                isaac_float3, 
                TAccDim, 
                TAcc, 
                TStream,
                alpaka::Buf< 
                    TDevAcc, 
                    isaac_functor_chain_pointer_N, 
                    TFraDim,
                    ISAAC_IDX_TYPE 
                >, 
                (
                    boost::mpl::size< TSourceList >::type::value +
                    boost::mpl::size< TParticleList >::type::value
                ) 
            > 
            ::call(
                myself->stream,
                myself->framebuffer,
                myself->framebufferDepth,
                myself->framebufferNormal,
                myself->framebuffer_size,
                framebuffer_start,
                myself->particle_sources,
                myself->sources,
                myself->step,
                bg_color,
                myself->transfer_d,
                myself->source_weight,
                myself->pointer_array,
                readback_viewport,
                myself->interpolation,
                myself->iso_surface,
                isaac_scale,
                myself->clipping,
                myself->ambientOcclusion
            );

            //wait until render kernel has finished
            alpaka::wait( myself->stream );

            //process color and depth values for depth simulation
            if(myself->ambientOcclusion.isEnabled && myself->ambientOcclusion.weight > 0.0f) {
                IsaacSSAOKernelCaller<
                    alpaka::Buf<
                        TDevAcc, 
                        isaac_float, 
                        TFraDim, 
                        ISAAC_IDX_TYPE
                    >,
                    alpaka::Buf<
                        TDevAcc, 
                        isaac_float3, 
                        TFraDim, 
                        ISAAC_IDX_TYPE
                    >,
                    alpaka::Buf<
                        TDevAcc, 
                        isaac_float3, 
                        TFraDim, 
                        ISAAC_IDX_TYPE
                    >,
                    TAccDim,
                    TAcc,
                    TStream
                >
                ::call (
                    myself->stream,
                    myself->framebufferAO,
                    myself->framebufferDepth,
                    myself->framebufferNormal,
                    myself->framebuffer_size,
                    framebuffer_start,
                    readback_viewport,
                    myself->ambientOcclusion
                );

                //wait until render kernel has finished
                alpaka::wait ( myself->stream );

                IsaacSSAOFilterKernelCaller
                <
                    alpaka::Buf<
                        TDevAcc, 
                        uint32_t, 
                        TFraDim, 
                        ISAAC_IDX_TYPE
                    >,
                    alpaka::Buf<
                        TDevAcc, 
                        isaac_float, 
                        TFraDim, 
                        ISAAC_IDX_TYPE
                    >,
                    alpaka::Buf<
                        TDevAcc, 
                        isaac_float3, 
                        TFraDim, 
                        ISAAC_IDX_TYPE
                    >,
                    TAccDim,
                    TAcc,
                    TStream
                >
                ::call (
                    myself->stream,
                    myself->framebuffer,
                    myself->framebufferAO,
                    myself->framebufferDepth,
                    myself->framebuffer_size,
                    framebuffer_start,
                    readback_viewport,
                    myself->ambientOcclusion
                );

                //wait until render kernel has finished
                alpaka::wait ( myself->stream );
            }

            //stop and restart time for delta calculation
            ISAAC_STOP_TIME_MEASUREMENT ( myself->kernel_time,
                +=,
                kernel,
                myself->getTicksUs( ) )
            ISAAC_START_TIME_MEASUREMENT ( copy,
                myself->getTicksUs( ) )

            //get memory view from IceT pixels on host
            alpaka::ViewPlainPtr <THost, uint32_t, TFraDim, ISAAC_IDX_TYPE>
                result_buffer(
                ( uint32_t * )( pixels ),
                myself->host,
                alpaka::Vec<
                    TFraDim,
                    ISAAC_IDX_TYPE
                >( myself->framebuffer_prod )
            );

            //copy device framebuffer to result IceT pixel buffer
            alpaka::memcpy(
                myself->stream,
                result_buffer,
                myself->framebuffer,
                alpaka::Vec<
                    TFraDim,
                    ISAAC_IDX_TYPE
                >( myself->framebuffer_prod )
            );
#else
            IsaacRenderKernelCaller<
                TSimDim,
                TParticleList,
                TSourceList,
                transfer_d_struct<
                    (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    )
                >,
                source_weight_struct<
                    (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    )
                >,
                pointer_array_struct<
                    (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    )
                >,
                mpl::vector< >,
                uint32_t *,
                isaac_float3*,
                isaac_float3*,
                TTransfer_size,
                isaac_float3,
                (
                    boost::mpl::size< TSourceList >::type::value
                    + boost::mpl::size< TParticleList >::type::value
                )
            >
            ::call(
                myself->framebuffer,
                myself->framebufferDepth,
                myself->framebufferNormal,
                myself->framebuffer_size,
                framebuffer_start,
                myself->particle_sources,
                myself->sources,
                myself->step,
                bg_color,
                myself->transfer_d,
                myself->source_weight,
                myself->pointer_array,
                readback_viewport,
                myself->interpolation,
                myself->iso_surface,
                isaac_scale,
                myself->clipping,
                myself->ambientOcclusion
            );

            //wait until render kernel has finished
            ISAAC_CUDA_CHECK ( cudaDeviceSynchronize() );

            //process depth/color information for depth simulation
            if(myself->ambientOcclusion.isEnabled && myself->ambientOcclusion.weight > 0.0f) {
                //call render kernel
                IsaacSSAOKernelCaller
                <
                isaac_float*,
                isaac_float3*,
                isaac_float3*
                >
                ::call (
                    myself->framebufferAO,
                    myself->framebufferDepth,
                    myself->framebufferNormal,
                    myself->framebuffer_size,
                    framebuffer_start,
                    readback_viewport,
                    myself->ambientOcclusion
                );

                //wait until render kernel has finished
                ISAAC_CUDA_CHECK ( cudaDeviceSynchronize() );

                //call render kernel
                IsaacSSAOFilterKernelCaller
                <
                uint32_t*,
                isaac_float*,
                isaac_float3*
                >
                ::call (
                    myself->framebuffer,
                    myself->framebufferAO,
                    myself->framebufferDepth,
                    myself->framebuffer_size,
                    framebuffer_start,
                    readback_viewport,
                    myself->ambientOcclusion
                );

                //wait until render kernel has finished
                ISAAC_CUDA_CHECK ( cudaDeviceSynchronize() );
            }

            //stop and restart time for delta calculation
            ISAAC_STOP_TIME_MEASUREMENT ( myself->kernel_time,
                +=,
                kernel,
                myself->getTicksUs( ) )
            ISAAC_START_TIME_MEASUREMENT ( copy,
                myself->getTicksUs( ) )

            //copy filled framebuffer to IceT result buffer
            ISAAC_CUDA_CHECK ( cudaMemcpy(
                ( uint32_t * )( pixels ),
                myself->framebuffer,
                sizeof( uint32_t ) * myself->framebuffer_prod,
                cudaMemcpyDeviceToHost
            ) );

#endif

            //stop timer and calculate copy time
            ISAAC_STOP_TIME_MEASUREMENT ( myself->copy_time,
                +=,
                copy,
                myself->getTicksUs( ) 
            )
        }


        static void * visualizationFunction( void * dummy )
        {
            //Message sending
            if( myself->rank == myself->master )
            {
                json_object_set_new(
                    myself->json_root,
                    "type",
                    json_string( "period" )
                );
                json_object_set_new(
                    myself->json_root,
                    "meta nr",
                    json_integer( myself->metaNr )
                );

                json_t * matrix;
                if( myself->send_projection )
                {
                    json_object_set_new(
                        myself->json_root,
                        "projection",
                        matrix = json_array( )
                    );
                    ISAAC_JSON_ADD_MATRIX ( matrix,
                        myself->projection,
                        16 * TController::pass_count )
                    json_object_set(
                        myself->json_init_root,
                        "projection",
                        matrix
                    );
                    myself->send_init_json = true;
                }
                if( myself->send_look_at )
                {
                    json_object_set_new(
                        myself->json_root,
                        "position",
                        matrix = json_array( )
                    );
                    ISAAC_JSON_ADD_MATRIX ( matrix,
                        myself->look_at,
                        3 )
                    json_object_set(
                        myself->json_init_root,
                        "position",
                        matrix
                    );
                    myself->send_init_json = true;
                }
                if( myself->send_rotation )
                {
                    json_object_set_new(
                        myself->json_root,
                        "rotation",
                        matrix = json_array( )
                    );
                    ISAAC_JSON_ADD_MATRIX ( matrix,
                        myself->rotation,
                        9 )
                    json_object_set(
                        myself->json_init_root,
                        "rotation",
                        matrix
                    );
                    myself->send_init_json = true;
                }
                if( myself->send_distance )
                {
                    json_object_set_new(
                        myself->json_root,
                        "distance",
                        json_real( myself->distance )
                    );
                    json_object_set_new(
                        myself->json_init_root,
                        "distance",
                        json_real( myself->distance )
                    );
                    myself->send_init_json = true;
                }
                if( myself->send_transfer )
                {
                    json_object_set_new(
                        myself->json_root,
                        "transfer array",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    ); i++ )
                    {
                        json_t * transfer = json_array( );
                        json_array_append_new(
                            matrix,
                            transfer
                        );
                        for( ISAAC_IDX_TYPE j = 0; j < TTransfer_size; j++ )
                        {
                            json_t * color = json_array( );
                            json_array_append_new(
                                transfer,
                                color
                            );
                            json_array_append_new(
                                color,
                                json_integer(
                                    isaac_uint(
                                        myself->transfer_h
                                            .pointer[i][j].x
                                        * isaac_float( 255 )
                                    )
                                )
                            );
                            json_array_append_new(
                                color,
                                json_integer(
                                    isaac_uint(
                                        myself->transfer_h
                                            .pointer[i][j].y
                                        * isaac_float( 255 )
                                    )
                                )
                            );
                            json_array_append_new(
                                color,
                                json_integer(
                                    isaac_uint(
                                        myself->transfer_h
                                            .pointer[i][j].z
                                        * isaac_float( 255 )
                                    )
                                )
                            );
                            json_array_append_new(
                                color,
                                json_integer(
                                    isaac_uint(
                                        myself->transfer_h
                                            .pointer[i][j].w
                                        * isaac_float( 255 )
                                    )
                                )
                            );
                        }
                    }
                    json_object_set_new(
                        myself->json_root,
                        "transfer points",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    ); i++ )
                    {
                        json_t * points = json_array( );
                        json_array_append_new(
                            matrix,
                            points
                        );
                        for( auto it = myself->transfer_h
                            .description[i].begin( );
                            it != myself->transfer_h
                                .description[i].end( );
                            it++ )
                        {
                            json_t * p = json_object( );
                            json_array_append_new(
                                points,
                                p
                            );
                            json_object_set_new(
                                p,
                                "value",
                                json_integer( it->first )
                            );
                            json_object_set_new(
                                p,
                                "r",
                                json_real(
                                    it->second
                                        .x
                                )
                            );
                            json_object_set_new(
                                p,
                                "g",
                                json_real(
                                    it->second
                                        .y
                                )
                            );
                            json_object_set_new(
                                p,
                                "b",
                                json_real(
                                    it->second
                                        .z
                                )
                            );
                            json_object_set_new(
                                p,
                                "a",
                                json_real(
                                    it->second
                                        .w
                                )
                            );
                        }
                    }
                }
                if( myself->send_functions )
                {
                    json_object_set_new(
                        myself->json_root,
                        "functions",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    ); i++ )
                    {
                        json_t * f = json_object( );
                        json_array_append_new(
                            matrix,
                            f
                        );
                        json_object_set_new(
                            f,
                            "source",
                            json_string(
                                myself->functions[i].source
                                    .c_str( )
                            )
                        );
                        json_object_set_new(
                            f,
                            "error",
                            json_integer( myself->functions[i].error_code )
                        );
                    }
                }
                if( myself->send_weight )
                {
                    json_object_set_new(
                        myself->json_root,
                        "weight",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    ); i++ )
                    {
                        json_array_append_new(
                            matrix,
                            json_real(
                                myself->source_weight
                                    .value[i]
                            )
                        );
                    }
                }
                if( myself->send_interpolation )
                {
                    json_object_set_new(
                        myself->json_root,
                        "interpolation",
                        json_boolean( myself->interpolation )
                    );
                    json_object_set_new(
                        myself->json_init_root,
                        "interpolation",
                        json_boolean( myself->interpolation )
                    );
                    myself->send_init_json = true;
                }
                if( myself->send_step )
                {
                    json_object_set_new(
                        myself->json_root,
                        "step",
                        json_real( myself->step )
                    );
                    json_object_set_new(
                        myself->json_init_root,
                        "step",
                        json_boolean( myself->step )
                    );
                    myself->send_init_json = true;
                }
                if( myself->send_iso_surface )
                {
                    json_object_set_new(
                        myself->json_root,
                        "iso surface",
                        json_boolean( myself->iso_surface )
                    );
                    json_object_set_new(
                        myself->json_init_root,
                        "iso surface",
                        json_boolean( myself->iso_surface )
                    );
                    myself->send_init_json = true;
                }
                if( myself->send_minmax )
                {
                    json_object_set_new(
                        myself->json_root,
                        "minmax",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < (
                        boost::mpl::size< TSourceList >::type::value
                        + boost::mpl::size< TParticleList >::type::value
                    ); i++ )
                    {
                        json_t * v = json_object( );
                        json_array_append_new(
                            matrix,
                            v
                        );
                        json_object_set_new(
                            v,
                            "min",
                            json_real(
                                myself->minmax_array
                                    .min[i]
                            )
                        );
                        json_object_set_new(
                            v,
                            "max",
                            json_real(
                                myself->minmax_array
                                    .max[i]
                            )
                        );
                    }
                }
                if( myself->send_background_color )
                {
                    json_object_set_new(
                        myself->json_root,
                        "background color",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < 3; i++ )
                    {
                        json_array_append_new(
                            matrix,
                            json_real( myself->background_color[i] )
                        );
                    }
                    json_object_set(
                        myself->json_init_root,
                        "background color",
                        matrix
                    );
                    myself->send_init_json = true;
                }
                if( myself->send_clipping )
                {
                    json_object_set_new(
                        myself->json_root,
                        "clipping",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < ISAAC_IDX_TYPE(
                        myself->clipping
                            .count
                    ); i++ )
                    {
                        json_t * f = json_object( );
                        json_array_append_new(
                            matrix,
                            f
                        );
                        json_t * inner = json_array( );
                        json_object_set_new(
                            f,
                            "position",
                            inner
                        );
                        json_array_append_new(
                            inner,
                            json_real(
                                myself->clipping
                                    .elem[i].position
                                    .x
                            )
                        );
                        json_array_append_new(
                            inner,
                            json_real(
                                myself->clipping
                                    .elem[i].position
                                    .y
                            )
                        );
                        json_array_append_new(
                            inner,
                            json_real(
                                myself->clipping
                                    .elem[i].position
                                    .z
                            )
                        );
                        inner = json_array( );
                        json_object_set_new(
                            f,
                            "normal",
                            inner
                        );
                        json_array_append_new(
                            inner,
                            json_real( myself->clipping_saved_normals[i].x )
                        );
                        json_array_append_new(
                            inner,
                            json_real( myself->clipping_saved_normals[i].y )
                        );
                        json_array_append_new(
                            inner,
                            json_real( myself->clipping_saved_normals[i].z )
                        );
                    }
                }
                if(myself->send_ao) {
                    //add ambient occlusion parameters
                    json_object_set_new(
                        myself->json_root, 
                        "ao isEnabled", 
                        json_boolean(myself->ambientOcclusion.isEnabled)
                    );
                    json_object_set_new(
                        myself->json_root, 
                        "ao weight", 
                        json_real(myself->ambientOcclusion.weight)
                    );       
                    //add ao params to initial response   
                    json_object_set_new(
                        myself->json_init_root, 
                        "ao isEnabled", 
                        json_boolean(myself->ambientOcclusion.isEnabled)
                    );
                    json_object_set_new(
                        myself->json_init_root, 
                        "ao weight", 
                        json_real(myself->ambientOcclusion.weight)
                    );
                    myself->send_init_json = true;
                }
                myself->controller
                    .sendFeedback(
                        myself->json_root,
                        myself->send_controller
                    );
                if( myself->send_init_json )
                {
                    json_object_set(
                        myself->json_root,
                        "init",
                        myself->json_init_root
                    );
                }
                char * buffer = json_dumps(
                    myself->json_root,
                    0
                );
                myself->communicator
                    ->serverSend( buffer );
                free( buffer );
            }
            json_decref( myself->json_root );
            myself->recreateJSON( );

            //Sending video
            ISAAC_START_TIME_MEASUREMENT ( video_send,
                myself->getTicksUs( ) )
            if( myself->communicator )
            {
                if( myself->image[0].opaque_internals )
                {
                    myself->communicator
                        ->serverSendFrame(
                            myself->compositor
                                .doCompositing( myself->image ),
                            myself->compbuffer_size
                                .x,
                            myself->compbuffer_size
                                .y,
                            4
                        );
                }
                else
                {
                    myself->communicator
                        ->serverSend(
                            NULL,
                            false,
                            true
                        );
                }
            }
            ISAAC_STOP_TIME_MEASUREMENT ( myself->video_send_time,
                +=,
                video_send,
                myself->getTicksUs( ) )
            myself->metaNr++;
            return 0;
        }


        void recreateJSON( )
        {
            json_root = json_object( );
            json_meta_root = json_object( );
            json_object_set_new(
                json_root,
                "metadata",
                json_meta_root
            );
        }


        void updateModelview( )
        {
            IceTDouble look_at_m[16];
            ISAAC_SET_IDENTITY ( 4,
                look_at_m )
            look_at_m[12] = look_at[0];
            look_at_m[13] = look_at[1];
            look_at_m[14] = look_at[2];

            IceTDouble rotation_m[16];
            for( isaac_int x = 0; x < 4; x++ )
            {
                for( isaac_int y = 0; y < 4; y++ )
                {
                    if( x < 3 && y < 3 )
                    {
                        rotation_m[x + y * 4] = rotation[x + y * 3];
                    }
                    else if( x != 3 || y != 3 )
                    {
                        rotation_m[x + y * 4] = 0.0;
                    }
                    else
                    {
                        rotation_m[x + y * 4] = 1.0;
                    }
                }
            }

            IceTDouble distance_m[16];
            ISAAC_SET_IDENTITY ( 4,
                distance_m )
            distance_m[14] = distance;

            IceTDouble temp[16];

            mulMatrixMatrix(
                temp,
                rotation_m,
                look_at_m
            );
            mulMatrixMatrix(
                modelview,
                distance_m,
                temp
            );
        }


#if ISAAC_ALPAKA == 1
        THost host;
        TDevAcc acc;
        TStream stream;
#endif
        std::string name;
        std::string server_url;
        isaac_uint server_port;
        isaac_size2 framebuffer_size;
        isaac_size2 compbuffer_size;
#if ISAAC_ALPAKA == 1
        alpaka::Vec<
            TFraDim,
            ISAAC_IDX_TYPE
        > framebuffer_prod;

        //framebuffer pixel values
        alpaka::Buf<
            TDevAcc,
            uint32_t,
            TFraDim,
            ISAAC_IDX_TYPE
        > framebuffer;

        //ambient occlusion factor values
        alpaka::Buf<
            TDevAcc, 
            isaac_float, 
            TFraDim, 
            ISAAC_IDX_TYPE
        > framebufferAO;

        //pixel depth information
        alpaka::Buf<
            TDevAcc, 
            isaac_float3, 
            TFraDim, 
            ISAAC_IDX_TYPE
        > framebufferDepth;

        //pixel normal information
        alpaka::Buf<
            TDevAcc, 
            isaac_float3, 
            TFraDim, 
            ISAAC_IDX_TYPE
        > framebufferNormal;  

        alpaka::Buf<
            TDevAcc,
            isaac_functor_chain_pointer_N,
            TFraDim,
            ISAAC_IDX_TYPE
        > functor_chain_d;
        alpaka::Buf<
            TDevAcc,
            isaac_functor_chain_pointer_N,
            TFraDim,
            ISAAC_IDX_TYPE
        > functor_chain_choose_d;
        alpaka::Buf<
            TDevAcc,
            minmax_struct,
            TFraDim,
            ISAAC_IDX_TYPE
        > local_minmax_array_d;
        alpaka::Buf<
            TDevAcc,
            minmax_struct,
            TFraDim,
            ISAAC_IDX_TYPE
        > local_particle_minmax_array_d;
#else
        ISAAC_IDX_TYPE framebuffer_prod;
        isaac_uint* framebuffer;                                //color values storage
        isaac_float* framebufferAO;                             //ao/depth indicator storage
        isaac_float3* framebufferDepth;                         //depth value storage
        isaac_float3* framebufferNormal;                        //normal values storage
        isaac_functor_chain_pointer_N * functor_chain_d;
        isaac_functor_chain_pointer_N * functor_chain_choose_d;
        minmax_struct * local_minmax_array_d;
        minmax_struct * local_particle_minmax_array_d;
#endif
        TDomainSize global_size;
        TDomainSize local_size;
        TDomainSize local_particle_size;
        TDomainSize position;
        std::vector <ISAAC_IDX_TYPE> global_size_scaled;
        std::vector <ISAAC_IDX_TYPE> local_size_scaled;
        std::vector <ISAAC_IDX_TYPE> position_scaled;
        MPI_Comm mpi_world;
        IceTDouble projection[16 * TController::pass_count];
        IceTDouble look_at[3];
        IceTDouble rotation[9];
        IceTDouble distance;

        //true if properties should be sent by server
        bool send_look_at;
        bool send_rotation;
        bool send_distance;
        bool send_projection;
        bool send_transfer;
        bool send_interpolation;
        bool send_step;
        bool send_iso_surface;
        bool send_functions;
        bool send_weight;
        bool send_minmax;
        bool send_background_color;
        bool send_clipping;
        bool send_controller;
        bool send_init_json;
        bool send_ao;


        bool interpolation;
        bool iso_surface;
        bool icet_bounding_box;
        isaac_float step;
        IceTDouble modelview[16];                           //modelview matrix
        IsaacCommunicator * communicator;
        json_t * json_root;
        json_t * json_init_root;
        json_t * json_meta_root;
        isaac_int rank;
        isaac_int master;
        isaac_int numProc;
        isaac_uint metaNr;
        TParticleList & particle_sources;
        TSourceList & sources;
        IceTContext icetContext[TController::pass_count];
        IsaacVisualizationMetaEnum thr_metaTargets;
        pthread_t visualizationThread;
        ao_struct ambientOcclusion;                         //state of ambient occlusion on client site
#if ISAAC_ALPAKA == 1
        std::vector <alpaka::Buf<
            TDevAcc,
            isaac_float4,
            TTexDim,
            ISAAC_IDX_TYPE
        >> transfer_d_buf;
        std::vector <alpaka::Buf<
            THost,
            isaac_float4,
            TTexDim,
            ISAAC_IDX_TYPE
        >> transfer_h_buf;
        std::vector <alpaka::Buf<
            TDevAcc,
            isaac_float,
            TFraDim,
            ISAAC_IDX_TYPE
        >> pointer_array_alpaka;
#endif
        transfer_d_struct<
            (
                boost::mpl::size< TSourceList >::type::value
                + boost::mpl::size< TParticleList >::type::value
            )
        > transfer_d;
        transfer_h_struct<
            (
                boost::mpl::size< TSourceList >::type::value
                + boost::mpl::size< TParticleList >::type::value
            )
        > transfer_h;
        source_weight_struct<
            (
                boost::mpl::size< TSourceList >::type::value
                + boost::mpl::size< TParticleList >::type::value
            )
        > source_weight;
        pointer_array_struct<
            (
                boost::mpl::size< TSourceList >::type::value
                + boost::mpl::size< TParticleList >::type::value
            )
        > pointer_array;
        minmax_array_struct<
            (
                boost::mpl::size< TSourceList >::type::value
                + boost::mpl::size< TParticleList >::type::value
            )
        > minmax_array;
        const static ISAAC_IDX_TYPE transfer_size = TTransfer_size;
        functions_struct functions[(
            boost::mpl::size< TSourceList >::type::value
            + boost::mpl::size< TParticleList >::type::value
        )];
        ISAAC_IDX_TYPE max_size;
        ISAAC_IDX_TYPE max_size_scaled;
        IceTFloat background_color[4];
        static IsaacVisualization * myself;
        TScale scale;
        clipping_struct clipping;
        isaac_float3 clipping_saved_normals[ISAAC_MAX_CLIPPING];
        TController controller;
        TCompositor compositor;
        IceTImage image[TController::pass_count];
    };

#if ISAAC_ALPAKA == 1
    template<
        typename THost,
        typename TAcc,
        typename TStream,
        typename TAccDim,
        typename TSimDim,
        typename TParticleList,
        typename TSourceList,
        typename TDomainSize, ISAAC_IDX_TYPE TTransfer_size,
        typename TScale,
        typename TController,
        typename TCompositor
    > IsaacVisualization<
        THost,
        TAcc,
        TStream,
        TAccDim,
        TSimDim,
        TParticleList,
        TSourceList,
        TDomainSize,
        TTransfer_size,
        TScale,
        TController,
        TCompositor
    > * IsaacVisualization<
        THost,
        TAcc,
        TStream,
        TAccDim,
        TSimDim,
        TParticleList,
        TSourceList,
        TDomainSize,
        TTransfer_size,
        TScale,
        TController,
        TCompositor
    >::myself = NULL;
#else
    template<
        typename TSimDim,
        typename TParticleList,
        typename TSourceList,
        typename TDomainSize, ISAAC_IDX_TYPE TTransfer_size,
        typename TScale,
        typename TController,
        typename TCompositor
    > IsaacVisualization<
        TSimDim,
        TParticleList,
        TSourceList,
        TDomainSize,
        TTransfer_size,
        TScale,
        TController,
        TCompositor
    > * IsaacVisualization<
        TSimDim,
        TParticleList,
        TSourceList,
        TDomainSize,
        TTransfer_size,
        TScale,
        TController,
        TCompositor
    >::myself = NULL;
#endif

} //namespace isaac;
