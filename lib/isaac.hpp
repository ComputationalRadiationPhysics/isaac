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




#include <alpaka/alpaka.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/not.hpp>



#include "isaac/isaac_communicator.hpp"
#include "isaac/isaac_helper.hpp"
#include "isaac/isaac_controllers.hpp"
#include "isaac/isaac_compositors.hpp"
#include "isaac/isaac_compositors.hpp"
#include "isaac/isaac_version.hpp"
// include all kernels
#include "isaac_particle_kernel.hpp"
#include "isaac_ssao_kernel.hpp"
#include "isaac_min_max_kernel.hpp"
#include "isaac_iso_kernel.hpp"
#include "isaac_volume_kernel.hpp"


namespace isaac
{

    template<
        typename T_Host,
        typename T_Acc,
        typename T_Stream,
        typename T_AccDim,
        typename T_ParticleList,
        typename T_SourceList,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_Controller,
        typename T_Compositor
    >
    class IsaacVisualization
    {
    public:

        IsaacCommunicator * getCommunicator( )
        {
            return communicator;
        }


        using DevAcc = alpaka::Dev< T_Acc >;
        using FraDim = alpaka::DimInt< 1 >;
        using TexDim = alpaka::DimInt< 1 >;

        struct source_2_json_iterator
        {
            template<
                typename T_Source,
                typename T_JsonRoot
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const T_Source & s,
                T_JsonRoot & jsonRoot
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
                    json_string( T_Source::getName( ).c_str( ) )
                );
                json_object_set_new(
                    content,
                    "feature dimension",
                    json_integer( s.featureDim )
                );
            }
        };

        struct ParticleSource2jsonIterator
        {
            template<
                typename T_Source,
                typename T_JsonRoot
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const T_Source & s,
                T_JsonRoot & jsonRoot
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
                    json_string( T_Source::getName( ).c_str( ) )
                );
                json_object_set_new(
                    content,
                    "feature dimension",
                    json_integer( 3 )
                );
            }
        };

        struct Functor2jsonIterator
        {
            template<
                typename T_Functor,
                typename T_JsonRoot
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const T_Functor & f,
                T_JsonRoot & jsonRoot
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
                    json_string( T_Functor::getName( ).c_str( ) )
                );
                json_object_set_new(
                    content,
                    "description",
                    json_string( T_Functor::getDescription( ).c_str( ) )
                );
                json_object_set_new(
                    content,
                    "uses parameter",
                    json_boolean( T_Functor::usesParameter )
                );
            }
        };

        struct ParseFunctorIterator
        {
            template<
                typename T_Functor,
                typename T_Name,
                typename T_Value,
                typename T_Found
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                T_Functor & f,
                const T_Name & name,
                T_Value & value,
                T_Found & found
            ) const
            {
                if( !found && name == T_Functor::getName( ) )
                {
                    value = I;
                    found = true;
                }
            }
        };

        struct UpdateFunctorChainIterator
        {
            template<
                typename T_Source,
                typename T_Functions,
                typename T_Offset,
                typename T_Dest
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const T_Source & source,
                const T_Functions & functions,
                const T_Offset & offset,
                T_Dest & dest
            ) const
            {
                isaac_int chain_nr = 0;
                for( int i = 0; i < ISAAC_MAX_FUNCTORS; i++ )
                {
                    chain_nr *= ISAAC_FUNCTOR_COUNT;
                    chain_nr += functions[I + offset].bytecode[i];
                }
                dest.nr[I + offset] = chain_nr * 4 + T_Source::featureDim - 1;
            }
        };

        struct AllocatePointerArrayIterator
        {
            template<
                typename T_Source,
                typename T_Array,
                typename T_LocalSize,
                typename T_Vector,
                typename T_DevAcc
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const T_Source & source,
                T_Array & pointerArray,
                const T_LocalSize & localSize,
                T_Vector & alpaka_vector,
                const T_DevAcc & acc
            ) const
            {
                if( T_Source::persistent )
                {
                    pointerArray.pointer[I] = NULL;
                }
                else
                {
                    alpaka_vector.push_back(
                        alpaka::Buf<
                            T_DevAcc,
                            isaac_float,
                            FraDim,
                            ISAAC_IDX_TYPE
                        >(
                            alpaka::allocBuf<
                                isaac_float,
                                ISAAC_IDX_TYPE
                            >(
                                acc,
                                alpaka::Vec<
                                    FraDim,
                                    ISAAC_IDX_TYPE
                                >(
                                    ISAAC_IDX_TYPE(
                                        T_Source::featureDim * (
                                            localSize[0] + 2 * ISAAC_GUARD_SIZE
                                        ) * (
                                            localSize[1] + 2 * ISAAC_GUARD_SIZE
                                        ) * (
                                            localSize[2] + 2 * ISAAC_GUARD_SIZE
                                        )
                                    )
                                )
                            )
                        )
                    );
                    pointerArray.pointer[I] =
                        alpaka::getPtrNative( alpaka_vector.back( ) );
                }
            }
        };


        struct UpdateParticleSourceIterator
        {
            /** Update iterator for particle sources
             *
             * Iterator for updating the particle sources with a boolean if the
             * Particle Source is enabled
             *
             * @tparam T_ParticleSource is the particle source type
             * @tparam T_Weight is weight type
             * @param I is the index of the current source
             * @param particleSource is the current particle source
             * @param weight is the array with all source weights
             * @param pointer is a pointer to user defined data for updating the source
             * @param weightArrayOffset is the offset in the array to the particle sources
             *
             */
            template<
                typename T_ParticleSource,
                typename T_Weight
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                T_ParticleSource & particleSource,
                const T_Weight & weight,
                void * pointer,
                const int weightArrayOffset
            ) const
            {
                bool enabled =
                    weight.value[I + weightArrayOffset] != isaac_float( 0 );
                particleSource.update(
                    enabled,
                    pointer
                );
            }

        };

        struct UpdatePointerArrayIterator
        {
            template<
                typename T_Source,
                typename T_Array,
                typename T_Weight,
                typename T_IsoTheshold,
                typename T_Stream__
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                T_Source & source,
                T_Array & pointerArray,
                const isaac_size3 & localSize,
                const T_Weight & weight,
                const T_IsoTheshold & isoThreshold,
                void * pointer,
                T_Stream__ & stream
            ) const
            {
                bool enabled = weight.value[I] != isaac_float( 0 ) || isoThreshold.value[I] != isaac_float( 0 );
                source.update(
                    enabled,
                    pointer
                );
                if( !T_Source::persistent && enabled )
                {
                    isaac_size2 gridSize = {
                        ISAAC_IDX_TYPE(
                            ( localSize[0] + ISAAC_GUARD_SIZE * 2 + 15 ) / 16
                        ),
                        ISAAC_IDX_TYPE(
                            ( localSize[1] + ISAAC_GUARD_SIZE * 2 + 15 ) / 16
                        ),
                    };
                    isaac_size2 blockSize = {
                        ISAAC_IDX_TYPE( 16 ),
                        ISAAC_IDX_TYPE( 16 ),
                    };
                    isaac_int3 localSizeArray = {
                        isaac_int( localSize[0] ),
                        isaac_int( localSize[1] ),
                        isaac_int( localSize[2] )
                    };
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
                    if ( boost::mpl::not_<boost::is_same<T_Acc, alpaka::AccGpuCudaRt<T_AccDim, ISAAC_IDX_TYPE> > >::value )
#endif
                    {
                        gridSize.x = ISAAC_IDX_TYPE(
                            localSize[0] + ISAAC_GUARD_SIZE * 2
                        );
                        gridSize.y = ISAAC_IDX_TYPE(
                            localSize[0] + ISAAC_GUARD_SIZE * 2
                        );
                        blockSize.x = ISAAC_IDX_TYPE( 1 );
                        blockSize.y = ISAAC_IDX_TYPE( 1 );
                    }
                    const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> threads(
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 )
                    );
                    const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> blocks(
                        ISAAC_IDX_TYPE( 1 ),
                        blockSize.x,
                        blockSize.y
                    );
                    const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> grid(
                        ISAAC_IDX_TYPE( 1 ),
                        gridSize.x,
                        gridSize.y
                    );
                    auto const workdiv(
                        alpaka::WorkDivMembers<
                            T_AccDim,
                            ISAAC_IDX_TYPE
                        >(
                            grid,
                            blocks,
                            threads
                        )
                    );
                    UpdateBufferKernel< T_Source > kernel;
                    auto const instance(
                        alpaka::createTaskKernel< T_Acc >(
                            workdiv,
                            kernel,
                            source,
                            pointerArray.pointer[I],
                            localSizeArray
                        )
                    );
                    alpaka::enqueue(
                        stream,
                        instance
                    );
                    alpaka::wait( stream );
                }
            }
        };

        struct CalcMinMaxIterator
        {
            template<
                typename T_Source,
                typename T_Array,
                typename T_Minmax,
                typename T_LocalMinmax,
                typename T_LocalSize,
                typename T_Stream__,
                typename T_Host__
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const T_Source & source,
                T_Array & pointerArray,
                T_Minmax & minMax,
                T_LocalMinmax & localMinMax,
                T_LocalSize & localSize,
                T_Stream__ & stream,
                const T_Host__ & host
            ) const
            {
                isaac_size2 gridSize = ( localSize + ISAAC_IDX_TYPE( 15 ) ) / ISAAC_IDX_TYPE( 16 );
                isaac_size2 blockSize( 16 );

                MinMax localMinMaxHostArray[localSize.x * localSize.y];

                if( localSize.x != 0 && localSize.y != 0 )
                {
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
                    if ( boost::mpl::not_<boost::is_same<T_Acc, alpaka::AccGpuCudaRt<T_AccDim, ISAAC_IDX_TYPE> > >::value )
#endif
                    {
                        gridSize = localSize;
                        blockSize = isaac_size2( 1 );
                    }
                    const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> threads(
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 )
                    );
                    const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> blocks(
                        ISAAC_IDX_TYPE( 1 ),
                        blockSize.x,
                        blockSize.y
                    );
                    const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> grid(
                        ISAAC_IDX_TYPE( 1 ),
                        gridSize.x,
                        gridSize.y
                    );
                    auto const workdiv(
                        alpaka::WorkDivMembers<
                            T_AccDim,
                            ISAAC_IDX_TYPE
                        >(
                            grid,
                            blocks,
                            threads
                        )
                    );
                    MinMaxKernel< T_Source > kernel;
                    auto const instance(
                        alpaka::createTaskKernel< T_Acc >(
                            workdiv,
                            kernel,
                            source,
                            I,
                            alpaka::getPtrNative( localMinMax ),
                            localSize,
                            pointerArray.pointer[I]
                        )
                    );
                    alpaka::enqueue(
                        stream,
                        instance
                    );
                    alpaka::wait( stream );
                    alpaka::ViewPlainPtr <T_Host, MinMax, FraDim, ISAAC_IDX_TYPE> minMaxBuffer(
                        localMinMaxHostArray,
                        host,
                        alpaka::Vec<
                            FraDim,
                            ISAAC_IDX_TYPE
                        >( localSize.x * localSize.y )
                    );
                    alpaka::memcpy(
                        stream,
                        minMaxBuffer,
                        localMinMax,
                        alpaka::Vec<
                            FraDim,
                            ISAAC_IDX_TYPE
                        >( localSize.x * localSize.y )
                    );
                }
                minMax.min[I] = std::numeric_limits<isaac_float>::max();
                minMax.max[I] = -std::numeric_limits<isaac_float>::max();
                for( ISAAC_IDX_TYPE i = 0; i < localSize.x * localSize.y; i++ )
                {
                    if( localMinMaxHostArray[i].min < minMax.min[I] )
                    {
                        minMax.min[I] = localMinMaxHostArray[i].min;
                    }
                    if( localMinMaxHostArray[i].max > minMax.max[I] )
                    {
                        minMax.max[I] = localMinMaxHostArray[i].max;
                    }
                }
            }
        };

        // calculate minMax for particles
        template<
            int T_offset
        >
        struct CalcParticleMinMaxIterator
        {
            template<
                typename T_ParticleSource,
                typename T_Minmax,
                typename T_LocalMinmax,
                typename T_LocalSize,
                typename T_Stream__,
                typename T_Host__
            >
            ISAAC_HOST_INLINE void operator()(
                const int I,
                const T_ParticleSource & particleSource,
                T_Minmax & minMax,
                T_LocalMinmax & localMinMax,
                T_LocalSize & localSize,
                T_Stream__ & stream,
                const T_Host__ & host
            ) const
            {
                // iterate over all cells and the particle lists

                isaac_size2 gridSize = ( localSize + ISAAC_IDX_TYPE( 15 ) ) / ISAAC_IDX_TYPE( 16 );
                isaac_size2 blockSize( 16 );

                MinMax localMinMaxHostArray[localSize.x * localSize.y];
                
                if( localSize.x != 0 && localSize.y != 0 )
                {
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
                    if ( boost::mpl::not_<boost::is_same<T_Acc, alpaka::AccGpuCudaRt<T_AccDim, ISAAC_IDX_TYPE> > >::value )
#endif
                    {
                        gridSize = localSize;
                        blockSize = isaac_size2( 1 );
                    }
                    const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> threads(
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 ),
                        ISAAC_IDX_TYPE( 1 )
                    );
                    const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> blocks(
                        ISAAC_IDX_TYPE( 1 ),
                        blockSize.x,
                        blockSize.y
                    );
                    const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> grid(
                        ISAAC_IDX_TYPE( 1 ),
                        gridSize.x,
                        gridSize.y
                    );
                    auto const workdiv(
                        alpaka::WorkDivMembers<
                            T_AccDim,
                            ISAAC_IDX_TYPE
                        >(
                            grid,
                            blocks,
                            threads
                        )
                    );
                    MinMaxParticleKernel< T_ParticleSource > kernel;
                    auto const instance(
                        alpaka::createTaskKernel< T_Acc >(
                            workdiv,
                            kernel,
                            particleSource,
                            I + T_offset,
                            alpaka::getPtrNative( localMinMax ),
                            localSize
                        )
                    );
                    alpaka::enqueue(
                        stream,
                        instance
                    );
                    alpaka::wait( stream );
                    alpaka::ViewPlainPtr <T_Host, MinMax, FraDim, ISAAC_IDX_TYPE> minMaxBuffer(
                        localMinMaxHostArray,
                        host,
                        alpaka::Vec<
                            FraDim,
                            ISAAC_IDX_TYPE
                        >( localSize.x * localSize.y )
                    );
                    alpaka::memcpy(
                        stream,
                        minMaxBuffer,
                        localMinMax,
                        alpaka::Vec<
                            FraDim,
                            ISAAC_IDX_TYPE
                        >( localSize.x * localSize.y )
                    );
                }
                minMax.min[I + T_offset] = std::numeric_limits<isaac_float>::max();
                minMax.max[I + T_offset] = -std::numeric_limits<isaac_float>::max();
                // find the min and max
                for( ISAAC_IDX_TYPE i = 0; i < localSize.x * localSize.y; i++ )
                {
                    if( localMinMaxHostArray[i].min < minMax.min[I + T_offset] )
                    {
                        minMax.min[I + T_offset] = localMinMaxHostArray[i].min;
                    }
                    if( localMinMaxHostArray[i].max > minMax.max[I + T_offset] )
                    {
                        minMax.max[I + T_offset] = localMinMaxHostArray[i].max;
                    }
                }
            }
        };


        IsaacVisualization(
            T_Host host,
            DevAcc acc,
            T_Stream stream,
            const std::string name,
            const isaac_int master,
            const std::string serverUrl,
            const isaac_uint serverPort,
            const isaac_size2 framebufferSize,
            const isaac_size3 globalSize,
            const isaac_size3 localSize,
            const isaac_size3 localParticleSize,
            const isaac_size3 position,
            T_ParticleList & particleSources,
            T_SourceList & sources,
            isaac_float3 scale

        ) :
            host( host ),
            acc( acc ),
            stream( stream ),
            globalSize( globalSize ),
            localSize( localSize ),
            localParticleSize( localParticleSize ),
            position( position ),
            name( name ),
            master( master ),
            serverUrl( serverUrl ),
            serverPort( serverPort ),
            framebufferSize( framebufferSize ),
            compbufferSize( T_Compositor::getCompositedbufferSize( framebufferSize ) ),
            compositor( framebufferSize ),
            metaNr( 0 ),
            visualizationThread( 0 ),
            kernelTime( 0 ),
            mergeTime( 0 ),
            videoSendTime( 0 ),
            copyTime( 0 ),
            sortingTime( 0 ),
            bufferTime( 0 ),
            interpolation( false ),
            step( isaac_float( ISAAC_DEFAULT_STEP ) ),
            framebufferProd(
                ISAAC_IDX_TYPE( framebufferSize.x )
                * ISAAC_IDX_TYPE( framebufferSize.y )
            ),
            particleSources( particleSources ),
            sources( sources ),
            scale( scale ),
            icetBoundingBox( true ),
            framebuffer(
                alpaka::allocBuf<
                    uint32_t,
                    ISAAC_IDX_TYPE
                >(
                    acc,
                    framebufferProd
                )
            ),
            framebufferAO(
                alpaka::allocBuf<
                    isaac_float,
                    ISAAC_IDX_TYPE
                >(
                    acc,
                    framebufferProd
                )
            ),
            framebufferDepth(
                alpaka::allocBuf<
                    isaac_float,
                    ISAAC_IDX_TYPE
                >(
                    acc,
                    framebufferProd
                )
            ),
            framebufferNormal(
                alpaka::allocBuf<
                    isaac_float3,
                    ISAAC_IDX_TYPE
                >(
                    acc,
                    framebufferProd
                )
            ),
            functor_chain_d(
                alpaka::allocBuf<
                    FunctorChainPointerN,
                    ISAAC_IDX_TYPE
                >(
                    acc,
                    ISAAC_IDX_TYPE( ISAAC_FUNCTOR_COMPLEX * 4 ) ) )
            ,

            functorChainChooseDevice ( alpaka::allocBuf<
                FunctorChainPointerN,
                ISAAC_IDX_TYPE
            >(
                acc,
                ISAAC_IDX_TYPE ( ( boost
                ::mpl::size< T_SourceList >::type::value
                + boost::mpl::size< T_ParticleList >::type::value ) ) ) )
            ,

            localMinMaxArrayDevice ( alpaka::allocBuf<
                MinMax,
                ISAAC_IDX_TYPE
            >(
                acc,
                ISAAC_IDX_TYPE ( localSize[0]
                * localSize[1] ) ) )
            ,

            localParticleMinMaxArrayDevice ( alpaka::allocBuf<
                MinMax,
                ISAAC_IDX_TYPE
            >(
                acc,
                ISAAC_IDX_TYPE ( localParticleSize[0]
                * localParticleSize[1] ) ) )
        {
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
            globalSizeScaled = isaac_float3( globalSize ) * scale;
            localSizeScaled = isaac_float3( localSize ) * scale;
            positionScaled = isaac_float3( position ) * scale;

            backgroundColor[0] = 0;
            backgroundColor[1] = 0;
            backgroundColor[2] = 0;
            backgroundColor[3] = 1;

            //INIT
            MPI_Comm_dup(
                MPI_COMM_WORLD,
                &mpiWorld
            );
            MPI_Comm_rank(
                mpiWorld,
                &rank
            );
            MPI_Comm_size(
                mpiWorld,
                &numProc
            );
            if( rank == master )
            {
                this->communicator = new IsaacCommunicator(
                    serverUrl,
                    serverPort
                );
            }
            else
            {
                this->communicator = NULL;
            }
            recreateJSON( );
            for(int i = 0; i < T_Controller::passCount; ++i)
            {
                projections.push_back(isaac_dmat4(1));
            }
            controller.updateProjection(
                projections,
                framebufferSize,
                NULL,
                true
            );
            lookAt = isaac_double3(0);
            rotation = isaac_dmat3(1);
            distance = -4.5f;
            updateModelview( );

            //Create functor chain pointer lookup table
            const alpaka::Vec<
                T_AccDim,
                ISAAC_IDX_TYPE
            > threads(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            const alpaka::Vec<
                T_AccDim,
                ISAAC_IDX_TYPE
            > blocks(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            const alpaka::Vec<
                T_AccDim,
                ISAAC_IDX_TYPE
            > grid(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            auto const workdiv(
                alpaka::WorkDivMembers<
                    T_AccDim,
                    ISAAC_IDX_TYPE
                >(
                    grid,
                    blocks,
                    threads
                )
            );
            FillFunctorChainPointerKernel kernel;
            auto const instance(
                alpaka::createTaskKernel< T_Acc >(
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
            //Init functions:
            for( int i = 0; i < (
                boost::mpl::size< T_SourceList >::type::value
                + boost::mpl::size< T_ParticleList >::type::value
            ); i++ )
            {
                functions[i].source = std::string( "idem" );
            }
            updateFunctions( );

            //non persistent buffer memory
            forEachParams(
                sources,
                AllocatePointerArrayIterator( ),
                pointerArray,
                localSize,
                pointerArrayAlpaka,
                acc
            );

            //Transfer func memory:
            for( int i = 0; i < (
                boost::mpl::size< T_SourceList >::type::value
                + boost::mpl::size< T_ParticleList >::type::value
            ); i++ )
            {
                sourceWeight.value[i] = ISAAC_DEFAULT_WEIGHT;
                transferDeviceBuf.push_back(
                    alpaka::Buf<
                        DevAcc,
                        isaac_float4,
                        TexDim,
                        ISAAC_IDX_TYPE
                    >(
                        alpaka::allocBuf<
                            isaac_float4,
                            ISAAC_IDX_TYPE
                        >(
                            acc,
                            alpaka::Vec<
                                TexDim,
                                ISAAC_IDX_TYPE
                            >( T_transferSize )
                        )
                    )
                );
                transferHostBuf.push_back(
                    alpaka::Buf<
                        T_Host,
                        isaac_float4,
                        TexDim,
                        ISAAC_IDX_TYPE
                    >(
                        alpaka::allocBuf<
                            isaac_float4,
                            ISAAC_IDX_TYPE
                        >(
                            host,
                            alpaka::Vec<
                                TexDim,
                                ISAAC_IDX_TYPE
                            >( T_transferSize )
                        )
                    )
                );
                transferDevice.pointer[i] =
                    alpaka::getPtrNative( transferDeviceBuf[i] );
                transferHost.pointer[i] =
                    alpaka::getPtrNative( transferHostBuf[i] );
                //Init volume transfer func with a alpha ramp from 0 -> 1
                if( i < boost::mpl::size< T_SourceList >::type::value )
                {
                    sourceIsoThreshold.value[i] = isaac_float( 0 );
                    transferHost.description[i].insert(
                        std::pair<
                            isaac_uint,
                            isaac_float4
                        >(
                            0,
                            getHSVA(
                                isaac_float( 2 * i ) * M_PI / isaac_float(
                                    (
                                        boost::mpl::size< T_SourceList >::type::value
                                        + boost::mpl::size< T_ParticleList >::type::value
                                    )
                                ),
                                1,
                                1,
                                0
                            )
                        )
                    );
                    transferHost.description[i].insert(
                        std::pair<
                            isaac_uint,
                            isaac_float4
                        >(
                            T_transferSize,
                            getHSVA(
                                isaac_float( 2 * i ) * M_PI / isaac_float(
                                    (
                                        boost::mpl::size< T_SourceList >::type::value
                                        + boost::mpl::size< T_ParticleList >::type::value
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
                    transferHost.description[i].insert(
                        std::pair<
                            isaac_uint,
                            isaac_float4
                        >(
                            0,
                            getHSVA(
                                isaac_float( 2 * i ) * M_PI / isaac_float(
                                    (
                                        boost::mpl::size< T_SourceList >::type::value
                                        + boost::mpl::size< T_ParticleList >::type::value
                                    )
                                ),
                                1,
                                1,
                                1
                            )
                        )
                    );
                    transferHost.description[i].insert(
                        std::pair<
                            isaac_uint,
                            isaac_float4
                        >(
                            T_transferSize,
                            getHSVA(
                                isaac_float( 2 * i ) * M_PI / isaac_float(
                                    (
                                        boost::mpl::size< T_SourceList >::type::value
                                        + boost::mpl::size< T_ParticleList >::type::value
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

            maxSize = glm::max( globalSize.x , glm::max( globalSize.y, globalSize.z ) );
            maxSizeScaled = glm::max( globalSizeScaled.x , glm::max( globalSizeScaled.y, globalSizeScaled.z ) );

            //ICET:
            IceTCommunicator icetComm;
            icetComm = icetCreateMPICommunicator( mpiWorld );
            for( int pass = 0; pass < T_Controller::passCount; pass++ )
            {
                icetContext[pass] = icetCreateContext( icetComm );
                icetResetTiles( );
                icetAddTile(
                    0,
                    0,
                    framebufferSize.x,
                    framebufferSize.y,
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
                    framebufferSize.x,
                    framebufferSize.y
                );
                icetDrawCallback( drawCallBack );
            }
            icetDestroyMPICommunicator( icetComm );
            updateBounding( );

            //JSON
            if( rank == master )
            {
                json_object_set_new(
                    jsonRoot,
                    "type",
                    json_string( "register" )
                );
                json_object_set_new(
                    jsonRoot,
                    "name",
                    json_string( name.c_str( ) )
                );
                json_object_set_new(
                    jsonRoot,
                    "nodes",
                    json_integer( numProc )
                );
                json_object_set_new(
                    jsonRoot,
                    "framebuffer width",
                    json_integer( compbufferSize.x )
                );
                json_object_set_new(
                    jsonRoot,
                    "framebuffer height",
                    json_integer( compbufferSize.y )
                );

                json_object_set_new(
                    jsonRoot,
                    "max functors",
                    json_integer( ISAAC_MAX_FUNCTORS )
                );
                json_t * jsonFunctorsArray = json_array( );
                json_object_set_new(
                    jsonRoot,
                    "functors",
                    jsonFunctorsArray
                );
                IsaacFunctorPool functors;
                forEachParams(
                    functors,
                    Functor2jsonIterator( ),
                    jsonFunctorsArray
                );

                json_t * matrix;
                json_object_set_new(
                    jsonRoot,
                    "projection count",
                    json_integer( T_Controller::passCount )
                );
                json_object_set_new(
                    jsonRoot,
                    "projection",
                    matrix = json_array( )
                );
                for(isaac_int p = 0; p < T_Controller::passCount; ++p)
                {
                    for (isaac_int i = 0; i < 16; ++i)
                    {
                        json_array_append_new( matrix, json_real( glm::value_ptr( projections[p] )[i] ) );
                    }
                }
                json_object_set_new(
                    jsonRoot,
                    "position",
                    matrix = json_array( )
                );
                for (isaac_int i = 0; i < 3; i++)
                {
                    json_array_append_new( matrix, json_real( glm::value_ptr( lookAt )[i] ) );
                }
                json_object_set_new(
                    jsonRoot,
                    "rotation",
                    matrix = json_array( )
                );
                for (isaac_int i = 0; i < 9; i++)
                {
                    json_array_append_new( matrix, json_real( glm::value_ptr( rotation )[i] ) );
                }
                json_object_set_new(
                    jsonRoot,
                    "distance",
                    json_real( distance )
                );

                json_t * jsonSourcesArray = json_array( );
                json_object_set_new(
                    jsonRoot,
                    "sources",
                    jsonSourcesArray
                );

                forEachParams(
                    sources,
                    source_2_json_iterator( ),
                    jsonSourcesArray
                );
                forEachParams(
                    particleSources,
                    source_2_json_iterator( ),
                    jsonSourcesArray
                );

                json_object_set_new(
                    jsonRoot,
                    "interpolation",
                    json_boolean( interpolation )
                );
                json_object_set_new(
                    jsonRoot,
                    "step",
                    json_real( step )
                );

                json_object_set_new(
                    jsonRoot,
                    "dimension",
                    json_integer( 3 )
                );
                json_object_set_new(
                    jsonRoot,
                    "width",
                    json_integer( globalSizeScaled.x )
                );
                json_object_set_new(
                    jsonRoot,
                    "height",
                    json_integer( globalSizeScaled.y )
                );
                json_object_set_new(
                    jsonRoot,
                    "depth",
                    json_integer( globalSizeScaled.z )
                );
                json_t * jsonVersionArray = json_array( );
                json_array_append_new(
                    jsonVersionArray,
                    json_integer( ISAAC_PROTOCOL_VERSION_MAJOR )
                );
                json_array_append_new(
                    jsonVersionArray,
                    json_integer( ISAAC_PROTOCOL_VERSION_MINOR )
                );
                json_object_set_new(
                    jsonRoot,
                    "protocol",
                    jsonVersionArray
                );

                //send inital ambientOcclusion settings
                json_object_set_new(
                    jsonRoot, 
                    "ao isEnabled", 
                    json_boolean(ambientOcclusion.isEnabled)
                );
                json_object_set_new(
                    jsonRoot, 
                    "ao weight", 
                    json_real(ambientOcclusion.weight)
                ); 
            }

            //allocate ssao kernel (16x16 matrix)
            alpaka::Buf<
                T_Host, 
                isaac_float3, 
                FraDim, 
                ISAAC_IDX_TYPE
            > ssaoKernelHostBuf (
                alpaka::allocBuf<
                    isaac_float3, 
                    ISAAC_IDX_TYPE
                > (
                    host, 
                    ISAAC_IDX_TYPE(64)
                )
            );
            isaac_float3* ssaoKernelHost = reinterpret_cast<isaac_float3*> ( alpaka::getPtrNative ( ssaoKernelHostBuf ) );

            //allocate ssao noise kernel (4x4 matrix)
            alpaka::Buf<
                T_Host, 
                isaac_float3, 
                FraDim, 
                ISAAC_IDX_TYPE
            > ssaoNoiseHostBuf (
                alpaka::allocBuf<
                    isaac_float3, 
                    ISAAC_IDX_TYPE
                > (
                    host, 
                    ISAAC_IDX_TYPE(16)
                )
            );
            isaac_float3* ssaoNoiseHost = reinterpret_cast<isaac_float3*> ( alpaka::getPtrNative ( ssaoNoiseHostBuf ) );

            
            std::uniform_real_distribution<isaac_float> randomFloats(0.0, 1.0);
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
                ssaoKernelHost[i] = sample;
            }

            //set ssao_noise values
            for(unsigned int i = 0; i < 16; i++) {
                isaac_float3 noise({
                    randomFloats(generator) * 2.0f - 1.0f,
                    randomFloats(generator) * 2.0f - 1.0f,
                    0.0f
                });
                ssaoNoiseHost[i] = noise;
            }

            //move ssao kernel to device
            //copy ssao kernel to constant memory
            alpaka::Vec<alpaka::DimInt<1u>, ISAAC_IDX_TYPE> const 
                ssaoKernelDeviceExtent(ISAAC_IDX_TYPE(64));

            auto ssaoKernelDeviceView (
                alpaka::createStaticDevMemView(
                    &SSAOKernelArray[0u], 
                    acc, 
                    ssaoKernelDeviceExtent
                )
            );
            alpaka::memcpy(
                stream, 
                ssaoKernelDeviceView, 
                ssaoKernelHostBuf, 
                ISAAC_IDX_TYPE(64)
            );

            //copy ssao noise kernel to constant memory
            alpaka::Vec<alpaka::DimInt<1u>, ISAAC_IDX_TYPE> const 
                ssaoNoiseDeviceExtent(ISAAC_IDX_TYPE(16));

            auto ssaoNoiseDeviceView ( 
                alpaka::createStaticDevMemView(
                    &SSAONoiseArray[0u], 
                    acc, 
                    ssaoNoiseDeviceExtent
                )
            );
            alpaka::memcpy ( 
                stream, 
                ssaoNoiseDeviceView, 
                ssaoNoiseHostBuf, 
                ISAAC_IDX_TYPE(16)
            );

        }


        void setJpegQuality( isaac_uint jpegQuality )
        {
            ISAAC_WAIT_VISUALIZATION
            if( communicator )
            {
                communicator->setJpegQuality( jpegQuality );
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
            isaac_float3 n = isaac_float3( nx, ny, nz );
            isaac_float3 nScaled = n * scale;
            isaac_float l = glm::length( nScaled );
            if( l == 0.0f )
            {
                return false;
            }
            nScaled /= l;
            clipping.elem[nr].position
                .x = px;
            clipping.elem[nr].position
                .y = py;
            clipping.elem[nr].position
                .z = pz;
            clipping.elem[nr].normal = nScaled;
            clippingSavedNormals[nr] = n;
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
                clippingSavedNormals[i] = clippingSavedNormals[i + 1];
            }
        }


        void updateBounding( )
        {
            ISAAC_WAIT_VISUALIZATION
            for( int pass = 0; pass < T_Controller::passCount; pass++ )
            {
                icetSetContext( icetContext[pass] );
                if( icetBoundingBox )
                {
                    isaac_float3 bbDimension = isaac_float3( localSizeScaled ) / isaac_float( maxSizeScaled ) * isaac_float( 2 );

                    isaac_float3 bbMin = isaac_float3( positionScaled ) / isaac_float( maxSizeScaled ) * isaac_float( 2 )
                                    - isaac_float3( globalSizeScaled ) / isaac_float( maxSizeScaled );

                    icetBoundingBoxf(
                        bbMin.x,
                        bbMin.x + bbDimension.x,
                        bbMin.y,
                        bbMin.y + bbDimension.y,
                        bbMin.z,
                        bbMin.z + bbDimension.z
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


        void updatePosition( const isaac_size3 position )
        {
            ISAAC_WAIT_VISUALIZATION
            this->position = position;
            this->positionScaled = isaac_float3( position ) * scale;
        }


        void updateLocalSize( const isaac_size3 localSize )
        {
            ISAAC_WAIT_VISUALIZATION
            this->localSize = localSize;
            this->localSizeScaled = isaac_float3( localSize ) * scale;
        }


        void updateLocalParticleSize( const isaac_size3 localParticleSize )
        {
            ISAAC_WAIT_VISUALIZATION
            this->localParticleSize = localParticleSize;
        }


        void updateFunctions( )
        {
            ISAAC_WAIT_VISUALIZATION
            IsaacFunctorPool functors;
            isaac_float4 functorParameterHost[(
                                               boost::mpl::size< T_SourceList >::type::value
                                               + boost::mpl::size< T_ParticleList >::type::value
                                           ) * ISAAC_MAX_FUNCTORS];
            for( int i = 0; i < (
                boost::mpl::size< T_SourceList >::type::value
                + boost::mpl::size< T_ParticleList >::type::value
            ); i++ )
            {
                functions[i].errorCode = 0;
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
                        functions[i].errorCode = 1;
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
                                functorParameterHost[i * ISAAC_MAX_FUNCTORS + elem]
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
                            functions[i].errorCode = -1;
                            break;
                        }
                        if( t_end - t_begin == 1 )
                        { //()
                            memset(
                                &(
                                    functorParameterHost[i * ISAAC_MAX_FUNCTORS
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
                            size_t pPos = 0;
                            bool pAgain = true;
                            int pElem = 0;
                            isaac_float * parameterArray = ( isaac_float * ) &(
                                functorParameterHost[i * ISAAC_MAX_FUNCTORS + elem]
                            );
                            while( pAgain && (
                                ( pPos = parameters.find( "," ) )
                                != std::string::npos
                                || ( ( pAgain = false ) == false )
                            ) )
                            {
                                if( pElem >= 4 )
                                {
                                    functions[i].errorCode = 2;
                                    break;
                                }
                                std::string par = parameters.substr(
                                    0,
                                    pPos
                                );
                                parameters.erase(
                                    0,
                                    pPos + 1
                                );
                                try
                                {
                                    parameterArray[pElem] = std::stof( par );
                                } catch( const std::invalid_argument & ia )
                                {
                                    std::cerr << "Invalid argument: "
                                              << ia.what( ) << '\n';
                                    functions[i].errorCode = -2;
                                    pElem++;
                                    break;
                                } catch( const std::out_of_range & oor )
                                {
                                    std::cerr << "Out of range: " << oor.what( )
                                              << '\n';
                                    functions[i].errorCode = -2;
                                    pElem++;
                                    break;
                                }

                                pElem++;
                            }
                            for( ; pElem < 4; pElem++ )
                            {
                                parameterArray[pElem] = parameterArray[pElem
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
                    forEachParams(
                        functors,
                        ParseFunctorIterator( ),
                        token,
                        functions[i].bytecode[elem],
                        found
                    );
                    if( !found )
                    {
                        functions[i].errorCode = -2;
                        break;
                    }

                    elem++;
                }
                for( ; elem < ISAAC_MAX_FUNCTORS; elem++ )
                {
                    functions[i].bytecode[elem] = 0; //last one idem
                    memset(
                        &(
                            functorParameterHost[i * ISAAC_MAX_FUNCTORS + elem]
                        ),
                        0,
                        sizeof( isaac_float4 )
                    );
                }
            }

            //Calculate functor chain nr per source
            DestArrayStruct<
                (
                    boost::mpl::size< T_SourceList >::type::value
                    + boost::mpl::size< T_ParticleList >::type::value
                )
            > dest;
            int zero = 0;
            forEachParams(
                sources,
                UpdateFunctorChainIterator( ),
                functions,
                zero,
                dest
            );
            forEachParams(
                particleSources,
                UpdateFunctorChainIterator( ),
                functions,
                boost::mpl::size< T_SourceList >::type::value,
                dest
            );
            alpaka::ViewPlainPtr <T_Host, isaac_float4, FraDim, ISAAC_IDX_TYPE>
                parameterBuffer(
                functorParameterHost,
                host,
                alpaka::Vec<
                    FraDim,
                    ISAAC_IDX_TYPE
                >(
                    ISAAC_IDX_TYPE(
                        ISAAC_MAX_FUNCTORS * (
                            boost::mpl::size< T_SourceList >::type::value
                            + boost::mpl::size< T_ParticleList >::type::value
                        )
                    )
                )
            );

            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                parameterDeviceExtent( ISAAC_IDX_TYPE( 16 ) );
            auto parameterDeviceView(
                alpaka::createStaticDevMemView ( & FunctorParameter[0u],
                acc,
                parameterDeviceExtent
            ) );
            alpaka::memcpy(
                stream,
                parameterDeviceView,
                parameterBuffer,
                alpaka::Vec<
                    FraDim,
                    ISAAC_IDX_TYPE
                >(
                    ISAAC_IDX_TYPE(
                        ISAAC_MAX_FUNCTORS * (
                            boost::mpl::size< T_SourceList >::type::value
                            + boost::mpl::size< T_ParticleList >::type::value
                        )
                    )
                )
            );

            const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> threads(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> blocks(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            const alpaka::Vec <T_AccDim, ISAAC_IDX_TYPE> grid(
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 ),
                ISAAC_IDX_TYPE( 1 )
            );
            auto const workdiv(
                alpaka::WorkDivMembers<
                    T_AccDim,
                    ISAAC_IDX_TYPE
                >(
                    grid,
                    blocks,
                    threads
                )
            );
            UpdateFunctorChainPointerKernel<
                (
                    boost::mpl::size< T_SourceList >::type::value
                    + boost::mpl::size< T_ParticleList >::type::value
                ),
                DestArrayStruct<
                    (
                        boost::mpl::size< T_SourceList >::type::value
                        + boost::mpl::size< T_ParticleList >::type::value
                    )
                >
            > kernel;
            auto const instance(
                alpaka::createTaskKernel< T_Acc >(
                    workdiv,
                    kernel,
                    alpaka::getPtrNative( functorChainChooseDevice ),
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
                functionChainDeviceExtent( ISAAC_IDX_TYPE( ISAAC_MAX_SOURCES ) );
            auto functionChainDeviceView(
                alpaka::createStaticDevMemView ( & FunctionChain[0u],
                acc,
                functionChainDeviceExtent
            ) );
            alpaka::memcpy(
                stream,
                functionChainDeviceView,
                functorChainChooseDevice,
                ISAAC_IDX_TYPE(
                    (
                        boost::mpl::size< T_SourceList >::type::value
                        + boost::mpl::size< T_ParticleList >::type::value
                    )
                )
            );
        }


        void updateTransfer( )
        {
            ISAAC_WAIT_VISUALIZATION
            for( int i = 0; i < (
                boost::mpl::size< T_SourceList >::type::value
                + boost::mpl::size< T_ParticleList >::type::value
            ); i++ )
            {
                auto next = transferHost.description[i].begin( );
                auto before = next;
                for( next++; next != transferHost.description[i].end( ); next++ )
                {
                    isaac_uint width = next->first - before->first;
                    for( ISAAC_IDX_TYPE j = 0;
                        j < ISAAC_IDX_TYPE( width )
                        && ISAAC_IDX_TYPE( j + before->first )
                           < ISAAC_IDX_TYPE( T_transferSize );
                        j++ )
                    {
                        transferHost.pointer[i][before->first + j] = (
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
                alpaka::memcpy(
                    stream,
                    transferDeviceBuf[i],
                    transferHostBuf[i],
                    T_transferSize
                );
            }
        }


        json_t * getJsonMetaRoot( )
        {
            ISAAC_WAIT_VISUALIZATION
            return jsonMetaRoot;
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
                mpiWorld
            );
            if( failed )
            {
                return -1;
            }
            if( rank == master )
            {
                jsonInitRoot = jsonRoot;
                communicator->serverSendRegister( &jsonInitRoot );
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
                forEachParams(
                    sources,
                    UpdatePointerArrayIterator( ),
                    pointerArray,
                    localSize,
                    sourceWeight,
                    sourceIsoThreshold,
                    pointer,
                    stream
                );
                forEachParams(
                    particleSources,
                    UpdateParticleSourceIterator( ),
                    sourceWeight,
                    pointer,
                    boost::mpl::size< T_SourceList >::type::value
                );
                ISAAC_STOP_TIME_MEASUREMENT ( bufferTime,
                    +=,
                    buffer,
                    getTicksUs( ) )

            }
            ISAAC_WAIT_VISUALIZATION

            myself = this;

            sendDistance = false;
            sendLookAt = false;
            sendProjection = false;
            sendRotation = false;
            sendTransfer = false;
            sendInterpolation = false;
            sendStep = false;
            sendIsoThreshold = false;
            sendFunctions = false;
            sendWeight = false;
            sendMinMax = false;
            sendBackgroundColor = false;
            sendClipping = false;
            sendController = false;
            sendInitJson = false;
            sendAO = false;

            //Handle messages
            json_t * message;
            char messageBuffer[ISAAC_MAX_RECEIVE] = "{}";
            //Master merges all messages and broadcasts it.

            if( rank == master )
            {
                message = json_object( );
                bool addModelview = false;
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
                            sendRotation = true;
                        }
                        if( strcmp(
                            target,
                            "position"
                        ) == 0 )
                        {
                            sendLookAt = true;
                        }
                        if( strcmp(
                            target,
                            "distance"
                        ) == 0 )
                        {
                            sendDistance = true;
                        }
                        if( strcmp(
                            target,
                            "projection"
                        ) == 0 )
                        {
                            sendProjection = true;
                        }
                        if( strcmp(
                            target,
                            "transfer"
                        ) == 0 )
                        {
                            sendTransfer = true;
                        }
                        if( strcmp(
                            target,
                            "interpolation"
                        ) == 0 )
                        {
                            sendInterpolation = true;
                        }
                        if( strcmp(
                            target,
                            "step"
                        ) == 0 )
                        {
                            sendStep = true;
                        }
                        if( strcmp(
                            target,
                            "iso mask"
                        ) == 0 )
                        {
                            sendIsoThreshold = true;
                        }
                        if( strcmp(
                            target,
                            "functions"
                        ) == 0 )
                        {
                            sendFunctions = true;
                        }
                        if( strcmp(
                            target,
                            "weight"
                        ) == 0 )
                        {
                            sendWeight = true;
                        }
                        if( strcmp(
                            target,
                            "background color"
                        ) == 0 )
                        {
                            sendBackgroundColor = true;
                        }
                        if( strcmp(
                            target,
                            "clipping"
                        ) == 0 )
                        {
                            sendClipping = true;
                        }
                        if( strcmp(
                            target,
                            "controller"
                        ) == 0 )
                        {
                            sendController = true;
                        }
                        if( strcmp(
                            target,
                            "init"
                        ) == 0 )
                        {
                            sendInitJson = true;
                        }
                        if(strcmp(target, "ao") == 0) {
                            sendAO = true;
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
                        addModelview = true;
                        sendRotation = true;
                        json_array_foreach(
                            js,
                            index,
                            value
                        )
                        glm::value_ptr( rotation )[index] = json_number_value( value );
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
                        addModelview = true;
                        sendRotation = true;
                        isaac_dmat3 relative;
                        json_array_foreach(
                            js,
                            index,
                            value
                        )
                        glm::value_ptr( relative )[index] = json_number_value( value );
                        rotation = relative * rotation;

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
                        isaac_double3 rotVec;
                        rotVec.x = json_number_value(
                            json_array_get(
                                js,
                                0
                            )
                        );
                        rotVec.y = json_number_value(
                            json_array_get(
                                js,
                                1
                            )
                        );
                        rotVec.z = json_number_value(
                            json_array_get(
                                js,
                                2
                            )
                        );
                        isaac_double deg = json_number_value(
                            json_array_get(
                                js,
                                3
                            )
                        );

                        isaac_dmat3 relative = glm::rotate( isaac_dmat4( 1 ), glm::radians( deg ), rotVec );
                        rotation = relative * rotation;

                        addModelview = true;
                        sendRotation = true;

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
                        addModelview = true;
                        sendLookAt = true;
                        json_array_foreach(
                            js,
                            index,
                            value
                        )
                        lookAt[index] = json_number_value( value );
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
                        addModelview = true;
                        sendLookAt = true;
                        isaac_double3 translation;
                        json_array_foreach(
                            js,
                            index,
                            value
                        )
                        translation[index] = json_number_value( value );

                        lookAt += glm::transpose( rotation ) * translation;
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
                        addModelview = true;
                        sendDistance = true;
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
                        addModelview = true;
                        sendDistance = true;
                        distance += json_number_value( js );
                        json_object_del(
                            last,
                            "distance relative"
                        );
                    }
                    //Giving the Controller the chance to grep for controller specific messages:
                    if( controller.updateProjection(
                        projections,
                        framebufferSize,
                        last
                    ) )
                    {
                        redraw = true;
                        sendProjection = true;
                        json_t * matrix;
                        json_object_set_new(
                            message,
                            "projection",
                            matrix = json_array( )
                        );
                        for(isaac_int p = 0; p < T_Controller::passCount; ++p)
                        {
                            for (isaac_int i = 0; i < 16; ++i)
                            {
                                json_array_append_new( matrix, json_real( glm::value_ptr( projections[p] )[i] ) );
                            }
                        }
                    }
                    mergeJSON(
                        message,
                        last
                    );
                    json_decref( last );
                }
                if( addModelview )
                {
                    redraw = true;
                    updateModelview( );
                    json_t * matrix;
                    json_object_set_new(
                        message,
                        "modelview",
                        matrix = json_array( )
                    );
                    for (isaac_int i = 0; i < 16; i++)
                    {
                        json_array_append_new( matrix, json_real( glm::value_ptr( modelview )[i] ) );
                    }
                }
                char * buffer = json_dumps(
                    message,
                    0
                );
                strncpy(
                    messageBuffer,
                    buffer,
                    ISAAC_MAX_RECEIVE - 1
                );
                messageBuffer[ISAAC_MAX_RECEIVE - 1] = 0;
                free( buffer );
                int l = strlen( messageBuffer );
                MPI_Bcast(
                    &l,
                    1,
                    MPI_INT,
                    master,
                    mpiWorld
                );
                MPI_Bcast(
                    messageBuffer,
                    l,
                    MPI_CHAR,
                    master,
                    mpiWorld
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
                    mpiWorld
                );
                MPI_Bcast(
                    messageBuffer,
                    l,
                    MPI_CHAR,
                    master,
                    mpiWorld
                );
                messageBuffer[l] = 0;
                message = json_loads(
                    messageBuffer,
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
                    sendMinMax = true;
                }
            }

            //Scene set?
            if( json_array_size(
                js = json_object_get(
                    message,
                    "projection"
                )
            ) == 16 * T_Controller::passCount )
            {
                redraw = true;
                sendProjection = true;
                json_array_foreach(
                    js,
                    index,
                    value
                )
                glm::value_ptr( projections[index / 16] )[index % 16] = json_number_value( value );
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
                glm::value_ptr( modelview )[index] = json_number_value( value );
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
                    transferHost.description[index].clear( );
                    size_t index2;
                    json_t * element;
                    json_array_foreach(
                        value,
                        index2,
                        element
                    )
                    {
                        transferHost.description[index].insert(
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
                sendTransfer = true;
            }
            if( js = json_object_get(
                message,
                "interpolation"
            ) )
            {
                redraw = true;
                interpolation = json_boolean_value( js );
                sendInterpolation = true;
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
                sendStep = true;
            }
            if( json_array_size(
                js = json_object_get(
                    message,
                    "iso threshold"
                )
            ) )
            {
                redraw = true;
                json_array_foreach(
                    js,
                    index,
                    value
                )
                sourceIsoThreshold.value[index] = json_number_value( value );
                sendIsoThreshold = true;
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
                sendFunctions = true;
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
                sourceWeight.value[index] = json_number_value( value );
                sendWeight = true;
            }
            if( js = json_object_get(
                message,
                "bounding box"
            ) )
            {
                redraw = true;
                icetBoundingBox = !icetBoundingBox;
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
                backgroundColor[index] = json_number_value( value );
                for( int pass = 0; pass < T_Controller::passCount; pass++ )
                {
                    icetSetContext( icetContext[pass] );
                    if( backgroundColor[0] == 0.0f
                        && backgroundColor[1] == 0.0f
                        && backgroundColor[2] == 0.0f )
                    {
                        icetDisable( ICET_CORRECT_COLORED_BACKGROUND );
                    }
                    else
                    {
                        icetEnable( ICET_CORRECT_COLORED_BACKGROUND );
                    }
                }
                sendBackgroundColor = true;
            }
            if( js = json_object_get(
                message,
                "clipping add"
            ) )
            {
                redraw = true;
                sendClipping = true;
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
                sendClipping = true;
                removeClipping( json_integer_value( js ) );
            }
            if( js = json_object_get(
                message,
                "clipping edit"
            ) )
            {
                redraw = true;
                sendClipping = true;
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

                sendAO = true;
            }

            if ( js = json_object_get ( message, "render_mode" ) ) {
                redraw = true;
                json_t * mode = json_object_get(js, "mode");
                myself->renderMode = (isaac_int)json_integer_value ( mode );
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
            thrMetaTargets = metaTargets;

            if( sendMinMax )
            {
                forEachParams(
                    sources,
                    CalcMinMaxIterator( ),
                    pointerArray,
                    minMaxArray,
                    localMinMaxArrayDevice,
                    localSize,
                    stream,
                    host
                );
                forEachParams(
                    particleSources,
                    CalcParticleMinMaxIterator< boost::mpl::size< T_SourceList >::type::value >( ),
                    minMaxArray,
                    localParticleMinMaxArrayDevice,
                    localParticleSize,
                    stream,
                    host
                );

                if( rank == master )
                {
                    MPI_Reduce(
                        MPI_IN_PLACE,
                        minMaxArray.min,
                        (
                            boost::mpl::size< T_SourceList >::type::value
                            + boost::mpl::size< T_ParticleList >::type::value
                        ),
                        MPI_FLOAT,
                        MPI_MIN,
                        master,
                        mpiWorld
                    );
                    MPI_Reduce(
                        MPI_IN_PLACE,
                        minMaxArray.max,
                        (
                            boost::mpl::size< T_SourceList >::type::value
                            + boost::mpl::size< T_ParticleList >::type::value
                        ),
                        MPI_FLOAT,
                        MPI_MAX,
                        master,
                        mpiWorld
                    );
                }
                else
                {
                    MPI_Reduce(
                        minMaxArray.min,
                        NULL,
                        (
                            boost::mpl::size< T_SourceList >::type::value
                            + boost::mpl::size< T_ParticleList >::type::value
                        ),
                        MPI_FLOAT,
                        MPI_MIN,
                        master,
                        mpiWorld
                    );
                    MPI_Reduce(
                        minMaxArray.max,
                        NULL,
                        (
                            boost::mpl::size< T_SourceList >::type::value
                            + boost::mpl::size< T_ParticleList >::type::value
                        ),
                        MPI_FLOAT,
                        MPI_MAX,
                        master,
                        mpiWorld
                    );
                }
            }

            for( int pass = 0; pass < T_Controller::passCount; pass++ )
            {
                image[pass].opaque_internals = NULL;
            }

            if( redraw )
            {
                for( int pass = 0; pass < T_Controller::passCount; pass++ )
                {
                    icetSetContext( icetContext[pass] );
                    //Calc order
                    ISAAC_START_TIME_MEASUREMENT ( sorting,
                        getTicksUs( ) )
                    //Every rank calculates it's distance to the camera
                    isaac_double4 point = isaac_double4( ( isaac_double3( positionScaled ) 
                                        + ( isaac_double3( localSizeScaled ) - isaac_double3( globalSizeScaled ) ) / isaac_double(2.0) )
                                        / isaac_double( maxSizeScaled / 2 ), 0 );
                    point.w = 1.0;

                    float pointDistance = glm::length(modelview * point);
                    //Allgather of the distances
                    float receiveBuffer[numProc];
                    MPI_Allgather(
                        &pointDistance,
                        1,
                        MPI_FLOAT,
                        receiveBuffer,
                        1,
                        MPI_FLOAT,
                        mpiWorld
                    );
                    //Putting to a std::multimap of {rank, distance}
                    std::multimap< float, isaac_int, std::less< float > >
                        distanceMap;
                    for( isaac_int i = 0; i < numProc; i++ )
                    {
                        distanceMap.insert(
                            std::pair<
                                float,
                                isaac_int
                            >(
                                receiveBuffer[i],
                                i
                            )
                        );
                    }
                    //Putting in an array for IceT
                    IceTInt icetOrderArray[numProc];
                    {
                        isaac_int i = 0;
                        for( auto it = distanceMap.begin( );
                            it != distanceMap.end( );
                            it++ )
                        {
                            icetOrderArray[i] = it->second;
                            i++;
                        }
                    }
                    icetCompositeOrder( icetOrderArray );
                    ISAAC_STOP_TIME_MEASUREMENT ( sortingTime,
                        +=,
                        sorting,
                        getTicksUs( ) )

                    //Drawing
                    ISAAC_START_TIME_MEASUREMENT ( merge,
                        getTicksUs( ) )
                    image[pass] = icetDrawFrame(
                        glm::value_ptr( projections[pass] ),
                        glm::value_ptr( modelview ),
                        backgroundColor
                    );
                    ISAAC_STOP_TIME_MEASUREMENT ( mergeTime,
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
                jsonRoot,
                0
            );
            strcpy(
                messageBuffer,
                buffer
            );
            free( buffer );
            if( metaTargets == META_MERGE )
            {
                if( rank == master )
                {
                    char receiveBuffer[numProc][ISAAC_MAX_RECEIVE];
                    MPI_Gather(
                        messageBuffer,
                        ISAAC_MAX_RECEIVE,
                        MPI_CHAR,
                        receiveBuffer,
                        ISAAC_MAX_RECEIVE,
                        MPI_CHAR,
                        master,
                        mpiWorld
                    );
                    for( isaac_int i = 0; i < numProc; i++ )
                    {
                        if( i == master )
                        {
                            continue;
                        }
                        json_t * js = json_loads(
                            receiveBuffer[i],
                            0,
                            NULL
                        );
                        mergeJSON(
                            jsonRoot,
                            js
                        );
                        json_decref( js );
                    }
                }
                else
                {
                    MPI_Gather(
                        messageBuffer,
                        ISAAC_MAX_RECEIVE,
                        MPI_CHAR,
                        NULL,
                        0,
                        MPI_CHAR,
                        master,
                        mpiWorld
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
            json_decref( jsonRoot );
            if( rank == master )
            {
                jsonRoot = json_object( );
                json_object_set_new(
                    jsonRoot,
                    "type",
                    json_string( "exit" )
                );
                char * buffer = json_dumps(
                    jsonRoot,
                    0
                );
                communicator->serverSend(
                    buffer,
                    true,
                    true
                );
                free( buffer );
                json_decref( jsonRoot );
            }
            for( int pass = 0; pass < T_Controller::passCount; pass++ )
            {
                icetDestroyContext( icetContext[pass] );
            }
            delete communicator;
            json_decref( jsonInitRoot );
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


        uint64_t kernelTime;
        uint64_t mergeTime;
        uint64_t videoSendTime;
        uint64_t copyTime;
        uint64_t sortingTime;
        uint64_t bufferTime;
    private:
        static void drawCallBack(
            const IceTDouble * projectionMatrix,
            const IceTDouble * modelviewMatrix,
            const IceTFloat * backgroundColor,
            const IceTInt * readbackViewport,
            IceTImage result
        )
        {
            //allocate memory for inverse mvp, mv and p matrix and simulation size properties

            //inverse mvp
            alpaka::Buf <T_Host, isaac_mat4, FraDim, ISAAC_IDX_TYPE>
                inverseMVPHostBuf(
                alpaka::allocBuf<
                    isaac_mat4,
                    ISAAC_IDX_TYPE
                >(
                    myself->host,
                    ISAAC_IDX_TYPE( 1 )
                )
            );
            isaac_mat4& inverseMVPHost =
                *(reinterpret_cast<isaac_mat4 *> ( alpaka::getPtrNative( inverseMVPHostBuf ) ) );


            //model-view matrix
            alpaka::Buf <T_Host, isaac_mat4, FraDim, ISAAC_IDX_TYPE>
                modelviewHostBuf(
                alpaka::allocBuf<
                    isaac_mat4,
                    ISAAC_IDX_TYPE
                >(
                    myself->host,
                    ISAAC_IDX_TYPE( 1 )
                )
            );
            isaac_mat4& modelviewHost =
                *( reinterpret_cast<isaac_mat4 *> ( alpaka::getPtrNative( modelviewHostBuf ) ) );


            //projection matrix
            alpaka::Buf <T_Host, isaac_mat4, FraDim, ISAAC_IDX_TYPE>
                projectionHostBuf(
                alpaka::allocBuf<
                    isaac_mat4,
                    ISAAC_IDX_TYPE
                >(
                    myself->host,
                    ISAAC_IDX_TYPE( 1 )
                )
            );
            isaac_mat4& projectionHost =
                *( reinterpret_cast<isaac_mat4 *> ( alpaka::getPtrNative( projectionHostBuf ) ) );


            //sim size values
            alpaka::Buf <T_Host, SimulationSizeStruct, FraDim, ISAAC_IDX_TYPE>
                sizeHostBuf(
                alpaka::allocBuf <SimulationSizeStruct ,
                ISAAC_IDX_TYPE> ( myself->host, ISAAC_IDX_TYPE( 1 ) )
            );
            SimulationSizeStruct & sizeHost =
                *( reinterpret_cast<SimulationSizeStruct*> ( alpaka::getPtrNative( sizeHostBuf ) ) );


            std::copy( projectionMatrix, projectionMatrix + 16, glm::value_ptr( projectionHost ) );
            std::copy( modelviewMatrix, modelviewMatrix + 16, glm::value_ptr( modelviewHost ) );
            //calculate inverse mvp matrix for render kernel
            isaac_mat4 inverse = glm::inverse( projectionHost * modelviewHost );
            std::copy( glm::value_ptr( inverse ), glm::value_ptr( inverse ) + 16, glm::value_ptr( inverseMVPHost ) );


            //set global simulation size
            sizeHost.globalSize = myself->globalSize;

            sizeHost.position = myself->position;

            //set subvolume size
            sizeHost.localSize = myself->localSize;
            sizeHost.localParticleSize = myself->localParticleSize;
            
            //get maximum size from biggest dimesnion MAX(x-dim, y-dim, z-dim)
            sizeHost.maxGlobalSize = myself->maxSize;

            //set global size with cellcount scaled
            sizeHost.globalSizeScaled = myself->globalSizeScaled;

            //set position in subvolume (adjusted to cellcount scale)
            sizeHost.positionScaled = myself->positionScaled;

            sizeHost.localSizeScaled = myself->localSizeScaled;

            //get maximum size from biggest dimesnion after scaling MAX(x-dim, y-dim, z-dim)
            sizeHost.maxGlobalSizeScaled = myself->maxSizeScaled;

            //set volume scale parameters
            isaac_float3 isaac_scale = myself->scale;

            //copy matrices and simulation size properties to constant memory

            //inverse matrix
            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                inverseMVPDeviceEextent( ISAAC_IDX_TYPE( 1 ) );
            //get view
            auto inverseMVPDeviceView(
                alpaka::createStaticDevMemView ( & InverseMVPMatrix,
                myself -> acc, inverseMVPDeviceEextent ) );
            //copy to constant memory
            alpaka::memcpy(
                myself->stream,
                inverseMVPDeviceView,
                inverseMVPHostBuf,
                ISAAC_IDX_TYPE( 1 )
            );

            //modelview matrix
            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                modelviewDeviceExtent( ISAAC_IDX_TYPE( 1 ) );
            //get view
            auto modelviewDeviceView(
                alpaka::createStaticDevMemView ( & ModelViewMatrix,
                myself -> acc, modelviewDeviceExtent ) );
            //copy to constant memory 
            alpaka::memcpy(
                myself->stream,
                modelviewDeviceView,
                modelviewHostBuf,
                ISAAC_IDX_TYPE( 1 )
            );


            //projection matrix
            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                projectionDeviceExtent( ISAAC_IDX_TYPE( 1 ) );
            //get view
            auto projectionDeviceView(
                alpaka::createStaticDevMemView ( & ProjectionMatrix,
                myself -> acc, projectionDeviceExtent ) );
            //copy to constant memory
            alpaka::memcpy(
                myself->stream,
                projectionDeviceView,
                projectionHostBuf,
                ISAAC_IDX_TYPE( 1 )
            );

            alpaka::Vec <alpaka::DimInt< 1u >, ISAAC_IDX_TYPE> const
                sizeDeviceExtent( ISAAC_IDX_TYPE( 1 ) );
            auto sizeDeviceView(
                alpaka::createStaticDevMemView ( & SimulationSize,
                myself -> acc, sizeDeviceExtent ) );
            alpaka::memcpy(
                myself->stream,
                sizeDeviceView,
                sizeHostBuf,
                ISAAC_IDX_TYPE( 1 )
            );

            //get pixel pointer from image as unsigned byte
            IceTUByte * pixels = icetImageGetColorub( result );

            //start time for performance measurment
            ISAAC_START_TIME_MEASUREMENT ( kernel,
                myself->getTicksUs( ) )

            //set color values for background color
            isaac_float4 bgColor = {
                isaac_float( backgroundColor[3] ),
                isaac_float( backgroundColor[2] ),
                isaac_float( backgroundColor[1] ),
                isaac_float( backgroundColor[0] )
            };

            //set framebuffer offset calculated from icet
            isaac_uint2 framebufferStart = {
                isaac_uint( readbackViewport[0] ),
                isaac_uint( readbackViewport[1] )
            };

            isaac_size2 blockSize= {
                ISAAC_IDX_TYPE ( 8 ),
                ISAAC_IDX_TYPE ( 16 )
            };
            isaac_size2 gridSize= {
                ISAAC_IDX_TYPE( ( readbackViewport[2] + blockSize.x - 1 ) / blockSize.x ),
                ISAAC_IDX_TYPE( ( readbackViewport[3] + blockSize.y - 1 ) / blockSize.y )
            };
#if ALPAKA_ACC_GPU_CUDA_ENABLED == 1
            if ( boost::mpl::not_<boost::is_same<T_Acc, alpaka::AccGpuCudaRt<T_AccDim, ISAAC_IDX_TYPE> > >::value )
#endif
            {
                gridSize.x = ISAAC_IDX_TYPE ( readbackViewport[2] );
                gridSize.y = ISAAC_IDX_TYPE ( readbackViewport[3] );
                blockSize.x = ISAAC_IDX_TYPE ( 1 );
                blockSize.y = ISAAC_IDX_TYPE ( 1 );
            }
            const alpaka::Vec<T_AccDim, ISAAC_IDX_TYPE> threads ( 
                ISAAC_IDX_TYPE ( 1 ), 
                ISAAC_IDX_TYPE ( 1 ), 
                ISAAC_IDX_TYPE ( 1 ) 
            );
            const alpaka::Vec<T_AccDim, ISAAC_IDX_TYPE> blocks ( 
                ISAAC_IDX_TYPE ( 1 ), 
                blockSize.y, 
                blockSize.x 
            );
            const alpaka::Vec<T_AccDim, ISAAC_IDX_TYPE> grid ( 
                ISAAC_IDX_TYPE ( 1 ), 
                gridSize.y, 
                gridSize.x 
            );
            alpaka::WorkDivMembers<T_AccDim, ISAAC_IDX_TYPE> const workdiv ( 
                grid,
                blocks,
                threads 
            );

            const int SourceListLength = boost::mpl::size< T_SourceList >::type::value
                                        + boost::mpl::size< T_ParticleList >::type::value;
            
            GBuffer gBuffer;
            gBuffer.size = myself->framebufferSize;
            gBuffer.startOffset = framebufferStart;
            gBuffer.color = alpaka::getPtrNative(myself->framebuffer);
            gBuffer.depth = alpaka::getPtrNative(myself->framebufferDepth);
            gBuffer.normal = alpaka::getPtrNative(myself->framebufferNormal);
            gBuffer.aoStrength = alpaka::getPtrNative(myself->framebufferAO);

            //reset the GBuffer to default values
            {
                ClearBufferKernel kernel;
                auto const instance
                (
                    alpaka::createTaskKernel<T_Acc>
                    (
                        workdiv,
                        kernel,
                        gBuffer,
                        bgColor
                    )
                );
                alpaka::enqueue(myself->stream, instance);
                alpaka::wait( myself->stream );
            }

            //call particle render kernel
            ParticleRenderKernelCaller<
                T_ParticleList,
                TransferDeviceStruct<SourceListLength>,
                SourceWeightStruct<SourceListLength>,
                boost::mpl::vector< >,
                T_transferSize,
                alpaka::WorkDivMembers<T_AccDim, ISAAC_IDX_TYPE>, 
                T_Acc, 
                T_Stream,
                boost::mpl::size< T_SourceList >::type::value,
                boost::mpl::size< T_ParticleList >::type::value
            > 
            ::call(
                myself->stream,
                gBuffer,
                myself->particleSources,
                myself->transferDevice,
                myself->sourceWeight,
                workdiv,
                isaac_float3( sizeHost.localSizeScaled ) / isaac_float3( sizeHost.localParticleSize ),
                myself->clipping
            );
            //wait until render kernel has finished
            alpaka::wait( myself->stream );

            //call iso render kernel
            IsoRenderKernelCaller<
                T_SourceList,
                TransferDeviceStruct<SourceListLength>,
                IsoThresholdStruct<boost::mpl::size< T_SourceList >::type::value>,
                PointerArrayStruct<boost::mpl::size< T_SourceList >::type::value>,
                boost::mpl::vector< >,
                T_transferSize,
                alpaka::WorkDivMembers<T_AccDim, ISAAC_IDX_TYPE>, 
                T_Acc, 
                T_Stream,
                boost::mpl::size< T_SourceList >::type::value
            > 
            ::call(
                myself->stream,
                gBuffer,
                myself->sources,
                myself->step,
                myself->transferDevice,
                myself->sourceIsoThreshold,
                myself->pointerArray,
                workdiv,
                myself->interpolation,
                isaac_scale,
                myself->clipping
            );
            //wait until render kernel has finished
            alpaka::wait( myself->stream );

            //process color and depth values for depth simulation
            if(myself->ambientOcclusion.isEnabled && myself->ambientOcclusion.weight > 0.0f) 
            {
                {
                    SSAOKernel kernel;
                    auto const instance
                    (
                        alpaka::createTaskKernel<T_Acc>
                        (
                            workdiv,
                            kernel,
                            gBuffer,
                            myself->ambientOcclusion
                        )
                    );
                    alpaka::enqueue(myself->stream, instance);
                }

                //wait until render kernel has finished
                alpaka::wait ( myself->stream );

                //deactivated until proper ao is implemented
                #if 0
                {
                    SSAOFilterKernel kernel;
                    auto const instance
                    (
                        alpaka::createTaskKernel<T_Acc>
                        (
                            workdiv,
                            kernel,
                            gBuffer,
                            myself->ambientOcclusion
                        )
                    );
                    alpaka::enqueue(myself->stream, instance);
                }

                //wait until render kernel has finished
                alpaka::wait ( myself->stream );
                #endif
            }
            
            //shade pixels
            {
                ShadingKernel kernel;
                auto const instance
                (
                    alpaka::createTaskKernel<T_Acc>
                    (
                        workdiv,
                        kernel,
                        gBuffer,
                        myself->ambientOcclusion,
                        bgColor,
                        myself->rank,
                        myself->renderMode
                    )
                );
                alpaka::enqueue(myself->stream, instance);
                alpaka::wait( myself->stream );
            }

            //call volume render kernel
            VolumeRenderKernelCaller<
                T_SourceList,
                TransferDeviceStruct<SourceListLength>,
                SourceWeightStruct<SourceListLength>,
                PointerArrayStruct<boost::mpl::size< T_SourceList >::type::value>,
                boost::mpl::vector< >,
                T_transferSize,
                alpaka::WorkDivMembers<T_AccDim, ISAAC_IDX_TYPE>, 
                T_Acc, 
                T_Stream,
                boost::mpl::size< T_SourceList >::type::value
            > 
            ::call(
                myself->stream,
                gBuffer,
                myself->sources,
                myself->step,
                myself->transferDevice,
                myself->sourceWeight,
                myself->pointerArray,
                workdiv,
                myself->interpolation,
                isaac_scale,
                myself->clipping
            );
            
            //wait until render kernel has finished
            alpaka::wait( myself->stream );

            //stop and restart time for delta calculation
            ISAAC_STOP_TIME_MEASUREMENT ( myself->kernelTime,
                +=,
                kernel,
                myself->getTicksUs( ) )
            ISAAC_START_TIME_MEASUREMENT ( copy,
                myself->getTicksUs( ) )

            //get memory view from IceT pixels on host
            alpaka::ViewPlainPtr <T_Host, uint32_t, FraDim, ISAAC_IDX_TYPE>
                result_buffer(
                ( uint32_t* )( pixels ),
                myself->host,
                alpaka::Vec<
                    FraDim,
                    ISAAC_IDX_TYPE
                >( myself->framebufferProd )
            );

            //copy device framebuffer to result IceT pixel buffer
            alpaka::memcpy(
                myself->stream,
                result_buffer,
                myself->framebuffer,
                alpaka::Vec<
                    FraDim,
                    ISAAC_IDX_TYPE
                >( myself->framebufferProd )
            );

            //stop timer and calculate copy time
            ISAAC_STOP_TIME_MEASUREMENT ( myself->copyTime,
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
                    myself->jsonRoot,
                    "type",
                    json_string( "period" )
                );
                json_object_set_new(
                    myself->jsonRoot,
                    "meta nr",
                    json_integer( myself->metaNr )
                );

                json_t * matrix;
                if( myself->sendProjection )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "projection",
                        matrix = json_array( )
                    );
                    for(isaac_int p = 0; p < T_Controller::passCount; ++p)
                    {
                        for (isaac_int i = 0; i < 16; ++i)
                        {
                            json_array_append_new( matrix, json_real( glm::value_ptr( myself->projections[p] )[i] ) );
                        }
                    }
                    json_object_set(
                        myself->jsonInitRoot,
                        "projection",
                        matrix
                    );
                    myself->sendInitJson = true;
                }
                if( myself->sendLookAt )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "position",
                        matrix = json_array( )
                    );
                    for (isaac_int i = 0; i < 3; i++)
                    {
                        json_array_append_new( matrix, json_real( glm::value_ptr( myself->lookAt )[i] ) );
                    }
                    json_object_set(
                        myself->jsonInitRoot,
                        "position",
                        matrix
                    );
                    myself->sendInitJson = true;
                }
                if( myself->sendRotation )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "rotation",
                        matrix = json_array( )
                    );
                    for (isaac_int i = 0; i < 9; i++)
                    {
                        json_array_append_new( matrix, json_real( glm::value_ptr( myself->rotation )[i] ) );
                    }
                    json_object_set(
                        myself->jsonInitRoot,
                        "rotation",
                        matrix
                    );
                    myself->sendInitJson = true;
                }
                if( myself->sendDistance )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "distance",
                        json_real( myself->distance )
                    );
                    json_object_set_new(
                        myself->jsonInitRoot,
                        "distance",
                        json_real( myself->distance )
                    );
                    myself->sendInitJson = true;
                }
                if( myself->sendTransfer )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "transfer array",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < (
                        boost::mpl::size< T_SourceList >::type::value
                        + boost::mpl::size< T_ParticleList >::type::value
                    ); i++ )
                    {
                        json_t * transfer = json_array( );
                        json_array_append_new(
                            matrix,
                            transfer
                        );
                        for( ISAAC_IDX_TYPE j = 0; j < T_transferSize; j++ )
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
                                        myself->transferHost
                                            .pointer[i][j].x
                                        * isaac_float( 255 )
                                    )
                                )
                            );
                            json_array_append_new(
                                color,
                                json_integer(
                                    isaac_uint(
                                        myself->transferHost
                                            .pointer[i][j].y
                                        * isaac_float( 255 )
                                    )
                                )
                            );
                            json_array_append_new(
                                color,
                                json_integer(
                                    isaac_uint(
                                        myself->transferHost
                                            .pointer[i][j].z
                                        * isaac_float( 255 )
                                    )
                                )
                            );
                            json_array_append_new(
                                color,
                                json_integer(
                                    isaac_uint(
                                        myself->transferHost
                                            .pointer[i][j].w
                                        * isaac_float( 255 )
                                    )
                                )
                            );
                        }
                    }
                    json_object_set_new(
                        myself->jsonRoot,
                        "transfer points",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < (
                        boost::mpl::size< T_SourceList >::type::value
                        + boost::mpl::size< T_ParticleList >::type::value
                    ); i++ )
                    {
                        json_t * points = json_array( );
                        json_array_append_new(
                            matrix,
                            points
                        );
                        for( auto it = myself->transferHost
                            .description[i].begin( );
                            it != myself->transferHost
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
                if( myself->sendFunctions )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "functions",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < (
                        boost::mpl::size< T_SourceList >::type::value
                        + boost::mpl::size< T_ParticleList >::type::value
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
                            json_integer( myself->functions[i].errorCode )
                        );
                    }
                }
                if( myself->sendWeight )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "weight",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < (
                        boost::mpl::size< T_SourceList >::type::value
                        + boost::mpl::size< T_ParticleList >::type::value
                    ); i++ )
                    {
                        json_array_append_new(
                            matrix,
                            json_real(
                                myself->sourceWeight
                                    .value[i]
                            )
                        );
                    }
                }
                if( myself->sendInterpolation )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "interpolation",
                        json_boolean( myself->interpolation )
                    );
                    json_object_set_new(
                        myself->jsonInitRoot,
                        "interpolation",
                        json_boolean( myself->interpolation )
                    );
                    myself->sendInitJson = true;
                }
                if( myself->sendStep )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "step",
                        json_real( myself->step )
                    );
                    json_object_set_new(
                        myself->jsonInitRoot,
                        "step",
                        json_boolean( myself->step )
                    );
                    myself->sendInitJson = true;
                }
                if( myself->sendIsoThreshold )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "iso threshold",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < boost::mpl::size< T_SourceList >::type::value; i++ )
                    {
                        json_array_append_new(
                            matrix,
                            json_real(
                                myself->sourceIsoThreshold
                                    .value[i]
                            )
                        );
                    }
                }
                if( myself->sendMinMax )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "minmax",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < (
                        boost::mpl::size< T_SourceList >::type::value
                        + boost::mpl::size< T_ParticleList >::type::value
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
                                myself->minMaxArray
                                    .min[i]
                            )
                        );
                        json_object_set_new(
                            v,
                            "max",
                            json_real(
                                myself->minMaxArray
                                    .max[i]
                            )
                        );
                    }
                }
                if( myself->sendBackgroundColor )
                {
                    json_object_set_new(
                        myself->jsonRoot,
                        "background color",
                        matrix = json_array( )
                    );
                    for( ISAAC_IDX_TYPE i = 0; i < 3; i++ )
                    {
                        json_array_append_new(
                            matrix,
                            json_real( myself->backgroundColor[i] )
                        );
                    }
                    json_object_set(
                        myself->jsonInitRoot,
                        "background color",
                        matrix
                    );
                    myself->sendInitJson = true;
                }
                if( myself->sendClipping )
                {
                    json_object_set_new(
                        myself->jsonRoot,
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
                            json_real( myself->clippingSavedNormals[i].x )
                        );
                        json_array_append_new(
                            inner,
                            json_real( myself->clippingSavedNormals[i].y )
                        );
                        json_array_append_new(
                            inner,
                            json_real( myself->clippingSavedNormals[i].z )
                        );
                    }
                }
                if(myself->sendAO) {
                    //add ambient occlusion parameters
                    json_object_set_new(
                        myself->jsonRoot, 
                        "ao isEnabled", 
                        json_boolean(myself->ambientOcclusion.isEnabled)
                    );
                    json_object_set_new(
                        myself->jsonRoot, 
                        "ao weight", 
                        json_real(myself->ambientOcclusion.weight)
                    );       
                    //add ao params to initial response   
                    json_object_set_new(
                        myself->jsonInitRoot, 
                        "ao isEnabled", 
                        json_boolean(myself->ambientOcclusion.isEnabled)
                    );
                    json_object_set_new(
                        myself->jsonInitRoot, 
                        "ao weight", 
                        json_real(myself->ambientOcclusion.weight)
                    );
                    myself->sendInitJson = true;
                }
                myself->controller
                    .sendFeedback(
                        myself->jsonRoot,
                        myself->sendController
                    );
                if( myself->sendInitJson )
                {
                    json_object_set(
                        myself->jsonRoot,
                        "init",
                        myself->jsonInitRoot
                    );
                }
                char * buffer = json_dumps(
                    myself->jsonRoot,
                    0
                );
                myself->communicator
                    ->serverSend( buffer );
                free( buffer );
            }
            json_decref( myself->jsonRoot );
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
                            myself->compbufferSize
                                .x,
                            myself->compbufferSize
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
            ISAAC_STOP_TIME_MEASUREMENT ( myself->videoSendTime,
                +=,
                video_send,
                myself->getTicksUs( ) )
            myself->metaNr++;
            return 0;
        }


        void recreateJSON( )
        {
            jsonRoot = json_object( );
            jsonMetaRoot = json_object( );
            json_object_set_new(
                jsonRoot,
                "metadata",
                jsonMetaRoot
            );
        }


        void updateModelview( )
        {
            isaac_dmat4 translationM = glm::translate( isaac_dmat4( 1 ), lookAt);

            isaac_dmat4 rotationM = rotation;
            rotationM[3][3] = 1.0;

            isaac_dmat4 distanceM = isaac_dmat4( 1 );
            distanceM[3][2] = distance;
            
            modelview = distanceM * rotationM * translationM;
        }


        T_Host host;
        DevAcc acc;
        T_Stream stream;
        std::string name;
        std::string serverUrl;
        isaac_uint serverPort;
        isaac_size2 framebufferSize;
        isaac_size2 compbufferSize;
        alpaka::Vec<
            FraDim,
            ISAAC_IDX_TYPE
        > framebufferProd;

        //framebuffer pixel values
        alpaka::Buf<
            DevAcc,
            uint32_t,
            FraDim,
            ISAAC_IDX_TYPE
        > framebuffer;

        //ambient occlusion factor values
        alpaka::Buf<
            DevAcc,
            isaac_float,
            FraDim,
            ISAAC_IDX_TYPE
        > framebufferAO;

        //pixel depth information
        alpaka::Buf<
            DevAcc,
            isaac_float,
            FraDim,
            ISAAC_IDX_TYPE
        > framebufferDepth;

        //pixel normal information
        alpaka::Buf<
            DevAcc,
            isaac_float3,
            FraDim,
            ISAAC_IDX_TYPE
        > framebufferNormal;  

        alpaka::Buf<
            DevAcc,
            FunctorChainPointerN,
            FraDim,
            ISAAC_IDX_TYPE
        > functor_chain_d;
        alpaka::Buf<
            DevAcc,
            FunctorChainPointerN,
            FraDim,
            ISAAC_IDX_TYPE
        > functorChainChooseDevice;
        alpaka::Buf<
            DevAcc,
            MinMax,
            FraDim,
            ISAAC_IDX_TYPE
        > localMinMaxArrayDevice;
        alpaka::Buf<
            DevAcc,
            MinMax,
            FraDim,
            ISAAC_IDX_TYPE
        > localParticleMinMaxArrayDevice;

        isaac_size3 globalSize;
        isaac_size3 localSize;
        isaac_size3 localParticleSize;
        isaac_size3 position;
        isaac_size3 globalSizeScaled;
        isaac_size3 localSizeScaled;
        isaac_size3 positionScaled;
        MPI_Comm mpiWorld;
        std::vector<isaac_dmat4> projections;
        isaac_dmat4 modelview;                           //modelview matrix
        isaac_double3 lookAt;
        isaac_dmat3 rotation;
        isaac_double distance;

        //true if properties should be sent by server
        bool sendLookAt;
        bool sendRotation;
        bool sendDistance;
        bool sendProjection;
        bool sendTransfer;
        bool sendInterpolation;
        bool sendStep;
        bool sendIsoThreshold;
        bool sendFunctions;
        bool sendWeight;
        bool sendMinMax;
        bool sendBackgroundColor;
        bool sendClipping;
        bool sendController;
        bool sendInitJson;
        bool sendAO;


        bool interpolation;
        bool icetBoundingBox;
        isaac_float step;
        IsaacCommunicator * communicator = nullptr;
        json_t * jsonRoot = nullptr;
        json_t * jsonInitRoot = nullptr;
        json_t * jsonMetaRoot = nullptr;
        isaac_int rank;
        isaac_int renderMode = 0;
        isaac_int master;
        isaac_int numProc;
        isaac_uint metaNr;
        T_ParticleList & particleSources;
        T_SourceList & sources;
        IceTContext icetContext[T_Controller::passCount];
        IsaacVisualizationMetaEnum thrMetaTargets;
        pthread_t visualizationThread;
        AOParams ambientOcclusion;                         //state of ambient occlusion on client site

        std::vector <alpaka::Buf<
            DevAcc,
            isaac_float4,
            TexDim,
            ISAAC_IDX_TYPE
        >> transferDeviceBuf;
        std::vector <alpaka::Buf<
            T_Host,
            isaac_float4,
            TexDim,
            ISAAC_IDX_TYPE
        >> transferHostBuf;
        std::vector <alpaka::Buf<
            DevAcc,
            isaac_float,
            FraDim,
            ISAAC_IDX_TYPE
        >> pointerArrayAlpaka;

        TransferDeviceStruct<
            (
                boost::mpl::size< T_SourceList >::type::value
                + boost::mpl::size< T_ParticleList >::type::value
            )
        > transferDevice;
        TransferHostStruct<
            (
                boost::mpl::size< T_SourceList >::type::value
                + boost::mpl::size< T_ParticleList >::type::value
            )
        > transferHost;
        SourceWeightStruct<
            (
                boost::mpl::size< T_SourceList >::type::value
                + boost::mpl::size< T_ParticleList >::type::value
            )
        > sourceWeight;
        IsoThresholdStruct<
            (
                boost::mpl::size< T_SourceList >::type::value
            )
        > sourceIsoThreshold;
        PointerArrayStruct<
            (
                boost::mpl::size< T_SourceList >::type::value
            )
        > pointerArray;
        MinMaxArray<
            (
                boost::mpl::size< T_SourceList >::type::value
                + boost::mpl::size< T_ParticleList >::type::value
            )
        > minMaxArray;
        FunctionsStruct functions[(
            boost::mpl::size< T_SourceList >::type::value
            + boost::mpl::size< T_ParticleList >::type::value
        )];
        ISAAC_IDX_TYPE maxSize;
        ISAAC_IDX_TYPE maxSizeScaled;
        IceTFloat backgroundColor[4];
        static IsaacVisualization * myself;
        isaac_float3 scale;
        ClippingStruct clipping;
        isaac_float3 clippingSavedNormals[ISAAC_MAX_CLIPPING];
        T_Controller controller;
        T_Compositor compositor;
        IceTImage image[T_Controller::passCount];
    };

    template<
        typename T_Host,
        typename T_Acc,
        typename T_Stream,
        typename T_AccDim,
        typename T_ParticleList,
        typename T_SourceList,
        ISAAC_IDX_TYPE T_transferSize,
        typename T_Controller,
        typename T_Compositor
    > IsaacVisualization<
        T_Host,
        T_Acc,
        T_Stream,
        T_AccDim,
        T_ParticleList,
        T_SourceList,
        T_transferSize,
        T_Controller,
        T_Compositor
    > * IsaacVisualization<
        T_Host,
        T_Acc,
        T_Stream,
        T_AccDim,
        T_ParticleList,
        T_SourceList,
        T_transferSize,
        T_Controller,
        T_Compositor
    >::myself = NULL;

} //namespace isaac;
