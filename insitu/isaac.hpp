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

//Hack for a bug, which occurs only in Cuda 7.0
#if (__CUDACC_VER__ < 70500) && !defined(BOOST_RESULT_OF_USE_TR1)
    #define BOOST_RESULT_OF_USE_TR1
#endif

#include <boost/config/select_compiler_config.hpp>

#include <string>
#include <string.h>
#include <jansson.h>
#include <pthread.h>
#include <list>
#include <vector>
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

namespace isaac
{

template <
#if ISAAC_ALPAKA == 1
    typename THost,
    typename TAcc,
    typename TStream,
    typename TAccDim,
#endif
    typename TSimDim,
    typename TSourceList,
    typename TDomainSize,
    size_t TTransfer_size,
    typename TScale
>
class IsaacVisualization 
{
    public:
        #if ISAAC_ALPAKA == 1
            using TDevAcc = alpaka::dev::Dev<TAcc>;
            using TFraDim = alpaka::dim::DimInt<1>;
            using TTexDim = alpaka::dim::DimInt<1>;
        #endif
        struct source_2_json_iterator
        {
            template
            <
                typename TSource,
                typename TJsonRoot
            >
            ISAAC_HOST_INLINE  void operator()( const int I,const TSource& s, TJsonRoot& jsonRoot) const
            {
                json_t *content = json_object();
                json_array_append_new( jsonRoot, content );
                json_object_set_new( content, "name", json_string ( s.name.c_str() ) );
                json_object_set_new( content, "feature dimension", json_integer ( s.feature_dim ) );
            }
        };

        struct functor_2_json_iterator
        {
            template
            <
                typename TFunctor,
                typename TJsonRoot
            >
            ISAAC_HOST_INLINE  void operator()( const int I,const TFunctor& f, TJsonRoot& jsonRoot) const
            {
                json_t *content = json_object();
                json_array_append_new( jsonRoot, content );
                json_object_set_new( content, "name", json_string ( TFunctor::name.c_str() ) );
                json_object_set_new( content, "description", json_string ( TFunctor::description.c_str() ) );
                json_object_set_new( content, "uses parameter", json_boolean ( TFunctor::uses_parameter ) );
            }
        };

        struct parse_functor_iterator
        {
            template
            <
                typename TFunctor,
                typename TName,
                typename TValue,
                typename TFound
            >
            ISAAC_HOST_INLINE  void operator()( const int I,TFunctor& f,const TName& name, TValue& value, TFound& found) const
            {
                if (!found && name == TFunctor::name)
                {
                    value = I;
                    found = true;
                }
            }
        };

        struct update_functor_chain_iterator
        {
            template
            <
                typename TSource,
                typename TFunctions,
                typename TDest
            >
            ISAAC_HOST_INLINE  void operator()(
                const int I,
                const TSource& source,
                const TFunctions& functions,
                TDest& dest
            ) const
            {
                isaac_int chain_nr = 0;
                for (int i = 0; i < ISAAC_MAX_FUNCTORS; i++)
                {
                    chain_nr *= ISAAC_FUNCTOR_COUNT;
                    chain_nr += functions[I].bytecode[i];
                }
                dest.nr[I] = chain_nr * 4 + TSource::feature_dim - 1;                
            }
        };
        
        struct allocate_pointer_array_iterator
        {
            template
            <
                typename TSource,
                typename TArray,
                typename TLocalSize
                #if ISAAC_ALPAKA == 1
                    ,typename TVector
                    ,typename TDevAcc__
                #endif
            >
            ISAAC_HOST_INLINE  void operator()(
                const int I,
                const TSource& source,
                TArray& pointer_array,
                const TLocalSize& local_size
                #if ISAAC_ALPAKA == 1
                    ,TVector& alpaka_vector
                    ,const TDevAcc__& acc
                #endif
            ) const
            {
                if (TSource::persistent)
                    pointer_array.pointer[I] = NULL;
                else
                {
                    #if ISAAC_ALPAKA == 1
                        alpaka_vector.push_back( alpaka::mem::buf::Buf< TDevAcc, isaac_float, TFraDim, size_t> ( alpaka::mem::buf::alloc<isaac_float, size_t>(acc, alpaka::Vec<TFraDim, size_t> ( TSource::feature_dim * (local_size[0] + 2 * ISAAC_GUARD_SIZE) * (local_size[1] + 2 * ISAAC_GUARD_SIZE) * (local_size[2] + 2 * ISAAC_GUARD_SIZE) ) ) ) );
                        pointer_array.pointer[I] = alpaka::mem::view::getPtrNative( alpaka_vector.back() );                        
                    #else
                        ISAAC_CUDA_CHECK(cudaMalloc((void**)&(pointer_array.pointer[I]), sizeof(isaac_float_dim< TSource::feature_dim >) * (local_size[0] + 2 * ISAAC_GUARD_SIZE) * (local_size[1] + 2 * ISAAC_GUARD_SIZE) * (local_size[2] + 2 * ISAAC_GUARD_SIZE) ) );
                    #endif
                }
            }
        };
        
        struct update_pointer_array_iterator
        {
            template
            <
                typename TSource,
                typename TArray,
                typename TLocalSize,
                typename TWeight,
                typename TPointer
                #if ISAAC_ALPAKA == 1
                    ,typename TStream__
                #endif
            >
            ISAAC_HOST_INLINE  void operator()(
                const int I,
                TSource& source,
                TArray& pointer_array,
                const TLocalSize& local_size,
                const TWeight& weight,
                const TPointer& pointer
                #if ISAAC_ALPAKA == 1
                    ,TStream__& stream
                #endif
            ) const
            {
                bool enabled = weight.value[ I ] != isaac_float(0);
                source.update( enabled, pointer );
                if (!TSource::persistent && enabled)
                {
                    isaac_size2 grid_size=
                    {
                        size_t((local_size[0] + ISAAC_GUARD_SIZE * 2 + 15)/16),
                        size_t((local_size[1] + ISAAC_GUARD_SIZE * 2 + 15)/16),
                    };
                    isaac_size2 block_size=
                    {
                        size_t(16),
                        size_t(16),
                    };
                    isaac_int3 local_size_array = { isaac_int(local_size[0]), isaac_int(local_size[1]), isaac_int(local_size[2]) };
                    #if ISAAC_ALPAKA == 1
                        if ( mpl::not_<boost::is_same<TAcc, alpaka::acc::AccGpuCudaRt<TAccDim, size_t> > >::value )
                        {
                            grid_size.x = size_t(local_size[0] + ISAAC_GUARD_SIZE * 2);
                            grid_size.y = size_t(local_size[0] + ISAAC_GUARD_SIZE * 2);
                            block_size.x = size_t(1);
                            block_size.y = size_t(1);                    
                        }                
                        const alpaka::Vec<TAccDim, size_t> threads (size_t(1), size_t(1), size_t(1));
                        const alpaka::Vec<TAccDim, size_t> blocks  (size_t(1), block_size.x, block_size.y);
                        const alpaka::Vec<TAccDim, size_t> grid    (size_t(1), grid_size.x, grid_size.y);
                        auto const workdiv(alpaka::workdiv::WorkDivMembers<TAccDim, size_t>(grid,blocks,threads));
                        updateBufferKernel<TSource> kernel;
                        auto const instance
                        (
                            alpaka::exec::create<TAcc>
                            (
                                workdiv,
                                kernel,
                                source,
                                pointer_array.pointer[ I ],
                                local_size_array
                            )
                        );
                        alpaka::stream::enqueue(stream, instance);                
                        alpaka::wait::wait(stream);
                    #else
                        dim3 block (block_size.x, block_size.y);
                        dim3 grid  (grid_size.x, grid_size.y);
                        updateBufferKernel<<<grid,block>>>( source, pointer_array.pointer[ I ], local_size_array );
                        ISAAC_CUDA_CHECK(cudaDeviceSynchronize());
                    #endif
                }
            }
        };
        
        struct calc_minmax_iterator
        {
            template
            <
                typename TSource,
                typename TArray,
                typename TMinmax,
                typename TLocalMinmax,
                typename TLocalSize
                #if ISAAC_ALPAKA == 1
                    ,typename TStream__
                    ,typename THost__
                    ,typename TFunctionChain
                    ,typename TParameter
                #endif
            >
            ISAAC_HOST_INLINE  void operator()(
                const int I,
                const TSource& source,
                TArray& pointer_array,
                TMinmax& minmax,
                TLocalMinmax& local_minmax,
                TLocalSize& local_size
                #if ISAAC_ALPAKA == 1
                    ,TStream__& stream
                    ,const THost__& host
                    ,const TFunctionChain& function_chain
                    ,const TParameter& parameter
                #endif
            ) const
            {
                isaac_size2 grid_size=
                {
                    size_t((local_size[0]+15)/16),
                    size_t((local_size[1]+15)/16),
                };
                isaac_size2 block_size=
                {
                    size_t(16),
                    size_t(16),
                };
                isaac_int3 local_size_array = { isaac_int(local_size[0]), isaac_int(local_size[1]), isaac_int(local_size[2]) };
                minmax_struct local_minmax_array_h[ local_size_array.x * local_size_array.y ];
                #if ISAAC_ALPAKA == 1
                    if ( mpl::not_<boost::is_same<TAcc, alpaka::acc::AccGpuCudaRt<TAccDim, size_t> > >::value )
                    {
                        grid_size.x = size_t(local_size[0]);
                        grid_size.y = size_t(local_size[0]);
                        block_size.x = size_t(1);
                        block_size.y = size_t(1);                    
                    }                
                    const alpaka::Vec<TAccDim, size_t> threads (size_t(1), size_t(1), size_t(1));
                    const alpaka::Vec<TAccDim, size_t> blocks  (size_t(1), block_size.x, block_size.y);
                    const alpaka::Vec<TAccDim, size_t> grid    (size_t(1), grid_size.x, grid_size.y);
                    auto const workdiv(alpaka::workdiv::WorkDivMembers<TAccDim, size_t>(grid,blocks,threads));
                    minMaxKernel<TSource> kernel;
                    auto const instance
                    (
                        alpaka::exec::create<TAcc>
                        (
                            workdiv,
                            kernel,
                            alpaka::mem::view::getPtrNative(function_chain),
                            alpaka::mem::view::getPtrNative(parameter),
                            source,
                            I,
                            alpaka::mem::view::getPtrNative(local_minmax),
                            local_size_array,
                            pointer_array.pointer[ I ]
                        )
                    );
                    alpaka::stream::enqueue(stream, instance);                
                    alpaka::wait::wait(stream);
                    alpaka::mem::buf::BufPlainPtrWrapper<THost, minmax_struct, TFraDim, size_t> minmax_buffer(local_minmax_array_h, host, size_t(local_size_array.x * local_size_array.y));
                    alpaka::mem::view::copy( stream, minmax_buffer, local_minmax, size_t(local_size_array.x * local_size_array.y));
                #else
                    dim3 block (block_size.x, block_size.y);
                    dim3 grid  (grid_size.x, grid_size.y);
                    minMaxKernel<<<grid,block>>>( source, I, local_minmax, local_size_array, pointer_array.pointer[ I ]);
                    ISAAC_CUDA_CHECK(cudaMemcpy( local_minmax_array_h, local_minmax, sizeof(minmax_struct)*local_size_array.x * local_size_array.y, cudaMemcpyDeviceToHost));
                #endif
                minmax.min[ I ] =  FLT_MAX;
                minmax.max[ I ] = -FLT_MAX;
                for (int i = 0; i < local_size_array.x * local_size_array.y; i++)
                {
                    if ( local_minmax_array_h[i].min < minmax.min[ I ])
                        minmax.min[ I ] = local_minmax_array_h[i].min;
                    if ( local_minmax_array_h[i].max > minmax.max[ I ])
                        minmax.max[ I ] = local_minmax_array_h[i].max;
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
            const TDomainSize position,
            TSourceList& sources,
            TScale scale
            ) :
            #if ISAAC_ALPAKA == 1
                host(host),
                acc(acc),
                stream(stream),
            #endif
            global_size(global_size),
            local_size(local_size),
            position(position),
            name(name),
            master(master),
            server_url(server_url),
            server_port(server_port),
            framebuffer_size(framebuffer_size),
            metaNr(0),
            visualizationThread(0),
            kernel_time(0),
            merge_time(0),
            video_send_time(0),
            copy_time(0),
            sorting_time(0),
            buffer_time(0),
            interpolation(false),
            iso_surface(false),
            step(isaac_float( ISAAC_DEFAULT_STEP )),
            framebuffer_prod(size_t(framebuffer_size.x) * size_t(framebuffer_size.y)),
            sources( sources ),
            scale( scale ),
            icet_bounding_box( true )
            #if ISAAC_ALPAKA == 1
                ,framebuffer(alpaka::mem::buf::alloc<uint32_t, size_t>(acc, framebuffer_prod))
                ,inverse_d(alpaka::mem::buf::alloc<isaac_float, size_t>(acc, size_t(16)))
                ,parameter_d(alpaka::mem::buf::alloc<isaac_float4, size_t>(acc, size_t(ISAAC_MAX_FUNCTORS * boost::mpl::size< TSourceList >::type::value)))
                ,size_d(alpaka::mem::buf::alloc<isaac_size_struct< TSimDim::value >, size_t>(acc, size_t(1)))
                ,functor_chain_d(alpaka::mem::buf::alloc<isaac_functor_chain_pointer_N, size_t>(acc, size_t( ISAAC_FUNCTOR_COMPLEX * 4)))
                ,functor_chain_choose_d(alpaka::mem::buf::alloc<isaac_functor_chain_pointer_N, size_t>(acc, size_t( boost::mpl::size< TSourceList >::type::value )))
                ,local_minmax_array_d(alpaka::mem::buf::alloc<minmax_struct, size_t>(acc, size_t( local_size[0] * local_size[1] )))
        {
            #else
        {
                ISAAC_CUDA_CHECK(cudaMalloc((uint32_t**)&framebuffer, sizeof(uint32_t)*framebuffer_prod));
                ISAAC_CUDA_CHECK(cudaMalloc((isaac_functor_chain_pointer_N**)&functor_chain_d, sizeof(isaac_functor_chain_pointer_N) * ISAAC_FUNCTOR_COMPLEX * 4));
                ISAAC_CUDA_CHECK(cudaMalloc((isaac_functor_chain_pointer_N**)&functor_chain_choose_d, sizeof(isaac_functor_chain_pointer_N) * boost::mpl::size< TSourceList >::type::value));
                ISAAC_CUDA_CHECK(cudaMalloc((minmax_struct**)&local_minmax_array_d, sizeof(minmax_struct) * local_size[0] * local_size[1]));
            #endif
            
            for (int i = 0; i < 3; i++)
            {
                global_size_scaled.push_back( isaac_int( (isaac_float)global_size[i] * (isaac_float)scale[i] ) );
                 local_size_scaled.push_back( isaac_int( (isaac_float) local_size[i] * (isaac_float)scale[i] ) );
                   position_scaled.push_back( isaac_int( (isaac_float)   position[i] * (isaac_float)scale[i] ) );
            }
            
            background_color[0] = 0;
            background_color[1] = 0;
            background_color[2] = 0;
            background_color[3] = 1;
            
            //INIT
            MPI_Comm_dup(MPI_COMM_WORLD, &mpi_world);
            MPI_Comm_rank(mpi_world, &rank);
            MPI_Comm_size(mpi_world, &numProc);
            if (rank == master)
            {
                this->communicator = new IsaacCommunicator(server_url,server_port);
            }
            else
            {
                this->communicator = NULL;
            }
            setPerspective( 45.0f, (isaac_float)framebuffer_size.x/(isaac_float)framebuffer_size.y,ISAAC_Z_NEAR, ISAAC_Z_FAR);
            look_at[0] = 0.0f;
            look_at[1] = 0.0f;
            look_at[2] = 0.0f;
            ISAAC_SET_IDENTITY(3,rotation)
            distance = -4.5f;
            updateModelview();

            //Create functor chain pointer lookup table
            #if ISAAC_ALPAKA == 1
                const alpaka::Vec<TAccDim, size_t> threads (size_t(1), size_t(1), size_t(1));
                const alpaka::Vec<TAccDim, size_t> blocks  (size_t(1), size_t(1), size_t(1));
                const alpaka::Vec<TAccDim, size_t> grid    (size_t(1), size_t(1), size_t(1));
                auto const workdiv(alpaka::workdiv::WorkDivMembers<TAccDim, size_t>(grid,blocks,threads));
                fillFunctorChainPointerKernel kernel;
                auto const instance
                (
                    alpaka::exec::create<TAcc>
                    (
                        workdiv,
                        kernel,
                        alpaka::mem::view::getPtrNative(functor_chain_d)
                    )
                );
                alpaka::stream::enqueue(stream, instance);                
                alpaka::wait::wait(stream);
            #else
                dim3 grid(1);
                dim3 block(1);
                fillFunctorChainPointerKernel<<<grid,block>>>( functor_chain_d );
                ISAAC_CUDA_CHECK(cudaDeviceSynchronize());
            #endif
            //Init functions:
            for (int i = 0; i < boost::mpl::size< TSourceList >::type::value; i++)
                functions[i].source = std::string("idem");
            updateFunctions();
            
            //non persistent buffer memory
            isaac_for_each_params(sources,allocate_pointer_array_iterator(),pointer_array,local_size
            #if ISAAC_ALPAKA == 1
                ,pointer_array_alpaka
                ,acc
            #endif
            );
            
            //Transfer func memory:
            for (int i = 0; i < boost::mpl::size< TSourceList >::type::value; i++)
            {
                source_weight.value[i] = isaac_float(1);
                #if ISAAC_ALPAKA == 1
                    transfer_d_buf.push_back( alpaka::mem::buf::Buf<TDevAcc, isaac_float4, TTexDim, size_t> ( alpaka::mem::buf::alloc<isaac_float4, size_t>( acc, alpaka::Vec<TTexDim, size_t> ( TTransfer_size ) ) ) );
                    transfer_h_buf.push_back( alpaka::mem::buf::Buf<  THost, isaac_float4, TTexDim, size_t> ( alpaka::mem::buf::alloc<isaac_float4, size_t>(host, alpaka::Vec<TTexDim, size_t> ( TTransfer_size ) ) ) );
                    transfer_d.pointer[i] = alpaka::mem::view::getPtrNative( transfer_d_buf[i] );
                    transfer_h.pointer[i] = alpaka::mem::view::getPtrNative( transfer_h_buf[i] );
                #else
                    ISAAC_CUDA_CHECK(cudaMalloc((isaac_float4**)&(transfer_d.pointer[i]), sizeof(isaac_float4)*TTransfer_size));
                    transfer_h.pointer[i] = (isaac_float4*)malloc( sizeof(isaac_float4)*TTransfer_size );
                #endif
                transfer_h.description[i].insert( std::pair< isaac_uint, isaac_float4> (0             , getHSVA(isaac_float(2*i)*M_PI/isaac_float(boost::mpl::size< TSourceList >::type::value),1,1,0) ));
                transfer_h.description[i].insert( std::pair< isaac_uint, isaac_float4> (TTransfer_size, getHSVA(isaac_float(2*i)*M_PI/isaac_float(boost::mpl::size< TSourceList >::type::value),1,1,1) ));
            }
            updateTransfer();

            //ISAAC:
            IceTCommunicator icetComm;
            icetComm = icetCreateMPICommunicator(mpi_world);
            icetContext = icetCreateContext(icetComm);
            icetDestroyMPICommunicator(icetComm);
            icetResetTiles();
            icetAddTile(0, 0, framebuffer_size.x, framebuffer_size.y, master);
            //icetStrategy(ICET_STRATEGY_DIRECT);
            icetStrategy(ICET_STRATEGY_SEQUENTIAL);
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

            icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
            icetSetDepthFormat(ICET_IMAGE_DEPTH_NONE);
            icetCompositeMode(ICET_COMPOSITE_MODE_BLEND);
            icetEnable(ICET_ORDERED_COMPOSITE);

            max_size = max(uint32_t(global_size[0]),uint32_t(global_size[1]));
            if (TSimDim::value > 2)
                max_size = max(uint32_t(global_size[2]),uint32_t(max_size));
            max_size_scaled = max(uint32_t(global_size_scaled[0]),uint32_t(global_size_scaled[1]));
            if (TSimDim::value > 2)
                max_size_scaled = max(uint32_t(global_size_scaled[2]),uint32_t(max_size_scaled));
            updateBounding( );
            icetPhysicalRenderSize(framebuffer_size.x, framebuffer_size.y);
            icetDrawCallback( drawCallBack );

            //JSON
            recreateJSON();
            if (rank == master)
            {
                json_object_set_new( json_root, "type", json_string( "register" ) );
                json_object_set_new( json_root, "name", json_string( name.c_str() ) );
                json_object_set_new( json_root, "nodes", json_integer( numProc ) );
                json_object_set_new( json_root, "framebuffer width", json_integer ( framebuffer_size.x ) );
                json_object_set_new( json_root, "framebuffer height", json_integer ( framebuffer_size.y ) );
                
                json_object_set_new( json_root, "max functors", json_integer( ISAAC_MAX_FUNCTORS ) );
                json_t *json_functors_array = json_array();
                json_object_set_new( json_root, "functors", json_functors_array );
                IsaacFunctorPool functors;
                isaac_for_each_params( functors, functor_2_json_iterator(), json_functors_array );

                json_t *matrix;
                json_object_set_new( json_root, "projection", matrix = json_array() );
                ISAAC_JSON_ADD_MATRIX(matrix,projection,16)
                json_object_set_new( json_root, "position", matrix = json_array() );
                ISAAC_JSON_ADD_MATRIX(matrix,look_at,3)
                json_object_set_new( json_root, "rotation", matrix = json_array() );
                ISAAC_JSON_ADD_MATRIX(matrix,rotation,9)
                json_object_set_new( json_root, "distance", json_real( distance ) );

                json_t *json_sources_array = json_array();
                json_object_set_new( json_root, "sources", json_sources_array );

                isaac_for_each_params( sources, source_2_json_iterator(), json_sources_array );

                json_object_set_new( json_root, "interpolation", json_boolean( interpolation ) );
                json_object_set_new( json_root, "iso surface", json_boolean( iso_surface ) );
                json_object_set_new( json_root, "step", json_real( step ) );
                
                json_object_set_new( json_root, "dimension", json_integer ( TSimDim::value ) );
                json_object_set_new( json_root, "width", json_integer ( global_size_scaled[0] ) );
                if (TSimDim::value > 1)
                    json_object_set_new( json_root, "height", json_integer ( global_size_scaled[1] ) );
                if (TSimDim::value > 2)
                    json_object_set_new( json_root, "depth", json_integer ( global_size_scaled[2] ) );
            }
        }
        void updateBounding()
        {
            if (icet_bounding_box)
            {
                isaac_float f_l_width = (isaac_float)local_size_scaled[0]/(isaac_float)max_size_scaled * 2.0f;
                isaac_float f_l_height = (isaac_float)local_size_scaled[1]/(isaac_float)max_size_scaled * 2.0f;
                isaac_float f_l_depth = 0.0f;
                if (TSimDim::value > 2)
                    f_l_depth = (isaac_float)local_size_scaled[2]/(isaac_float)max_size_scaled * 2.0f;
                isaac_float f_x = (isaac_float)position_scaled[0]/(isaac_float)max_size_scaled * 2.0f - (isaac_float)global_size_scaled[0]/(isaac_float)max_size_scaled;
                isaac_float f_y = (isaac_float)position_scaled[1]/(isaac_float)max_size_scaled * 2.0f - (isaac_float)global_size_scaled[1]/(isaac_float)max_size_scaled;
                isaac_float f_z = 0.0f;
                if (TSimDim::value > 2)
                    f_z = (isaac_float)position_scaled[2]/(isaac_float)max_size_scaled * isaac_float(2) - (isaac_float)global_size_scaled[2]/(isaac_float)max_size_scaled;
                icetBoundingBoxf( f_x, f_x + f_l_width, f_y, f_y + f_l_height, f_z, f_z + f_l_depth);
            }
            else
                icetBoundingVertices(0,0,0,0,NULL);
        }
        void updatePosition( const TDomainSize position )
        {
            this->position = position;
            for (int i = 0; i < 3; i++)
                position_scaled[i] = isaac_int( (isaac_float) position[i] * (isaac_float)scale[i] );
        }
        void updateLocalSize( const TDomainSize local_size )
        {
            this->local_size = local_size;
            for (int i = 0; i < 3; i++)
                local_size_scaled[i] = isaac_int( (isaac_float) local_size[i] * (isaac_float)scale[i] );
        }
        void updateFunctions()
        {
            IsaacFunctorPool functors;
            isaac_float4 isaac_parameter_h[boost::mpl::size< TSourceList >::type::value * ISAAC_MAX_FUNCTORS];
            for (int i = 0; i < boost::mpl::size< TSourceList >::type::value; i++)
            {
                functions[i].error_code = 0;
                //Going from | to |...
                std::string source = functions[i].source;
                size_t pos = 0;
                bool again = true;
                int elem = 0;
                while (again && ((pos = source.find("|")) != std::string::npos || ((again = false) == false)) )
                {
                    if (elem >= ISAAC_MAX_FUNCTORS)
                    {
                        functions[i].error_code = 1;
                        break;
                    }
                    std::string token = source.substr(0, pos);
                    source.erase(0, pos + 1);
                    //ignore " " in token
                    token.erase(remove_if(token.begin(), token.end(), isspace), token.end());
                    //search "(" and parse parameters
                    size_t t_begin = token.find("(");
                    if (t_begin == std::string::npos)
                        memset(&(isaac_parameter_h[i * ISAAC_MAX_FUNCTORS + elem]), 0, sizeof(isaac_float4));
                    else
                    {
                        size_t t_end = token.find(")");
                        if (t_end == std::string::npos)
                        {
                            functions[i].error_code = -1;
                            break;
                        }
                        if (t_end - t_begin == 1) //()
                            memset(&(isaac_parameter_h[i * ISAAC_MAX_FUNCTORS + elem]), 0, sizeof(isaac_float4));
                        else
                        {
                            std::string parameters = token.substr(t_begin+1, t_end-t_begin-1);
                            size_t p_pos = 0;
                            bool p_again = true;
                            int p_elem = 0;
                            isaac_float* parameter_array = (float*)&(isaac_parameter_h[i * ISAAC_MAX_FUNCTORS + elem]);
                            while (p_again && ((p_pos = parameters.find(",")) != std::string::npos || ((p_again = false) == false)) )
                            {
                                if (p_elem >= 4)
                                {
                                    functions[i].error_code = 2;
                                    break;
                                }
                                std::string par = parameters.substr(0, p_pos);
                                parameters.erase(0, p_pos + 1);
                                parameter_array[p_elem] = std::stof( par );
                                p_elem++;
                            }
                            for (;p_elem < 4; p_elem++)
                                parameter_array[p_elem] = parameter_array[p_elem - 1]; //last one repeated
                        }
                    }
                    //parse token itself
                    if (t_begin != std::string::npos)
                        token = token.substr(0, t_begin);
                    bool found = false;
                    isaac_for_each_params( functors, parse_functor_iterator(), token, functions[i].bytecode[elem], found );
                    if (!found)
                    {
                        functions[i].error_code = -2;
                        break;
                    }
                    
                    elem++;
                }
                for (;elem < ISAAC_MAX_FUNCTORS; elem++)
                {
                    functions[i].bytecode[elem] = 0; //last one idem
                    memset(&(isaac_parameter_h[i * ISAAC_MAX_FUNCTORS + elem]), 0, sizeof(isaac_float4));
                }
            }

            //Calculate functor chain nr per source
            dest_array_struct< boost::mpl::size< TSourceList >::type::value > dest;
            isaac_for_each_params( sources, update_functor_chain_iterator(), functions, dest);
            #if ISAAC_ALPAKA == 1
                alpaka::mem::buf::BufPlainPtrWrapper<THost, isaac_float4, TFraDim, size_t> parameter_buffer(isaac_parameter_h, host, size_t(ISAAC_MAX_FUNCTORS * boost::mpl::size< TSourceList >::type::value));
                alpaka::mem::view::copy( stream, parameter_d, parameter_buffer, size_t(ISAAC_MAX_FUNCTORS * boost::mpl::size< TSourceList >::type::value));
                const alpaka::Vec<TAccDim, size_t> threads (size_t(1), size_t(1), size_t(1));
                const alpaka::Vec<TAccDim, size_t> blocks  (size_t(1), size_t(1), size_t(1));
                const alpaka::Vec<TAccDim, size_t> grid    (size_t(1), size_t(1), size_t(1));
                auto const workdiv(alpaka::workdiv::WorkDivMembers<TAccDim, size_t>(grid,blocks,threads));
                updateFunctorChainPointerKernel
                <
                    boost::mpl::size< TSourceList >::type::value,
                    dest_array_struct< boost::mpl::size< TSourceList >::type::value >
                > kernel;
                auto const instance
                (
                    alpaka::exec::create<TAcc>
                    (
                        workdiv,
                        kernel,
                        alpaka::mem::view::getPtrNative(functor_chain_choose_d),
                        alpaka::mem::view::getPtrNative(functor_chain_d),
                        dest
                    )
                );
                alpaka::stream::enqueue(stream, instance);                
                alpaka::wait::wait(stream);                
            #else
                ISAAC_CUDA_CHECK(cudaMemcpyToSymbol( isaac_parameter_d, isaac_parameter_h, sizeof( isaac_parameter_h )));
                dim3 grid(1);
                dim3 block(1);
                updateFunctorChainPointerKernel< boost::mpl::size< TSourceList >::type::value > <<<grid,block>>>(functor_chain_choose_d, functor_chain_d, dest);
                ISAAC_CUDA_CHECK(cudaDeviceSynchronize());
                isaac_functor_chain_pointer_N* constant_ptr;
                ISAAC_CUDA_CHECK(cudaGetSymbolAddress((void **)&constant_ptr, isaac_function_chain_d));
                ISAAC_CUDA_CHECK(cudaMemcpy(constant_ptr, functor_chain_choose_d, boost::mpl::size< TSourceList >::type::value * sizeof( isaac_functor_chain_pointer_N ), cudaMemcpyDeviceToDevice));
            #endif
        }
        void updateTransfer()
        {
            for (int i = 0; i < boost::mpl::size< TSourceList >::type::value; i++)
            {
                auto next = transfer_h.description[i].begin();
                auto before = next;
                for(next++; next != transfer_h.description[i].end(); next++)
                {
                    isaac_uint width = next->first - before->first;
                    for (size_t j = 0; j < width && j + before->first < TTransfer_size; j++)
                    {
                        transfer_h.pointer[i][before->first + j] = (
                            before->second * isaac_float(width-j) +
                              next->second * isaac_float(j)
                            ) / isaac_float( width );
                    }
                    before = next;
                }
                #if ISAAC_ALPAKA == 1
                    alpaka::mem::view::copy(stream, transfer_d_buf[i], transfer_h_buf[i], TTransfer_size );
                #else
                    ISAAC_CUDA_CHECK(cudaMemcpy(transfer_d.pointer[i], transfer_h.pointer[i], sizeof(isaac_float4)*TTransfer_size, cudaMemcpyHostToDevice));
                #endif
            }
        }
        json_t* getJsonMetaRoot()
        {
            ISAAC_WAIT_VISUALIZATION
            return json_meta_root;
        }
        int init()
        {
            int failed = 0;
            if (communicator && communicator->serverConnect())
                failed = 1;
            MPI_Bcast(&failed,1, MPI_INT, master, mpi_world);
            if (failed)
                return -1;
            if (rank == master)
            {
                char* buffer = json_dumps( json_root, 0 );
                communicator->serverSend(buffer);
                free(buffer);
                json_decref( json_root );
                recreateJSON();
            }
            return 0;
        }
        json_t* doVisualization( const IsaacVisualizationMetaEnum metaTargets = META_MASTER, void* pointer = NULL, bool redraw = true)
        {
            if (redraw)
            {
                ISAAC_START_TIME_MEASUREMENT( buffer, getTicksUs() )
                isaac_for_each_params( sources, update_pointer_array_iterator(), pointer_array, local_size, source_weight, pointer
                #if ISAAC_ALPAKA == 1
                    ,stream
                #endif
                );
                ISAAC_STOP_TIME_MEASUREMENT( buffer_time, +=, buffer, getTicksUs() )
            }
            //if (rank == master)
            //    printf("-----\n");
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

            //Handle messages
            json_t* message;
            char message_buffer[ISAAC_MAX_RECEIVE] = "{}";
            //Master merges all messages and broadcasts it.

            if (rank == master)
            {
                message = json_object();
                bool add_modelview = false;
                while (json_t* last = communicator->getLastMessage())
                {
                    json_t* js;
                    size_t index;
                    json_t *value;
                    //search for update requests
                    if ( js = json_object_get(last, "request") )
                    {
                        const char* target = json_string_value( js );
                        if ( strcmp( target, "rotation" ) == 0 )
                            send_rotation = true;
                        if ( strcmp( target, "position" ) == 0 )
                            send_look_at = true;
                        if ( strcmp( target, "distance" ) == 0 )
                            send_distance = true;
                        if ( strcmp( target, "projection" ) == 0 )
                            send_projection = true;
                        if ( strcmp( target, "transfer" ) == 0 )
                            send_transfer = true;
                        if ( strcmp( target, "interpolation" ) == 0 )
                            send_interpolation = true;
                        if ( strcmp( target, "step" ) == 0 )
                            send_step = true;
                        if ( strcmp( target, "iso surface" ) == 0 )
                            send_iso_surface = true;
                        if ( strcmp( target, "functions" ) == 0 )
                            send_functions = true;
                        if ( strcmp( target, "weight" ) == 0 )
                            send_weight = true;
                        if ( strcmp( target, "background color" ) == 0 )
                            send_background_color = true;
                    }
                    //Search for scene changes
                    if (json_array_size( js = json_object_get(last, "rotation absolute") ) == 9)
                    {
                        add_modelview = true;
                        send_rotation = true;
                        json_array_foreach(js, index, value)
                            rotation[index] = json_number_value( value );
                        json_object_del( last, "rotation absolute" );
                    }
                    if (json_array_size( js = json_object_get(last, "rotation relative") ) == 9)
                    {
                        add_modelview = true;
                        send_rotation = true;
                        IceTDouble relative[9];
                        IceTDouble new_rotation[9];
                        json_array_foreach(js, index, value)
                            relative[index] = json_number_value( value );
                        for (isaac_int x = 0; x < 3; x++)
                            for (isaac_int y = 0; y < 3; y++)
                                new_rotation[y+x*3] = relative[y+0*3] * rotation[0+x*3]
                                                    + relative[y+1*3] * rotation[1+x*3]
                                                    + relative[y+2*3] * rotation[2+x*3];
                        memcpy(rotation, new_rotation, 9 * sizeof(IceTDouble) );
                        json_object_del( last, "rotation relative" );
                    }
                    if (json_array_size( js = json_object_get(last, "rotation axis") ) == 4)
                    {
                        IceTDouble relative[9];
                        IceTDouble x = json_number_value( json_array_get( js, 0 ) );
                        IceTDouble y = json_number_value( json_array_get( js, 1 ) );
                        IceTDouble z = json_number_value( json_array_get( js, 2 ) );
                        IceTDouble rad = json_number_value( json_array_get( js, 3 ) );
                        IceTDouble s = sin( rad * M_PI / 180.0);
                        IceTDouble c = cos( rad * M_PI / 180.0);
                        IceTDouble l = sqrt( x * x + y * y + z * z);
                        if ( l != 0.0 )
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
                            for (isaac_int x = 0; x < 3; x++)
                                for (isaac_int y = 0; y < 3; y++)
                                    new_rotation[y+x*3] = relative[y+0*3] * rotation[0+x*3]
                                                        + relative[y+1*3] * rotation[1+x*3]
                                                        + relative[y+2*3] * rotation[2+x*3];
                            memcpy(rotation, new_rotation, 9 * sizeof(IceTDouble) );
                        }
                        json_object_del( last, "rotation axis" );
                    }
                    if (json_array_size( js = json_object_get(last, "position absolute") ) == 3)
                    {
                        add_modelview = true;
                        send_look_at = true;
                        json_array_foreach(js, index, value)
                            look_at[index] = json_number_value( value );
                        json_object_del( last, "position absolute" );
                    }
                    if (json_array_size( js = json_object_get(last, "position relative") ) == 3)
                    {
                        add_modelview = true;
                        send_look_at = true;
                        IceTDouble add[3];
                        json_array_foreach(js, index, value)
                            add[index] = json_number_value( value );
                        IceTDouble add_p[3] =
                        {
                            rotation[0] * add[0] + rotation[1] * add[1] + rotation[2] * add[2],
                            rotation[3] * add[0] + rotation[4] * add[1] + rotation[5] * add[2],
                            rotation[6] * add[0] + rotation[7] * add[1] + rotation[8] * add[2]
                        };
                        look_at[0] += add_p[0];
                        look_at[1] += add_p[1];
                        look_at[2] += add_p[2];
                        json_object_del( last, "position relative" );
                    }
                    if ( js = json_object_get(last, "distance absolute") )
                    {
                        add_modelview = true;
                        send_distance = true;
                        distance = json_number_value( js );
                        json_object_del( last, "distance absolute" );
                    }
                    if ( js = json_object_get(last, "distance relative") )
                    {
                        add_modelview = true;
                        send_distance = true;
                        distance += json_number_value( js );
                        json_object_del( last, "distance relative" );
                    }
                    mergeJSON(message,last);
                    json_decref( last );
                }
                if (add_modelview)
                {
                    redraw = true;
                    updateModelview();
                    json_t *matrix;
                    json_object_set_new( message, "modelview", matrix = json_array() );
                    ISAAC_JSON_ADD_MATRIX(matrix,modelview,16)
                }
                char* buffer = json_dumps( message, 0 );
                strncpy( message_buffer, buffer, ISAAC_MAX_RECEIVE-1);
                message_buffer[ ISAAC_MAX_RECEIVE-1 ] = 0;
                free(buffer);
                int l = strlen( message_buffer );
                MPI_Bcast( &l, 1, MPI_INT, master, mpi_world);
                MPI_Bcast( message_buffer, l, MPI_CHAR, master, mpi_world);
            }
            else //The others just get the message
            {
                int l;
                MPI_Bcast( &l, 1, MPI_INT, master, mpi_world);
                MPI_Bcast( message_buffer, l, MPI_CHAR, master, mpi_world);
                message_buffer[l] = 0;
                message = json_loads(message_buffer, 0, NULL);
            }

            json_t* js;
            size_t index;
            json_t *value;

            //search for requests for all ranks
            if ( js = json_object_get(message, "request") )
            {
                const char* target = json_string_value( js );
                if ( strcmp( target, "redraw" ) == 0 )
                    redraw = true;
                if ( strcmp( target, "minmax" ) == 0 )
                    send_minmax = true;
            }
            
            //Scene set?
            if (json_array_size( js = json_object_get(message, "projection") ) == 16)
            {
                redraw = true;
                send_projection = true;
                json_array_foreach(js, index, value)
                    projection[index] = json_number_value( value );
            }
            if (rank!= master && json_array_size( js = json_object_get(message, "modelview") ) == 16)
            {
                redraw = true;
                json_array_foreach(js, index, value)
                    modelview[index] = json_number_value( value );
            }
            if (json_array_size( js = json_object_get(message, "transfer points") ) )
            {
                redraw = true;
                json_array_foreach(js, index, value)
                {
                    transfer_h.description[index].clear();
                    size_t index_2;
                    json_t *element;
                    json_array_foreach(value, index_2, element)
                    {
                        transfer_h.description[index].insert( std::pair< isaac_uint, isaac_float4> (
                            isaac_uint( json_number_value( json_object_get( element, "value" ) ) ), {
                                isaac_float( json_number_value( json_object_get( element, "r" ) ) ),
                                isaac_float( json_number_value( json_object_get( element, "g" ) ) ),
                                isaac_float( json_number_value( json_object_get( element, "b" ) ) ),
                                isaac_float( json_number_value( json_object_get( element, "a" ) ) ) } ) );
                    }
                }
                updateTransfer();
                send_transfer = true;
            }
            if ( js = json_object_get(message, "interpolation") )
            {
                redraw = true;
                interpolation = json_boolean_value ( js );
                send_interpolation = true;
            }
            if ( js = json_object_get(message, "step") )
            {
                redraw = true;
                step = json_number_value ( js );
                if (step < isaac_float(0.01))
                    step = isaac_float(0.01);
                send_step = true;
            }
            if ( js = json_object_get(message, "iso surface") )
            {
                redraw = true;
                iso_surface = json_boolean_value ( js );
                send_iso_surface = true;
            }
            if (json_array_size( js = json_object_get(message, "functions") ) )
            {
                redraw = true;
                json_array_foreach(js, index, value)
                    functions[index].source = std::string(json_string_value(value));
                updateFunctions();
                send_functions = true;
            }
            if (json_array_size( js = json_object_get(message, "weight") ) )
            {
                redraw = true;
                json_array_foreach(js, index, value)
                    source_weight.value[index] = json_number_value(value);
                send_weight = true;
            }
            if (js = json_object_get(message, "bounding box") )
            {
                redraw = true;
                icet_bounding_box = !icet_bounding_box;
                updateBounding( );
            }
            if (json_array_size( js = json_object_get(message, "background color") ) == 3 )
            {
                redraw = true;
                json_array_foreach(js, index, value)
                    background_color[index] = json_number_value( value );
                if (background_color[0] == 0.0f &&
                    background_color[1] == 0.0f && 
                    background_color[2] == 0.0f )
                    icetDisable(ICET_CORRECT_COLORED_BACKGROUND);
                else
                    icetEnable(ICET_CORRECT_COLORED_BACKGROUND);
                send_background_color = true;
            }
            
            json_t* metadata = json_object_get( message, "metadata" );
            if (metadata)
                json_incref(metadata);
            json_decref(message);
            thr_metaTargets = metaTargets;
            
            if (send_minmax)
            {
                isaac_for_each_params( sources, calc_minmax_iterator(), pointer_array, minmax_array, local_minmax_array_d, local_size
                #if ISAAC_ALPAKA == 1
                    ,stream
                    ,host
                    ,functor_chain_choose_d
                    ,parameter_d
                #endif
                );
                if (rank == master)
                {
                    MPI_Reduce( MPI_IN_PLACE, minmax_array.min, boost::mpl::size< TSourceList >::type::value, MPI_FLOAT, MPI_MIN, master, mpi_world);
                    MPI_Reduce( MPI_IN_PLACE, minmax_array.max, boost::mpl::size< TSourceList >::type::value, MPI_FLOAT, MPI_MAX, master, mpi_world);
                }
                else
                {
                    MPI_Reduce( minmax_array.min, NULL, boost::mpl::size< TSourceList >::type::value, MPI_FLOAT, MPI_MIN, master, mpi_world);
                    MPI_Reduce( minmax_array.max, NULL, boost::mpl::size< TSourceList >::type::value, MPI_FLOAT, MPI_MAX, master, mpi_world);
                }
            }
            
            IceTImage image = { NULL };
            
            if (redraw)
            {
                //Calc order
                ISAAC_START_TIME_MEASUREMENT( sorting, getTicksUs() )
                //Every rank calculates it's distance to the camera
                IceTDouble point[4] =
                {
                    (IceTDouble(position_scaled[0]) + (IceTDouble(local_size_scaled[0]) - IceTDouble(global_size_scaled[0])) / 2.0) / IceTDouble(max_size_scaled/2),
                    (IceTDouble(position_scaled[1]) + (IceTDouble(local_size_scaled[1]) - IceTDouble(global_size_scaled[1])) / 2.0) / IceTDouble(max_size_scaled/2),
                    (IceTDouble(position_scaled[2]) + (IceTDouble(local_size_scaled[2]) - IceTDouble(global_size_scaled[2])) / 2.0) / IceTDouble(max_size_scaled/2),
                    1.0
                };
                IceTDouble result[4];
                mulMatrixVector(result,modelview,point);
                float point_distance = sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);
                //Allgather of the distances
                float receive_buffer[numProc];
                MPI_Allgather( &point_distance, 1, MPI_FLOAT, receive_buffer, 1, MPI_FLOAT, mpi_world);
                //Putting to a std::multimap of {rank, distance}
                std::multimap<float, isaac_int, std::less< float > > distance_map;
                for (isaac_int i = 0; i < numProc; i++)
                    distance_map.insert( std::pair<float, isaac_int>( receive_buffer[i], i ) );
                //Putting in an array for IceT
                IceTInt icet_order_array[numProc];
                {
                    isaac_int i = 0;
                    for (auto it = distance_map.begin(); it != distance_map.end(); it++)
                    {
                        icet_order_array[i] = it->second;
                        i++;
                    }
                }
                icetCompositeOrder( icet_order_array );
                ISAAC_STOP_TIME_MEASUREMENT( sorting_time, +=, sorting, getTicksUs() )

                //Drawing
                ISAAC_START_TIME_MEASUREMENT( merge, getTicksUs() )
                image = icetDrawFrame(projection,modelview,background_color);
                ISAAC_STOP_TIME_MEASUREMENT( merge_time, +=, merge, getTicksUs() )
            }
            else
                usleep(10000);

            //Message merging
            char* buffer = json_dumps( json_root, 0 );
            strcpy( message_buffer, buffer );
            free(buffer);
            if ( metaTargets == META_MERGE )
            {
                if (rank == master)
                {
                    char receive_buffer[numProc][ISAAC_MAX_RECEIVE];
                    MPI_Gather( message_buffer, ISAAC_MAX_RECEIVE, MPI_CHAR, receive_buffer, ISAAC_MAX_RECEIVE, MPI_CHAR, master, mpi_world);
                    for (isaac_int i = 0; i < numProc; i++)
                    {
                        if (i == master)
                            continue;
                        json_t* js = json_loads(receive_buffer[i], 0, NULL);
                        mergeJSON( json_root, js );
                        json_decref( js );
                    }
                }
                else
                    MPI_Gather( message_buffer, ISAAC_MAX_RECEIVE, MPI_CHAR, NULL, 0,  MPI_CHAR, master, mpi_world);
            }
            
            #ifdef ISAAC_THREADING
                pthread_create(&visualizationThread,NULL,visualizationFunction,image.opaque_internals);
            #else
                visualizationFunction(image.opaque_internals);
            #endif
            return metadata;
        }
        ~IsaacVisualization()
        {
            ISAAC_WAIT_VISUALIZATION
            json_decref( json_root );
            if (rank == master)
            {
                json_root = json_object();
                json_object_set_new( json_root, "type", json_string( "exit" ) );
                char* buffer = json_dumps( json_root, 0 );
                communicator->serverSend(buffer);
                free(buffer);
                json_decref( json_root );
            }
            icetDestroyContext(icetContext);
            delete communicator;
        }    
        uint64_t getTicksUs()
        {
            struct timespec ts;
            if (clock_gettime(CLOCK_MONOTONIC_RAW,&ts) == 0)
                return ts.tv_sec*1000000 + ts.tv_nsec/1000;
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
            IceTImage result)
        {
            #if ISAAC_ALPAKA == 1
                alpaka::mem::buf::Buf<THost, isaac_float, TFraDim, size_t> inverse_h_buf ( alpaka::mem::buf::alloc<isaac_float, size_t>(myself->host, size_t(16)));
                alpaka::mem::buf::Buf<THost, isaac_size_struct< TSimDim::value >, TFraDim, size_t> size_h_buf ( alpaka::mem::buf::alloc<isaac_size_struct< TSimDim::value >, size_t>(myself->host, size_t(1)));
                isaac_float* inverse_h = reinterpret_cast<float*>(alpaka::mem::view::getPtrNative(inverse_h_buf));
                isaac_size_struct< TSimDim::value >* size_h = reinterpret_cast<isaac_size_struct< TSimDim::value >*>(alpaka::mem::view::getPtrNative(size_h_buf));
            #else
                isaac_float inverse_h[16];
                isaac_size_struct< TSimDim::value > size_h[1];
            #endif
            IceTDouble inverse[16];
            calcInverse(inverse,projection_matrix,modelview_matrix);
            for (int i = 0; i < 16; i++)
                inverse_h[i] = static_cast<float>(inverse[i]);

            size_h[0].global_size.value.x = myself->global_size[0];
            size_h[0].global_size.value.y = myself->global_size[1];
            if (TSimDim::value > 2)
                size_h[0].global_size.value.z = myself->global_size[2];
            size_h[0].position.value.x = myself->position[0];
            size_h[0].position.value.y = myself->position[1];
            if (TSimDim::value > 2)
                size_h[0].position.value.z = myself->position[2];
            size_h[0].local_size.value.x = myself->local_size[0];
            size_h[0].local_size.value.y = myself->local_size[1];
            if (TSimDim::value > 2)
                size_h[0].local_size.value.z = myself->local_size[2];
            size_h[0].max_global_size = static_cast<float>(max(max(uint32_t(myself->global_size[0]),uint32_t(myself->global_size[1])),uint32_t(myself->global_size[2])));

            size_h[0].global_size_scaled.value.x = myself->global_size_scaled[0];
            size_h[0].global_size_scaled.value.y = myself->global_size_scaled[1];
            if (TSimDim::value > 2)
                size_h[0].global_size_scaled.value.z = myself->global_size_scaled[2];
            size_h[0].position_scaled.value.x = myself->position_scaled[0];
            size_h[0].position_scaled.value.y = myself->position_scaled[1];
            if (TSimDim::value > 2)
                size_h[0].position_scaled.value.z = myself->position_scaled[2];
            size_h[0].local_size_scaled.value.x = myself->local_size_scaled[0];
            size_h[0].local_size_scaled.value.y = myself->local_size_scaled[1];
            if (TSimDim::value > 2)
                size_h[0].local_size_scaled.value.z = myself->local_size_scaled[2];
            size_h[0].max_global_size_scaled = static_cast<float>(max(max(uint32_t(myself->global_size_scaled[0]),uint32_t(myself->global_size_scaled[1])),uint32_t(myself->global_size_scaled[2])));
            
            isaac_float3 isaac_scale =
            {
                myself->scale[0],
                myself->scale[1],
                myself->scale[2]
            };
            
            #if ISAAC_ALPAKA == 1
                alpaka::mem::view::copy(myself->stream, myself->inverse_d, inverse_h_buf, size_t(16));
                alpaka::mem::view::copy(myself->stream, myself->size_d, size_h_buf, size_t(1));
            #else
                ISAAC_CUDA_CHECK(cudaMemcpyToSymbol( isaac_inverse_d, inverse_h, 16 * sizeof(float)));
                ISAAC_CUDA_CHECK(cudaMemcpyToSymbol( isaac_size_d, size_h, sizeof(isaac_size_struct< TSimDim::value >)));
            #endif
            IceTUByte* pixels = icetImageGetColorub(result);
            ISAAC_START_TIME_MEASUREMENT( kernel, myself->getTicksUs() )
            isaac_float4 bg_color =
            {
                isaac_float(background_color[3]),
                isaac_float(background_color[2]),
                isaac_float(background_color[1]),
                isaac_float(background_color[0])
            };
            isaac_uint2 framebuffer_start =
            {
                isaac_uint( readback_viewport[0] ),
                isaac_uint( readback_viewport[1] )
            };
            #if ISAAC_ALPAKA == 1
                IsaacFillRectKernelStruct
                <
                    TSimDim,
                    TSourceList,
                    transfer_d_struct< boost::mpl::size< TSourceList >::type::value >,
                    source_weight_struct< boost::mpl::size< TSourceList >::type::value >,
                    pointer_array_struct< boost::mpl::size< TSourceList >::type::value >,
                    mpl::vector<>,
                    alpaka::mem::buf::Buf<TDevAcc, uint32_t, TFraDim, size_t>,
                    TTransfer_size,
                    isaac_float3,
                    TAccDim,
                    TAcc,
                    TStream,
                    alpaka::mem::buf::Buf<TDevAcc, float, TFraDim, size_t>,
                    alpaka::mem::buf::Buf<TDevAcc, isaac_size_struct< TSimDim::value >, TFraDim, size_t>,
                    alpaka::mem::buf::Buf<TDevAcc, isaac_float4, TFraDim, size_t>,                    
                    alpaka::mem::buf::Buf<TDevAcc, isaac_functor_chain_pointer_N, TFraDim, size_t>,                    
                    boost::mpl::size< TSourceList >::type::value
                >
                ::call(
                    myself->inverse_d,
                    myself->size_d,
                    myself->parameter_d,
                    myself->stream,
                    myself->functor_chain_choose_d,
                    myself->framebuffer,
                    myself->framebuffer_size,
                    framebuffer_start,
                    myself->sources,
                    myself->step,
                    bg_color,
                    myself->transfer_d,
                    myself->source_weight,
                    myself->pointer_array,
                    readback_viewport,
                    myself->interpolation,
                    myself->iso_surface,
                    isaac_scale
                );
                alpaka::wait::wait(myself->stream);
                ISAAC_STOP_TIME_MEASUREMENT( myself->kernel_time, +=, kernel, myself->getTicksUs() )
                ISAAC_START_TIME_MEASUREMENT( copy, myself->getTicksUs() )
                alpaka::mem::buf::BufPlainPtrWrapper<THost, uint32_t, TFraDim, size_t> result_buffer((uint32_t*)(pixels), myself->host, myself->framebuffer_prod);
                alpaka::mem::view::copy(myself->stream, result_buffer, myself->framebuffer, myself->framebuffer_prod);
            #else
                IsaacFillRectKernelStruct
                <
                    TSimDim,
                    TSourceList,
                    transfer_d_struct< boost::mpl::size< TSourceList >::type::value >,
                    source_weight_struct< boost::mpl::size< TSourceList >::type::value >,
                    pointer_array_struct< boost::mpl::size< TSourceList >::type::value >,
                    mpl::vector<>,
                    uint32_t*,
                    TTransfer_size,
                    isaac_float3,
                    boost::mpl::size< TSourceList >::type::value
                >
                ::call(
                    myself->framebuffer,
                    myself->framebuffer_size,
                    framebuffer_start,
                    myself->sources,
                    myself->step,
                    bg_color,
                    myself->transfer_d,
                    myself->source_weight,
                    myself->pointer_array,
                    readback_viewport,
                    myself->interpolation,
                    myself->iso_surface,
                    isaac_scale
                );
                ISAAC_CUDA_CHECK(cudaDeviceSynchronize());
                ISAAC_STOP_TIME_MEASUREMENT( myself->kernel_time, +=, kernel, myself->getTicksUs() )
                ISAAC_START_TIME_MEASUREMENT( copy, myself->getTicksUs() )
                ISAAC_CUDA_CHECK(cudaMemcpy((uint32_t*)(pixels), myself->framebuffer, sizeof(uint32_t)*myself->framebuffer_prod, cudaMemcpyDeviceToHost));
            #endif
            ISAAC_STOP_TIME_MEASUREMENT( myself->copy_time, +=, copy, myself->getTicksUs() )
        }

        static void* visualizationFunction(void* dummy)
        {
            //Message sending
            if (myself->rank == myself->master)
            {
                json_object_set_new( myself->json_root, "type", json_string( "period" ) );
                json_object_set_new( myself->json_root, "meta nr", json_integer( myself->metaNr ) );

                json_t *matrix;
                if ( myself->send_projection )
                {
                    json_object_set_new( myself->json_root, "projection", matrix = json_array() );
                    ISAAC_JSON_ADD_MATRIX(matrix,myself->projection,16)
                }
                if ( myself->send_look_at )
                {
                    json_object_set_new( myself->json_root, "position", matrix = json_array() );
                    ISAAC_JSON_ADD_MATRIX(matrix,myself->look_at,3)
                }
                if ( myself->send_rotation )
                {
                    json_object_set_new( myself->json_root, "rotation", matrix = json_array() );
                    ISAAC_JSON_ADD_MATRIX(matrix, myself->rotation,9)
                }
                if ( myself->send_distance )
                    json_object_set_new( myself->json_root, "distance", json_real( myself->distance ) );
                if ( myself->send_transfer )
                {
                    json_object_set_new( myself->json_root, "transfer array", matrix = json_array() );
                    for (size_t i = 0; i < boost::mpl::size< TSourceList >::type::value; i++)
                    {
                        json_t* transfer = json_array();
                        json_array_append_new( matrix, transfer );
                        for (size_t j = 0; j < TTransfer_size; j++)
                        {
                            json_t* color = json_array();
                            json_array_append_new( transfer, color );
                            json_array_append_new( color, json_integer( isaac_uint( myself->transfer_h.pointer[i][j].x * isaac_float(255) ) ) );
                            json_array_append_new( color, json_integer( isaac_uint( myself->transfer_h.pointer[i][j].y * isaac_float(255) ) ) );
                            json_array_append_new( color, json_integer( isaac_uint( myself->transfer_h.pointer[i][j].z * isaac_float(255) ) ) );
                            json_array_append_new( color, json_integer( isaac_uint( myself->transfer_h.pointer[i][j].w * isaac_float(255) ) ) );
                        }
                    }
                    json_object_set_new( myself->json_root, "transfer points", matrix = json_array() );
                    for (size_t i = 0; i < boost::mpl::size< TSourceList >::type::value; i++)
                    {
                        json_t* points = json_array();
                        json_array_append_new( matrix, points );
                        for(auto it = myself->transfer_h.description[i].begin(); it != myself->transfer_h.description[i].end(); it++)
                        {
                            json_t* p = json_object();
                            json_array_append_new( points, p);
                            json_object_set_new(p, "value", json_integer( it->first ) );
                            json_object_set_new(p, "r", json_real( it->second.x ) );
                            json_object_set_new(p, "g", json_real( it->second.y ) );
                            json_object_set_new(p, "b", json_real( it->second.z ) );
                            json_object_set_new(p, "a", json_real( it->second.w ) );
                        }
                    }
                }
                if ( myself->send_functions )
                {
                    json_object_set_new( myself->json_root, "functions", matrix = json_array() );
                    for (size_t i = 0; i < boost::mpl::size< TSourceList >::type::value; i++)
                    {
                        json_t* f = json_object();
                        json_array_append_new( matrix, f );
                        json_object_set_new(f, "source", json_string( myself->functions[i].source.c_str() ) );
                        json_object_set_new(f, "error", json_integer( myself->functions[i].error_code ) );
                    }
                }
                if ( myself->send_weight )
                {
                    json_object_set_new( myself->json_root, "weight", matrix = json_array() );
                    for (size_t i = 0; i < boost::mpl::size< TSourceList >::type::value; i++)
                        json_array_append_new( matrix, json_real( myself->source_weight.value[i] ) );
                }
                if ( myself->send_interpolation )
                    json_object_set_new( myself->json_root, "interpolation", json_boolean( myself->interpolation ) );
                if ( myself->send_step )
                    json_object_set_new( myself->json_root, "step", json_real( myself->step ) );
                if ( myself->send_iso_surface )
                    json_object_set_new( myself->json_root, "iso surface", json_boolean( myself->iso_surface ) );
                if ( myself->send_minmax )
                {
                    json_object_set_new( myself->json_root, "minmax", matrix = json_array() );
                    for (size_t i = 0; i < boost::mpl::size< TSourceList >::type::value; i++)
                    {
                        json_t* v = json_object();
                        json_array_append_new( matrix, v );
                        json_object_set_new(v, "min", json_real( myself->minmax_array.min[i] ) );
                        json_object_set_new(v, "max", json_real( myself->minmax_array.max[i] ) );
                    }
                }
                if ( myself->send_background_color )
                {
                    json_object_set_new( myself->json_root, "background color", matrix = json_array() );
                    for (size_t i = 0; i < 3; i++)
                        json_array_append_new( matrix, json_real( myself->background_color[i] ) );
                }
                char* buffer = json_dumps( myself->json_root, 0 );
                myself->communicator->serverSend(buffer);
                free(buffer);
            }

            json_decref( myself->json_root );
            myself->recreateJSON();
            
            //Sending video
            if (dummy)
            {
                IceTImage image = { dummy };
                ISAAC_START_TIME_MEASUREMENT( video_send, myself->getTicksUs() )
                if (myself->communicator)
                    myself->communicator->serverSendFrame(icetImageGetColorui(image),myself->framebuffer_size.x,myself->framebuffer_size.y,4);
                ISAAC_STOP_TIME_MEASUREMENT( myself->video_send_time, +=, video_send, myself->getTicksUs() )
            }
            myself->metaNr++;
            return 0;
        }
        void setFrustum(const isaac_float left,const isaac_float right,const isaac_float bottom,const isaac_float top,const isaac_float znear,const isaac_float zfar )
        {
            isaac_float  znear2 = znear * isaac_float(2);
            isaac_float  width = right - left;
            isaac_float  height = top - bottom;
            isaac_float  zRange = znear - zfar;
            projection[ 0] = znear2 / width;
            projection[ 1] = isaac_float( 0);
            projection[ 2] = isaac_float( 0);
            projection[ 3] = isaac_float( 0);
            projection[ 4] = isaac_float( 0);
            projection[ 5] = znear2 / height;
            projection[ 6] = isaac_float( 0);
            projection[ 7] = isaac_float( 0);
            projection[ 8] = ( right + left ) / width;
            projection[ 9] = ( top + bottom ) / height;
            projection[10] = ( zfar + znear) / zRange;
            projection[11] = isaac_float(-1);
            projection[12] = isaac_float( 0);
            projection[13] = isaac_float( 0);
            projection[14] = ( -znear2 * zfar ) / -zRange;
            projection[15] = isaac_float( 0);
        }
        void setPerspective(const isaac_float fovyInDegrees,const isaac_float aspectRatio,const isaac_float __znear,const isaac_float zfar )
        {
            isaac_float znear = __znear;
            isaac_float ymax = znear * tan( fovyInDegrees * M_PI / isaac_float(360) );
            isaac_float xmax = ymax * aspectRatio;
            setFrustum( -xmax, xmax, -ymax, ymax, znear, zfar );
        }
        void recreateJSON()
        {
            json_root = json_object();
            json_meta_root = json_object();
            json_object_set_new( json_root, "metadata", json_meta_root );
        }
        void updateModelview()
        {
            IceTDouble look_at_m[16];
            ISAAC_SET_IDENTITY(4,look_at_m)
            look_at_m[12] = look_at[0];
            look_at_m[13] = look_at[1];
            look_at_m[14] = look_at[2];
            
            IceTDouble rotation_m[16];
            for (isaac_int x = 0; x < 4; x++)
                for (isaac_int y = 0; y < 4; y++)
                {
                    if (x < 3 && y < 3)
                        rotation_m[x+y*4] = rotation[x+y*3];
                    else
                    if (x!=3 || y!=3)
                        rotation_m[x+y*4] = 0.0;
                    else
                        rotation_m[x+y*4] = 1.0;
                }

            IceTDouble distance_m[16];
            ISAAC_SET_IDENTITY(4,distance_m)
            distance_m[14] = distance;
            
            IceTDouble temp[16];
            
            mulMatrixMatrix( temp, rotation_m, look_at_m );
            mulMatrixMatrix( modelview, distance_m, temp );
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
        #if ISAAC_ALPAKA == 1
            alpaka::Vec<TFraDim, size_t> framebuffer_prod;
            alpaka::mem::buf::Buf<TDevAcc, uint32_t, TFraDim, size_t> framebuffer;
            alpaka::mem::buf::Buf<TDevAcc, float, TFraDim, size_t> inverse_d;
            alpaka::mem::buf::Buf<TDevAcc, isaac_size_struct< TSimDim::value >, TFraDim, size_t> size_d;
            alpaka::mem::buf::Buf<TDevAcc, isaac_functor_chain_pointer_N, TFraDim, size_t> functor_chain_d;
            alpaka::mem::buf::Buf<TDevAcc, isaac_functor_chain_pointer_N, TFraDim, size_t> functor_chain_choose_d;
            alpaka::mem::buf::Buf<TDevAcc, minmax_struct, TFraDim, size_t> local_minmax_array_d;
        #else
            size_t framebuffer_prod;
            isaac_uint* framebuffer;
            isaac_functor_chain_pointer_N* functor_chain_d;
            isaac_functor_chain_pointer_N* functor_chain_choose_d;
            minmax_struct* local_minmax_array_d;
        #endif
        TDomainSize global_size;
        TDomainSize local_size;
        TDomainSize position;
        std::vector<size_t> global_size_scaled;
        std::vector<size_t> local_size_scaled;
        std::vector<size_t> position_scaled;
        MPI_Comm mpi_world;
        IceTDouble projection[16];
        IceTDouble look_at[3];
        IceTDouble rotation[9];
        IceTDouble distance;
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
        bool interpolation;
        bool iso_surface;
        bool icet_bounding_box;
        isaac_float step;
        IceTDouble modelview[16];
        IsaacCommunicator* communicator;
        json_t *json_root;
        json_t *json_meta_root;
        isaac_int rank;
        isaac_int master;
        isaac_int numProc;
        isaac_uint metaNr;
        TSourceList& sources;
        IceTContext icetContext;
        IsaacVisualizationMetaEnum thr_metaTargets;
        pthread_t visualizationThread;
        #if ISAAC_ALPAKA == 1
            std::vector< alpaka::mem::buf::Buf<TDevAcc, isaac_float4, TTexDim, size_t> > transfer_d_buf;
            std::vector< alpaka::mem::buf::Buf<  THost, isaac_float4, TTexDim, size_t> > transfer_h_buf;
            std::vector< alpaka::mem::buf::Buf< TDevAcc, isaac_float, TFraDim, size_t> > pointer_array_alpaka;
            alpaka::mem::buf::Buf<TDevAcc, isaac_float4, TFraDim, size_t> parameter_d;
        #endif
        transfer_d_struct< boost::mpl::size< TSourceList >::type::value > transfer_d;
        transfer_h_struct< boost::mpl::size< TSourceList >::type::value > transfer_h;
        source_weight_struct< boost::mpl::size< TSourceList >::type::value > source_weight;
        pointer_array_struct< boost::mpl::size< TSourceList >::type::value > pointer_array;
        minmax_array_struct< boost::mpl::size< TSourceList >::type::value > minmax_array;
        const static size_t transfer_size = TTransfer_size;
        functions_struct functions[boost::mpl::size< TSourceList >::type::value];
        size_t max_size;
        size_t max_size_scaled;
        IceTFloat background_color[4];
        static IsaacVisualization *myself;
        TScale scale;
};

#if ISAAC_ALPAKA == 1
    template <typename THost,typename TAcc,typename TStream,typename TAccDim,typename TSimDim, typename TSourceList, typename TDomainSize, size_t TTransfer_size,typename TScale>
    IsaacVisualization<THost,TAcc,TStream,TAccDim,TSimDim,TSourceList,TDomainSize,TTransfer_size,TScale>* IsaacVisualization<THost,TAcc,TStream,TAccDim,TSimDim,TSourceList,TDomainSize,TTransfer_size,TScale>::myself = NULL;
#else
    template <typename TSimDim, typename TSourceList, typename TDomainSize, size_t TTransfer_size,typename TScale>
    IsaacVisualization<TSimDim,TSourceList,TDomainSize,TTransfer_size,TScale>* IsaacVisualization<TSimDim,TSourceList,TDomainSize,TTransfer_size,TScale>::myself = NULL;
#endif

} //namespace isaac;
