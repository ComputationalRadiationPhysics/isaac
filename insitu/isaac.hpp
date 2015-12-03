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

#include <boost/config/select_compiler_config.hpp>

#include <string>
#include <string.h>
#include <jansson.h>
#include <pthread.h>
#include <list>
#include <vector>
#include <memory>
#include <mpi.h>
#include <IceT.h>
#include <IceTMPI.h>
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
    size_t TTransfer_func_size
>
class IsaacVisualization 
{
    public:
        #if ISAAC_ALPAKA == 1
            using TDevAcc = alpaka::dev::Dev<TAcc>;
            using TFraDim = alpaka::dim::DimInt<1>;
            using TTexDim = alpaka::dim::DimInt<1>;
        #endif
        using TChainList = boost::fusion::list< IsaacChainLength >;
        
        struct source_2_json_iterator
        {
            template
            <
                typename TSource,
                typename TJsonRoot,
                typename TUnused1,
                typename TUnused2
            >
            ISAAC_HOST_DEVICE_INLINE  void operator()( const int N,TSource& s, TJsonRoot& jsonRoot, TUnused1& unused1, TUnused2& unused2) const
            {
                #ifndef __CUDA_ARCH__
                    json_t *content = json_object();
                    json_array_append_new( jsonRoot, content );
                    json_object_set_new( content, "name", json_string ( s.name.c_str() ) );
                    json_object_set_new( content, "feature dimension", json_integer ( s.feature_dim ) );
                #endif
            }
        };
        
        IsaacVisualization(
            #if ISAAC_ALPAKA == 1
                THost host,
                TDevAcc acc,
                TStream stream,
            #endif
            std::string name,
            isaac_int master,
            std::string server_url,
            isaac_uint server_port,
            isaac_size2 framebuffer_size,
            const TDomainSize global_size,
            const TDomainSize local_size,
            const TDomainSize position,
            TSourceList sources
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
            framebuffer_prod(size_t(framebuffer_size.x) * size_t(framebuffer_size.y)),
            sources( sources )
            #if ISAAC_ALPAKA == 1
                ,framebuffer(alpaka::mem::buf::alloc<uint32_t, size_t>(acc, framebuffer_prod))
                ,inverse_d(alpaka::mem::buf::alloc<isaac_float, size_t>(acc, size_t(16)))
                ,size_d(alpaka::mem::buf::alloc<isaac_size_struct< TSimDim::value >, size_t>(acc, size_t(1)))
        {
            #else
        {
                ISAAC_CUDA_CHECK(cudaMalloc((uint32_t**)&framebuffer, sizeof(uint32_t)*framebuffer_prod));
            #endif
            //INIT
            myself = this;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &numProc);
            if (rank == master)
            {
                this->communicator = new IsaacCommunicator(server_url,server_port);
                this->video_communicator = new IsaacCommunicator(server_url,server_port);
            }
            else
            {
                this->communicator = NULL;
                this->video_communicator = NULL;
            }
            setPerspective( 45.0f, (isaac_float)framebuffer_size.x/(isaac_float)framebuffer_size.y,ISAAC_Z_NEAR, ISAAC_Z_FAR);
            look_at[0] = 0.0f;
            look_at[1] = 0.0f;
            look_at[2] = 0.0f;
            ISAAC_SET_IDENTITY(3,rotation)
            distance = -5.0f;
            updateModelview();
            
            //Transfer func memory:
            for (int i = 0; i < boost::mpl::size< TSourceList >::type::value; i++)
            {
                #if ISAAC_ALPAKA == 1
                    transfer_func_vec[i] = alpaka::mem::buf::Buf<TDevAcc, isaac_float4, TTexDim, size_t> ( alpaka::mem::buf::alloc<isaac_float4, size_t>(acc, alpaka::Vec<TTexDim, size_t> ( TTransfer_func_size ) ) );
                    transfer_funcs.pointer[i] = alpaka::mem::view::getPtrNative( transfer_func_vec[i] );
                    alpaka::mem::buf::Buf<THost, isaac_float4, TTexDim, size_t> transfer_func_h_buf ( alpaka::mem::buf::alloc<isaac_float4, size_t>(host, TTransfer_func_size ) );
                    isaac_float4* transfer_func_h = reinterpret_cast<isaac_float4*>(alpaka::mem::view::getPtrNative(transfer_func_h_buf));
                #else
                    ISAAC_CUDA_CHECK(cudaMalloc((isaac_float4**)&(transfer_funcs.pointer[i]), sizeof(isaac_float4)*TTransfer_func_size));
                    isaac_float4 transfer_func_h[ transfer_func_size ];
                #endif
                for (size_t j = 0; j < TTransfer_func_size; j++)
                {
                    transfer_func_h[j].x = isaac_float(i%2 == 0 ? 1 : 0);
                    transfer_func_h[j].y = isaac_float(i%2 == 0 ? 0 : 1);
                    transfer_func_h[j].z = isaac_float(0);
                    transfer_func_h[j].w = isaac_float(j) / isaac_float( TTransfer_func_size-1);
                }
                #if ISAAC_ALPAKA == 1
                    alpaka::mem::view::copy(stream, transfer_func_vec[i], transfer_func_h_buf, TTransfer_func_size );
                #else
                    ISAAC_CUDA_CHECK(cudaMemcpy(transfer_funcs.pointer[i], transfer_func_h, sizeof(isaac_float4)*TTransfer_func_size, cudaMemcpyHostToDevice));
                #endif
            }

            //ISAAC:
            IceTCommunicator icetComm;
            icetComm = icetCreateMPICommunicator(MPI_COMM_WORLD);
            icetContext = icetCreateContext(icetComm);
            icetDestroyMPICommunicator(icetComm);
            icetResetTiles();
            icetAddTile(0, 0, framebuffer_size.x, framebuffer_size.y, master);
            icetStrategy(ICET_STRATEGY_DIRECT);
            //icetStrategy(ICET_STRATEGY_SEQUENTIAL);
            //icetStrategy(ICET_STRATEGY_REDUCE);
            
            icetSingleImageStrategy( ICET_SINGLE_IMAGE_STRATEGY_AUTOMATIC );
            //icetSingleImageStrategy( ICET_SINGLE_IMAGE_STRATEGY_BSWAP );
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
            
            size_t max_size = max(uint32_t(global_size[0]),uint32_t(global_size[1]));
            if (TSimDim::value > 2)
                max_size = max(uint32_t(global_size[2]),uint32_t(max_size));
            isaac_float f_l_width = (isaac_float)local_size[0]/(isaac_float)max_size * 2.0f;
            isaac_float f_l_height = (isaac_float)local_size[1]/(isaac_float)max_size * 2.0f;
            isaac_float f_l_depth = 0.0f;
            if (TSimDim::value > 2)
                f_l_depth = (isaac_float)local_size[2]/(isaac_float)max_size * 2.0f;
            isaac_float f_x = (isaac_float)position[0]/(isaac_float)max_size * 2.0f - (isaac_float)global_size[0]/(isaac_float)max_size;
            isaac_float f_y = (isaac_float)position[1]/(isaac_float)max_size * 2.0f - (isaac_float)global_size[1]/(isaac_float)max_size;
            isaac_float f_z = 0.0f;
            if (TSimDim::value > 2)
                f_z = (isaac_float)position[2]/(isaac_float)max_size * isaac_float(2) - (isaac_float)global_size[2]/(isaac_float)max_size;
            icetBoundingBoxf( f_x, f_x + f_l_width, f_y, f_y + f_l_height, f_z, f_z + f_l_depth);
            icetPhysicalRenderSize(framebuffer_size.x, framebuffer_size.y);
            icetDrawCallback( drawCallBack );
            
            //JSON
            recreateJSON();
            if (rank == master)
            {
                json_object_set_new( json_root, "name", json_string( name.c_str() ) );
                json_object_set_new( json_root, "nodes", json_integer( numProc ) );
                json_object_set_new( json_root, "framebuffer width", json_integer ( framebuffer_size.x ) );
                json_object_set_new( json_root, "framebuffer height", json_integer ( framebuffer_size.y ) );
                //TODO: Read real values
                json_object_set_new( json_root, "max chain", json_integer( 5 ) );
                json_t *operators = json_array();
                json_object_set_new( json_root, "operators", operators );
                json_array_append_new( operators, json_string( "length(x)" ) );
                json_array_append_new( operators, json_string( "pow(x,c)" ) );
                json_array_append_new( operators, json_string( "add(x,c)" ) );
                json_array_append_new( operators, json_string( "mul(x,c)" ) );

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
                
                isaac_for_each_1_params( sources, source_2_json_iterator(), json_sources_array );

                json_object_set_new( json_root, "dimension", json_integer ( TSimDim::value ) );
                json_object_set_new( json_root, "width", json_integer ( global_size[0] ) );
                if (TSimDim::value > 1)
                    json_object_set_new( json_root, "height", json_integer ( global_size[1] ) );
                if (TSimDim::value > 2)
                    json_object_set_new( json_root, "depth", json_integer ( global_size[2] ) );
                json_object_set_new( json_root, "type", json_string( "register" ) );
            }
        }
        json_t* getJsonMetaRoot()
        {
            return json_meta_root;
        }
        int init()
        {
            isaac_int failed = 0;
            if (communicator && communicator->serverConnect())
                failed = 1;
            if (failed == 0 && video_communicator && video_communicator->serverConnect(true))
                failed = 1;
            MPI_Bcast(&failed,sizeof(failed), MPI_INT, master, MPI_COMM_WORLD);
            if (failed)
                return -1;
            if (rank == master)
            {
                char* buffer = json_dumps( json_root, 0 );
                communicator->serverSend(buffer);
                free(buffer);
                json_decref( json_root );
            
                if (video_communicator)
                {
                    json_root = json_object();
                    json_object_set_new( json_root, "type", json_string( "register video" ) );
                    json_object_set_new( json_root, "name", json_string( name.c_str() ) );
                    char* buffer = json_dumps( json_root, 0 );
                    video_communicator->serverSend(buffer);
                    free(buffer);
                    json_decref( json_root );
                }            
                recreateJSON();
            }
            return 0;
        }
        json_t* doVisualization( IsaacVisualizationMetaEnum metaTargets = META_MASTER )
        {
            //if (rank == master)
            //    printf("-----\n");
            ISAAC_WAIT_VISUALIZATION
            
            send_distance = false;
            send_look_at = false;
            send_projection = false;
            send_rotation = false;

            //Handle messages
            json_t* message;
            char message_buffer[ISAAC_MAX_RECEIVE];
            //Master merges all messages and broadcasts it.
            if (rank == master)
            {
                message = json_object();
                bool add_modelview = false;
                while (json_t* last = communicator->getLastMessage())
                {
                    //Search for scene changes
                    json_t* js;
                    size_t index;
                    json_t *value;
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
                        IceTDouble s = sin( rad * M_PI / 360.0);
                        IceTDouble c = cos( rad * M_PI / 360.0);
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
                    updateModelview();
                    json_t *matrix;
                    json_object_set_new( message, "modelview", matrix = json_array() );
                    ISAAC_JSON_ADD_MATRIX(matrix,modelview,16)
                }
                char* buffer = json_dumps( message, 0 );
                strcpy( message_buffer, buffer );
                free(buffer);
                MPI_Bcast( message_buffer, ISAAC_MAX_RECEIVE, MPI_CHAR, master, MPI_COMM_WORLD);
            }
            else //The others just get the message
            {
                MPI_Bcast( message_buffer, ISAAC_MAX_RECEIVE, MPI_CHAR, master, MPI_COMM_WORLD);
                message = json_loads(message_buffer, 0, NULL);
            }
            
            json_t* js;
            size_t index;
            json_t *value;
            
            //Scene set?
            if (json_array_size( js = json_object_get(message, "projection") ) == 16)
            {
                send_projection = true;
                json_array_foreach(js, index, value)
                    projection[index] = json_number_value( value );
            }
            if (rank!= master && json_array_size( js = json_object_get(message, "modelview") ) == 16)
                json_array_foreach(js, index, value)
                    modelview[index] = json_number_value( value );
                    
            json_t* metadata = json_object_get( message, "metadata" );
            if (metadata)
                json_incref(metadata);
            json_decref(message);
            thr_metaTargets = metaTargets;

           //Calc order
            ISAAC_START_TIME_MEASUREMENT( sorting, myself->getTicksUs() )
            //Every rank calculates it's distance to the camera
            IceTDouble point[4] =
            {
                IceTDouble(myself->position[0]) + (IceTDouble(myself->local_size[0]) - IceTDouble(myself->global_size[0])) / 2.0,
                IceTDouble(myself->position[1]) + (IceTDouble(myself->local_size[1]) - IceTDouble(myself->global_size[1])) / 2.0,
                IceTDouble(myself->position[2]) + (IceTDouble(myself->local_size[2]) - IceTDouble(myself->global_size[2])) / 2.0,
                1.0
            };
            IceTDouble result[4];
            mulMatrixVector(result,myself->modelview,point);
            isaac_float point_distance = sqrt( result[0]*result[0] + result[1]*result[1] + result[2]*result[2] );
            //Allgather of the distances
            isaac_float receive_buffer[myself->numProc];
            MPI_Allgather( &point_distance, 1, MPI_FLOAT, receive_buffer, 1, MPI_FLOAT, MPI_COMM_WORLD);
            //Putting to a std::multimap of {rank, distance}
            std::multimap<isaac_float, isaac_int, std::less< isaac_float > > distance_map;
            for (isaac_int i = 0; i < myself->numProc; i++)
                distance_map.insert( std::pair<isaac_float, isaac_int>( receive_buffer[i], i ) );
            //Putting in an array for IceT
            IceTInt icet_order_array[myself->numProc];
            {
                isaac_int i = 0;
                for (auto it = distance_map.begin(); it != distance_map.end(); it++)
                {
                    icet_order_array[i] = it->second;
                    i++;
                }
            }
            icetCompositeOrder( icet_order_array );
            ISAAC_STOP_TIME_MEASUREMENT( myself->sorting_time, +=, sorting, myself->getTicksUs() )

            //Drawing
            IceTFloat background_color[4] = {0.0f,0.0f,0.0f,1.0f};
            ISAAC_START_TIME_MEASUREMENT( merge, myself->getTicksUs() )
            IceTImage image = icetDrawFrame(myself->projection,myself->modelview,background_color);
            ISAAC_STOP_TIME_MEASUREMENT( myself->merge_time, +=, merge, myself->getTicksUs() )

            //Message sending
            char* buffer = json_dumps( myself->json_root, 0 );
            strcpy( message_buffer, buffer );
            free(buffer);
            if (myself->thr_metaTargets == META_MERGE)
            {
                if (myself->rank == myself->master)
                {
                    char receive_buffer[myself->numProc][ISAAC_MAX_RECEIVE];
                    MPI_Gather( message_buffer, ISAAC_MAX_RECEIVE, MPI_CHAR, receive_buffer, ISAAC_MAX_RECEIVE, MPI_CHAR, myself->master, MPI_COMM_WORLD);
                    for (isaac_int i = 0; i < myself->numProc; i++)
                    {
                        if (i == myself->master)
                            continue;
                        json_t* js = json_loads(receive_buffer[i], 0, NULL);
                        mergeJSON( myself->json_root, js );
                    }
                }
                else
                    MPI_Gather( message_buffer, ISAAC_MAX_RECEIVE, MPI_CHAR, NULL, 0,  MPI_CHAR, myself->master, MPI_COMM_WORLD);
            }
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
                    
                char* buffer = json_dumps( myself->json_root, 0 );
                myself->communicator->serverSend(buffer);
                free(buffer);
            }
            json_decref( myself->json_root );
            myself->recreateJSON();

            #ifdef ISAAC_THREADING
                pthread_create(&visualizationThread,NULL,visualizationFunction,&image);
            #else
                visualizationFunction(&image);
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
    private:        
        static IsaacVisualization *myself;
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
            size_h[0].global_size.x = myself->global_size[0];
            size_h[0].global_size.y = myself->global_size[1];
            if (TSimDim::value > 2)
                size_h[0].global_size.z = myself->global_size[2];
            size_h[0].position.x = myself->position[0];
            size_h[0].position.y = myself->position[1];
            if (TSimDim::value > 2)
                size_h[0].position.z = myself->position[2];
            size_h[0].local_size.x = myself->local_size[0];
            size_h[0].local_size.y = myself->local_size[1];
            if (TSimDim::value > 2)
                size_h[0].local_size.z = myself->local_size[2];
            size_h[0].max_global_size = static_cast<float>(max(max(uint32_t(myself->global_size[0]),uint32_t(myself->global_size[1])),uint32_t(myself->global_size[2])));
            
            #if ISAAC_ALPAKA == 1
                alpaka::mem::view::copy(myself->stream, myself->inverse_d, inverse_h_buf, size_t(16));
                alpaka::mem::view::copy(myself->stream, myself->size_d, size_h_buf, size_t(1));
            #else
                ISAAC_CUDA_CHECK(cudaMemcpyToSymbol( isaac_inverse_d, inverse_h, 16 * sizeof(float)));
                ISAAC_CUDA_CHECK(cudaMemcpyToSymbol( isaac_size_d, size_h, sizeof(isaac_size_struct< TSimDim::value >)));
            #endif
            IceTUByte* pixels = icetImageGetColorub(result);
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
            ISAAC_START_TIME_MEASUREMENT( kernel, myself->getTicksUs() )
            isaac_float step = 1.0f;
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
                if ( boost::mpl::not_<boost::is_same<TAcc, alpaka::acc::AccGpuCudaRt<TAccDim, size_t> > >::value )
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
                auto const test (alpaka::exec::create<TAcc> (workdiv,
                    myself->fillRectKernel,
                    alpaka::mem::view::getPtrNative(myself->inverse_d),
                    alpaka::mem::view::getPtrNative(myself->size_d),
                    alpaka::mem::view::getPtrNative(myself->framebuffer),
                    myself->framebuffer_size,
                    framebuffer_start,
                    myself->sources,
                    step,
                    bg_color,
                    myself->transfer_funcs));
                alpaka::stream::enqueue(myself->stream, test);
                alpaka::wait::wait(myself->stream);
                ISAAC_STOP_TIME_MEASUREMENT( myself->kernel_time, +=, kernel, myself->getTicksUs() )
                ISAAC_START_TIME_MEASUREMENT( copy, myself->getTicksUs() )
                alpaka::mem::buf::BufPlainPtrWrapper<THost, uint32_t, TFraDim, size_t> result_buffer((uint32_t*)(pixels), myself->host, myself->framebuffer_prod);
                alpaka::mem::view::copy(myself->stream, result_buffer, myself->framebuffer, myself->framebuffer_prod);
            #else
                dim3 block (block_size.x, block_size.y);
                dim3 grid  (grid_size.x, grid_size.y);
                IsaacFillRectKernel<TSimDim,TSourceList,TChainList,transfer_func_struct< boost::mpl::size< TSourceList >::type::value >, TTransfer_func_size > <<<grid, block>>>(
                    myself->framebuffer,
                    myself->framebuffer_size,
                    framebuffer_start,
                    myself->sources,
                    step,
                    bg_color,
                    myself->transfer_funcs);
                ISAAC_CUDA_CHECK(cudaDeviceSynchronize());
                ISAAC_STOP_TIME_MEASUREMENT( myself->kernel_time, +=, kernel, myself->getTicksUs() )
                ISAAC_START_TIME_MEASUREMENT( copy, myself->getTicksUs() )
                ISAAC_CUDA_CHECK(cudaMemcpy((uint32_t*)(pixels), myself->framebuffer, sizeof(uint32_t)*myself->framebuffer_prod, cudaMemcpyDeviceToHost));
            #endif
            ISAAC_STOP_TIME_MEASUREMENT( myself->copy_time, +=, copy, myself->getTicksUs() )
        }
        
        static void* visualizationFunction(void* dummy)
        {
            IceTImage* image = (IceTImage*)dummy;
            //Sending
            ISAAC_START_TIME_MEASUREMENT( video_send, myself->getTicksUs() )
            if (myself->video_communicator)
                myself->video_communicator->serverSendFrame(icetImageGetColorui(*image),icetImageGetNumPixels(*image)*4);
            ISAAC_STOP_TIME_MEASUREMENT( myself->video_send_time, +=, video_send, myself->getTicksUs() )

            myself->metaNr++;
            return 0;
        }
        void setFrustum(isaac_float left,isaac_float right,isaac_float bottom,isaac_float top,isaac_float znear,isaac_float zfar )
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
        void setPerspective(isaac_float fovyInDegrees,isaac_float aspectRatio,isaac_float __znear,isaac_float zfar )
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
            TDomainSize global_size;
            TDomainSize local_size;
            TDomainSize position;
            alpaka::mem::buf::Buf<TDevAcc, uint32_t, TFraDim, size_t> framebuffer;
            alpaka::mem::buf::Buf<TDevAcc, float, TFraDim, size_t> inverse_d;
            alpaka::mem::buf::Buf<TDevAcc, isaac_size_struct< TSimDim::value >, TFraDim, size_t> size_d;
        #else
            size_t framebuffer_prod;
            TDomainSize global_size;
            TDomainSize local_size;
            TDomainSize position;        
            isaac_uint* framebuffer;
        #endif
        IceTDouble projection[16];
        IceTDouble look_at[3];
        IceTDouble rotation[9];
        IceTDouble distance;
        bool send_look_at;
        bool send_rotation;
        bool send_distance;
        bool send_projection;
        IceTDouble modelview[16];
        IsaacCommunicator* communicator;
        IsaacCommunicator* video_communicator;
        json_t *json_root;
        json_t *json_meta_root;
        isaac_int rank;
        isaac_int master;
        isaac_int numProc;
        isaac_uint metaNr;
        TSourceList sources;
        IceTContext icetContext;
        IsaacVisualizationMetaEnum thr_metaTargets;
        pthread_t visualizationThread;
        #if ISAAC_ALPAKA == 1
            IsaacFillRectKernel<TSimDim,TSourceList,TChainList,transfer_func_struct< boost::mpl::size< TSourceList >::type::value >, TTransfer_func_size > fillRectKernel;
            std::vector< alpaka::mem::buf::Buf<TDevAcc, isaac_float4, TTexDim, size_t> > transfer_func_vec;
        #endif
        transfer_func_struct< boost::mpl::size< TSourceList >::type::value > transfer_funcs;
        const static size_t transfer_func_size = TTransfer_func_size;
};

#if ISAAC_ALPAKA == 1
    template <typename THost,typename TAcc,typename TStream,typename TAccDim,typename TSimDim, typename TSourceList, typename TDomainSize, size_t TTransfer_func_size>
    IsaacVisualization<THost,TAcc,TStream,TAccDim,TSimDim,TSourceList,TDomainSize,TTransfer_func_size>* IsaacVisualization<THost,TAcc,TStream,TAccDim,TSimDim,TSourceList,TDomainSize,TTransfer_func_size>::myself = NULL;
#else
    template <typename TSimDim, typename TSourceList, typename TDomainSize, size_t TTransfer_func_size>
    IsaacVisualization<TSimDim,TSourceList,TDomainSize,TTransfer_func_size>* IsaacVisualization<TSimDim,TSourceList,TDomainSize,TTransfer_func_size>::myself = NULL;
#endif

} //namespace isaac;
