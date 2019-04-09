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

#define ISAAC_IDX_TYPE int


#include <isaac.hpp>

#include "example_details.hpp"


using namespace isaac;

#define VOLUME_X 64
#define VOLUME_Y 64
#define VOLUME_Z 64

#define PARTICLE_VOLUME_X 64
#define PARTICLE_VOLUME_Y 64
#define PARTICLE_VOLUME_Z 64

#define PARTICLE_COUNT 64


// Volume Source 1

ISAAC_NO_HOST_DEVICE_WARNING
#if ISAAC_ALPAKA == 1

template<
    typename TDevAcc,
    typename THost,
    typename TStream
>
#endif
class TestSource1
{
public:
    static const ISAAC_IDX_TYPE feature_dim = 3;
    static const bool has_guard = false;
    static const bool persistent = true;


    ISAAC_NO_HOST_DEVICE_WARNING TestSource1(
#if ISAAC_ALPAKA == 1
        TDevAcc acc,
        THost host,
        TStream stream,
#endif
        isaac_float3 * ptr
    ) :
        ptr( ptr )
    {}


    ISAAC_HOST_INLINE static std::string getName( )
    {
        return std::string( "Test Source 1" );
    }


    ISAAC_HOST_INLINE void update(
        bool enabled,
        void * pointer
    )
    {}


    isaac_float3 * ptr;

    ISAAC_NO_HOST_DEVICE_WARNING ISAAC_HOST_DEVICE_INLINE
    isaac_float_dim <feature_dim> operator[]( const isaac_int3 & nIndex ) const
    {
        isaac_float3 value = ptr[nIndex.x + nIndex.y * VOLUME_X
                                 + nIndex.z * VOLUME_X * VOLUME_Y];
        isaac_float_dim< 3 > result;
        result.value = value;
        return result;
    }
};


// Volume Source 2

ISAAC_NO_HOST_DEVICE_WARNING
#if ISAAC_ALPAKA == 1

template<
    typename TDevAcc,
    typename THost,
    typename TStream
>
#endif
class TestSource2
{
public:
    static const ISAAC_IDX_TYPE feature_dim = 1;
    static const bool has_guard = false;
    static const bool persistent = false;


    ISAAC_NO_HOST_DEVICE_WARNING TestSource2(
#if ISAAC_ALPAKA == 1
        TDevAcc acc,
        THost host,
        TStream stream,
#endif
        isaac_float * ptr
    ) :
        ptr( ptr )
    {}


    ISAAC_HOST_INLINE static std::string


    getName( )
    {
        return std::string( "Test Source 2" );
    }


    ISAAC_HOST_INLINE void update(
        bool enabled,
        void * pointer
    )
    {}


    isaac_float * ptr;

    ISAAC_NO_HOST_DEVICE_WARNING ISAAC_HOST_DEVICE_INLINE
    isaac_float_dim <feature_dim> operator[]( const isaac_int3 & nIndex ) const
    {
        isaac_float value = ptr[nIndex.x + nIndex.y * VOLUME_X
                                + nIndex.z * VOLUME_X * VOLUME_Y];
        isaac_float_dim< 1 > result;
        result.value
            .x = value;
        return result;
    }
};


// Particle Iterator

template< size_t feature_dim, typename ElemType >
class ParticleIterator1
{
public:
    size_t size;

    ISAAC_NO_HOST_DEVICE_WARNING ISAAC_HOST_DEVICE_INLINE ParticleIterator1(
        ElemType * first_element,
        size_t size,
        const isaac_uint3 & local_grid_coord
    ) :
        current_element( first_element ),
        size( size ),
        local_grid_coord( local_grid_coord )
    {}


    ISAAC_HOST_DEVICE_INLINE void next( )
    {
        current_element = &current_element[1];
    }


    ISAAC_HOST_DEVICE_INLINE ElemType getPosition( ) const
    {
        return *current_element;
    }


    ISAAC_HOST_DEVICE_INLINE isaac_float_dim< feature_dim > getAttribute( ) const
    {
        return {
            isaac_float( local_grid_coord.x ),
            isaac_float( local_grid_coord.y ),
            isaac_float( local_grid_coord.z )
        };
    }


    ISAAC_HOST_DEVICE_INLINE isaac_float getRadius( ) const
    {
        return 0.05f;
    }


private:
    ElemType * current_element;
    isaac_uint3 local_grid_coord;

};


// Particle Source

ISAAC_NO_HOST_DEVICE_WARNING
#if ISAAC_ALPAKA == 1

template<
    typename TDevAcc,
    typename THost,
    typename TStream
>
#endif
class ParticleSource1
{
public:
    static const ISAAC_IDX_TYPE feature_dim = 3;


    ISAAC_NO_HOST_DEVICE_WARNING ParticleSource1(
#if ISAAC_ALPAKA == 1
        TDevAcc acc,
        THost host,
        TStream stream,
#endif
        isaac_float3 * ptr,
        size_t size
    ) :
        ptr( ptr ),
        size( size )
    {}


    ISAAC_HOST_INLINE static std::string


    getName( )
    {
        return std::string( "Particle Source 1" );
    }


    ISAAC_HOST_INLINE void update(
        bool enabled,
        void * pointer
    )
    {}


    isaac_float3 * ptr;
    size_t size;


    // Returns correct particle iterator for the requested cell (in the example the same particle list for each cell)
    ISAAC_NO_HOST_DEVICE_WARNING ISAAC_HOST_DEVICE_INLINE


    ParticleIterator1<
        feature_dim,
        isaac_float3
    > getIterator( const isaac_uint3 & local_grid_coord ) const
    {

        return ParticleIterator1<
            feature_dim,
            isaac_float3
        >(
            ptr,
            size,
            local_grid_coord
        );
    }
};


// Main program
int main(
    int argc,
    char ** argv
)
{
    //Settings the parameters for the example
    char __server[] = "localhost";
    char * server = __server;
    char * filename = NULL;
    //If existend first parameter is the server. Default: "localhost"
    if( argc > 1 )
    {
        server = argv[1];
    }
    int port = 2460;
    //If existend second parameter is the port. Default: 2460
    if( argc > 2 )
    {
        port = atoi( argv[2] );
    }
    if( argc > 3 )
    {
        filename = argv[3];
    }

    //MPI Init
    int rank, numProc;
    MPI_Init(
        &argc,
        &argv
    );
    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );
    MPI_Comm_size(
        MPI_COMM_WORLD,
        &numProc
    );

    //Let's calculate the best spatial distribution of the dimensions so that d[0]*d[1]*d[2] = numProc
    ISAAC_IDX_TYPE d[3] = {
        1,
        1,
        1
    };
    recursive_kgv(
        d,
        numProc,
        2
    );
    ISAAC_IDX_TYPE p[3] = {
        rank % d[0],
        ( rank / d[0] ) % d[1],
        ( rank / d[0] / d[1] ) % d[2]
    };

    //Let's generate some unique name for the simulation and broadcast it
    int id;
    if( rank == 0 )
    {
        srand( time( NULL ) );
        id = rand( ) % 1000000;
    }
    MPI_Bcast(
        &id,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );
    char name[32];
    sprintf(
        name,
        "Example_%i",
        id
    );
    printf(
        "Using name %s\n",
        name
    );

    //This defines the size of the generated rendering
    isaac_size2 framebuffer_size = {
        ISAAC_IDX_TYPE( 800 ),
        ISAAC_IDX_TYPE( 600 )
    };

#if ISAAC_ALPAKA == 1

    // Alpaka specific initialization
    using AccDim = alpaka::dim::DimInt< 3 >;
    using SimDim = alpaka::dim::DimInt< 3 >;
    using DatDim = alpaka::dim::DimInt< 1 >;

    //using Acc = alpaka::acc::AccGpuCudaRt<AccDim, ISAAC_IDX_TYPE>;
    //using Stream  = alpaka::queue::StreamCudaRtSync;
    using Acc = alpaka::acc::AccCpuOmp2Blocks<
        AccDim,
        ISAAC_IDX_TYPE
    >;
    using Stream  = alpaka::queue::StreamCpuSync;
    //using Acc = alpaka::acc::AccCpuOmp2Threads<AccDim, ISAAC_IDX_TYPE>;
    //using Stream  = alpaka::queue::StreamCpuSync;

    using DevAcc = alpaka::dev::Dev< Acc >;
    using DevHost = alpaka::dev::DevCpu;
    using PltfHost = alpaka::pltf::Pltf< DevHost >;
    using PltfAcc = alpaka::pltf::Pltf< DevAcc >;

    DevAcc devAcc(
        alpaka::pltf::getDevByIdx< PltfAcc >(
            rank % alpaka::pltf::getDevCount< PltfAcc >( )
        )
    );
    DevHost devHost( alpaka::pltf::getDevByIdx< PltfHost >( 0u ) );
    Stream stream( devAcc );

    const alpaka::vec::Vec< SimDim, ISAAC_IDX_TYPE > global_size(
        d[0] * VOLUME_X,
        d[1] * VOLUME_Y,
        d[2] * VOLUME_Z
    );
    const alpaka::vec::Vec< SimDim, ISAAC_IDX_TYPE > local_size(
        ISAAC_IDX_TYPE( VOLUME_X ),
        ISAAC_IDX_TYPE( VOLUME_Y ),
        ISAAC_IDX_TYPE( VOLUME_Z )
    );
    const alpaka::vec::Vec< DatDim, ISAAC_IDX_TYPE > data_size(
        ISAAC_IDX_TYPE( VOLUME_X ) * ISAAC_IDX_TYPE( VOLUME_Y )
        * ISAAC_IDX_TYPE( VOLUME_Z )
    );
    const alpaka::vec::Vec< SimDim, ISAAC_IDX_TYPE > position(
        p[0] * VOLUME_X,
        p[1] * VOLUME_Y,
        p[2] * VOLUME_Z
    );
#else //CUDA

    // Cuda specific initialization
    int devCount;
    cudaGetDeviceCount( &devCount );
    cudaSetDevice( rank % devCount );
    typedef boost::mpl::int_< 3 > SimDim;
    std::vector< ISAAC_IDX_TYPE > global_size;
    global_size.push_back( d[0] * VOLUME_X );
    global_size.push_back( d[1] * VOLUME_Y );
    global_size.push_back( d[2] * VOLUME_Z );
    std::vector< ISAAC_IDX_TYPE > local_size;
    local_size.push_back( VOLUME_X );
    local_size.push_back( VOLUME_Y );
    local_size.push_back( VOLUME_Z );
    std::vector< ISAAC_IDX_TYPE > position;
    position.push_back( p[0] * VOLUME_X );
    position.push_back( p[1] * VOLUME_Y );
    position.push_back( p[2] * VOLUME_Z );
    int stream = 0;
#endif

    //The whole size of the rendered sub volumes
    ISAAC_IDX_TYPE prod = local_size[0] * local_size[1] * local_size[2];

    // Init memory
#if ISAAC_ALPAKA == 1
    alpaka::mem::buf::Buf< DevHost, float3_t, DatDim, ISAAC_IDX_TYPE >
        hostBuffer1(
        alpaka::mem::buf::alloc<
            float3_t,
            ISAAC_IDX_TYPE
        >(
            devHost,
            data_size
        )
    );
    alpaka::mem::buf::Buf< DevAcc, float3_t, DatDim, ISAAC_IDX_TYPE >
        deviceBuffer1(
        alpaka::mem::buf::alloc<
            float3_t,
            ISAAC_IDX_TYPE
        >(
            devAcc,
            data_size
        )
    );
    alpaka::mem::buf::Buf<
        DevHost, float, DatDim,
        ISAAC_IDX_TYPE
    > hostBuffer2(
        alpaka::mem::buf::alloc<
            float,
            ISAAC_IDX_TYPE
        >(
            devHost,
            data_size
        )
    );
    alpaka::mem::buf::Buf<
        DevAcc, float, DatDim,
        ISAAC_IDX_TYPE
    > deviceBuffer2(
        alpaka::mem::buf::alloc<
            float,
            ISAAC_IDX_TYPE
        >(
            devAcc,
            data_size
        )
    );
#else //CUDA
    float3_t * hostBuffer1 = ( float3_t * ) malloc( sizeof( float3_t ) * prod );
    float3_t * deviceBuffer1;
    cudaMalloc(
        ( float3_t ** ) &deviceBuffer1,
        sizeof( float3_t ) * prod
    );
    float * hostBuffer2 = ( float * ) malloc( sizeof( float ) * prod );
    float * deviceBuffer2;
    cudaMalloc(
        ( float ** ) &deviceBuffer2,
        sizeof( float ) * prod
    );
    float3_t * hostBuffer3 =
        ( float3_t * ) malloc( sizeof( float3_t ) * PARTICLE_COUNT );
    float3_t * deviceBuffer3;
    cudaMalloc(
        ( float3_t ** ) &deviceBuffer3,
        sizeof( float3_t ) * PARTICLE_COUNT
    );
#endif

    // Creating source list
#if ISAAC_ALPAKA == 1
    TestSource1<
        DevAcc,
        DevHost,
        Stream
    > testSource1(
        devAcc,
        devHost,
        stream,
        reinterpret_cast<isaac_float3 *> ( alpaka::mem::view::getPtrNative( deviceBuffer1 ) )
    );
    TestSource2<
        DevAcc,
        DevHost,
        Stream
    > testSource2(
        devAcc,
        devHost,
        stream,
        reinterpret_cast<isaac_float *> ( alpaka::mem::view::getPtrNative( deviceBuffer2 ) )
    );
    using SourceList = boost::fusion::list<
        TestSource1<
            DevAcc,
            DevHost,
            Stream
        >,
        TestSource2<
            DevAcc,
            DevHost,
            Stream
        >
    >;
#else //CUDA
    TestSource1
        testSource1( reinterpret_cast<isaac_float3 *> ( deviceBuffer1 ) );
    TestSource2
        testSource2( reinterpret_cast<isaac_float *> ( deviceBuffer2 ) );

    ParticleSource1 particleTestSource1(
        reinterpret_cast<isaac_float3 *> ( deviceBuffer3 ),
        PARTICLE_COUNT
    );
    using SourceList = boost::fusion::list<
        TestSource1,
        TestSource2
    >;
    using ParticleList = boost::fusion::list<
        ParticleSource1
    >;
#endif

    ParticleList particle_sources( particleTestSource1 );
    SourceList sources(
        testSource1,
        testSource2
    );

#if ISAAC_NO_SIMULATION == 1
    if ( !filename ) {
        update_data(
            stream,
            hostBuffer1,
            deviceBuffer1,
            hostBuffer2,
            deviceBuffer2,
            prod,
            0.0f,
            local_size,
            position,
            global_size
        );
    }

#endif
    int s_x = 1, s_y = 1, s_z = 3;
    if( filename )
    {
        read_vtk_to_memory(
            filename,
            stream,
            hostBuffer1,
            deviceBuffer1,
            hostBuffer2,
            deviceBuffer2,
            prod,
            0.0f,
            local_size,
            position,
            global_size,
            s_x,
            s_y,
            s_z
        );
    }

    std::vector< float > scaling;
    scaling.push_back( s_x );
    scaling.push_back( s_y );
    scaling.push_back( s_z );

    // Create isaac visualization object
    auto visualization = new IsaacVisualization<
#if ISAAC_ALPAKA == 1
        DevHost, //Alpaka specific Host Dev Type
        Acc, //Alpaka specific Accelerator Dev Type
        Stream, //Alpaka specific Stream Type
        AccDim, //Alpaka specific Acceleration Dimension Type
#endif
        SimDim, //Dimension of the Simulation. In this case: 3D
        ParticleList, SourceList, //The boost::fusion list of Source Types
#if ISAAC_ALPAKA == 1
        alpaka::vec::Vec<
            SimDim,
            ISAAC_IDX_TYPE
        >, //Type of the 3D vectors used later
#else //CUDA
        std::vector< ISAAC_IDX_TYPE >, //Type of the 3D vectors used later
#endif
        1024, //Size of the transfer functions
        std::vector< float >, //user defined type of scaling

#if ( ISAAC_STEREO == 0 )
        isaac::DefaultController,
        //isaac::OrthoController,
        isaac::DefaultCompositor
#else
    isaac::StereoController,
#if (ISAAC_STEREO == 1)
    isaac::StereoCompositorSideBySide<isaac::StereoController>
#else
    isaac::StereoCompositorAnaglyph<isaac::StereoController,0x000000FF,0x00FFFF00>
#endif
#endif
    >(
#if ISAAC_ALPAKA == 1
        devHost, //Alpaka specific host dev instance
        devAcc, //Alpaka specific accelerator dev instance
        stream, //Alpaka specific stream instance
#endif
        name, //Name of the visualization shown to the client
        0, //Master rank, which will opens the connection to the server
        server, //Address of the server
        port, //Inner port of the server
        framebuffer_size, //Size of the rendered image
        global_size, //Size of the whole volumen including all nodes
        local_size, //Local size of the subvolume
        {
            PARTICLE_VOLUME_X,
            PARTICLE_VOLUME_Y,
            PARTICLE_VOLUME_Z
        },
        position, //Position of the subvolume in the globale volume
        particle_sources,
        sources, //instances of the sources to render
        scaling
    );

    //Setting up the metadata description (only master, but however slaves could then add metadata, too, it would be merged)
    if( rank == 0 )
    {
        json_object_set_new(
            visualization->getJsonMetaRoot( ),
            "counting variable",
            json_string( "counting" )
        );
        json_object_set_new(
            visualization->getJsonMetaRoot( ),
            "drawing_time",
            json_string( "drawing_time" )
        );
        json_object_set_new(
            visualization->getJsonMetaRoot( ),
            "simulation_time",
            json_string( "simulation_time" )
        );
        json_object_set_new(
            visualization->getJsonMetaRoot( ),
            "sorting_time",
            json_string( "sorting_time" )
        );
        json_object_set_new(
            visualization->getJsonMetaRoot( ),
            "merge_time",
            json_string( "merge_time" )
        );
        json_object_set_new(
            visualization->getJsonMetaRoot( ),
            "kernel_time",
            json_string( "kernel_time" )
        );
        json_object_set_new(
            visualization->getJsonMetaRoot( ),
            "copy_time",
            json_string( "copy_time" )
        );
        json_object_set_new(
            visualization->getJsonMetaRoot( ),
            "video_send_time",
            json_string( "video_send_time" )
        );
        json_object_set_new(
            visualization->getJsonMetaRoot( ),
            "buffer_time",
            json_string( "buffer_time" )
        );
    }

    //finish init and sending the meta data scription to the isaac server
    if( visualization->init( RetryEverySend ) )
    {
        fprintf(
            stderr,
            "Isaac init failed.\n"
        );
        return -1;
    }


    // Program flow and time mesaurment variables
    float a = 0.0f;
    volatile int force_exit = 0;
    int start = visualization->getTicksUs( );
    int count = 0;
    int drawing_time = 0;
    int simulation_time = 0;
    int full_drawing_time = 0;
    int full_simulation_time = 0;
    int sorting_time = 0;
    int merge_time = 0;
    int kernel_time = 0;
    int copy_time = 0;
    int video_send_time = 0;
    int buffer_time = 0;
    bool pause = false;
    //How often should the visualization be updated?
    int interval = 1;
    int step = 0;
    if( rank == 0 )
    {
        json_object_set_new(
            visualization->getJsonMetaRoot( ),
            "interval",
            json_integer( interval )
        );
    }

    // Main loop
    while( !force_exit )
    {

        // "Simulation"
        if( !pause )
        {
            a += 0.01f;
            int start_simulation = visualization->getTicksUs( );
#if ISAAC_NO_SIMULATION == 0
            if( !filename )
            {
                update_data(
                    stream,
                    hostBuffer1,
                    deviceBuffer1,
                    hostBuffer2,
                    deviceBuffer2,
                    prod,
                    a,
                    local_size,
                    position,
                    global_size
                );
            }
#endif
            update_particles(
                hostBuffer3,
                deviceBuffer3,
                PARTICLE_COUNT,
                a
            );
            simulation_time += visualization->getTicksUs( ) - start_simulation;
        }
        step++;
        if( step >= interval )
        {
            step = 0;

            // Metadata fill
            if( rank == 0 )
            {
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "counting variable",
                    json_real( a )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "drawing_time",
                    json_integer( drawing_time )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "simulation_time",
                    json_integer( simulation_time )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "sorting_time",
                    json_integer( visualization->sorting_time )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "merge_time",
                    json_integer( visualization->merge_time )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "kernel_time",
                    json_integer( visualization->kernel_time )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "copy_time",
                    json_integer( visualization->copy_time )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "video_send_time",
                    json_integer( visualization->video_send_time )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "buffer_time",
                    json_integer( visualization->buffer_time )
                );
                full_drawing_time += drawing_time;
                full_simulation_time += simulation_time;
                sorting_time += visualization->sorting_time;
                merge_time += visualization->merge_time;
                kernel_time += visualization->kernel_time;
                copy_time += visualization->copy_time;
                video_send_time += visualization->video_send_time;
                buffer_time += visualization->buffer_time;
                drawing_time = 0;
                simulation_time = 0;
                visualization->sorting_time = 0;
                visualization->merge_time = 0;
                visualization->kernel_time = 0;
                visualization->copy_time = 0;
                visualization->video_send_time = 0;
                visualization->buffer_time = 0;
            }

            // Visualization
            int start_drawing = visualization->getTicksUs( );
            json_t * meta = visualization->doVisualization(
                META_MASTER,
                NULL,
                !pause
            );
            drawing_time += visualization->getTicksUs( ) - start_drawing;

            // Message check
            if( meta )
            {
                //Let's print it to stdout
                char * buffer = json_dumps(
                    meta,
                    0
                );
                printf(
                    "META (%i): %s\n",
                    rank,
                    buffer
                );
                free( buffer );
                //And let's also check for an exit message
                if( json_integer_value(
                    json_object_get(
                        meta,
                        "exit"
                    )
                ) )
                {
                    force_exit = 1;
                }
                if( json_boolean_value(
                    json_object_get(
                        meta,
                        "pause"
                    )
                ) )
                {
                    pause = !pause;
                }
                //Deref the jansson json root! Otherwise we would get a memory leak
                json_t * js;
                if( js = json_object_get(
                    meta,
                    "interval"
                ) )
                {
                    interval = std::max(
                        int( 1 ),
                        int( json_integer_value( js ) )
                    );
                    //Feedback for other clients than the changing one
                    if( rank == 0 )
                    {
                        json_object_set_new(
                            visualization->getJsonMetaRoot( ),
                            "interval",
                            json_integer( interval )
                        );
                    }
                }
                json_decref( meta );
            }
            usleep( 100 );
            count++;

            // Debug output
            if( rank == 0 )
            {
                int end = visualization->getTicksUs( );
                int diff = end - start;
                if( diff >= 1000000 )
                {
                    merge_time -= kernel_time + copy_time;
                    printf(
                        "FPS: %.1f \n\tSimulation: %.1f ms\n\t"
                        "Drawing: %.1f ms\n\t\tSorting: %.1f ms\n\t\t"
                        "Merge: %.1f ms\n\t\tKernel: %.1f ms\n\t\t"
                        "Copy: %.1f ms\n\t\tVideo: %.1f ms\n\t\t"
                        "Buffer: %.1f ms\n",
                        ( float ) count * 1000000.0f / ( float ) diff,
                        ( float ) full_simulation_time / 1000.0f
                        / ( float ) count,
                        ( float ) full_drawing_time / 1000.0f / ( float ) count,
                        ( float ) sorting_time / 1000.0f / ( float ) count,
                        ( float ) merge_time / 1000.0f / ( float ) count,
                        ( float ) kernel_time / 1000.0f / ( float ) count,
                        ( float ) copy_time / 1000.0f / ( float ) count,
                        ( float ) video_send_time / 1000.0f / ( float ) count,
                        ( float ) buffer_time / 1000.0f / ( float ) count
                    );
                    sorting_time = 0;
                    merge_time = 0;
                    kernel_time = 0;
                    copy_time = 0;
                    video_send_time = 0;
                    buffer_time = 0;
                    full_drawing_time = 0;
                    full_simulation_time = 0;
                    start = end;
                    count = 0;
                }
            }
        }
    }
    MPI_Barrier( MPI_COMM_WORLD );
    printf(
        "%i finished\n",
        rank
    );

    // Winter wrap up
    delete ( visualization );

#if ISAAC_ALPAKA == 0
    free( hostBuffer1 );
    free( hostBuffer2 );
    cudaFree( deviceBuffer1 );
    cudaFree( deviceBuffer2 );
    cudaFree( deviceBuffer3 );
#endif

    MPI_Finalize( );
    return 0;
}

// Not necessary, just for the example
