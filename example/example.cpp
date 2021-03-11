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
class TestSource1
{
public:
    static const ISAAC_IDX_TYPE featureDim = 3;
    static const bool hasGuard = false;
    static const bool persistent = true;


    ISAAC_NO_HOST_DEVICE_WARNING TestSource1(
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
    isaac_float_dim <featureDim> operator[]( const isaac_int3 & nIndex ) const
    {
        isaac_float3 value = ptr[nIndex.x + nIndex.y * VOLUME_X
                                 + nIndex.z * VOLUME_X * VOLUME_Y];
        return value;
    }
};


// Volume Source 2

ISAAC_NO_HOST_DEVICE_WARNING
class TestSource2
{
public:
    static const ISAAC_IDX_TYPE featureDim = 1;
    static const bool hasGuard = false;
    static const bool persistent = false;


    ISAAC_NO_HOST_DEVICE_WARNING TestSource2(
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
    isaac_float_dim <featureDim> operator[]( const isaac_int3 & nIndex ) const
    {
        isaac_float value = ptr[nIndex.x + nIndex.y * VOLUME_X
                                + nIndex.z * VOLUME_X * VOLUME_Y];
        return isaac_float_dim<featureDim>( value );
    }
};


// Particle Iterator

template< ISAAC_IDX_TYPE T_featureDim, typename T_ElemType >
class ParticleIterator1
{
public:
    size_t size;

    ISAAC_NO_HOST_DEVICE_WARNING ISAAC_HOST_DEVICE_INLINE ParticleIterator1(
        T_ElemType * firstElement,
        size_t size,
        const isaac_uint3 & localGridCoord
    ) :
        currentElement( firstElement ),
        size( size ),
        localGridCoord( localGridCoord )
    {}


    ISAAC_HOST_DEVICE_INLINE void next( )
    {
        currentElement = &currentElement[1];
    }


    ISAAC_HOST_DEVICE_INLINE T_ElemType getPosition( ) const
    {
        return *currentElement;
    }


    ISAAC_HOST_DEVICE_INLINE isaac_float_dim< T_featureDim > getAttribute( ) const
    {
        return {
            isaac_float( localGridCoord.x ),
            isaac_float( localGridCoord.y ),
            isaac_float( localGridCoord.z )
        };
    }


    ISAAC_HOST_DEVICE_INLINE isaac_float getRadius( ) const
    {
        return 0.05f;
    }


private:
    T_ElemType * currentElement;
    isaac_uint3 localGridCoord;

};


// Particle Source

ISAAC_NO_HOST_DEVICE_WARNING
class ParticleSource1
{
public:
    static const ISAAC_IDX_TYPE featureDim = 3;


    ISAAC_NO_HOST_DEVICE_WARNING ParticleSource1(
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
        featureDim,
        isaac_float3
    > getIterator( const isaac_uint3 & localGridCoord ) const
    {

        return ParticleIterator1<
            featureDim,
            isaac_float3
        >(
            ptr,
            size,
            localGridCoord
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
    isaac_size2 framebufferSize = {
        ISAAC_IDX_TYPE( 1920 ),
        ISAAC_IDX_TYPE( 1080 )
    };

    // Alpaka specific initialization
    using AccDim = alpaka::DimInt< 3 >;
    using SimDim = alpaka::DimInt< 3 >;
    using DatDim = alpaka::DimInt< 1 >;
    
    using Acc = alpaka::AccGpuCudaRt<
        AccDim, 
        ISAAC_IDX_TYPE
    >;

    //using Acc = alpaka::AccCpuOmp2Blocks<
    //    AccDim,
    //    ISAAC_IDX_TYPE
    //>;
    using Stream = alpaka::Queue<Acc, alpaka::Blocking>;

    using DevAcc = alpaka::Dev< Acc >;
    using DevHost = alpaka::DevCpu;
    using PltfHost = alpaka::Pltf< DevHost >;
    using PltfAcc = alpaka::Pltf< DevAcc >;

    DevAcc devAcc(
        alpaka::getDevByIdx< PltfAcc >(
            rank % alpaka::getDevCount< PltfAcc >( )
        )
    );
    DevHost devHost( alpaka::getDevByIdx< PltfHost >( 0u ) );
    Stream stream( devAcc );

    const isaac_size_dim<SimDim::value> globalSize(
        d[0] * VOLUME_X,
        d[1] * VOLUME_Y,
        d[2] * VOLUME_Z
    );
    const isaac_size_dim<SimDim::value> localSize(
        ISAAC_IDX_TYPE( VOLUME_X ),
        ISAAC_IDX_TYPE( VOLUME_Y ),
        ISAAC_IDX_TYPE( VOLUME_Z )
    );
    const alpaka::Vec< DatDim, ISAAC_IDX_TYPE > dataSize(
        ISAAC_IDX_TYPE( VOLUME_X ) * ISAAC_IDX_TYPE( VOLUME_Y )
        * ISAAC_IDX_TYPE( VOLUME_Z )
    );
    const isaac_size_dim<SimDim::value> position(
        p[0] * VOLUME_X,
        p[1] * VOLUME_Y,
        p[2] * VOLUME_Z
    );
    const alpaka::Vec< alpaka::DimInt< 1 >, ISAAC_IDX_TYPE > particleCount(
        ISAAC_IDX_TYPE( PARTICLE_COUNT )
    );

    //The whole size of the rendered sub volumes
    ISAAC_IDX_TYPE prod = localSize[0] * localSize[1] * localSize[2];

    // Init memory

    auto hostBuffer1 = 
        alpaka::allocBuf<
            isaac_float3,
            ISAAC_IDX_TYPE
        >(
            devHost,
            dataSize
        );
    auto deviceBuffer1 =
        alpaka::allocBuf<
            isaac_float3,
            ISAAC_IDX_TYPE
        >(
            devAcc,
            dataSize
        );
    auto hostBuffer2 = 
        alpaka::allocBuf<
            isaac_float,
            ISAAC_IDX_TYPE
        >(
            devHost,
            dataSize
        );
    auto deviceBuffer2 =
        alpaka::allocBuf<
            isaac_float,
            ISAAC_IDX_TYPE
        >(
            devAcc,
            dataSize
        );
    auto hostBuffer3 =
        alpaka::allocBuf<
            isaac_float3,
            ISAAC_IDX_TYPE
        >(
            devHost,
            particleCount
        );
    auto deviceBuffer3 = 
        alpaka::allocBuf<
            isaac_float3,
            ISAAC_IDX_TYPE
        >(
            devAcc,
            particleCount
        );

    // Creating source list

    TestSource1 testSource1(
        alpaka::getPtrNative( deviceBuffer1 )
    );
    TestSource2 testSource2(
        alpaka::getPtrNative( deviceBuffer2 )
    );

    ParticleSource1 particleTestSource1(
        alpaka::getPtrNative( deviceBuffer3 ),
        PARTICLE_COUNT
    );

    using SourceList = boost::fusion::list<
        TestSource1,
        TestSource2
    >;

    using ParticleList = boost::fusion::list<
        ParticleSource1
    >;

    ParticleList particleSources( particleTestSource1 );
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
            localSize,
            position,
            globalSize
        );
    }

    update_particles(
        stream,
        hostBuffer3,
        deviceBuffer3,
        particleCount,
        0.0f
    );


#endif
    int sX = 1, sY = 1, sZ = 3;
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
            localSize,
            position,
            globalSize,
            sX,
            sY,
            sZ
        );
    }

    isaac_float3 scaling( sX, sY, sZ );

    // Create isaac visualization object
    auto visualization = new IsaacVisualization<
        DevHost, //Alpaka specific Host Dev Type
        Acc, //Alpaka specific Accelerator Dev Type
        Stream, //Alpaka specific Stream Type
        AccDim, //Alpaka specific Acceleration Dimension Type
        ParticleList, decltype(sources), //The boost::fusion list of Source Types
        1024, //Size of the transfer functions
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
        devHost, //Alpaka specific host dev instance
        devAcc, //Alpaka specific accelerator dev instance
        stream, //Alpaka specific stream instance
        name, //Name of the visualization shown to the client
        0, //Master rank, which will opens the connection to the server
        server, //Address of the server
        port, //Inner port of the server
        framebufferSize, //Size of the rendered image
        globalSize, //Size of the whole volumen including all nodes
        localSize, //Local size of the subvolume
        {
            PARTICLE_VOLUME_X,
            PARTICLE_VOLUME_Y,
            PARTICLE_VOLUME_Z
        },
        position, //Position of the subvolume in the globale volume
        particleSources,
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
    volatile int forceExit = 0;
    int start = visualization->getTicksUs( );
    int count = 0;
    int drawingTime = 0;
    int simulationTime = 0;
    int fullDrawingTime = 0;
    int fullSimulationTime = 0;
    int sortingTime = 0;
    int mergeTime = 0;
    int kernelTime = 0;
    int copyTime = 0;
    int videoSendTime = 0;
    int bufferTime = 0;
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
    while( !forceExit )
    {

        // "Simulation"
        if( !pause )
        {
            a += 0.01f;
            int startSimulation = visualization->getTicksUs( );
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
                    localSize,
                    position,
                    globalSize
                );
            }

            update_particles(
                stream,
                hostBuffer3,
                deviceBuffer3,
                particleCount,
                a
            );

#endif



            simulationTime += visualization->getTicksUs( ) - startSimulation;
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
                    json_integer( drawingTime )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "simulation_time",
                    json_integer( simulationTime )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "sorting_time",
                    json_integer( visualization->sortingTime )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "merge_time",
                    json_integer( visualization->mergeTime )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "kernel_time",
                    json_integer( visualization->kernelTime )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "copy_time",
                    json_integer( visualization->copyTime )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "video_send_time",
                    json_integer( visualization->videoSendTime )
                );
                json_object_set_new(
                    visualization->getJsonMetaRoot( ),
                    "buffer_time",
                    json_integer( visualization->bufferTime )
                );
                fullDrawingTime += drawingTime;
                fullSimulationTime += simulationTime;
                sortingTime += visualization->sortingTime;
                mergeTime += visualization->mergeTime;
                kernelTime += visualization->kernelTime;
                copyTime += visualization->copyTime;
                videoSendTime += visualization->videoSendTime;
                bufferTime += visualization->bufferTime;
                drawingTime = 0;
                simulationTime = 0;
                visualization->sortingTime = 0;
                visualization->mergeTime = 0;
                visualization->kernelTime = 0;
                visualization->copyTime = 0;
                visualization->videoSendTime = 0;
                visualization->bufferTime = 0;
            }

            // Visualization
            int start_drawing = visualization->getTicksUs( );
            json_t * meta = visualization->doVisualization(
                META_MASTER,
                NULL,
                !pause
            );
            drawingTime += visualization->getTicksUs( ) - start_drawing;

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
                    forceExit = 1;
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
                    mergeTime -= kernelTime + copyTime;
                    printf(
                        "FPS: %.1f \n\tSimulation: %.1f ms\n\t"
                        "Drawing: %.1f ms\n\t\tSorting: %.1f ms\n\t\t"
                        "Merge: %.1f ms\n\t\tKernel: %.1f ms\n\t\t"
                        "Copy: %.1f ms\n\t\tVideo: %.1f ms\n\t\t"
                        "Buffer: %.1f ms\n",
                        ( float ) count * 1000000.0f / ( float ) diff,
                        ( float ) fullSimulationTime / 1000.0f
                        / ( float ) count,
                        ( float ) fullDrawingTime / 1000.0f / ( float ) count,
                        ( float ) sortingTime / 1000.0f / ( float ) count,
                        ( float ) mergeTime / 1000.0f / ( float ) count,
                        ( float ) kernelTime / 1000.0f / ( float ) count,
                        ( float ) copyTime / 1000.0f / ( float ) count,
                        ( float ) videoSendTime / 1000.0f / ( float ) count,
                        ( float ) bufferTime / 1000.0f / ( float ) count
                    );
                    sortingTime = 0;
                    mergeTime = 0;
                    kernelTime = 0;
                    copyTime = 0;
                    videoSendTime = 0;
                    bufferTime = 0;
                    fullDrawingTime = 0;
                    fullSimulationTime = 0;
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

    MPI_Finalize( );
    return 0;
}

// Not necessary, just for the example
