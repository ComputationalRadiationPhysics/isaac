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
#include <string>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h> 
#include <unistd.h>
#include <jansson.h>
#include <pthread.h>
#include <list>
#include <vector>
#include <memory>
#include <mpi.h>
#include <boost/preprocessor.hpp>
#include <IceT.h>
#include <IceTMPI.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <map>

#ifdef ISAAC_ALPAKA
	#include <alpaka/alpaka.hpp>
	#include <boost/type_traits.hpp>
	#include <boost/mpl/not.hpp>
#else
	#include <boost/mpl/int.hpp>
#endif

#ifndef __CUDACC__
	struct isaac_float4
	{
		float x,y,z,w;
	};
	struct isaac_float3
	{
		float x,y,z;
	};
	struct isaac_uint4
	{
		uint32_t x,y,z,w;
	};
	struct isaac_uint3
	{
		uint32_t x,y,z;
	};
	struct isaac_int4
	{
		int32_t x,y,z,w;
	};
	struct isaac_int3
	{
		int32_t x,y,z;
	};
#else
	typedef float4 isaac_float4;
	typedef float3 isaac_float3;
	typedef uint4 isaac_uint4;
	typedef uint3 isaac_uint3;
	typedef int4 isaac_int4;
	typedef int3 isaac_int3;
#endif

struct isaac_size_type
{
	isaac_float3 global_size;
	float max_global_size;
	isaac_float3 position;
	isaac_float3 local_size;
};

#define ISAAC_CUDA_CHECK(call)                                                 \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define ISAAC_CALL_FOR_XYZ(start,end) \
	start.x end \
	start.y end \
	start.z end

#define ISAAC_CALL_FOR_XYZ_ITERATE(start,first,end) \
	start.x first end \
	start.y first+1 end \
	start.z first+2 end

#define ISAAC_CALL_FOR_XYZ_TWICE(start,middle,end) \
	start.x middle.x end \
	start.y middle.y end \
	start.z middle.z end

#define ISAAC_CALL_FOR_XYZ_TRIPLE(start,middle1,middle2,end) \
	start.x middle1.x middle2.x end \
	start.y middle1.y middle2.y end \
	start.z middle1.z middle2.z end

#define ISAAC_SWITCH_IF_SMALLER(left,right) \
	if (left < right) \
	{ \
		auto temp = left; \
		left = right; \
		right = temp; \
	}

#define ISAAC_SET_COLOR( dest, color ) \
	{ \
		isaac_uint4 result; \
		result.x = min(1.0f,color.x) * 255.0f; \
		result.y = min(1.0f,color.y) * 255.0f; \
		result.z = min(1.0f,color.z) * 255.0f; \
		result.w = min(1.0f,color.w) * 255.0f; \
		dest = (result.w << 24) | (result.z << 16) | (result.y << 8) | (result.x << 0); \
	}


#define ISAAC_START_TIME_MEASUREMENT( unique_name, time_function ) \
	uint64_t BOOST_PP_CAT( __tm_start_, unique_name ) = time_function;
#define ISAAC_STOP_TIME_MEASUREMENT( result, operand, unique_name, time_function ) \
	result operand time_function - BOOST_PP_CAT( __tm_start_, unique_name );
			
#define MAX_RECEIVE 32768 //32kb
#define Z_NEAR 1.0f
#define Z_FAR 100.0f

__constant__ float isaac_inverse_d[16];
__constant__ float isaac_modelview_d[16];
__constant__ isaac_size_type isaac_size_d[1]; //[1] to access it same for cuda and alpaka


#ifdef ISAAC_ALPAKA
	struct IsaacFillRectKernel
	{
		template <typename TAcc__>
		ALPAKA_FN_ACC void operator()(
			TAcc__ const &acc,
			float* isaac_inverse_d,
			float* isaac_modelview_d,
			isaac_size_type* isaac_size_d,
#else
		__global__ void IsaacFillRectKernel(
#endif
			uint32_t* pixels,
			uint32_t value,
			size_t framebuffer_width,
			size_t framebuffer_height,
			size_t startx,
			size_t starty,
			float* source,
			float step,
			isaac_float4 background_color)
#ifdef ISAAC_ALPAKA
		const
#endif
		{
			#ifdef ISAAC_ALPAKA
				auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
				auto x = threadIdx[2];
				auto y = threadIdx[1];
			#else
				auto x = threadIdx.x + blockIdx.x * blockDim.x;
				auto y = threadIdx.y + blockIdx.y * blockDim.y;
			#endif
			x+= startx;
			y+= starty;
			if (x >= framebuffer_width || y >= framebuffer_height)
				return;
			
			//Debug output of the bounding box
			//isaac_float4 foobar = {0.5,0.5,0.5,0.5};
			//ISAAC_SET_COLOR( pixels[x + y * framebuffer_width], foobar )
			//return;
			
			float f_x = x/(float)framebuffer_width*2.0f-1.0f;
			float f_y = y/(float)framebuffer_height*2.0f-1.0f;
			isaac_float4 start_p = {f_x*Z_NEAR,f_y*Z_NEAR,-1.0f*Z_NEAR,1.0f*Z_NEAR}; //znear
			isaac_float4 end_p = {f_x*Z_FAR,f_y*Z_FAR,1.0f*Z_FAR,1.0f*Z_FAR}; //zfar
			isaac_float3 start;
			isaac_float3 end;
			start.x =  isaac_inverse_d[ 0] * start_p.x + isaac_inverse_d[ 4] * start_p.y +  isaac_inverse_d[ 8] * start_p.z + isaac_inverse_d[12] * start_p.w;
			start.y =  isaac_inverse_d[ 1] * start_p.x + isaac_inverse_d[ 5] * start_p.y +  isaac_inverse_d[ 9] * start_p.z + isaac_inverse_d[13] * start_p.w;
			start.z =  isaac_inverse_d[ 2] * start_p.x + isaac_inverse_d[ 6] * start_p.y +  isaac_inverse_d[10] * start_p.z + isaac_inverse_d[14] * start_p.w;
			  end.x =  isaac_inverse_d[ 0] *   end_p.x + isaac_inverse_d[ 4] *   end_p.y +  isaac_inverse_d[ 8] *   end_p.z + isaac_inverse_d[12] *   end_p.w;
			  end.y =  isaac_inverse_d[ 1] *   end_p.x + isaac_inverse_d[ 5] *   end_p.y +  isaac_inverse_d[ 9] *   end_p.z + isaac_inverse_d[13] *   end_p.w;
			  end.z =  isaac_inverse_d[ 2] *   end_p.x + isaac_inverse_d[ 6] *   end_p.y +  isaac_inverse_d[10] *   end_p.z + isaac_inverse_d[14] *   end_p.w;

			float max_size = isaac_size_d[0].max_global_size / 2.0f;

			//scale to globale grid size
			ISAAC_CALL_FOR_XYZ( start , *=max_size; )
			ISAAC_CALL_FOR_XYZ( end , *=max_size; )
			
			//move to local grid
			ISAAC_CALL_FOR_XYZ_TRIPLE( start, += isaac_size_d[0].global_size, / 2.0f - isaac_size_d[0].position, ; )
			ISAAC_CALL_FOR_XYZ_TRIPLE(   end, += isaac_size_d[0].global_size, / 2.0f - isaac_size_d[0].position, ; )

			isaac_float3 vec;
			ISAAC_CALL_FOR_XYZ_TRIPLE( vec, = end, - start, ; )
			float l = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
			
			isaac_float3 step_vec = vec;
			ISAAC_CALL_FOR_XYZ( step_vec , /= l; )
			ISAAC_CALL_FOR_XYZ( step_vec , *= step; )

			isaac_float3 count_start;
			ISAAC_CALL_FOR_XYZ_TRIPLE( count_start, = -start, / step_vec, ; )
			isaac_float3 moved_start;
			ISAAC_CALL_FOR_XYZ_TRIPLE( moved_start, = -start, + isaac_size_d[0].local_size, ; )
			isaac_float3 count_end;
			ISAAC_CALL_FOR_XYZ_TRIPLE( count_end, = moved_start, / step_vec, ; )

			//count_start shall have the smaller values
			ISAAC_SWITCH_IF_SMALLER( count_end.x, count_start.x )
			ISAAC_SWITCH_IF_SMALLER( count_end.y, count_start.y )
			ISAAC_SWITCH_IF_SMALLER( count_end.z, count_start.z )
			
			//calc intersection of all three super planes and save in [count_start.x ; count_end.x]
			count_start.x = max( max( count_start.x, count_start.y ), count_start.z );
			  count_end.x = min( min(   count_end.x,   count_end.y ),   count_end.z );
			if ( count_start.x > count_end.x || count_end.x <= 0.0f )
			{
				ISAAC_SET_COLOR( pixels[x + y * framebuffer_width], background_color )
				return;
			}
			
			//Starting the main loop
			int32_t first = ceil( count_start.x );
			int32_t last = floor( count_end.x );
			int32_t count = last - first + 1;
			float count_reciprocal = 1.0f/(float)count/2.0f;
			isaac_float4 color = background_color;
			isaac_float3 pos = start;
			isaac_uint3 local_size_uint =
			{
				uint32_t( isaac_size_d[0].local_size.x ),
				uint32_t( isaac_size_d[0].local_size.y ),
				uint32_t( isaac_size_d[0].local_size.z )
			};
			ISAAC_CALL_FOR_XYZ_TWICE( pos, += step_vec, * float(first); )
			
			for (int32_t i = 0; i < count; i++)
			{
				isaac_uint3 coord;
				ISAAC_CALL_FOR_XYZ_TWICE( coord, = (uint32_t)pos, ; )				
				if ( ISAAC_CALL_FOR_XYZ( coord, >= 64 || ) 0 )
					break;

				int32_t source_pos = (
					coord.x +
					coord.y * local_size_uint.x +
					coord.z * local_size_uint.x * local_size_uint.y ) * 3;
				ISAAC_CALL_FOR_XYZ_ITERATE( color, += source[source_pos + 0, ] * count_reciprocal; )					
				ISAAC_CALL_FOR_XYZ_TWICE( pos, += step_vec, ; )
			}
			color.w = 0.5f;
			ISAAC_SET_COLOR( pixels[x + y * framebuffer_width], color )

		}
#ifdef ISAAC_ALPAKA
	};
#endif


typedef enum
{
	META_NONE = -1,
	META_MERGE = 0,
	META_MASTER = 1
} IsaacVisualizationMetaEnum;

#ifdef ISAAC_ALPAKA
	template <typename TDevAcc,typename TAccDim>
#endif
class IsaacSource
{
	#ifdef ISAAC_ALPAKA
		template <typename THost__,typename TAcc__,typename TStream__,typename TAccDim__,typename TSimDim__> friend class IsaacVisualization;
	#else
		template <typename TSimDim__> friend class IsaacVisualization;
	#endif
	public:
		IsaacSource(std::string name,float* ptr,unsigned int f_dim)
		{
			this->name = name;
			this->ptr = ptr;
			this->f_dim = f_dim;
		}
		float* getPtr()
		{
			return ptr;
		}
	private:
		std::string name;
		float* ptr;
		unsigned int f_dim;
};

#define ISAAC_SET_IDENTITY(matrix) \
	for (int x = 0; x < 4; x++) \
		for (int y = 0; y < 4; y++) \
			(matrix)[x+y*4] = (x==y)?1.0f:0.0f;

#define ISAAC_JSON_ADD_MATRIX(array,matrix) \
	for (int i = 0; i < 16; i++) \
		json_array_append_new( array, json_real( (matrix)[i] ) );


#ifdef ISAAC_SET_IDENTITY
	#define ISAAC_WAIT_VISUALIZATION \
		if (visualizationThread) \
		{ \
			pthread_join(visualizationThread,NULL); \
			visualizationThread = 0; \
		}
#else
	#define ISAAC_WAIT_VISUALIZATION BOOST_PP_EMPTY
#endif

#ifdef ISAAC_ALPAKA
	template <typename THost,typename TAcc,typename TStream,typename TAccDim,typename TSimDim>
#else
	template <typename TSimDim>
#endif
class IsaacVisualization 
{
	public:
		#ifdef ISAAC_ALPAKA
			using TDevAcc = alpaka::dev::Dev<TAcc>;
			using TFraDim = alpaka::dim::DimInt<1>;
		#endif
		IsaacVisualization(
			#ifdef ISAAC_ALPAKA
				THost host,
				TDevAcc acc,
				TStream stream,
			#endif
			std::string name,
			int master,
			std::string server_url,
			int server_port,
			int framebuffer_width,
			int framebuffer_height,
			#ifdef ISAAC_ALPAKA
				const alpaka::Vec<TSimDim, size_t> global_size,
				const alpaka::Vec<TSimDim, size_t> local_size,
				const alpaka::Vec<TSimDim, size_t> position
			#else
				const std::vector<size_t> global_size,
				const std::vector<size_t> local_size,
				const std::vector<size_t> position			
			#endif
			) :
			#ifdef ISAAC_ALPAKA
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
			framebuffer_width(framebuffer_width),
			framebuffer_height(framebuffer_height),
			metaNr(0),
			visualizationThread(0),
			kernel_time(0),
			merge_time(0),
			video_send_time(0),
			copy_time(0),
			sorting_time(0),
			framebuffer_size(size_t(framebuffer_width) * size_t(framebuffer_height))
			#ifdef ISAAC_ALPAKA
				,framebuffer(alpaka::mem::buf::alloc<uint32_t, size_t>(acc, framebuffer_size))
				,inverse_d(alpaka::mem::buf::alloc<float, size_t>(acc, size_t(16)))
				,modelview_d(alpaka::mem::buf::alloc<float, size_t>(acc, size_t(16)))
				,size_d(alpaka::mem::buf::alloc<isaac_size_type, size_t>(acc, size_t(1)))
		{
			#else
		{
				ISAAC_CUDA_CHECK(cudaMalloc((uint32_t**)&framebuffer, sizeof(uint32_t)*framebuffer_size));
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
			setPerspective( 45.0f, (float)framebuffer_width/(float)framebuffer_height,Z_NEAR, Z_FAR);
			ISAAC_SET_IDENTITY(modelview)
			modelview[14] = -5.0f; //glTranslate(0,0,-5);
			ISAAC_SET_IDENTITY(rotation)
			
			//Fill framebuffer with test values:
			uint32_t value = (255 << 24);
			#ifdef ISAAC_ALPAKA
				alpaka::mem::buf::Buf<THost, uint32_t, TFraDim, size_t> framebuffer_host(alpaka::mem::buf::alloc<uint32_t, size_t>(host, framebuffer_size));
				for (size_t i = 0; i < framebuffer_size.prod(); ++i)
					alpaka::mem::view::getPtrNative(framebuffer_host)[i] = value;
				alpaka::mem::view::copy(stream, framebuffer, framebuffer_host, framebuffer_size);
			#else
				uint32_t* framebuffer_host = (uint32_t*)malloc(framebuffer_size*sizeof(uint32_t));
				for (size_t i = 0; i < framebuffer_size; ++i)
					framebuffer_host[i] = value;
				ISAAC_CUDA_CHECK(cudaMemcpy(framebuffer, framebuffer_host, sizeof(uint32_t)*framebuffer_size, cudaMemcpyHostToDevice));
				free(framebuffer_host);
			#endif
			
			//ISAAC:
			IceTCommunicator icetComm;
			icetComm = icetCreateMPICommunicator(MPI_COMM_WORLD);
			icetContext = icetCreateContext(icetComm);
			icetDestroyMPICommunicator(icetComm);
			icetResetTiles();
			icetAddTile(0, 0, framebuffer_width, framebuffer_height, master);
			//icetStrategy(ICET_STRATEGY_SPLIT);
			icetStrategy(ICET_STRATEGY_SEQUENTIAL);
			icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
			icetSetDepthFormat(ICET_IMAGE_DEPTH_NONE);
			icetCompositeMode(ICET_COMPOSITE_MODE_BLEND);
			icetEnable(ICET_ORDERED_COMPOSITE);
			
			size_t max_size = max(global_size[0],global_size[1]);
			if (TSimDim::value > 2)
				max_size = max(global_size[2],max_size);
			float f_l_width = (float)local_size[0]/(float)max_size * 2.0f;
			float f_l_height = (float)local_size[1]/(float)max_size * 2.0f;
			float f_l_depth = 0.0f;
			if (TSimDim::value > 2)
				f_l_depth = (float)local_size[2]/(float)max_size * 2.0f;
			float f_x = (float)position[0]/(float)max_size * 2.0f - (float)global_size[0]/(float)max_size;
			float f_y = (float)position[1]/(float)max_size * 2.0f - (float)global_size[1]/(float)max_size;
			float f_z = 0.0f;
			if (TSimDim::value > 2)
				f_z = (float)position[2]/(float)max_size * 2.0f - (float)global_size[2]/(float)max_size;
			icetBoundingBoxf( f_x, f_x + f_l_width, f_y, f_y + f_l_height, f_z, f_z + f_l_depth);
			icetPhysicalRenderSize(framebuffer_width, framebuffer_height);
			icetDrawCallback( drawCallBack );
			
			//JSON
			recreateJSON();
			if (rank == master)
			{
				json_object_set_new( json_root, "name", json_string( name.c_str() ) );
				json_object_set_new( json_root, "nodes", json_integer( numProc ) );
				json_object_set_new( json_root, "framebuffer width", json_integer ( framebuffer_width ) );
				json_object_set_new( json_root, "framebuffer height", json_integer ( framebuffer_height ) );
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
				ISAAC_JSON_ADD_MATRIX(matrix,projection)
				json_object_set_new( json_root, "modelview", matrix = json_array() );
				ISAAC_JSON_ADD_MATRIX(matrix,modelview)
				json_object_set_new( json_root, "rotation", matrix = json_array() );
				ISAAC_JSON_ADD_MATRIX(matrix,rotation)
				
				json_sources_array = json_array();
				json_object_set_new( json_root, "sources", json_sources_array );

				json_object_set_new( json_root, "dimension", json_integer ( TSimDim::value ) );
				json_object_set_new( json_root, "width", json_integer ( global_size[0] ) );
				if (TSimDim::value > 1)
					json_object_set_new( json_root, "height", json_integer ( global_size[1] ) );
				if (TSimDim::value > 2)
					json_object_set_new( json_root, "depth", json_integer ( global_size[2] ) );
				json_object_set_new( json_root, "type", json_string( "register" ) );
			}
		}
		
		void registerSource(std::string name,float* ptr,unsigned int f_dim)
		{
			#ifdef ISAAC_ALPAKA
				IsaacSource<TDevAcc,TAccDim>* source = new IsaacSource<TDevAcc,TAccDim>(name,ptr,f_dim);
				sources.push_back(std::shared_ptr< IsaacSource<TDevAcc,TAccDim> >( source ) );
			#else
				IsaacSource* source = new IsaacSource(name,ptr,f_dim);
				sources.push_back(std::shared_ptr< IsaacSource >( source ) );
			#endif
			if (rank == master)
			{
				json_t *content = json_object();
				json_array_append_new( json_sources_array, content );
				json_object_set_new( content, "name", json_string ( name.c_str() ) );
				json_object_set_new( content, "feature dimension", json_integer ( source->f_dim ) );
			}
		}
		
		json_t* getJsonMetaRoot()
		{
			return json_meta_root;
		}
		int init()
		{
			int failed = 0;
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
			//	printf("-----\n");
			ISAAC_WAIT_VISUALIZATION

			//Handle messages
			json_t* message;
			char message_buffer[MAX_RECEIVE];
			//Master merges all messages and broadcasts it.
			if (rank == master)
			{
				message = json_object();
				while (json_t* last = communicator->getLastMessage())
				{
					mergeJSON(message,last);
					json_decref( last );
				}
				char* buffer = json_dumps( message, 0 );
				strcpy( message_buffer, buffer );
				free(buffer);
				MPI_Bcast( message_buffer, MAX_RECEIVE, MPI_CHAR, master, MPI_COMM_WORLD);
			}
			else //The others just get the message
			{
				MPI_Bcast( message_buffer, MAX_RECEIVE, MPI_CHAR, master, MPI_COMM_WORLD);
				message = json_loads(message_buffer, 0, NULL);
			}
			
			json_t* js;
			size_t index;
			json_t *value;
			
			//Scene set?
			if (json_array_size( js = json_object_get(message, "projection") ) == 16)
				json_array_foreach(js, index, value)
					projection[index] = json_number_value( value );
			if (json_array_size( js = json_object_get(message, "modelview") ) == 16)
				json_array_foreach(js, index, value)
					modelview[index] = json_number_value( value );
			if (json_array_size( js = json_object_get(message, "rotation") ) == 16)
				json_array_foreach(js, index, value)
					rotation[index] = json_number_value( value );
					
			json_t* metadata = json_object_get( message, "metadata" );
			if (metadata)
				json_incref(metadata);
			json_decref(message);
			thr_metaTargets = metaTargets;
			#ifdef ISAAC_THREADING
				pthread_create(&visualizationThread,NULL,visualizationFunction,NULL);
			#else
				visualizationFunction(NULL);
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
	//private:		
		static IsaacVisualization *myself;
		static void drawCallBack(
			const IceTDouble * projection_matrix,
			const IceTDouble * modelview_matrix,
			const IceTFloat * background_color,
			const IceTInt * readback_viewport,
			IceTImage result)
		{
			#ifdef ISAAC_ALPAKA
				alpaka::mem::buf::Buf<THost, float, TFraDim, size_t> inverse_h_buf ( alpaka::mem::buf::alloc<float, size_t>(myself->host, size_t(16)));
				alpaka::mem::buf::Buf<THost, float, TFraDim, size_t> modelview_h_buf ( alpaka::mem::buf::alloc<float, size_t>(myself->host, size_t(16)));
				alpaka::mem::buf::Buf<THost, isaac_size_type, TFraDim, size_t> size_h_buf ( alpaka::mem::buf::alloc<isaac_size_type, size_t>(myself->host, size_t(1)));
				float* inverse_h = reinterpret_cast<float*>(alpaka::mem::view::getPtrNative(inverse_h_buf));
				float* modelview_h = reinterpret_cast<float*>(alpaka::mem::view::getPtrNative(modelview_h_buf));
				isaac_size_type* size_h = reinterpret_cast<isaac_size_type*>(alpaka::mem::view::getPtrNative(size_h_buf));
			#else
				float inverse_h[16];
				float modelview_h[16];
				isaac_size_type size_h[1];
			#endif
			IceTDouble inverse[16];
			myself->calcInverse(inverse,projection_matrix,modelview_matrix);
			for (int i = 0; i < 16; i++)
			{
				inverse_h[i] = static_cast<float>(inverse[i]);
				modelview_h[i] = static_cast<float>(modelview_matrix[i]);
			}
			ISAAC_CALL_FOR_XYZ_ITERATE( size_h[0].global_size, = myself->global_size[ 0, ]; )
			ISAAC_CALL_FOR_XYZ_ITERATE( size_h[0].position, = myself->position[ 0, ]; )
			ISAAC_CALL_FOR_XYZ_ITERATE( size_h[0].local_size, = myself->local_size[ 0, ]; )
			size_h[0].max_global_size = static_cast<float>(max(max(myself->global_size[0],myself->global_size[1]),myself->global_size[2]));
			
			#ifdef ISAAC_ALPAKA
				alpaka::mem::view::copy(myself->stream, myself->inverse_d, inverse_h_buf, size_t(16));
				alpaka::mem::view::copy(myself->stream, myself->modelview_d, modelview_h_buf, size_t(16));
				alpaka::mem::view::copy(myself->stream, myself->size_d, size_h_buf, size_t(1));
			#else
				ISAAC_CUDA_CHECK(cudaMemcpyToSymbol( isaac_inverse_d, inverse_h, 16 * sizeof(float)));
				ISAAC_CUDA_CHECK(cudaMemcpyToSymbol( isaac_modelview_d, modelview_h, 16 * sizeof(float)));
				ISAAC_CUDA_CHECK(cudaMemcpyToSymbol( isaac_size_d, size_h, sizeof(isaac_size_type)));
			#endif
			uint32_t value = (255 << 24) | (myself->rank*255/myself->numProc << 16) | (255 - myself->rank*255/myself->numProc << 8) | 255;
			IceTUByte* pixels = icetImageGetColorub(result);
			size_t g_width = (readback_viewport[2]+15)/16;
			size_t g_height = (readback_viewport[3]+15)/16;
			size_t b_width = 16;
			size_t b_height = 16;
			ISAAC_START_TIME_MEASUREMENT( kernel, myself->getTicksUs() )
			float* source = myself->sources[0]->getPtr();
			float step = 1.0f;
			isaac_float4 bg_color = { background_color[3], background_color[2], background_color[1], background_color[0] };
			#ifdef ISAAC_ALPAKA
				if ( boost::mpl::not_<boost::is_same<TAcc, alpaka::acc::AccGpuCudaRt<TAccDim, size_t> > >::value )
				{
					g_width = readback_viewport[2];
					g_height = readback_viewport[3];
					b_width = 1;
					b_height = 1;					
				}
				const alpaka::Vec<TAccDim, size_t> threads (size_t(1), size_t(1), size_t(1));
				const alpaka::Vec<TAccDim, size_t> blocks  (size_t(1), size_t(b_height), size_t(b_width));
				const alpaka::Vec<TAccDim, size_t> grid    (size_t(1), size_t(g_height), size_t(g_width));
				auto const workdiv(alpaka::workdiv::WorkDivMembers<TAccDim, size_t>(grid,blocks,threads));
				/*const alpaka::Vec<TAccDim, size_t> grid    (size_t(1), size_t(readback_viewport[3]), size_t(readback_viewport[2]));
				auto const workdiv(
					alpaka::workdiv::getValidWorkDiv<TAcc>(
					myself->acc,
					grid,
					alpaka::Vec<TAccDim, size_t>::ones(),
					false,
					alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));*/
				auto const test (alpaka::exec::create<TAcc> (workdiv,
					myself->fillRectKernel,
					alpaka::mem::view::getPtrNative(myself->inverse_d),
					alpaka::mem::view::getPtrNative(myself->modelview_d),
					alpaka::mem::view::getPtrNative(myself->size_d),
					alpaka::mem::view::getPtrNative(myself->framebuffer),
					value,
					myself->framebuffer_width,
					myself->framebuffer_height,
					readback_viewport[0],
					readback_viewport[1],
					source,
					step,
					bg_color));
				alpaka::stream::enqueue(myself->stream, test);
				alpaka::wait::wait(myself->stream);
				ISAAC_STOP_TIME_MEASUREMENT( myself->kernel_time, +=, kernel, myself->getTicksUs() )
				ISAAC_START_TIME_MEASUREMENT( copy, myself->getTicksUs() )
				alpaka::mem::buf::BufPlainPtrWrapper<THost, uint32_t, TFraDim, size_t> result_buffer((uint32_t*)(pixels), myself->host, myself->framebuffer_size);
				alpaka::mem::view::copy(myself->stream, result_buffer, myself->framebuffer, myself->framebuffer_size);
			#else
				dim3 block (b_width,b_height);
				dim3 grid  (g_width,g_height);
				IsaacFillRectKernel<<<grid, block>>>(
					myself->framebuffer,
					value,
					myself->framebuffer_width,
					myself->framebuffer_height,
					readback_viewport[0],
					readback_viewport[1],
					source,
					step,
					bg_color);
				ISAAC_CUDA_CHECK(cudaDeviceSynchronize());
				ISAAC_STOP_TIME_MEASUREMENT( myself->kernel_time, +=, kernel, myself->getTicksUs() )
				ISAAC_START_TIME_MEASUREMENT( copy, myself->getTicksUs() )
				ISAAC_CUDA_CHECK(cudaMemcpy((uint32_t*)(pixels), myself->framebuffer, sizeof(uint32_t)*myself->framebuffer_size, cudaMemcpyDeviceToHost));
			#endif
			ISAAC_STOP_TIME_MEASUREMENT( myself->copy_time, +=, copy, myself->getTicksUs() )
		}
		void mergeJSON(json_t* result,json_t* candidate)
		{
			const char *c_key;
			const char *r_key;
			json_t *c_value;
			json_t *r_value;
			//metadata merge, old values stay, arrays are merged
			json_t* m_candidate = json_object_get(candidate, "metadata");
			json_t* m_result = json_object_get(result, "metadata");
			void *temp,*temp2;
			if (m_candidate && m_result)
			{
				json_object_foreach_safe( m_candidate, temp, c_key, c_value )
				{
					bool found_array = false;
					json_object_foreach_safe( m_result, temp2, r_key, r_value )
					{
						if (strcmp(r_key,c_key) == 0)
						{
							if (json_is_array(r_value) && json_is_array(c_value))
							{
								json_array_extend(r_value,c_value);
								found_array = true;
							}
							break;
						}
					}
					if (!found_array)
						json_object_set( m_result, c_key, c_value );
				}
			}			
			//general merge, new values stay
			json_object_foreach_safe( candidate, temp, c_key, c_value )
			{
				bool found_meta = false;
				json_object_foreach_safe( result, temp2, r_key, r_value )
				{
					if (strcmp(r_key,c_key) == 0 && strcmp(r_key,"metadata") == 0)
					{
						found_meta = true;
						break;
					}
				}
				if (!found_meta)
					json_object_set( result, c_key, c_value );
			}
		}
		void mulMatrixMatrix(IceTDouble* result,const IceTDouble* matrix1,const IceTDouble* matrix2)
		{
			for (int x = 0; x < 4; x++)
				for (int y = 0; y < 4; y++)
					result[y+x*4] = matrix1[y+0*4] * matrix2[0+x*4]
								  + matrix1[y+1*4] * matrix2[1+x*4]
								  + matrix1[y+2*4] * matrix2[2+x*4]
								  + matrix1[y+3*4] * matrix2[3+x*4];
		}
		void mulMatrixVector(IceTDouble* result,const IceTDouble* matrix,const IceTDouble* vector)
		{
			result[0] =  matrix[ 0] * vector[0] + matrix[ 4] * vector[1] +  matrix[ 8] * vector[2] + matrix[12] * vector[3];
			result[1] =  matrix[ 1] * vector[0] + matrix[ 5] * vector[1] +  matrix[ 9] * vector[2] + matrix[13] * vector[3];
			result[2] =  matrix[ 2] * vector[0] + matrix[ 6] * vector[1] +  matrix[10] * vector[2] + matrix[14] * vector[3];
			result[3] =  matrix[ 3] * vector[0] + matrix[ 7] * vector[1] +  matrix[11] * vector[2] + matrix[15] * vector[3];
		}
		static void* visualizationFunction(void* dummy)
		{
			//Message sending
			char message_buffer[MAX_RECEIVE];
			char* buffer = json_dumps( myself->json_root, 0 );
			strcpy( message_buffer, buffer );
			free(buffer);
			if (myself->thr_metaTargets == META_MERGE)
			{
				if (myself->rank == myself->master)
				{
					char receive_buffer[myself->numProc][MAX_RECEIVE];
					MPI_Gather( message_buffer, MAX_RECEIVE, MPI_CHAR, receive_buffer, MAX_RECEIVE, MPI_CHAR, myself->master, MPI_COMM_WORLD);
					for (int i = 0; i < myself->numProc; i++)
					{
						if (i == myself->master)
							continue;
						json_t* js = json_loads(receive_buffer[i], 0, NULL);
						myself->mergeJSON( myself->json_root, js );
					}
				}
				else
					MPI_Gather( message_buffer, MAX_RECEIVE, MPI_CHAR, NULL, 0,  MPI_CHAR, myself->master, MPI_COMM_WORLD);
			}
			if (myself->rank == myself->master && myself->thr_metaTargets != META_NONE)
			{
				json_object_set_new( myself->json_root, "type", json_string( "period" ) );
				json_object_set_new( myself->json_root, "meta nr", json_integer( myself->metaNr ) );
				char* buffer = json_dumps( myself->json_root, 0 );
				myself->communicator->serverSend(buffer);
				free(buffer);
			}
			json_decref( myself->json_root );
			myself->recreateJSON();
			//Calculating the whole modelview matrix
			IceTDouble real_modelview[16];
			myself->mulMatrixMatrix(real_modelview,myself->modelview,myself->rotation);

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
			myself->mulMatrixVector(result,real_modelview,point);
			float point_distance = sqrt( result[0]*result[0] + result[1]*result[1] + result[2]*result[2] );
			//Allgather of the distances
			float receive_buffer[myself->numProc];
			MPI_Allgather( &point_distance, 1, MPI_FLOAT, receive_buffer, 1, MPI_FLOAT, MPI_COMM_WORLD);
			//Putting to a std::multimap of {rank, distance}
			std::multimap<float, int, std::less< float > > distance_map;
			for (int i = 0; i < myself->numProc; i++)
				distance_map.insert( std::pair<float, int>( receive_buffer[i], i ) );
			//Putting in an array for IceT
			IceTInt icet_order_array[myself->numProc];
			{
				int i = 0;
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
			IceTImage image = icetDrawFrame(myself->projection,real_modelview,background_color);
			ISAAC_STOP_TIME_MEASUREMENT( myself->merge_time, +=, merge, myself->getTicksUs() )
			ISAAC_START_TIME_MEASUREMENT( video_send, myself->getTicksUs() )
			if (myself->video_communicator)
				myself->video_communicator->serverSendFrame(icetImageGetColorui(image),icetImageGetNumPixels(image)*4);
			ISAAC_STOP_TIME_MEASUREMENT( myself->video_send_time, +=, video_send, myself->getTicksUs() )

			myself->metaNr++;
			return 0;
		}
		void setFrustum(float left,float  right,float  bottom,float  top,float  znear,float  zfar )
		{
			float  znear2 = znear * 2.0f;
			float  width = right - left;
			float  height = top - bottom;
			float  zRange = znear - zfar;
			projection[ 0] = znear2 / width;
			projection[ 1] = 0.0f;
			projection[ 2] = 0.0f;
			projection[ 3] = 0.0f;
			projection[ 4] = 0.0f;
			projection[ 5] = znear2 / height;
			projection[ 6] = 0.0f;
			projection[ 7] = 0.0f;
			projection[ 8] = ( right + left ) / width;
			projection[ 9] = ( top + bottom ) / height;
			projection[10] = ( zfar + znear) / zRange;
			projection[11] = -1.0f;
			projection[12] = 0.0f;
			projection[13] = 0.0f;
			projection[14] = ( -znear2 * zfar ) / -zRange;
			projection[15] = 0.0f;
		}

		void setPerspective(float fovyInDegrees,float aspectRatio,float __znear,float zfar )
		{
			float znear = __znear;
			float ymax = znear * tan( fovyInDegrees * M_PI / 360.0f );
			float xmax = ymax * aspectRatio;
			setFrustum( -xmax, xmax, -ymax, ymax, znear, zfar );
		}

		void calcInverse(IceTDouble* inv,const IceTDouble* projection,const IceTDouble* modelview)
		{
			IceTDouble m[16];
			mulMatrixMatrix(m,projection,modelview);
			inv[0] = m[5]  * m[10] * m[15] - 
					 m[5]  * m[11] * m[14] - 
					 m[9]  * m[6]  * m[15] + 
					 m[9]  * m[7]  * m[14] +
					 m[13] * m[6]  * m[11] - 
					 m[13] * m[7]  * m[10];

			inv[4] = -m[4]  * m[10] * m[15] + 
					  m[4]  * m[11] * m[14] + 
					  m[8]  * m[6]  * m[15] - 
					  m[8]  * m[7]  * m[14] - 
					  m[12] * m[6]  * m[11] + 
					  m[12] * m[7]  * m[10];

			inv[8] = m[4]  * m[9] * m[15] - 
					 m[4]  * m[11] * m[13] - 
					 m[8]  * m[5] * m[15] + 
					 m[8]  * m[7] * m[13] + 
					 m[12] * m[5] * m[11] - 
					 m[12] * m[7] * m[9];

			inv[12] = -m[4]  * m[9] * m[14] + 
					   m[4]  * m[10] * m[13] +
					   m[8]  * m[5] * m[14] - 
					   m[8]  * m[6] * m[13] - 
					   m[12] * m[5] * m[10] + 
					   m[12] * m[6] * m[9];

			inv[1] = -m[1]  * m[10] * m[15] + 
					  m[1]  * m[11] * m[14] + 
					  m[9]  * m[2] * m[15] - 
					  m[9]  * m[3] * m[14] - 
					  m[13] * m[2] * m[11] + 
					  m[13] * m[3] * m[10];

			inv[5] = m[0]  * m[10] * m[15] - 
					 m[0]  * m[11] * m[14] - 
					 m[8]  * m[2] * m[15] + 
					 m[8]  * m[3] * m[14] + 
					 m[12] * m[2] * m[11] - 
					 m[12] * m[3] * m[10];

			inv[9] = -m[0]  * m[9] * m[15] + 
					  m[0]  * m[11] * m[13] + 
					  m[8]  * m[1] * m[15] - 
					  m[8]  * m[3] * m[13] - 
					  m[12] * m[1] * m[11] + 
					  m[12] * m[3] * m[9];

			inv[13] = m[0]  * m[9] * m[14] - 
					  m[0]  * m[10] * m[13] - 
					  m[8]  * m[1] * m[14] + 
					  m[8]  * m[2] * m[13] + 
					  m[12] * m[1] * m[10] - 
					  m[12] * m[2] * m[9];

			inv[2] = m[1]  * m[6] * m[15] - 
					 m[1]  * m[7] * m[14] - 
					 m[5]  * m[2] * m[15] + 
					 m[5]  * m[3] * m[14] + 
					 m[13] * m[2] * m[7] - 
					 m[13] * m[3] * m[6];

			inv[6] = -m[0]  * m[6] * m[15] + 
					  m[0]  * m[7] * m[14] + 
					  m[4]  * m[2] * m[15] - 
					  m[4]  * m[3] * m[14] - 
					  m[12] * m[2] * m[7] + 
					  m[12] * m[3] * m[6];

			inv[10] = m[0]  * m[5] * m[15] - 
					  m[0]  * m[7] * m[13] - 
					  m[4]  * m[1] * m[15] + 
					  m[4]  * m[3] * m[13] + 
					  m[12] * m[1] * m[7] - 
					  m[12] * m[3] * m[5];

			inv[14] = -m[0]  * m[5] * m[14] + 
					   m[0]  * m[6] * m[13] + 
					   m[4]  * m[1] * m[14] - 
					   m[4]  * m[2] * m[13] - 
					   m[12] * m[1] * m[6] + 
					   m[12] * m[2] * m[5];

			inv[3] = -m[1] * m[6] * m[11] + 
					  m[1] * m[7] * m[10] + 
					  m[5] * m[2] * m[11] - 
					  m[5] * m[3] * m[10] - 
					  m[9] * m[2] * m[7] + 
					  m[9] * m[3] * m[6];

			inv[7] = m[0] * m[6] * m[11] - 
					 m[0] * m[7] * m[10] - 
					 m[4] * m[2] * m[11] + 
					 m[4] * m[3] * m[10] + 
					 m[8] * m[2] * m[7] - 
					 m[8] * m[3] * m[6];

			inv[11] = -m[0] * m[5] * m[11] + 
					   m[0] * m[7] * m[9] + 
					   m[4] * m[1] * m[11] - 
					   m[4] * m[3] * m[9] - 
					   m[8] * m[1] * m[7] + 
					   m[8] * m[3] * m[5];

			inv[15] = m[0] * m[5] * m[10] - 
					  m[0] * m[6] * m[9] - 
					  m[4] * m[1] * m[10] + 
					  m[4] * m[2] * m[9] + 
					  m[8] * m[1] * m[6] - 
					  m[8] * m[2] * m[5];

			IceTDouble det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

			if (det == 0)
				return;
				
			det = 1.0 / det;

			for (int i = 0; i < 16; i++)
				inv[i] = inv[i] * det;
		}
		void recreateJSON()
		{
			json_root = json_object();
			json_meta_root = json_object();
			json_object_set_new( json_root, "metadata", json_meta_root );
		}
		class IsaacCommunicator
		{
			public:
				IsaacCommunicator(std::string url,int port)
				{
					pthread_mutex_init (&deleteMessageMutex, NULL);
					this->url = url;
					this->port = port;
					this->sockfd = 0;
				}
				static size_t json_load_callback_function (void *buffer, size_t buflen, void *data)
				{
					return read(*((int*)data),buffer,1);
				}
				void readAndSetMessages()
				{
					while (json_t * content = json_load_callback(json_load_callback_function,&sockfd,JSON_DISABLE_EOF_CHECK,NULL))
					{
						pthread_mutex_lock(&deleteMessageMutex);
						messageList.push_back(content);
						pthread_mutex_unlock(&deleteMessageMutex);
					}
				}
				static void* run_readAndSetMessages(void* communicator)
				{
					((IsaacCommunicator*)communicator)->readAndSetMessages();
					return 0;
				}
				json_t* getLastMessage()
				{
					json_t* result = NULL;
					pthread_mutex_lock(&deleteMessageMutex);
					if (!messageList.empty())
					{
						result = messageList.front();
						messageList.pop_front();
					}
					pthread_mutex_unlock(&deleteMessageMutex);
					return result;
				}
				int serverConnect(bool video = false)
				{
					struct hostent *server;
					server = gethostbyname(url.c_str());
					if (!server)
					{
						fprintf(stderr,"Could not resolve %s.\n",url.c_str());
						return -1;
					}
					sockfd = socket(AF_INET, SOCK_STREAM, 0);
					if (sockfd < 0)
					{
						fprintf(stderr,"Could not create socket.\n");
						return -2;
					}
					struct sockaddr_in serv_addr;
					memset(&serv_addr,0, sizeof(serv_addr));
					serv_addr.sin_family = AF_INET;
					bcopy((char *)server->h_addr,(char *)&serv_addr.sin_addr.s_addr,server->h_length);
					serv_addr.sin_port = htons(port);
					if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
					{
						close(sockfd);
						fprintf(stderr,"Could not connect to %s.\n",url.c_str());
						return -3;
					}
					if (!video)
						pthread_create(&readThread,NULL,run_readAndSetMessages,this);
					return 0;
				}
				int serverSend(const char* content)
				{
					int n = send(sockfd,content,strlen(content),0);
					return n;
				}
				int serverSendFrame(void* ptr,int count)
				{
					int n = 0;
					int div = count / 262144; //256kb per message
					int rest = count % 262144; //rest
					for (int i = 0; i <=  div; i++)
					{
						int r = -1;
						while (r < 0)
							r = send(sockfd,&(((char*)ptr)[i*262144]),i == div ? rest : 262144,0);
						n += r;
					}
					return n;
				}
				void serverDisconnect()
				{
					close(sockfd);
				}
				~IsaacCommunicator()
				{
					if (sockfd)
						serverDisconnect();
					usleep(100000); //100ms
					pthread_cancel(readThread);
					pthread_mutex_destroy(&deleteMessageMutex);
				}
			private:
				std::string url;
				int port;
				int sockfd;
				std::list<json_t*> messageList;
				pthread_mutex_t deleteMessageMutex;
				pthread_t readThread;
		};
		#ifdef ISAAC_ALPAKA
			THost host;
			TDevAcc acc;
			TStream stream;
		#endif
		std::string name;
		std::string server_url;
		int server_port;
		int framebuffer_width;
		int framebuffer_height;
		#ifdef ISAAC_ALPAKA
			alpaka::Vec<TFraDim, size_t> framebuffer_size;
			alpaka::Vec<TSimDim, size_t> global_size;
			alpaka::Vec<TSimDim, size_t> local_size;
			alpaka::Vec<TSimDim, size_t> position;
			alpaka::mem::buf::Buf<TDevAcc, uint32_t, TFraDim, size_t> framebuffer;
			alpaka::mem::buf::Buf<TDevAcc, float, TFraDim, size_t> inverse_d;
			alpaka::mem::buf::Buf<TDevAcc, float, TFraDim, size_t> modelview_d;
			alpaka::mem::buf::Buf<TDevAcc, isaac_size_type, TFraDim, size_t> size_d;
		#else
			size_t framebuffer_size;
			std::vector<size_t> global_size;
			std::vector<size_t> local_size;
			std::vector<size_t> position;		
			uint32_t* framebuffer;
		#endif
		IceTDouble projection[16];
		IceTDouble modelview[16];
		IceTDouble rotation[16];
		IsaacCommunicator* communicator;
		IsaacCommunicator* video_communicator;
		json_t *json_root;
		json_t *json_meta_root;
		json_t *json_sources_array;
		int rank;
		int master;
		int numProc;
		int metaNr;
		#ifdef ISAAC_ALPAKA
			std::vector< std::shared_ptr< IsaacSource<TDevAcc,TAccDim> > > sources;
		#else
			std::vector< std::shared_ptr< IsaacSource > > sources;		
		#endif
		IceTContext icetContext;
		IsaacVisualizationMetaEnum thr_metaTargets;
		pthread_t visualizationThread;
		#ifdef ISAAC_ALPAKA
			IsaacFillRectKernel fillRectKernel;
		#endif
};

#ifdef ISAAC_ALPAKA
	template <typename THost,typename TAcc,typename TStream,typename TAccDim,typename TSimDim>
	IsaacVisualization<THost,TAcc,TStream,TAccDim,TSimDim>* IsaacVisualization<THost,TAcc,TStream,TAccDim,TSimDim>::myself = NULL;
#else
	template <typename TSimDim>
	IsaacVisualization<TSimDim>* IsaacVisualization<TSimDim>::myself = NULL;
#endif
