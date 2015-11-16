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

#ifdef ISAAC_ALPAKA
	#include <alpaka/alpaka.hpp>
#else
	#include <boost/mpl/int.hpp>
#endif

#define MAX_RECEIVE 32768 //32kb

#ifdef ISAAC_ALPAKA
	struct IsaacFillRectKernel
	{
		template <typename TAcc__, typename __TBuffer, typename __TValue>
		ALPAKA_FN_ACC void operator()( TAcc__ const &acc, __TBuffer pixels, __TValue value, size_t framebuffer_width, size_t start, size_t length) const
#else
		__global__ void IsaacFillRectKernel( uint32_t* pixels, uint32_t value, size_t framebuffer_width, size_t start, size_t length)
#endif
		{
			#ifdef ISAAC_ALPAKA
				auto threadIdx = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
				size_t local_start = start + threadIdx[2] * framebuffer_width;
			#else
				size_t local_start = start + threadIdx.x * framebuffer_width;
			#endif
			for(size_t i = local_start; i < local_start+length; i++)
				pixels[i] = value;
			pixels[local_start] = (255 << 24) | (255 << 16) | (0 << 8) | 0;
			pixels[local_start+length-1] = (255 << 24) | (255 << 16) | (255 << 8) | 0;
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
			framebuffer_size(size_t(framebuffer_width) * size_t(framebuffer_height))
			#ifdef ISAAC_ALPAKA
				,framebuffer(alpaka::mem::buf::alloc<uint32_t, size_t>(acc, framebuffer_size))
		{
			#else
		{
				cudaMalloc((uint32_t**)&framebuffer, sizeof(uint32_t)*framebuffer_size);
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
			setPerspective( 45.0f, (float)framebuffer_width/(float)framebuffer_height,1.0f, 100.0f);
			ISAAC_SET_IDENTITY(modelview)
			modelview[14] = -5.0f; //glTranslate(0,0,-5);
			ISAAC_SET_IDENTITY(rotation)
			
			//Fill framebuffer with test values:
			uint32_t value = (255 << 24) | (rank*255/numProc << 16) | (255 - rank*255/numProc << 8) | 0;
			#ifdef ISAAC_ALPAKA
				alpaka::mem::buf::Buf<THost, uint32_t, TFraDim, size_t> framebuffer_host(alpaka::mem::buf::alloc<uint32_t, size_t>(host, framebuffer_size));
				for (size_t i = 0; i < framebuffer_size.prod(); ++i)
					alpaka::mem::view::getPtrNative(framebuffer_host)[i] = value;
				alpaka::mem::view::copy(stream, framebuffer, framebuffer_host, framebuffer_size);
			#else
				uint32_t* framebuffer_host = (uint32_t*)malloc(framebuffer_size*sizeof(uint32_t));
				for (size_t i = 0; i < framebuffer_size; ++i)
					framebuffer_host[i] = value;
				cudaMemcpy(framebuffer, framebuffer_host, sizeof(uint32_t)*framebuffer_size, cudaMemcpyHostToDevice);
				free(framebuffer_host);
			#endif
			
			//ISAAC:
			IceTCommunicator icetComm;
			icetComm = icetCreateMPICommunicator(MPI_COMM_WORLD);
			icetContext = icetCreateContext(icetComm);
			icetDestroyMPICommunicator(icetComm);
			icetResetTiles();
			icetAddTile(0, 0, framebuffer_width, framebuffer_height, master);
			icetStrategy(ICET_STRATEGY_SPLIT);
			icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
			icetSetDepthFormat(ICET_IMAGE_DEPTH_NONE);
			icetCompositeMode(ICET_COMPOSITE_MODE_BLEND);
			
			size_t max = global_size[0] > global_size[1] ? global_size[0] : global_size[1];
			if (TSimDim::value > 2)
				max = global_size[2] > max ? global_size[2] : max;
			float f_l_width = (float)local_size[0]/(float)max * 2.0f;
			float f_l_height = (float)local_size[1]/(float)max * 2.0f;
			float f_l_depth = 0.0f;
			if (TSimDim::value > 2)
				f_l_depth = (float)local_size[2]/(float)max * 2.0f;
			float f_x = (float)position[0]/(float)max * 2.0f - (float)global_size[0]/(float)max;
			float f_y = (float)position[1]/(float)max * 2.0f - (float)global_size[1]/(float)max;
			float f_z = 0.0f;
			if (TSimDim::value > 2)
				f_z = (float)position[2]/(float)max * 2.0f - (float)global_size[2]/(float)max;
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
			if (failed == 0 && video_communicator && video_communicator->serverConnect())
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
	//private:		
		static IsaacVisualization *myself;
		static void drawCallBack(
			const IceTDouble * projection_matrix,
			const IceTDouble * modelview_matrix,
			const IceTFloat * background_color,
			const IceTInt * readback_viewport,
			IceTImage result)
		{
			uint32_t value = (255 << 24) | (myself->rank*255/myself->numProc << 16) | (255 - myself->rank*255/myself->numProc << 8) | 255;
			IceTUByte* pixels = icetImageGetColorub(result);
			#ifdef ISAAC_ALPAKA
				IsaacFillRectKernel fillRectKernel;				
				const alpaka::Vec<TAccDim, size_t> threads (size_t(1), size_t(1), size_t(1));
				const alpaka::Vec<TAccDim, size_t> blocks  (size_t(1), size_t(1), size_t(readback_viewport[3]));
				const alpaka::Vec<TAccDim, size_t> grid    (size_t(1), size_t(1), size_t(1));
				auto const workdiv(alpaka::workdiv::WorkDivMembers<TAccDim, size_t>(grid,blocks,threads));
				auto const test (alpaka::exec::create<TAcc> (workdiv,
					fillRectKernel,
					alpaka::mem::view::getPtrNative(myself->framebuffer),
					value,
					myself->framebuffer_width,
					readback_viewport[0]+readback_viewport[1]*myself->framebuffer_width,
					readback_viewport[2]));
				alpaka::stream::enqueue(myself->stream, test);
				alpaka::mem::buf::BufPlainPtrWrapper<THost, uint32_t, TFraDim, size_t> result_buffer((uint32_t*)(pixels), myself->host, myself->framebuffer_size);
				alpaka::mem::view::copy(myself->stream, result_buffer, myself->framebuffer, myself->framebuffer_size);
				//alpaka::wait::wait(myself->stream);
			#else
				dim3 block (readback_viewport[3]);
				dim3 grid  (1);
				IsaacFillRectKernel<<<grid, block>>>(
					myself->framebuffer,
					value,
					myself->framebuffer_width,
					readback_viewport[0]+readback_viewport[1]*myself->framebuffer_width,
					readback_viewport[2]);
				cudaMemcpy((uint32_t*)(pixels), myself->framebuffer, sizeof(uint32_t)*myself->framebuffer_size, cudaMemcpyDeviceToHost);
			#endif
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
		static void* visualizationFunction(void* dummy)
		{
			//Drawing
			IceTFloat background_color[4] = {0.0f,0.0f,0.0f,0.0};
			IceTDouble real_modelview[16];
			for (int x = 0; x < 4; x++)
				for (int y = 0; y < 4; y++)
					real_modelview[y+x*4]
						= myself->modelview[y+0*4] * myself->rotation[0+x*4]
						+ myself->modelview[y+1*4] * myself->rotation[1+x*4]
						+ myself->modelview[y+2*4] * myself->rotation[2+x*4]
						+ myself->modelview[y+3*4] * myself->rotation[3+x*4];
			IceTImage image = icetDrawFrame(myself->projection,real_modelview,background_color);
			if (myself->video_communicator)
				myself->video_communicator->serverSendFrame(icetImageGetColorui(image),icetImageGetNumPixels(image)*4);
			
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
					this->videofd = 0;
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
				int serverConnect()
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
					pthread_create(&readThread,NULL,run_readAndSetMessages,this);
					return 0;
				}
				int serverSend(const char* content)
				{
					int n = write(sockfd,content,strlen(content));
					return n;
				}
				int serverSendFrame(void* ptr,int count)
				{
					int n = write(sockfd,ptr,count);
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
				}
			private:
				std::string url;
				int port;
				int sockfd;
				int videofd;
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
};

#ifdef ISAAC_ALPAKA
	template <typename THost,typename TAcc,typename TStream,typename TAccDim,typename TSimDim>
	IsaacVisualization<THost,TAcc,TStream,TAccDim,TSimDim>* IsaacVisualization<THost,TAcc,TStream,TAccDim,TSimDim>::myself = NULL;
#else
	template <typename TSimDim>
	IsaacVisualization<TSimDim>* IsaacVisualization<TSimDim>::myself = NULL;
#endif
