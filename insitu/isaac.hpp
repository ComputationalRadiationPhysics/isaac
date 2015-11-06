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

#define MAX_RECEIVE 262144 //256kb

typedef enum
{
	META_NONE = -1,
	META_MERGE = 0,
	META_MASTER = 1
} IsaacVisualizationMetaEnum;

class IsaacSource
{
	friend class IsaacVisualization;
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
		int f_dim;
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

class IsaacVisualization 
{
	public:
		IsaacVisualization(
			std::string name,
			int master,
			std::string server_url,
			int server_port,
			int framebuffer_width,
			int framebuffer_height,
			unsigned int g_width,
			unsigned int g_height,
			unsigned int g_depth,
			unsigned int l_width,
			unsigned int l_height,
			unsigned int l_depth,
			unsigned int x,
			unsigned int y = 0,
			unsigned int z = 0)
		{
			//INIT
			myself = this;
			this->name = name;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			MPI_Comm_size(MPI_COMM_WORLD, &numProc);
			this->master = master;
			this->server_url = server_url;
			this->server_port = server_port;
			this->framebuffer_width = framebuffer_width;
			this->framebuffer_height = framebuffer_height;
			this->metaNr = 0;
			this->communicator = new IsaacCommunicator(server_url,server_port);
			if (rank == master)
				this->video_communicator = new IsaacCommunicator(server_url,server_port);
			else
				this->video_communicator = NULL;
			this->visualizationThread = 0;
			this->sources_g_size[0] = g_width;
			this->sources_g_size[1] = g_height;
			this->sources_g_size[2] = g_depth;
			this->sources_l_size[0] = l_width;
			this->sources_l_size[1] = l_height;
			this->sources_l_size[2] = l_depth;
			this->sources_pos[0] = x;
			this->sources_pos[1] = y;
			this->sources_pos[2] = z;
			if (l_height == 0)
				this->dim = 1;
			else
			if (l_depth == 0)
				this->dim = 2;
			else
				this->dim = 3;
			setPerspective( 45.0f, (float)framebuffer_width/(float)framebuffer_height,1.0f, 100.0f);
			ISAAC_SET_IDENTITY(modelview)
			modelview[14] = -5.0f; //glTranslate(0,0,-5);
			ISAAC_SET_IDENTITY(rotation)
			
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
			
			int max = g_width > g_height ? g_width : g_height;
			float f_l_width = (float)l_width/(float)max * 2.0f;
			float f_l_height = (float)l_height/(float)max * 2.0f;
			float f_l_depth = (float)l_depth/(float)max * 2.0f;
			float f_x = (float)x/(float)max * 2.0f - 1.0f;
			float f_y = (float)y/(float)max * 2.0f - 1.0f;
			float f_z = (float)z/(float)max * 2.0f - 1.0f;
			icetBoundingBoxf( f_x, f_x + f_l_width, f_y, f_y + f_l_height, f_z, f_z + f_l_depth);
			icetPhysicalRenderSize(framebuffer_width, framebuffer_height);
			icetDrawCallback( drawCallBack );
			
			//JSON
			recreateJSON();
			json_object_set_new( json_root, "name", json_string( name.c_str() ) );
			json_object_set_new( json_root, "rank", json_integer( rank ) );
			if (rank == master)
			{
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

				json_object_set_new( json_root, "dimension", json_integer ( dim ) );
				json_object_set_new( json_root, "width", json_integer ( sources_g_size[0] ) );
				if (dim > 1)
					json_object_set_new( json_root, "height", json_integer ( sources_g_size[1] ) );
				if (dim > 2)
					json_object_set_new( json_root, "depth", json_integer ( sources_g_size[2] ) );

			}
		}
		
		void registerSource(std::string name,float* ptr,unsigned int f_dim)
		{
			IsaacSource* source = new IsaacSource(name,ptr,f_dim);
			sources.push_back(std::shared_ptr< IsaacSource >( source ) );
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
			if (communicator->serverConnect())
				return -1;
			if (video_communicator && video_communicator->serverConnect())
				return -1;
			if (rank == master)
				json_object_set_new( json_root, "type", json_string( "register master" ) );
			else
				json_object_set_new( json_root, "type", json_string( "register slave" ) );
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
			
			//Let's wait for the group completely registered at the server
			json_t* ready;
			while ((ready = communicator->getLastMessage()) == NULL)
				usleep(5000);
			json_decref( ready ); //Let's just assume the server sends a correct message
			recreateJSON();
			return 0;
		}
		void doVisualization(
			IsaacVisualizationMetaEnum metaTargets = META_MASTER,
			int metaCount = -1 )
		{
			ISAAC_WAIT_VISUALIZATION
			thr_metaTargets = metaTargets;
			thr_metaCount = metaCount;
			#ifdef ISAAC_THREADING
				pthread_create(&visualizationThread,NULL,viszualisationFunction,NULL);
			#else
				viszualisationFunction(NULL);
			#endif
		}
		json_t* getMeta()
		{
			ISAAC_WAIT_VISUALIZATION
			return communicator->getLastMeta();
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
	private:
		static IsaacVisualization *myself;
		static void drawCallBack(
			const IceTDouble * projection_matrix,
			const IceTDouble * modelview_matrix,
			const IceTFloat * background_color,
			const IceTInt * readback_viewport,
			IceTImage result)
		{
			IceTUByte* pixels = icetImageGetColorub(result);
			int w = icetImageGetWidth(result);
			for (int x = readback_viewport[0]; x < readback_viewport[0]+readback_viewport[2]; x++)
				for (int y = readback_viewport[1]; y < readback_viewport[1]+readback_viewport[3]; y++)
				{
					int base = (x+y*w)*4;
					pixels[base + 0] = 0; //B
					pixels[base + 1] = myself->rank*255/myself->numProc; //G
					pixels[base + 2] = 255-myself->rank*255/myself->numProc; //R
					pixels[base + 3] = 255; //A
				}
		}
		static void* viszualisationFunction(void* dummy)
		{
			//Handle messages
			while (json_t* message = myself->communicator->getLastMessage())
			{
				json_t* js;
				size_t index;
				json_t *value;
				//Scene set?
				if (json_array_size( js = json_object_get(message, "projection") ) == 16)
					json_array_foreach(js, index, value)
						myself->projection[index] = json_number_value( value );
				if (json_array_size( js = json_object_get(message, "modelview") ) == 16)
					json_array_foreach(js, index, value)
						myself->modelview[index] = json_number_value( value );
				if (json_array_size( js = json_object_get(message, "rotation") ) == 16)
					json_array_foreach(js, index, value)
						myself->rotation[index] = json_number_value( value );
				json_decref( message );
			}
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
			if (myself->rank == myself->master)
				myself->video_communicator->serverSendFrame(icetImageGetColorui(image),icetImageGetNumPixels(image)*4);
			//Sending messages to isaac server
			if (myself->thr_metaTargets == META_MERGE || (myself->thr_metaTargets == META_MASTER && myself->rank == myself->master))
			{
				if (myself->thr_metaTargets == META_MERGE)
					json_object_set_new( myself->json_root, "type", json_string( "period merge" ) );
				else
				if (myself->thr_metaTargets == META_MASTER && myself->rank == myself->master)
					json_object_set_new( myself->json_root, "type", json_string( "period master" ) );
				if (myself->thr_metaCount > 0)
					json_object_set_new( myself->json_root, "count", json_integer( myself->thr_metaCount ) );
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
					pthread_mutex_init (&deleteMetaMutex, NULL);
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
						json_t* metaElement = json_object_get(content, "metadata");
						if (json_is_object(metaElement))
						{
							pthread_mutex_lock(&deleteMetaMutex);
							metaList.push_back(json_incref(metaElement));
							pthread_mutex_unlock(&deleteMetaMutex);
						}
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
				json_t* getLastMeta()
				{
					json_t* result = NULL;
					pthread_mutex_lock(&deleteMetaMutex);
					if (!metaList.empty())
					{
						result = metaList.front();
						metaList.pop_front();
					}
					pthread_mutex_unlock(&deleteMetaMutex);
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
				std::list<json_t*> metaList;
				pthread_mutex_t deleteMetaMutex;
				std::list<json_t*> messageList;
				pthread_mutex_t deleteMessageMutex;
				pthread_t readThread;
		};	
		std::string name;
		std::string server_url;
		int server_port;
		int framebuffer_width;
		int framebuffer_height;
		int sources_g_size[3];
		int sources_l_size[3];
		int sources_pos[3];
		int dim;
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
		std::vector< std::shared_ptr< IsaacSource > > sources;
		IceTContext icetContext;
		IsaacVisualizationMetaEnum thr_metaTargets;
		int thr_metaCount;
		pthread_t visualizationThread;
};

IsaacVisualization* IsaacVisualization::myself = NULL;
