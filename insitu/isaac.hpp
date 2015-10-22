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
#include <mpi.h>

#define MAX_RECEIVE 262144 //256kb

typedef enum
{
	META_MERGE = 0,
	META_MASTER = 1
} IsaacVisualizationMetaEnum;

class IsaacVisualization 
{
	public:
		IsaacVisualization(
			std::string name,
			int master,
			std::string server_url,
			int server_port,
			int framebuffer_count,
			int framebuffer_size )
		{
			this->name = name;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			MPI_Comm_size(MPI_COMM_WORLD, &numProc);
			this->master = master;
			this->server_url = server_url;
			this->server_port = server_port;
			this->framebuffer_count = framebuffer_count;
			this->framebuffer_size = framebuffer_size;
			this->communicator = new IsaacCommunicator(server_url,server_port);
			//TODO: Alloc framebuffer
			recreateJSON();
			json_object_set_new( json_root, "name", json_string( name.c_str() ) );
			json_object_set_new( json_root, "rank", json_integer( rank ) );
			if (rank == master)
			{
				json_object_set_new( json_root, "nodes", json_integer( numProc ) );
				json_object_set_new( json_root, "framebuffer count",  json_integer ( framebuffer_count ) );
				json_object_set_new( json_root, "framebuffer size", json_integer ( framebuffer_size ) );
				//TODO: Read real values
				json_object_set_new( json_root, "max chain", json_integer( 5 ) );
				json_t *operators = json_array();
				json_object_set_new( json_root, "operators", operators );
				json_array_append( operators, json_string( "dummy1" ) );
				json_array_append( operators, json_string( "dummy2" ) );
				json_array_append( operators, json_string( "dummy3" ) );
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
			if (rank == master)
				json_object_set_new( json_root, "type", json_string( "register master" ) );
			else
				json_object_set_new( json_root, "type", json_string( "register slave" ) );
			char* buffer = json_dumps( json_root, 0 );
			communicator->serverSend(std::string(buffer) + " ");
			free(buffer);
			json_decref( json_root );
			recreateJSON();
			return 0;
		}
		void doVisualization(
			IsaacVisualizationMetaEnum metaTargets)
		{
			//Getting messages
			while (json_t* message = communicator->getLastMessage())
			{
				//TODO: handling message
				json_decref( message );
			}
			//Drawing
			//TODO: Drawing ;)
			//Sending messages
			if (metaTargets == META_MERGE || rank == master)
			{
				if (metaTargets == META_MERGE)
					json_object_set_new( json_root, "type", json_string( "period merge" ) );
				else
					json_object_set_new( json_root, "type", json_string( "period master" ) );
				char* buffer = json_dumps( json_root, 0 );
				communicator->serverSend(std::string(buffer) + " ");
				free(buffer);
			}
			json_decref( json_root );
			recreateJSON();
		}
		json_t* getMeta()
		{
			return communicator->getLastMeta();
		}
		~IsaacVisualization()
		{
			json_decref( json_root );
			if (rank == master)
			{
				json_root = json_object();
				json_object_set_new( json_root, "type", json_string( "exit" ) );
				char* buffer = json_dumps( json_root, 0 );
				communicator->serverSend(std::string(buffer) + " ");
				free(buffer);
				json_decref( json_root );
			}
			delete communicator;
		}	
	private:
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
				int serverSend(std::string content)
				{
					const char* c_content = content.c_str();
					int n = write(sockfd,c_content,strlen(c_content));
					return n;
				}
				std::string serverReceive()
				{
					char buffer[MAX_RECEIVE];
					read(sockfd,buffer,MAX_RECEIVE);
					return std::string(buffer);
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
				std::list<json_t*> metaList;
				pthread_mutex_t deleteMetaMutex;
				std::list<json_t*> messageList;
				pthread_mutex_t deleteMessageMutex;
				pthread_t readThread;
		};	
		std::string name;
		std::string server_url;
		int server_port;
		int framebuffer_count;
		int framebuffer_size;
		IsaacCommunicator* communicator;
		json_t *json_root;
		json_t *json_meta_root;
		int rank;
		int master;
		int numProc;
};
