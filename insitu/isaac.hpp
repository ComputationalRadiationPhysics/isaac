/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ISAAC.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include <string>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h> 
#include <unistd.h>
#include <jansson.h>

#define MAX_RECEIVE 262144 //256kb

class IsaacCommunicator
{
	public:
		IsaacCommunicator(std::string url,int port)
		{
			this->url = url;
			this->port = port;
			this->sockfile = NULL;
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
			int sockfd = socket(AF_INET, SOCK_STREAM, 0);
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
			sockfile = fdopen(sockfd,"r+");
			return 0;
		}
		int serverSend(std::string content)
		{
			const char* c_content = content.c_str();
			int n = fwrite(c_content,strlen(c_content),1,sockfile);
			fflush(sockfile);
			return n;
		}
		std::string serverReceive()
		{
			char buffer[MAX_RECEIVE];
			fread(buffer,MAX_RECEIVE,1,sockfile);
			return std::string(buffer);
		}
		void serverDisconnect()
		{
			fclose(sockfile);
		}
		~IsaacCommunicator()
		{
			if (sockfile)
				serverDisconnect();
		}
	private:
		std::string url;
		int port;
		FILE* sockfile;
};

class IsaacVisualization 
{
	public:
		IsaacVisualization(
			std::string name,
			std::string server_url,
			int server_port,
			int framebuffer_count,
			int framebuffer_size )
		{
			this->name = name;
			this->server_url = server_url;
			this->server_port = server_port;
			this->framebuffer_count = framebuffer_count;
			this->framebuffer_size = framebuffer_size;
			this->communicator = new IsaacCommunicator(server_url,server_port);
			//TODO: Alloc framebuffer
			recreateJSON(false);
			json_object_set_new( json_root, "name", json_string( name.c_str() ) );
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
		json_t* getJsonMetaRoot()
		{
			return json_meta_root;
		}
		int init()
		{
			if (communicator->serverConnect())
				return -1;
			char* buffer = json_dumps( json_root, 0 );
			communicator->serverSend(std::string(buffer) + " ");
			free(buffer);
			json_decref( json_root );
			recreateJSON(true);
			return 0;
		}
		int drawNow()
		{
			char* buffer = json_dumps( json_root, 0 );
			communicator->serverSend(std::string(buffer) + " ");
			free(buffer);
			json_decref( json_root );
			recreateJSON(true);
		}
		~IsaacVisualization()
		{
			json_decref( json_root );
			json_root = json_object();
			json_object_set_new( json_root, "type", json_string( "exit" ) );
			char* buffer = json_dumps( json_root, 0 );
			communicator->serverSend(std::string(buffer) + " ");
			free(buffer);
			json_decref( json_root );
			delete communicator;
		}	
	private:
		void recreateJSON(bool period)
		{
			json_root = json_object();
			if (period)
				json_object_set_new( json_root, "type", json_string( "period" ) );
			else
				json_object_set_new( json_root, "type", json_string( "register" ) );
			json_meta_root = json_object();
			json_object_set_new( json_root, "metadata", json_meta_root );
		}
		std::string name;
		std::string server_url;
		int server_port;
		int framebuffer_count;
		int framebuffer_size;
		IsaacCommunicator* communicator;
		json_t *json_root;
		json_t *json_meta_root;
};
