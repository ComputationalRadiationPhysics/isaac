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

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h> 
#include <unistd.h>

#if ISAAC_JPEG == 1
    #include <jpeglib.h>
#endif

#include "isaac_macros.hpp"

namespace isaac
{

class IsaacCommunicator
{
	public:
		IsaacCommunicator(std::string url,isaac_uint port)
		{
			pthread_mutex_init (&deleteMessageMutex, NULL);
			this->url = url;
			this->port = port;
			this->sockfd = 0;
		}
		static size_t json_load_callback_function (void *buffer, size_t buflen, void *data)
		{
			return recv(*((isaac_int*)data),buffer,1,0);
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
		isaac_int serverConnect(bool video = false)
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
		isaac_int serverSend(const char* content)
		{
			uint32_t l = strlen(content);
			send(sockfd,&l,4,0);
			isaac_int n = send(sockfd,content,l,0);
			return n;
		}
		static void isaac_init_destination(j_compress_ptr cinfo)
		{
		}
		static boolean isaac_jpeg_empty_output_buffer(j_compress_ptr cinfo)
		{
			return true;
		}
		static void isaac_jpeg_term_destination(j_compress_ptr cinfo)
		{
		}
		isaac_int serverSendFrame(void* ptr,isaac_uint width,isaac_uint height,isaac_uint depth)
		{
			//let's first wait for message, whether the master is reading
			uint32_t count = width*height*depth;
			char go;
			recv(sockfd,&go,1,0);
			if (go != 42)
				return 0;
			//First the size
			#if ISAAC_JPEG == 1
				struct jpeg_compress_struct cinfo;
				struct jpeg_error_mgr jerr;
				jpeg_destination_mgr dest;
				dest.init_destination = &isaac_init_destination;
				dest.empty_output_buffer = &isaac_jpeg_empty_output_buffer;
				dest.term_destination = &isaac_jpeg_term_destination;
				cinfo.err = jpeg_std_error(&jerr);
				jpeg_create_compress(&cinfo);
				cinfo.dest = &dest;
				std::vector<char> jpeg_buffer;
				jpeg_buffer.resize( count );
				cinfo.dest->next_output_byte = (JOCTET*)(jpeg_buffer.data());
				cinfo.dest->free_in_buffer = count;
				cinfo.image_width = width;
				cinfo.image_height = height;
				cinfo.input_components = depth;
				cinfo.in_color_space = JCS_EXT_RGBX;
				jpeg_set_defaults(&cinfo);
				jpeg_start_compress(&cinfo, TRUE);
				while (cinfo.next_scanline < cinfo.image_height)
				{
					JSAMPROW row_pointer[1];
					row_pointer[0] = & ((JSAMPROW)ptr)[cinfo.next_scanline * width * depth];
					(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
				}
				jpeg_finish_compress(&cinfo);
				count -= cinfo.dest->free_in_buffer;
				ptr = jpeg_buffer.data();
				jpeg_destroy_compress(&cinfo);
				send(sockfd,&count,4,0);
			#else
				uint32_t s = 0;
				send(sockfd,&s,4,0);
			#endif
			isaac_int n = 0;
			isaac_int div = count / ISAAC_MAX_RECEIVE; //256kb per message
			isaac_int rest = count % ISAAC_MAX_RECEIVE; //rest
			for (isaac_int i = 0; i <=  div; i++)
			{
				isaac_int r = -1;
				while (r < 0)
					r = send(sockfd,&(((char*)ptr)[i*ISAAC_MAX_RECEIVE]),i == div ? rest : ISAAC_MAX_RECEIVE,0);
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
		isaac_uint port;
		isaac_int sockfd;
		std::list<json_t*> messageList;
		pthread_mutex_t deleteMessageMutex;
		pthread_t readThread;
};

} //namespace isaac;