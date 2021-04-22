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
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include "isaac_macros.hpp"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <jansson.h>
#include <list>

#if ISAAC_JPEG == 1
#    include <jpeglib.h>
#endif

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/ostream_iterator.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <sstream>

namespace isaac
{
    enum CommunicatorSetting
    {
        ReturnAtError = 0,
        RetryEverySend = 1
    };

    class IsaacCommunicator
    {
    public:
        IsaacCommunicator(const std::string url, const isaac_uint port)
            : id(0)
            , server_id(0)
            , url(url)
            , port(port)
            , sockfd(0)
            , jpeg_quality(90)
            , registerMessage(NULL)
        {
            pthread_mutex_init(&deleteMessageMutex, NULL);
        }
        json_t* getLastMessage()
        {
            json_t* result = NULL;
            pthread_mutex_lock(&deleteMessageMutex);
            if(!messageList.empty())
            {
                result = messageList.front();
                messageList.pop_front();
            }
            pthread_mutex_unlock(&deleteMessageMutex);
            return result;
        }
        isaac_int serverConnect(CommunicatorSetting setting)
        {
            struct hostent* server;
            server = gethostbyname(url.c_str());
            if(!server)
            {
                if(setting == ReturnAtError)
                {
                    fprintf(stderr, "Could not resolve %s.\n", url.c_str());
                    return -1;
                }
                else
                {
                    sockfd = 0;
                    return 1;
                }
            }
            sockfd = socket(AF_INET, SOCK_STREAM, 0);
            if(sockfd < 0)
            {
                if(setting == ReturnAtError)
                {
                    fprintf(stderr, "Could not create socket.\n");
                    return -2;
                }
                else
                {
                    sockfd = 0;
                    return 1;
                }
            }
            struct sockaddr_in serv_addr;
            memset(&serv_addr, 0, sizeof(serv_addr));
            serv_addr.sin_family = AF_INET;
            bcopy((char*) server->h_addr, (char*) &serv_addr.sin_addr.s_addr, server->h_length);
            serv_addr.sin_port = htons(port);
            if(connect(sockfd, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) < 0)
            {
                close(sockfd);
                if(setting == ReturnAtError)
                {
                    fprintf(stderr, "Could not connect to %s.\n", url.c_str());
                    return -3;
                }
                else
                {
                    sockfd = 0;
                    return 1;
                }
            }
            pthread_create(&readThread, NULL, run_readAndSetMessages, this);
            return 0;
        }

        isaac_int serverSend(char* content, bool starting = true, bool finishing = false)
        {
            if(sockfd < 0)
                return 0;
            if(sockfd == 0) // Connection lost or never established
            {
                if(serverConnect(RetryEverySend))
                    return 0;
                char* content = json_dumps(*registerMessage, 0);
                isaac_int result = serverSend(content, true, true);
                free(content);
            }
            int n = 0;
            if(starting)
            {
                int c = 0;
                while(id > server_id + ISAAC_MAX_DIFFERENCE)
                {
                    usleep(1000);
                    c++;
                    if(c > 1000) // 1s!
                    {
                        id = server_id - 1;
                        break;
                    }
                }
                char id_string[32];
                sprintf(id_string, "{\"uid\": %i", id);
                int add = send(sockfd, id_string, strlen(id_string), MSG_MORE | MSG_NOSIGNAL);
                ISAAC_HANDLE_EPIPE(add, n, sockfd, readThread)
                n += add;
                id++;
            }
            if(content)
            {
                content[0] = ',';
                uint32_t l = strlen(content) - 1; // without closing }
                // content[l] = 0;
                int amount = (l + 4095) / 4096;
                for(int i = 0; i < amount; i++)
                {
                    if(i == amount - 1)
                    {
                        int add = send(sockfd, &content[i * 4096], l - i * 4096, MSG_MORE | MSG_NOSIGNAL);
                        ISAAC_HANDLE_EPIPE(add, n, sockfd, readThread)
                        n += add;
                    }
                    else
                    {
                        int add = send(sockfd, &content[i * 4096], 4096, MSG_MORE | MSG_NOSIGNAL);
                        ISAAC_HANDLE_EPIPE(add, n, sockfd, readThread)
                        n += add;
                    }
                }
            }
            if(finishing)
            {
                char finisher[] = "} ";
                int add = send(sockfd, finisher, 2, MSG_NOSIGNAL);
                ISAAC_HANDLE_EPIPE(add, n, sockfd, readThread)
                n += add;
            }
            return n;
        }

        isaac_int serverSendRegister(json_t** registerMessage)
        {
            this->registerMessage = registerMessage;
            char* content = json_dumps(*registerMessage, 0);
            isaac_int result = serverSend(content, true, true);
            free(content);
            return result;
        }

#if ISAAC_JPEG == 1
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
#endif
        void setJpegQuality(isaac_uint jpeg_quality)
        {
            if(jpeg_quality > 100)
                jpeg_quality = 100;
            this->jpeg_quality = jpeg_quality;
        }
        void serverSendFrame(void* ptr, const isaac_uint width, const isaac_uint height, const isaac_uint depth)
        {
            // First the size
            uint32_t count = width * height * depth;
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
            jpeg_buffer.resize(count);
            cinfo.dest->next_output_byte = (JOCTET*) (jpeg_buffer.data());
            cinfo.dest->free_in_buffer = count;
            cinfo.image_width = width;
            cinfo.image_height = height;
            cinfo.input_components = depth;
            cinfo.in_color_space = JCS_EXT_RGBX;
            jpeg_set_defaults(&cinfo);
            jpeg_set_quality(&cinfo, jpeg_quality, false);
            jpeg_start_compress(&cinfo, TRUE);
            while(cinfo.next_scanline < cinfo.image_height)
            {
                JSAMPROW row_pointer[1];
                row_pointer[0] = &((JSAMPROW) ptr)[cinfo.next_scanline * width * depth];
                (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
            }
            jpeg_finish_compress(&cinfo);
            count -= cinfo.dest->free_in_buffer;
            ptr = jpeg_buffer.data();
            jpeg_destroy_compress(&cinfo);
#endif

            using namespace boost::archive::iterators;
            std::stringstream payload;
            typedef base64_from_binary<transform_width<
                const unsigned char*,
                6,
                8>>
                base64_text; // compose all the above operations in to a new iterator

            std::copy(
                base64_text((char*) ptr),
                base64_text((char*) ptr + count),
                boost::archive::iterators::ostream_iterator<char>(payload));

#if ISAAC_JPEG == 1
            char header[] = "{\"payload\": \"data:image/jpeg;base64,";
#else
            char header[] = "{\"payload\": \"data:image/raw-rgba;base64,";
#endif
            char footer[] = "\"}";
            int hl = strlen(header);
            int pl = payload.str().length();
            int fl = strlen(footer);
#if ISAAC_VALGRIND_TWEAKS == 1
            // Allocating one letter more for \0 and 4 letter more because of
            // strlen (of glib) always reading 4 aligned bytes - even after \0.
            // It should never crash because of the missing 4 bytes - but
            // valgrind does complain nevertheless.
            char* message = (char*) malloc(hl + pl + fl + 1 + 4);
#else
            char* message = (char*) malloc(hl + pl + fl + 1);
#endif
            memcpy(message, header, hl);
            memcpy(&(message[hl]), payload.str().c_str(), pl);
            memcpy(&(message[hl + pl]), footer, fl + 1); // with 0
            serverSend(message, false, true);
            free(message);
        }
        void serverDisconnect()
        {
            close(sockfd);
        }
        ~IsaacCommunicator()
        {
            if(sockfd)
                serverDisconnect();
            usleep(100000); // 100ms
            pthread_cancel(readThread);
            pthread_mutex_destroy(&deleteMessageMutex);
        }
        void setMessage(json_t* content)
        {
            pthread_mutex_lock(&deleteMessageMutex);
            messageList.push_back(content);
            pthread_mutex_unlock(&deleteMessageMutex);
        }

    private:
        void readAndSetMessages()
        {
            while(json_t* content
                  = json_load_callback(json_load_callback_function, &sockfd, JSON_DISABLE_EOF_CHECK, NULL))
            {
                // Search for ready messages:
                json_t* js;
                if(js = json_object_get(content, "fatal error"))
                {
                    const char* fatal_error = json_string_value(js);
                    fprintf(stderr, "Fatal error: \"%s\".\n", fatal_error);
                    if(strcmp(fatal_error, "protocol mismatch") == 0)
                    {
                        close(sockfd);
                        sockfd = -1;
                    }
                    json_decref(content);
                }
                else if(js = json_object_get(content, "done"))
                {
                    isaac_uint new_server_id = json_integer_value(js);
                    if(new_server_id > server_id)
                        server_id = new_server_id;
                    json_decref(content);
                }
                else
                {
                    pthread_mutex_lock(&deleteMessageMutex);
                    messageList.push_back(content);
                    pthread_mutex_unlock(&deleteMessageMutex);
                }
            }
        }
        static size_t json_load_callback_function(void* buffer, size_t buflen, void* data)
        {
            return recv(*((isaac_int*) data), buffer, 1, 0);
        }
        static void* run_readAndSetMessages(void* communicator)
        {
            ((IsaacCommunicator*) communicator)->readAndSetMessages();
            return 0;
        }
        isaac_uint id;
        isaac_uint server_id;
        std::string url;
        isaac_uint port;
        isaac_int sockfd;
        isaac_uint jpeg_quality;
        std::list<json_t*> messageList;
        pthread_mutex_t deleteMessageMutex;
        pthread_t readThread;
        json_t** registerMessage;
    };

} // namespace isaac
