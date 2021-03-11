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

#include "InsituConnectorMaster.hpp"

#include "NetworkInterfaces.hpp"
#include "version.hpp"

#include <errno.h>
#include <jansson.h>
#include <netinet/in.h>
#include <pthread.h>
#include <string.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <vector>

InsituConnectorMaster::InsituConnectorMaster()
{
    sockfd = 0;
    nextFreeNumber = 0;
    force_exit = false;
}

errorCode InsituConnectorMaster::init(int port, std::string interface)
{
    sockfd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if(sockfd < 0)
        return -1;
    int enable = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    NetworkInterfaces::bindInterface(serv_addr.sin_addr.s_addr, interface);
    serv_addr.sin_port = htons(port);
    if(bind(sockfd, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) < 0)
    {
        printf("Bind failed with error %i\n", errno);
        return -2;
    }
    return 0;
}

int InsituConnectorMaster::getSockFD()
{
    return sockfd;
}

size_t json_load_callback_function(void* buffer, size_t buflen, void* data)
{
    json_load_callback_struct* jlcb = (json_load_callback_struct*) data;
    if(jlcb->pos < jlcb->count)
    {
        ((char*) buffer)[0] = jlcb->buffer[jlcb->pos];
        jlcb->pos++;
        return 1;
    }
    return 0;
}

errorCode InsituConnectorMaster::run()
{
    listen(sockfd, 5);
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);

    struct pollfd fd_array[MAX_SOCKETS];
    memset(fd_array, 0, sizeof(fd_array));
    std::vector<InsituConnectorContainer*> con_array = std::vector<InsituConnectorContainer*>(MAX_SOCKETS, NULL);

    fd_array[0].fd = sockfd;
    fd_array[0].events = POLLIN;
    int fdnum = 1;

    while(!force_exit)
    {
        int rv = poll(fd_array, fdnum, 1000); // 1s timeout
        if(rv < 0)
        {
            fprintf(stderr, "Error while calling poll\n");
            return -1;
        }
        if(rv)
        {
            // First some extra sausage for the listening sockfd
            if(fd_array[0].revents == POLLIN)
            {
                int newsockfd = 1;
                while(newsockfd >= 0)
                {
                    newsockfd = accept(sockfd, (struct sockaddr*) &cli_addr, &clilen);
                    if(newsockfd >= 0)
                    {
                        InsituConnector* insituConnector = new InsituConnector(newsockfd, nextFreeNumber++);
                        insituConnector->jlcb.count = 0;
                        InsituConnectorContainer* d = new InsituConnectorContainer();
                        d->connector = insituConnector;
                        con_array[fdnum] = d;
                        fd_array[fdnum].fd = newsockfd;
                        fd_array[fdnum].events = POLLIN;
                        insituConnectorList.push_back(d);
                        fdnum++;
                    }
                }
            }
            for(int i = 1; i < fdnum; i++)
            {
                if(fd_array[i].revents & POLLIN)
                {
                    while(1)
                    {
                        int add = recv(
                            fd_array[i].fd,
                            &(con_array[i]->connector->jlcb.buffer[con_array[i]->connector->jlcb.count]),
                            4096,
                            MSG_DONTWAIT);
                        if(add > 0)
                        {
                            con_array[i]->connector->jlcb.count += add;
                            if(con_array[i]->connector->jlcb.count > ISAAC_MAX_RECEIVE)
                            {
                                fprintf(
                                    stderr,
                                    "Fatal error: Socket received %d bytes but buffer is only %d bytes! To increase "
                                    "the allowed size set ISAAC_MAX_RECEIVE to a higher value.\n",
                                    con_array[i]->connector->jlcb.count,
                                    ISAAC_MAX_RECEIVE);
                                return -1;
                            }
                        }
                        else
                            break;
                    }
                    con_array[i]->connector->jlcb.pos = 0;
                    con_array[i]->connector->jlcb.buffer[con_array[i]->connector->jlcb.count] = 0;
                    bool closed = false;
                    if(con_array[i]->connector->jlcb.count > 0)
                    {
                        con_array[i]->connector->jlcb.buffer[con_array[i]->connector->jlcb.count] = 0;
                        json_error_t error;
                        int last_working_pos = 0;
                        while(json_t* content = json_load_callback(
                                  json_load_callback_function,
                                  &con_array[i]->connector->jlcb,
                                  JSON_DISABLE_EOF_CHECK,
                                  &error))
                        {
                            last_working_pos = con_array[i]->connector->jlcb.pos;
                            MessageContainer* message = new MessageContainer(NONE, content);
                            MessageType type = message->type;
                            if(type == REGISTER)
                            {
                                json_object_set_new(
                                    message->json_root,
                                    "id",
                                    json_integer(con_array[i]->connector->getID()));
                                json_t* protocol_version = json_object_get(message->json_root, "protocol");
                                long version[2]
                                    = {json_integer_value(json_array_get(protocol_version, 0)),
                                       json_integer_value(json_array_get(protocol_version, 1))};
                                if(version[0] != ISAAC_PROTOCOL_VERSION_MAJOR)
                                {
                                    printf(
                                        "Fatal error: Protocol version mismatch: Library has %ld.%ld, server needs "
                                        "%i.%i!\n",
                                        version[0],
                                        version[1],
                                        ISAAC_PROTOCOL_VERSION_MAJOR,
                                        ISAAC_PROTOCOL_VERSION_MINOR);
                                    const char buffer[] = "{ \"fatal error\": \"protocol mismatch\" }";
                                    send(fd_array[i].fd, buffer, strlen(buffer), MSG_NOSIGNAL);
                                    closed = true;
                                }
                                else if(version[1] != ISAAC_PROTOCOL_VERSION_MINOR)
                                    printf(
                                        "Warning: Protocol minor version mismatch: Library has %ld.%ld, server can "
                                        "%i.%i!\n",
                                        version[0],
                                        version[1],
                                        ISAAC_PROTOCOL_VERSION_MAJOR,
                                        ISAAC_PROTOCOL_VERSION_MINOR);
                            }
                            if(!closed)
                            {
                                long long uid = json_integer_value(json_object_get(content, "uid"));
                                con_array[i]->connector->clientSendMessage(message);
                                if(type == EXIT_PLUGIN)
                                {
                                    closed = true;
                                    break;
                                }
                                else
                                {
                                    // send, which uid we just got
                                    char buffer[32];
                                    sprintf(buffer, "{\"done\": %lld}", uid);
                                    int l = strlen(buffer);
                                    if(send(fd_array[i].fd, buffer, l, MSG_NOSIGNAL) < l)
                                    {
                                        MessageContainer* message = new MessageContainer(EXIT_PLUGIN, json_object());
                                        json_object_set_new(message->json_root, "type", json_string("exit"));
                                        con_array[i]->connector->clientSendMessage(message);
                                        closed = true;
                                        break;
                                    }
                                }
                            }
                        }
                        // If the whole json message was not received yet, we need to keep the start
                        if(error.position != 1 || strcmp(error.text, "'[' or '{' expected near end of file") != 0)
                        {
                            for(int j = 0; j < con_array[i]->connector->jlcb.count - last_working_pos; j++)
                                con_array[i]->connector->jlcb.buffer[j]
                                    = con_array[i]->connector->jlcb.buffer[j + last_working_pos];
                            con_array[i]->connector->jlcb.count -= last_working_pos;
                        }
                        else
                            con_array[i]->connector->jlcb.count = 0;
                    }
                    else // Closed
                    {
                        MessageContainer* message = new MessageContainer(EXIT_PLUGIN, json_object());
                        json_object_set_new(message->json_root, "type", json_string("exit"));
                        con_array[i]->connector->clientSendMessage(message);
                        closed = true;
                    }
                    if(closed)
                    {
                        close(fd_array[i].fd);
                        fdnum--;
                        for(int j = i; j < fdnum; j++)
                        {
                            fd_array[j] = fd_array[j + 1];
                            con_array[j] = con_array[j + 1];
                        }
                        memset(&(fd_array[fdnum]), 0, sizeof(fd_array[fdnum]));
                    }
                }
            }
        }
    }
    return 0;
}

void InsituConnectorMaster::setExit()
{
    force_exit = true;
}

InsituConnectorMaster::~InsituConnectorMaster()
{
    InsituConnectorContainer* mom;
    while((mom = insituConnectorList.pop_front()))
    {
        shutdown(mom->connector->getSockFD(), SHUT_RDWR);
        printf("Waiting for Connections %i to finish... ", mom->connector->getID());
        fflush(stdout);
        delete mom->connector;
        delete mom;
        printf("Done\n");
    }
}
