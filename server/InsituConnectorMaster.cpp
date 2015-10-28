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

#include "InsituConnectorMaster.hpp"
#include <jansson.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <netinet/in.h>
#include <string.h>
#include <pthread.h>
#include <vector>

InsituConnectorMaster::InsituConnectorMaster()
{
	sockfd = 0;
	nextFreeNumber = 0;
	force_exit = false;
}

errorCode InsituConnectorMaster::init(int port)
{
	sockfd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
	if (sockfd < 0)
		return -1;
	int enable = 1;
	setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
	struct sockaddr_in serv_addr;
	memset(&serv_addr,0, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = INADDR_ANY;
	serv_addr.sin_port = htons(port);
	if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
	{
		printf("Bind failed with error %i\n",errno);
		return -2;
	}
}

int InsituConnectorMaster::getSockFD()
{
	return sockfd;
}

typedef struct
{
	char buffer[MAX_RECEIVE];
	int pos;
	int count;
} json_load_callback_struct;

size_t json_load_callback_function (void *buffer, size_t buflen, void *data)
{
	json_load_callback_struct* jlcb = (json_load_callback_struct*)data;
	if (jlcb->pos < jlcb->count)
	{
		((char*)buffer)[0] = jlcb->buffer[jlcb->pos];
		jlcb->pos++;
		return 1;
	}
	return 0;
	//return recv(jlcb->sockfd,buffer,1,MSG_DONTWAIT);
}

errorCode InsituConnectorMaster::run()
{
	json_load_callback_struct jlcb;
	
	listen(sockfd,5);
	struct sockaddr_in cli_addr;
	socklen_t clilen = sizeof(cli_addr);
	
	struct pollfd fd_array[MAX_SOCKETS];
	memset(fd_array,0,sizeof(fd_array));
	std::vector< InsituConnectorContainer* > con_array = std::vector< InsituConnectorContainer* >(MAX_SOCKETS,NULL);
	
	fd_array[0].fd = sockfd;
	fd_array[0].events = POLLIN;
	int fdnum = 1;
	
	while (!force_exit)
	{
		int rv = poll(fd_array, fdnum, 1000); //1s timeout
		if (rv < 0)
		{
			fprintf(stderr,"Error while calling poll\n");
			return -1;
		}
		if (rv)
		{
			//First some extra sausage for the listening sockfd
			if (fd_array[0].revents == POLLIN)
			{
				int newsockfd = 1;
				while (newsockfd >= 0)
				{
					newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
					if (newsockfd >= 0)
					{
						printf("New connection from Plugin established\n");
						InsituConnector* insituConnector = new InsituConnector(newsockfd,nextFreeNumber++);
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
			for (int i = 1; i < fdnum; i++)
			{
				if (fd_array[i].revents == POLLIN)
				{
					jlcb.pos = 0;
					jlcb.count = recv(fd_array[i].fd,jlcb.buffer,MAX_RECEIVE,MSG_DONTWAIT);
					bool closed = false;
					if (jlcb.count > 0)
					{
						jlcb.buffer[jlcb.count] = 0;
						while (json_t * content = json_load_callback(json_load_callback_function,&jlcb,JSON_DISABLE_EOF_CHECK,NULL))
						{
							MessageContainer* message = new MessageContainer(NONE,content);
							MessageType type = message->type;
							if (type == REGISTER_MASTER)
								json_object_set_new( message->json_root, "id", json_integer( con_array[i]->connector->getID() ) );
							con_array[i]->connector->clientSendMessage(message);
							if (type == EXIT_PLUGIN)
							{
								closed = true;
								break;
							}
						}
					}
					else //Closed
					{
						MessageContainer* message = new MessageContainer(EXIT_PLUGIN,json_object());
						json_object_set_new( message->json_root, "type", json_string( "exit" ) );
						con_array[i]->connector->clientSendMessage(message);
						closed = true;
					}
					if (closed)
					{
						fdnum--;
						for (int j = i; j < fdnum; j++)
						{
							fd_array[j] = fd_array[j+1];
							con_array[j] = con_array[j+1];
						}
					}
				}
			}
		}
	}
}

void InsituConnectorMaster::setExit()
{
	force_exit = true;
}

InsituConnectorMaster::~InsituConnectorMaster()
{
	InsituConnectorContainer* mom;
	while (mom = insituConnectorList.pop_front())
	{
		shutdown(mom->connector->getSockFD(),SHUT_RDWR);
		printf("Waiting for InsituConnectorThread %i to finish... ",mom->connector->getID());
		fflush(stdout);
		delete mom;
		printf("Done\n");
	}
}

