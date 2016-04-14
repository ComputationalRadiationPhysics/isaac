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

#include "TCPDataConnector.hpp"
#include "MetaDataClient.hpp"

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <netinet/in.h>

TCPDataConnector::TCPDataConnector()
{
}

std::string TCPDataConnector::getName()
{
	return "TCPDataConnector";
}

errorCode TCPDataConnector::init(int port)
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
		printf("TCPDataConnector: Bind failed with error %i\n",errno);
		return -2;
	}
}

struct jlcb_container
{
	json_load_callback_struct jlcb;
};

errorCode TCPDataConnector::run()
{
	listen(sockfd,5);
	struct sockaddr_in cli_addr;
	socklen_t clilen = sizeof(cli_addr);
	
	struct pollfd fd_array[MAX_SOCKETS];
	memset(fd_array,0,sizeof(fd_array));
	std::vector< MetaDataClient* > client_array = std::vector< MetaDataClient* >(MAX_SOCKETS,NULL);
	std::vector< jlcb_container* > jlcb_array = std::vector< jlcb_container* >(MAX_SOCKETS,NULL);
	
	fd_array[0].fd = sockfd;
	fd_array[0].events = POLLIN;
	int fdnum = 1;
	
	volatile bool force_exit = false;
	
	while (!force_exit)
	{
		//Own messages
		while (MessageContainer* message = clientGetMessage())
		{
			if (message->type == FORCE_EXIT)
				force_exit = true;
			delete message;
		}
		int rv = poll(fd_array, fdnum, 10); //10ms timeout
		if (rv < 0)
		{
			fprintf(stderr,"TCPDataConnector: Error while calling poll\n");
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
						printf("TCPDataConnector: New connection, giving id %i\n",fdnum);
						client_array[fdnum] = broker->addDataClient();
						jlcb_array[fdnum] = new jlcb_container();
						fd_array[fdnum].fd = newsockfd;
						fd_array[fdnum].events = POLLIN;
						fdnum++;
					}
				}
			}
		}
		for (int i = 1; i < fdnum; i++)
		{
			bool closed = false;
			//Messages of children
			while (MessageContainer* message = client_array[i]->clientGetMessage())
			{
				char* buffer = json_dumps( message->json_root, 0 );
				int l = strlen(buffer);
				int n = 0;
				int amount = (l+4095)/4096;
				for (int j = 0; j < amount; j++)
				{
					if (j == amount - 1)
						n += send(fd_array[i].fd,&buffer[j*4096],l - j * 4096,0);
					else
						n += send(fd_array[i].fd,&buffer[j*4096],4096,MSG_MORE);
				}
				if (n < l)
				{
					printf("TCPDataConnector: ERROR %d writing to socket %i\n", n, fd_array[i].fd);
					client_array[i]->clientSendMessage(new MessageContainer(CLOSED));
					closed = true;
				}
				delete message;
				free(buffer);
			}
			if (fd_array[i].revents & POLLIN)
			{
				while (1)
				{
					int add = recv(fd_array[i].fd,&(jlcb_array[i]->jlcb.buffer[jlcb_array[i]->jlcb.count]),4096,MSG_DONTWAIT);
					if (add > 0)
						jlcb_array[i]->jlcb.count += add;
					else
						break;
				}
				jlcb_array[i]->jlcb.pos = 0;
				jlcb_array[i]->jlcb.buffer[jlcb_array[i]->jlcb.count] = 0;
				if (jlcb_array[i]->jlcb.count > 0)
				{
					jlcb_array[i]->jlcb.buffer[jlcb_array[i]->jlcb.count] = 0;
					json_error_t error;
					int last_working_pos = 0;
					while (json_t * content = json_load_callback(json_load_callback_function,&jlcb_array[i]->jlcb,JSON_DISABLE_EOF_CHECK,&error))
					{
						last_working_pos = jlcb_array[i]->jlcb.pos;
						MessageContainer* message = new MessageContainer(NONE,content);
						MessageType type = message->type;
						client_array[i]->clientSendMessage(message);
					}
					//If the whole json message was not received yet, we need to keep the start
					if ( error.position != 1 || strcmp(error.text,"'[' or '{' expected near end of file") != 0 )
					{
						for (int j = 0; j < jlcb_array[i]->jlcb.count - last_working_pos; j++)
							jlcb_array[i]->jlcb.buffer[j] = jlcb_array[i]->jlcb.buffer[j + last_working_pos];
						jlcb_array[i]->jlcb.count -= last_working_pos;
					}
					else
						jlcb_array[i]->jlcb.count = 0;
				}
				else //Closed
				{
					client_array[i]->clientSendMessage(new MessageContainer(CLOSED));
					closed = true;
				}
			}
			if (closed)
			{
				close(fd_array[i].fd);
				printf("TCPDataConnector: Closed connection %i\n",i);
				delete(jlcb_array[i]);
				fdnum--;
				for (int j = i; j < fdnum; j++)
				{
					fd_array[j] = fd_array[j+1];
					client_array[j] = client_array[j+1];
				}
				memset(&(fd_array[fdnum]),0,sizeof(fd_array[fdnum]));
			}
		}
	}
	return 0;
}
