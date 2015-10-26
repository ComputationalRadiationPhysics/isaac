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
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <pthread.h>

InsituConnectorMaster::InsituConnectorMaster()
{
	sockfd = 0;
	nextFreeNumber = 0;
}

errorCode InsituConnectorMaster::init(int port)
{
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
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

errorCode InsituConnectorMaster::run()
{
	listen(sockfd,5);
	struct sockaddr_in cli_addr;
	socklen_t clilen = sizeof(cli_addr);
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
			pthread_create(&(d->thread),NULL,Runable::run_runable,insituConnector);
			insituConnectorList.push_back(d);
		}
	}
}

InsituConnectorMaster::~InsituConnectorMaster()
{
	InsituConnectorContainer* mom;
	while (mom = insituConnectorList.pop_front())
	{
		shutdown(mom->connector->getSockFD(),SHUT_RDWR);
		printf("Waiting for InsituConnectorThread %i to finish... ",mom->connector->getID());
		fflush(stdout);
		pthread_join(mom->thread,NULL);
		delete mom;
		printf("Done\n");
	}
}

