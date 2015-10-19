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

#include "Master.hpp"
#include <stdio.h>
#include "InsituConnectorMaster.hpp"
#include <pthread.h>
#include "ThreadList.hpp"

volatile sig_atomic_t Master::force_exit = 0;

void sighandler(int sig)
{
	printf("\n");
	Master::force_exit = 1;
}

Master::Master(std::string name,int inner_port)
{
	this->name = name;
	this->inner_port = inner_port;
	this->insituThread = 0;
}

errorCode Master::addDataConnector(MetaDataConnector *dataConnector)
{
	MetaDataConnectorList d;
	d.connector = dataConnector;
	d.thread = 0;
	dataConnectorList.push_back(d);
	dataConnector->setMaster(this);
	return 0;
}

errorCode Master::run()
{
	printf("Running ISAAC Master\n");
	signal(SIGINT, sighandler);
	printf("Starting insitu plugin listener\n");
	InsituConnectorMaster insituMaster = InsituConnectorMaster();
	if (insituMaster.init(inner_port))
	{
		fprintf(stderr,"Error while starting insitu plugin listener\n");
		signal(SIGINT, SIG_DFL);
		return -1;
	}
	pthread_create(&insituThread,NULL,Runable::run_runable,&insituMaster);
	
	for (std::list<MetaDataConnectorList>::iterator it = dataConnectorList.begin(); it != dataConnectorList.end(); it++)
	{
		printf("Launching %s\n",(*it).connector->getName().c_str());
		pthread_create(&((*it).thread),NULL,Runable::run_runable,(*it).connector);
	}		
	int c = 0;
	while (force_exit == 0)
	{
		ThreadList<InsituConnectorList>::ThreadListContainer_ptr mom = insituMaster.insituConnectorList.getFront();
		while (mom)
		{
			//Check for new messages for every insituConnector
			MessageContainer* message = mom->t.connector->getMessage();
			if (message)
			{
				//c++;
				//printf("Received message %i %i (%.3f)\n",message->type,c,json_real_value(json_object_get(json_object_get(message->json_root, "metadata"), "speed")));
				if (message->type == REGISTER_PLUGIN) //Saving the metadata description for later
				{
					mom->t.initData = message->json_root;
					for (std::list<MetaDataConnectorList>::iterator it = dataConnectorList.begin(); it != dataConnectorList.end(); it++)
					{
						printf("Telling %s about new plugin\n",(*it).connector->getName().c_str());
						(*it).connector->addMessage(new MessageContainer(REGISTER_PLUGIN,mom->t.initData));
					}
				}
				free(message);
			}
			mom = mom->next;
		}
		usleep(1);
	}
	for (std::list<MetaDataConnectorList>::iterator it = dataConnectorList.begin(); it != dataConnectorList.end(); it++)
	{
		printf("Asking %s to exit\n",(*it).connector->getName().c_str());
		(*it).connector->addMessage(new MessageContainer(FORCE_EXIT));
	}
	for (std::list<MetaDataConnectorList>::iterator it = dataConnectorList.begin(); it != dataConnectorList.end(); it++)
	{
		pthread_join((*it).thread,NULL);
		printf("%s finished\n",(*it).connector->getName().c_str());
	}
	signal(SIGINT, SIG_DFL);
	return 0;
}

Master::~Master()
{
	dataConnectorList.clear();
}
