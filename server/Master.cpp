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
	masterHello = json_object();
	json_object_set_new( masterHello, "type", json_string( "hello" ) );
	json_object_set_new( masterHello, "name", json_string( name.c_str() ) );
}

errorCode Master::addDataConnector(MetaDataConnector *dataConnector)
{
	dataConnector->setMaster(this);
	MetaDataConnectorList d;
	d.connector = dataConnector;
	d.thread = 0;
	dataConnectorList.push_back(d);
	return 0;
}

MetaDataClient* Master::addDataClient()
{
	MetaDataClient* client = new MetaDataClient();
	dataClientList.push_back(client);
	client->masterSendMessage(new MessageContainer(MASTER_HELLO,masterHello));
	//Send all registered visualizations
	ThreadList<InsituConnectorList*>::ThreadListContainer_ptr mom = insituMaster.insituConnectorList.getFront();
	while (mom)
	{
		if (!mom->deleted)
			client->masterSendMessage(new MessageContainer(REGISTER_PLUGIN,mom->t->initData));
		mom = mom->next;
	}
	return client;
}

errorCode Master::run()
{
	printf("Running ISAAC Master\n");
	signal(SIGINT, sighandler);
	printf("Starting insitu plugin listener\n");
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
	while (force_exit == 0)
	{
		/////////////////////////////////////
		// Iterate over all insitu clients //
		/////////////////////////////////////
		ThreadList<InsituConnectorList*>::ThreadListContainer_ptr insitu = insituMaster.insituConnectorList.getFront();
		while (insitu)
		{
			if (insitu->deleted)
			{
				insitu = insitu->next;
				continue;
			}
			//Check for new messages for every insituConnector
			while (MessageContainer* message = insitu->t->connector->masterGetMessage())
			{
				if (message->type == REGISTER_PLUGIN) //Saving the metadata description for later
				{
					insitu->t->initData = message->json_root;
					//Adding a field with the number
					json_object_set_new( insitu->t->initData, "id", json_integer( insitu->t->connector->getID() ) );
					printf("New connection, giving id %i\n",insitu->t->connector->getID());
					ThreadList<MetaDataClient*>::ThreadListContainer_ptr dc = dataClientList.getFront();
					while (dc)
					{
						dc->t->masterSendMessage(new MessageContainer(REGISTER_PLUGIN,insitu->t->initData));
						dc = dc->next;
					}
				}
				else
				if (message->type == EXIT_PLUGIN) //Let's tell everybody and remove it from the list
				{
					int id = json_integer_value( json_object_get(message->json_root, "id") );
					printf("Connection %i closed.\n",id);
					ThreadList<MetaDataClient*>::ThreadListContainer_ptr dc = dataClientList.getFront();
					while (dc)
					{
						dc->t->masterSendMessage(new MessageContainer(EXIT_PLUGIN,message->json_root));
						dc = dc->next;
					}
					ThreadList<InsituConnectorList*>::ThreadListContainer_ptr insitu = insituMaster.insituConnectorList.getFront();
					while (insitu)
					{
						if (insitu->t->connector->getID() == id)
							insitu->deleted = true;
						insitu = insitu->next;
					}	
				}
				free(message);
			}
			insitu = insitu->next;
		}
		///////////////////////////////////////
		// Iterate over all metadata clients //
		///////////////////////////////////////
		ThreadList<MetaDataClient*>::ThreadListContainer_ptr client = dataClientList.getFront();
		while (client)
		{
			if (client->deleted)
			{
				client = client->next;
				continue;
			}
			//Check for new messages for every client
			while (MessageContainer* message = client->t->masterGetMessage())
			{
				if (message->type == CLOSED)
					client->deleted = true;
				free(message);
			}
			client = client->next;
		}
		usleep(1);
	}
	for (std::list<MetaDataConnectorList>::iterator it = dataConnectorList.begin(); it != dataConnectorList.end(); it++)
	{
		printf("Asking %s to exit\n",(*it).connector->getName().c_str());
		(*it).connector->masterSendMessage(new MessageContainer(FORCE_EXIT));
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
	json_decref( masterHello );
	dataConnectorList.clear();
	MetaDataClient* mom;
	while (mom = dataClientList.pop_front())
		delete mom;	
}
