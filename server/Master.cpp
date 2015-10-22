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

#include "Master.hpp"
#include <stdio.h>
#include <pthread.h>
#include "ThreadList.hpp"
#include <sys/socket.h>

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
	MetaDataConnectorContainer d;
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
	ThreadList< InsituConnectorGroup* >::ThreadListContainer_ptr mom = insituConnectorGroupList.getFront();
	while (mom)
	{
		if (mom->t->nodes == mom->t->elements.length())
			client->masterSendMessage(new MessageContainer(TELL_PLUGIN,mom->t->initData));
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
	
	for (std::list<MetaDataConnectorContainer>::iterator it = dataConnectorList.begin(); it != dataConnectorList.end(); it++)
	{
		printf("Launching %s\n",(*it).connector->getName().c_str());
		pthread_create(&((*it).thread),NULL,Runable::run_runable,(*it).connector);
	}
	while (force_exit == 0)
	{
		/////////////////////////////////////
		// Iterate over all insitu clients //
		/////////////////////////////////////
		ThreadList< InsituConnectorContainer* >::ThreadListContainer_ptr insitu = insituMaster.insituConnectorList.getFront();
		while (insitu)
		{
			ThreadList< InsituConnectorContainer* >::ThreadListContainer_ptr next = insitu->next;
			//Check for new messages for every insituConnector
			while (MessageContainer* message = insitu->t->connector->masterGetMessage())
			{
				if (message->type == PERIOD_MASTER)
				{
					//Iterate over all metaDataClient and send to them, which observe
					ThreadList<MetaDataClient*>::ThreadListContainer_ptr dc = dataClientList.getFront();
					while (dc)
					{
						if (dc->t->doesObserve(insitu->t->connector->getID()))
							dc->t->masterSendMessage(new MessageContainer(PERIOD_MASTER,message->json_root));
						dc = dc->next;
					}
				}
				else
				if (message->type == PERIOD_MERGE)
				{
					if (insitu->t->group->merge_count == 0)
					{
						//Reset merge data
						json_incref( message->json_root );
						insitu->t->group->mergeData = message->json_root;
						insitu->t->group->merge_count++;
						insitu->t->group->meta_merge_count++;
						insitu->t->meta_merge_count++;
					}
					else
					if (insitu->t->meta_merge_count < insitu->t->group->meta_merge_count)
					{
						insitu->t->group->mergeJSON( insitu->t->group->mergeData, message->json_root, NULL );
						insitu->t->group->merge_count++;
						insitu->t->meta_merge_count++;
					}
					//Iterate over all metaDataClient and send to them, which observe
					if (insitu->t->group->merge_count == insitu->t->group->nodes)
					{
						ThreadList<MetaDataClient*>::ThreadListContainer_ptr dc = dataClientList.getFront();
						while (dc)
						{
							if (dc->t->doesObserve(insitu->t->group->getID()))
								dc->t->masterSendMessage(new MessageContainer(PERIOD_MERGE,insitu->t->group->mergeData));
							dc = dc->next;
						}
						json_decref( insitu->t->group->mergeData );
						insitu->t->group->merge_count = 0;
					}
				}
				else
				if (message->type == REGISTER_MASTER || message->type == REGISTER_SLAVE) //Saving the metadata description for later
				{
					//Get group
					std::string name( json_string_value( json_object_get(message->json_root, "name") ) );
					InsituConnectorGroup* group = NULL;
					ThreadList< InsituConnectorGroup* >::ThreadListContainer_ptr it = insituConnectorGroupList.getFront();
					while (it)
					{
						if (it->t->name == name)
						{
							group = it->t;
							break;
						}
						it = it->next;
					}
					if (group == NULL)
					{
						group = new InsituConnectorGroup(name);
						insituConnectorGroupList.push_back( group );
					}
					insitu->t->group = group;
					group->mergeJSON( group->initData, message->json_root, insitu->t );
					group->elements.push_back( insitu->t );

					//Adding a field with the number
					printf("New connection, giving id %i\n",insitu->t->connector->getID());
					
					if (group->nodes == group->elements.length())
					{
						printf("Group complete, sending to connected interfaces\n");
						ThreadList<MetaDataClient*>::ThreadListContainer_ptr dc = dataClientList.getFront();
						while (dc)
						{
							dc->t->masterSendMessage(new MessageContainer(TELL_PLUGIN,group->initData));
							dc = dc->next;
						}
					}
				}
				else
				if (message->type == EXIT_PLUGIN) //Let's tell everybody and remove it from the list
				{
					InsituConnectorContainer* insituContainer = insitu->t;
					int id = insituContainer->connector->getID();
					printf("Connection %i closed.\n",id);
					//Group does still exist?
					if (insituContainer->group)
					{
						InsituConnectorGroup* group = insituContainer->group;
						//Add id of group
						json_object_set_new( message->json_root, "id", json_integer( group->getID() ) );
						ThreadList<MetaDataClient*>::ThreadListContainer_ptr dc = dataClientList.getFront();
						while (dc)
						{
							dc->t->masterSendMessage(new MessageContainer(EXIT_PLUGIN,message->json_root));
							dc = dc->next;
						}
						//Now let's remove the whole group
						ThreadList< InsituConnectorContainer* >::ThreadListContainer_ptr element = group->elements.getFront();
						while (element)
						{
							element->t->group = NULL;
							element = element->next;
						}
						ThreadList< InsituConnectorGroup* >::ThreadListContainer_ptr it = insituConnectorGroupList.getFront();
						printf("Removed group %i\n",group->id);
						while (it)
						{
							if (it->t == group)
							{
								delete insituConnectorGroupList.remove(it);
								break;
							}
							it = it->next;
						}
					}
					delete insituMaster.insituConnectorList.remove(insitu);
					break;
				}
				free(message);
			}
			insitu = next;
		}
		///////////////////////////////////////
		// Iterate over all metadata clients //
		///////////////////////////////////////
		ThreadList<MetaDataClient*>::ThreadListContainer_ptr dc = dataClientList.getFront();
		while (dc)
		{
			ThreadList<MetaDataClient*>::ThreadListContainer_ptr next = dc->next;
			//Check for new messages for every client
			while (MessageContainer* message = dc->t->masterGetMessage())
			{
				if (message->type == FEEDBACK_MASTER || message->type == FEEDBACK_ALL)
				{
					json_t* observe_id = json_object_get(message->json_root, "observe id");
					if (observe_id)
					{
						int id = json_integer_value( observe_id );
						//Send feedback to observing insitu
						ThreadList< InsituConnectorGroup* >::ThreadListContainer_ptr group = insituConnectorGroupList.getFront();
						while (group)
						{
							char* buffer = json_dumps( message->json_root, 0 );
							if ( group->t->master->connector->getID() == id)
							{
								if (message->type == FEEDBACK_MASTER)
									write(group->t->master->connector->getSockFD(),buffer,strlen(buffer));
								else //FEEDBACK_ALL
								{
									ThreadList< InsituConnectorContainer* >::ThreadListContainer_ptr insitu = group->t->elements.getFront();
									while (insitu)
									{
										write(insitu->t->connector->getSockFD(),buffer,strlen(buffer));
										insitu = insitu->next;
									}					
								}
								break;
							}
							free(buffer);
							group = group->next;
						}
					}
				}
				if (message->type == OBSERVE)
					dc->t->observe( json_integer_value( json_object_get(message->json_root, "observe id") ) );
				if (message->type == STOP)
					dc->t->stopObserve( json_integer_value( json_object_get(message->json_root, "observe id") ) );
				if (message->type == CLOSED)
				{
					dataClientList.remove(dc);
					break;
				}
				free(message);
			}
			dc = next;
		}
		usleep(1);
	}
	shutdown(insituMaster.getSockFD(),SHUT_RDWR);
	printf("Waiting for insitu Master thread to finish... ");
	fflush(stdout);
	//Yeah... "finish"
	pthread_join(insituThread,NULL);
	printf("Done\n");
	for (std::list<MetaDataConnectorContainer>::iterator it = dataConnectorList.begin(); it != dataConnectorList.end(); it++)
	{
		printf("Asking %s to exit\n",(*it).connector->getName().c_str());
		(*it).connector->masterSendMessage(new MessageContainer(FORCE_EXIT));
	}
	for (std::list<MetaDataConnectorContainer>::iterator it = dataConnectorList.begin(); it != dataConnectorList.end(); it++)
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
