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

volatile sig_atomic_t Master::force_exit = 0;

void sighandler(int sig)
{
	printf("\n");
	Master::force_exit = 1;
}

Master::Master(std::string name)
{
	this->name = name;
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

void* run_data_connector(void* ptr)
{
	MetaDataConnector* dataConnector = (MetaDataConnector*)ptr;
	dataConnector->run();
}

errorCode Master::run()
{
	printf("Running ISAAC Master\n");
	signal(SIGINT, sighandler);
	for (std::list<MetaDataConnectorList>::iterator it = dataConnectorList.begin(); it != dataConnectorList.end(); it++)
	{
		printf("Launching %s\n",(*it).connector->getName().c_str());
		pthread_create(&((*it).thread),NULL,run_data_connector,(*it).connector);
	}		
	while (force_exit == 0)
		usleep(1);
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
