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

#ifndef __MASTER
#define __MASTER

#include <string>
#include <list>
#include "Common.hpp"
#include "MetaDataConnector.hpp"
#include <signal.h>
#include "MetaDataClient.hpp"
#include "InsituConnectorMaster.hpp"

class MetaDataConnector;

typedef struct MetaDataConnectorList_struct
{
	MetaDataConnector* connector;
	pthread_t thread;
} MetaDataConnectorList;

class Master
{
	public:
		Master(std::string name,int inner_port);
		~Master();
		errorCode addDataConnector(MetaDataConnector *dataConnector);
		MetaDataClient* addDataClient();
		errorCode run();
		static volatile sig_atomic_t force_exit;
	private:
		InsituConnectorMaster insituMaster;
		json_t* masterHello;
		std::string name;
		std::list<MetaDataConnectorList> dataConnectorList;
		ThreadList<MetaDataClient*> dataClientList;
		int inner_port;
		pthread_t insituThread;
};

#endif
