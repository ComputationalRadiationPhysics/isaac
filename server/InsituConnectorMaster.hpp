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

#pragma once
#include "Common.hpp"
#include "Runable.hpp"
#include "ThreadList.hpp"
#include "InsituConnector.hpp"
#include <memory>
class InsituConnectorGroup;

typedef struct InsituConnectorContainer_struct
{
	InsituConnector* connector;
	pthread_t thread;
	InsituConnectorGroup* group;
	int meta_merge_count;
} InsituConnectorContainer;

class InsituConnectorMaster : public Runable
{
	public:
		InsituConnectorMaster();
		errorCode init(int port);
		errorCode run();
		~InsituConnectorMaster();
		ThreadList< InsituConnectorContainer* > insituConnectorList;
		int getSockFD();
	private:
		int nextFreeNumber;
		int sockfd;
};
