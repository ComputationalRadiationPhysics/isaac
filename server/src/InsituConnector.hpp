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

#pragma once
#include "Runable.hpp"
#include "MessageAble.hpp"
#include "ThreadList.hpp"

typedef struct
{
	int pos;
	int count;
	char buffer[ISAAC_MAX_RECEIVE];
} json_load_callback_struct;

class InsituConnector : public MessageAble<MessageContainer>
{
	friend class InsituConnectorMaster;
	public:
		InsituConnector(int sockfd,int id);
		~InsituConnector();
		int getID();
		int getSockFD();
	private:
		json_load_callback_struct jlcb;
		int id;
		int sockfd;
};
