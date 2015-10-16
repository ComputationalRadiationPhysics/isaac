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

#pragma once

#include "Common.hpp"
#include <string>
#include "ThreadList.hpp"
class Master;

typedef enum
{
	FORCE_EXIT = -1,
	NONE = 0,
	ADD_CLIENT,
	REMOVE_CLIENT,
	ADD_OBSERVER,
	REMOVE_OBSERVER
} MessageType;

class MessageContainer
{
	public:
		MessageContainer(MessageType type = NONE,int ref1 = 0,int ref2 = 0,std::string content = "")
		{
			this->type = type;
			this->ref1 = ref1;
			this->ref2 = ref2;
			this->content = content;
		}
		
		MessageType type;
		int ref1;
		int ref2;
		std::string content;
};

/** This class is used for the connection between the isaac server and
 * some frontend. It defines and abstracts an interface isaac will use.*/
class MetaDataConnector
{
	public:
		MetaDataConnector();
		
		//To be used, the message must be freed!
		MessageContainer* getLastMessage();

		//To be overwritten
		virtual errorCode init(int port) = 0;
		virtual errorCode run() = 0;
		virtual std::string getName() = 0;

		//Called from the Master
		void setMaster(Master* master);		
		errorCode addMessage(MessageContainer* message);
		bool force_exit;
		ThreadList<MessageContainer*> messages;
	private:
		Master* master;
};
