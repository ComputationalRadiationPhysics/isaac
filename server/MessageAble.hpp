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

#include "Common.hpp"
#include "ThreadList.hpp"

template <typename MessageTemplate>
class MessageAble
{
	public:
		virtual ~MessageAble()
		{
			MessageTemplate* mom;
			while (mom = messagesIn.pop_front())
				delete mom;
			while (mom = messagesOut.pop_front())
				delete mom;
		}
		//Called from MetaDataConnector / Client
		errorCode clientSendMessage(MessageTemplate* message)
		{
			messagesOut.push_back(message);
		}
		MessageTemplate* clientGetMessage()
		{
			return messagesIn.pop_front();
		}
		//Called from Master
		errorCode masterSendMessage(MessageTemplate* message)
		{
			messagesIn.push_back(message);
		}
		MessageTemplate* masterGetMessage()
		{
			return messagesOut.pop_front();
		}
	//protected:
		ThreadList<MessageTemplate*> messagesIn; //From master to the client
		ThreadList<MessageTemplate*> messagesOut; //From client to the master
};
