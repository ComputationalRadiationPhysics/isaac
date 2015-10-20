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

#include "MessageAble.hpp"

errorCode MessageAble::masterSendMessage(MessageContainer* message)
{
	messagesIn.push_back(message);
}

MessageContainer* MessageAble::masterGetMessage()
{
	return messagesOut.pop_front();
}

errorCode MessageAble::clientSendMessage(MessageContainer* message)
{
	messagesOut.push_back(message);
}

MessageContainer* MessageAble::clientGetMessage()
{
	return messagesIn.pop_front();
}

MessageAble::~MessageAble()
{
	MessageContainer* mom;
	while (mom = messagesIn.pop_front())
		delete mom;
	while (mom = messagesOut.pop_front())
		delete mom;
}
