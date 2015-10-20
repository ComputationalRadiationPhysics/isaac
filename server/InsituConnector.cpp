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

#include "InsituConnector.hpp"
#include "stdio.h"

InsituConnector::InsituConnector(int sockfd,int id)
{
	this->sockfd = sockfd;
	this->sockfile = fdopen(sockfd,"r");
	this->id = id;
}

int InsituConnector::getID()
{
	return id;
}

errorCode InsituConnector::run()
{
	MessageContainer* message = NULL;
	//Get init sequence of insitu plugin
	while (json_t * content = json_loadf(sockfile,JSON_DISABLE_EOF_CHECK,NULL))
	{
		message = new MessageContainer(NONE,content);
		if (message->type == EXIT_PLUGIN)
			break;
		clientSendMessage(message);
		message = NULL;
	}
	if (!message) //We ended because of closes connection
	{
		message = new MessageContainer(EXIT_PLUGIN,json_object());
		json_object_set_new( message->json_root, "type", json_string( "exit" ) );
	}
	json_object_set_new( message->json_root, "id", json_integer( id) );
	clientSendMessage(message);
	messagesOut.spin_over_delete();
}

InsituConnector::~InsituConnector()
{
	fclose(sockfile);
}
