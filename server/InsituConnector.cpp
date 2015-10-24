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

#include "InsituConnector.hpp"
#include "stdio.h"

InsituConnector::InsituConnector(int sockfd,int id)
{
	this->sockfd = sockfd;
	this->id = id;
}

int InsituConnector::getID()
{
	return id;
}

int InsituConnector::getSockFD()
{
	return sockfd;
}

#define LOAD_BUFFER 128
typedef struct
{
	char buffer[LOAD_BUFFER];
	int sockfd;
	int pos;
	int count;
} json_load_callback_struct;

size_t json_load_callback_function (void *buffer, size_t buflen, void *data)
{
	json_load_callback_struct* jlcb = (json_load_callback_struct*)data;
	if (jlcb->pos >= jlcb->count)
	{
		jlcb->count = read(jlcb->sockfd,jlcb->buffer,LOAD_BUFFER);
		if (jlcb->count > 0)
			jlcb->pos = 0;
	}
	if (jlcb->pos < jlcb->count)
	{
		((char*)buffer)[0] = jlcb->buffer[jlcb->pos];
		jlcb->pos++;
		return 1;
	}
	return 0;
	//return read(jlcb->sockfd,buffer,1);
}

errorCode InsituConnector::run()
{
	MessageContainer* message = NULL;
	json_load_callback_struct jlcb;
	jlcb.sockfd = sockfd;
	jlcb.pos = 0;
	jlcb.count = 0;
	while (json_t * content = json_load_callback(json_load_callback_function,&jlcb,JSON_DISABLE_EOF_CHECK,NULL))
	{
		message = new MessageContainer(NONE,content);
		if (message->type == EXIT_PLUGIN)
			break;
		if (message->type == REGISTER_MASTER)
			json_object_set_new( message->json_root, "id", json_integer( id) );
		clientSendMessage(message);
		message = NULL;
	}
	if (!message) //We ended because of closed connection
	{
		message = new MessageContainer(EXIT_PLUGIN,json_object());
		json_object_set_new( message->json_root, "type", json_string( "exit" ) );
	}
	clientSendMessage(message);
	//messagesOut.spin_over_delete();
}

InsituConnector::~InsituConnector()
{
	//fclose(sockfile);
}
