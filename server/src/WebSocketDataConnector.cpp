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

#include "WebSocketDataConnector.hpp"


#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>

#include <syslog.h>
#include <sys/time.h>
#include <unistd.h>

#include <pthread.h>

#include "MetaDataClient.hpp"

#include <sstream>
#include <iostream>

WebSocketDataConnector::WebSocketDataConnector()
{
	context = NULL;
}

std::string WebSocketDataConnector::getName()
{
	return "WebSocketDataConnector";
}

static int callback_http(
		struct lws *wsi,
		enum lws_callback_reasons reason,
		void *user,
		void *in,
		size_t len )
{
	Broker** broker_ptr = (Broker**)lws_context_user(lws_get_context(wsi));
	Broker* broker = NULL;
	if (broker_ptr)
		broker = *broker_ptr;
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wswitch"
	switch (reason)
	{
		case LWS_CALLBACK_HTTP:
		{
			std::istringstream request( (char*)in );
			std::string left,middle,right;
			std::getline(request, left, '/'); //first /
			std::getline(request, left, '/');
			std::getline(request, middle, '/');
			std::getline(request, right, '/');
			std::string description = broker->getStream(left,middle,right);
			char buf[LWS_SEND_BUFFER_PRE_PADDING + 2048 + LWS_SEND_BUFFER_POST_PADDING];
			char* use = &(buf[LWS_SEND_BUFFER_PRE_PADDING]);
			sprintf(use,"HTTP/1.1 200 OK\n\n%s",description.c_str());
			lws_write(wsi, (unsigned char*) use, strlen(use), LWS_WRITE_HTTP);
			char name[256];
			char rip[256];
			lws_get_peer_addresses(wsi,lws_get_socket_fd(wsi),name,256,rip,256);
			printf("HTTP Connection from %s (%s)!\n",name,rip);
			return -1;
		}
	}
	#pragma GCC diagnostic pop
	return 0;
}

struct per_session_data__isaac {
	MetaDataClient* client;
	char url[32];
};

static int callback_isaac(
		struct lws *wsi,
		enum lws_callback_reasons reason,
		void *user,
		void *in,
		size_t len )
{
	int n, m;
     
	struct per_session_data__isaac *pss = (struct per_session_data__isaac *)user;
	Broker** broker_ptr = (Broker**)lws_context_user(lws_get_context(wsi));
	Broker* broker = NULL;
	if (broker_ptr)
		broker = *broker_ptr;
	
	switch (reason) {

	case LWS_CALLBACK_ESTABLISHED:
		printf("callback_isaac: LWS_CALLBACK_ESTABLISHED\n");
		char dummy[32];
		lws_get_peer_addresses(wsi,lws_get_socket_fd(wsi),dummy,32,pss->url,32);
		break;

	case LWS_CALLBACK_SERVER_WRITEABLE:
		if (pss->client)
		{
            char* buf = new char[LWS_SEND_BUFFER_PRE_PADDING + ISAAC_MAX_RECEIVE +
                        LWS_SEND_BUFFER_POST_PADDING];
                         
            char *p = &buf[LWS_SEND_BUFFER_PRE_PADDING];
			MessageContainer* message = NULL;
			int l = 0;
			do
			{
				l = pss->client->messagesIn.length();
				while ( (message = pss->client->clientGetMessage()) != NULL && //new message
						l > 1 && //at least two
						message->drop_able ) //only skip if dropable!
				{
					printf("WebSocketDataConnector: Dropped one dropable package!\n");
					delete message;
					l--;
				}
				if (message) //New message from master sama!
				{
					pthread_mutex_lock(&MessageContainer::deep_copy_mutex);
					char* buffer = json_dumps( message->json_root, 0 );
					pthread_mutex_unlock(&MessageContainer::deep_copy_mutex);
					n = strlen(buffer);
					sprintf(p,"%s",buffer);
					m = lws_write(wsi, (unsigned char*)p, n, LWS_WRITE_TEXT);
					free(buffer);
					if (m < n)
					{
						lwsl_err("ERROR %d writing to socket\n", n);
						pss->client->clientSendMessage(new MessageContainer(CLOSED));
						return -1;
					}
					delete message;
				}
			}
			while (l > 1 && !lws_send_pipe_choked(wsi));
            delete[] buf;
		}
		break;
	//case LWS_CALLBACK_CLOSED:
	//	pss->client->clientSendMessage(new MessageContainer(CLOSED));
	//	return -1;	
	case LWS_CALLBACK_RECEIVE:
		if (pss->client)
		{
			json_error_t error;
			json_t* input = json_loadb((const char *)in, len, 0, &error);
			if(!input)
				printf("JSON ERROR: %s", error.text);
			MessageContainer* message = new MessageContainer(NONE,input);
			int finish = (message->type == CLOSED);
			json_object_set_new( message->json_root, "url", json_string( pss->url ) );
			pss->client->clientSendMessage(message);
			if (finish)
				return -1;
		}
		break;

	case LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION:
	{
		printf("callback_isaac: LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION\n");
		char name[256];
		char rip[256];
		lws_get_peer_addresses(wsi,lws_get_socket_fd(wsi),name,256,rip,256);
		printf("ISAAC Connection from %s (%s)!\n",name,rip);
		pss->client = broker->addDataClient();
		break;
	}

	default:
		break;
	}
	return 0;
}

static struct lws_protocols protocols[] = {
	{
		"http-only",		/* name */
		callback_http,		/* callback */
		0,	/* per_session_data_size */
		0,  /* max frame size / rx buffer */
	},
	{
		"isaac-json-protocol",
		callback_isaac,
		sizeof(struct per_session_data__isaac),
		ISAAC_MAX_RECEIVE,
	},
	{ NULL, NULL, 0, 0 } /* terminator */
};

errorCode WebSocketDataConnector::init(int port,std::string interface)
{
	setlogmask(LOG_UPTO (LOG_DEBUG));
	openlog("lwsts", LOG_PID | LOG_PERROR, LOG_DAEMON);
	int logs = LLL_USER | LLL_ERR | LLL_WARN | LLL_NOTICE;
	lws_set_log_level(logs, lwsl_emit_syslog);
	struct lws_context_creation_info info;
	memset(&info, 0, sizeof info);
	info.protocols = protocols;
	#ifndef LWS_NO_EXTENSIONS
		info.extensions = NULL;
	#endif
	info.user = (void*)(&broker);
	info.port = port;
	info.gid = -1;
	info.uid = -1;
	if (interface.compare(std::string("*")) != 0)
		info.iface = interface.c_str();
	context = lws_create_context(&info);
	if (context == NULL) {
		lwsl_err("libwebsocket init failed\n");
		return -1;
	}
	return 0;
}

errorCode WebSocketDataConnector::run()
{
	int n = 0;
	bool force_exit = false;
	while (n >= 0 && !force_exit)
	{
		n = lws_service(context, 0);
		lws_callback_on_writable_all_protocol(context,&protocols[1]);
		while (MessageContainer* message = clientGetMessage())
		{
			if (message->type == FORCE_EXIT)
				force_exit = true;
			delete message;
		}
		usleep(100);
	}
	lws_context_destroy(context);
	return 0;
}
