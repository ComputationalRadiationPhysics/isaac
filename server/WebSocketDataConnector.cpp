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

std::string WebSocketDataConnector::getName()
{
	return "WebSocketDataConnector";
}

static int callback_http(
		struct libwebsocket_context *context,
		struct libwebsocket *wsi,
		enum libwebsocket_callback_reasons reason,
		void *user,
		void *in,
		size_t len )
{
	switch (reason)
	{
		case LWS_CALLBACK_HTTP:
		{
			char universal_response[] = "Please use websockets to connect to this server";
			libwebsocket_write(wsi, (unsigned char*) universal_response, strlen(universal_response), LWS_WRITE_HTTP);
			char name[256];
			char rip[256];
			libwebsockets_get_peer_addresses(context,wsi,libwebsocket_get_socket_fd(wsi),name,256,rip,256);
			printf("HTTP Connection from %s (%s)!\n",name,rip);
			return -1;
		}
	}	
	return 0;
}



struct per_session_data__dumb_increment {
	int number;
};

static int
callback_dumb_increment(
		struct libwebsocket_context *context,
		struct libwebsocket *wsi,
		enum libwebsocket_callback_reasons reason,
		void *user,
		void *in,
		size_t len )
{
	int n, m;
	unsigned char buf[LWS_SEND_BUFFER_PRE_PADDING + 512 +
						  LWS_SEND_BUFFER_POST_PADDING];
	unsigned char *p = &buf[LWS_SEND_BUFFER_PRE_PADDING];
	struct per_session_data__dumb_increment *pss = (struct per_session_data__dumb_increment *)user;

	switch (reason) {

	case LWS_CALLBACK_ESTABLISHED:
		lwsl_info("callback_dumb_increment: "
						 "LWS_CALLBACK_ESTABLISHED\n");
		pss->number = 0;
		break;

	case LWS_CALLBACK_SERVER_WRITEABLE:
		n = sprintf((char *)p, "%d", pss->number++);
		m = libwebsocket_write(wsi, p, n, LWS_WRITE_TEXT);
		if (m < n) {
			lwsl_err("ERROR %d writing to socket\n", n);
			return -1;
		}
		break;

	case LWS_CALLBACK_RECEIVE:
		if (len < 6)
			break;
		if (strcmp((const char *)in, "reset\n") == 0)
			pss->number = 0;
		break;

	case LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION:
	{
		char name[256];
		char rip[256];
		libwebsockets_get_peer_addresses(context,wsi,libwebsocket_get_socket_fd(wsi),name,256,rip,256);
		printf("DUMB Connection from %s (%s)!\n",name,rip);
		break;
	}

	default:
		break;
	}

	return 0;
}

static struct libwebsocket_protocols protocols[] = {
	{
		"http-only",		/* name */
		callback_http,		/* callback */
		0,	/* per_session_data_size */
		0,  /* max frame size / rx buffer */
	},
	{
		"dumb-increment-protocol",
		callback_dumb_increment,
		sizeof(struct per_session_data__dumb_increment),
		10,
	},
	{ NULL, NULL, 0, 0 } /* terminator */
};

errorCode WebSocketDataConnector::init(int port)
{
	setlogmask(LOG_UPTO (LOG_DEBUG));
	openlog("lwsts", LOG_PID | LOG_PERROR, LOG_DAEMON);
	lws_set_log_level(7, lwsl_emit_syslog);
	struct lws_context_creation_info info;
	memset(&info, 0, sizeof info);
	info.protocols = protocols;
	#ifndef LWS_NO_EXTENSIONS
		info.extensions = libwebsocket_get_internal_extensions();
	#endif
	info.port = port;
	info.gid = -1;
	info.uid = -1;

	context = libwebsocket_create_context(&info);
	if (context == NULL) {
		lwsl_err("libwebsocket init failed\n");
		return -1;
	}
	return 0;
}

errorCode WebSocketDataConnector::run()
{
	int n = 0;
	unsigned int ms, oldms = 0;	
	while (n >= 0 && !force_exit) {
		struct timeval tv;

		gettimeofday(&tv, NULL);

		ms = (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
		if ((ms - oldms) > 50) {
			libwebsocket_callback_on_writable_all_protocol(&protocols[1]);
			oldms = ms;
		}
		n = libwebsocket_service(context, 50);
		while (MessageContainer* message = getLastMessage())
		{
			//Do something with the message
			delete message;
		}
	}
	libwebsocket_context_destroy(context);
}