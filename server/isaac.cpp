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

#include "Broker.hpp"
#include "WebSocketDataConnector.hpp"
#ifdef ISAAC_SDL
	#include "SDLImageConnector.hpp"
#endif
#ifdef ISAAC_GST
	#include "RTPImageConnector.hpp"
	#include "RTMPImageConnector.hpp"
#endif
#ifdef ISAAC_JPEG
	#include "URIImageConnector.hpp"
#endif

#define ISAAC_VERSION "1.0"

#define ISAAC_INCREASE_NR_OR_DIE \
	nr++; \
	if (nr >= argc) \
	{ \
		printf("Not enough arguments!\n"); \
		return 1; \
	}

int main(int argc, char **argv)
{
	int outer_port = 2459;
	int inner_port = 2460;
	const char __name[] = "ISAAC Visualization server";
	const char __url[] = "127.0.0.1";
	const char* name = __name;
	const char* url = __url;
	#ifdef ISAAC_GST
		char* twitch_apikey = NULL;
		char* twitch_url = NULL;
	#endif
	int nr = 1;
	while (nr < argc)
	{
		if (strcmp(argv[nr],"--help") == 0)
		{
			printf("ISAAC - In Situ Animation of Accelerated Computations " ISAAC_VERSION"\n");
			printf("Usage:\n");
			printf("isaac [--help] [--outer_port <X>] [--inner_port <X>] [--url <X>] [--name <X>]\n");
			printf("      [--jpeg] [--version]\n");
			printf("       --help: Shows this help\n");
			printf(" --outer_port: Set port for the clients to connect. Default 2459\n");
			printf(" --inner_port: Set port for the simulations to connect to. Default 2460\n");
			printf("        --url: Set the url to connect to from outside. Default 127.0.0.1\n");
			printf("       --name: Set the name of the server.\n");
			printf("    --version: Shows the version\n");
		#ifdef ISAAC_GST
			printf("     --twitch: Set twitch apikey for twitch live streaming\n");
			printf(" --twitch_url: Set twitch rtmp-url for ssh forwarding or another rtmp service\n");
		#endif
			return 0;
		}
		else
		if (strcmp(argv[nr],"--version") == 0)
		{
			printf("Isaac version " ISAAC_VERSION "\n");
			return 0;
		}
		else
		if (strcmp(argv[nr],"--outer_port") == 0)
		{
			ISAAC_INCREASE_NR_OR_DIE
			outer_port = atoi(argv[nr]);
		}
		else
		if (strcmp(argv[nr],"--inner_port") == 0)
		{
			ISAAC_INCREASE_NR_OR_DIE
			inner_port = atoi(argv[nr]);
		}
		else
		if (strcmp(argv[nr],"--url") == 0)
		{
			ISAAC_INCREASE_NR_OR_DIE
			url = argv[nr];
		}
		else
		if (strcmp(argv[nr],"--name") == 0)
		{
			ISAAC_INCREASE_NR_OR_DIE
			name = argv[nr];
		}
		#ifdef ISAAC_GST
			else
			if (strcmp(argv[nr],"--twitch") == 0)
			{
				ISAAC_INCREASE_NR_OR_DIE
				twitch_apikey = argv[nr];
			}
			else
			if (strcmp(argv[nr],"--twitch_url") == 0)
			{
				ISAAC_INCREASE_NR_OR_DIE
				twitch_url = argv[nr];
			}
		#endif
		else
		{
			printf("Don't know argument %s\n",argv[nr]);
			return 1;
		}
		nr++;
	}
	
	printf("Using outer_port=%i and inner_port=%i\n",outer_port,inner_port);
	
	printf("\n");
	Broker broker(name,inner_port);
	WebSocketDataConnector* webSocketDataConnector = new WebSocketDataConnector();
	if (webSocketDataConnector->init(outer_port))
		delete webSocketDataConnector;
	else
		broker.addDataConnector(webSocketDataConnector);
	#ifdef ISAAC_GST
		RTPImageConnector* rTPImageConnector = new RTPImageConnector(url,false,false);
		if (rTPImageConnector->init(5000,5099))
			delete rTPImageConnector;
		else
			broker.addImageConnector(rTPImageConnector);
		rTPImageConnector = new RTPImageConnector(url,false,true);
		if (rTPImageConnector->init(5100,5199))
			delete rTPImageConnector;
		else
			broker.addImageConnector(rTPImageConnector);
	#endif
	#ifdef ISAAC_JPEG
		URIImageConnector* uRIImageConnector = new URIImageConnector();
		if (uRIImageConnector->init(0,0))
			delete uRIImageConnector;
		else
			broker.addImageConnector(uRIImageConnector);
	#endif
	#ifdef ISAAC_GST
		if (twitch_apikey)
		{
			RTMPImageConnector* twitchImageConnector;
			if (twitch_url)
				twitchImageConnector = new RTMPImageConnector( std::string("Twitch"), std::string(twitch_apikey), std::string(twitch_url) );
			else
				twitchImageConnector = new RTMPImageConnector( std::string("Twitch"), std::string(twitch_apikey), std::string("live-fra.twitch.tv/app") );
			if (twitchImageConnector->init(0,0))
				delete twitchImageConnector;
			else
				broker.addImageConnector(twitchImageConnector);
		}
	#endif
	#ifdef ISAAC_SDL
		SDLImageConnector* sDLImageConnector = new SDLImageConnector();
		if (sDLImageConnector->init(0,0))
			delete sDLImageConnector;
		else
			broker.addImageConnector(sDLImageConnector);
	#endif
	if (broker.run())
	{
		printf("Error while running isaac\n");
		return -1;
	}
	return 0;
}
