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

#include "Master.hpp"
#include "WebSocketDataConnector.hpp"

#define ISAAC_VERSION "1.0"

int main(int argc, char **argv)
{
	int outer_port = 2459;
	int inner_port = 2460;
	if (argc > 1)
	{
		if (strcmp(argv[1],"--help") == 0)
		{
			printf("ISAAC - In Situ Animation of Accelerated Computations "ISAAC_VERSION"\n");
			printf("Usage:\n");
			printf("\tisaac --help\n");
			printf("\t\tShows this help\n");
			printf("\tisaac [outer_port] [inner_port]\n");
			printf("\t\touter_port default: 2459\n");
			printf("\t\tinner_port default: 2460\n");
			printf("\tisaac --version\n");
			printf("\t\tShows the version\n");
			return 0;
		}
		if (strcmp(argv[1],"--version") == 0)
		{
			printf("Isaac version "ISAAC_VERSION"\n");
			return 0;
		}
		outer_port = atoi(argv[1]);
	}
	if (argc > 2)
		outer_port = atoi(argv[2]);
	
	printf("Using outer_port=%i and inner_port=%i\n",outer_port,inner_port);
	
	printf("\n");
	Master master("ISAAC Test",inner_port);
	WebSocketDataConnector* webSocketDataConnector = new WebSocketDataConnector();
	if (webSocketDataConnector->init(outer_port))
		delete webSocketDataConnector;
	else
		master.addDataConnector(webSocketDataConnector);
	if (master.run())
	{
		printf("Error while running isaac\n");
		return -1;
	}
	return 0;
}
