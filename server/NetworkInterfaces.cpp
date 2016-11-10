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

#include "NetworkInterfaces.hpp"
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>

struct ifaddrs * NetworkInterfaces::ifaddr = NULL;

void NetworkInterfaces::initIfaddr()
{
	if (ifaddr == NULL)
		getifaddrs(&ifaddr);
}

void NetworkInterfaces::bindInterface(in_addr_t &s_addr,std::string interface,bool ipv6)
{
	if (interface.compare(std::string("*")) == 0)
	{
		s_addr = INADDR_ANY;
		return;
	}
	else
		s_addr = INADDR_NONE;
	initIfaddr();
	int correct_family = AF_INET;
	int size_of_struct = sizeof(struct sockaddr_in);
	if (ipv6)
	{
		correct_family = AF_INET6;
		size_of_struct = sizeof(struct sockaddr_in6);
	}
	struct ifaddrs * it = ifaddr;
	while (it)
	{
		int family, s, n;
		char host[NI_MAXHOST];
		family = it->ifa_addr->sa_family;
		if (family == correct_family)
		{
			s = getnameinfo(it->ifa_addr,size_of_struct,host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
			if (interface.compare(std::string(it->ifa_name)) == 0)
			{
				s_addr = inet_addr(host);
				return;
			}
		}
		it = it->ifa_next;
	}
	printf("Warning: interface %s does not exist, therefore no connection can be established!\n",interface.c_str());
}
