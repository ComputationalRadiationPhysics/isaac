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

#ifndef __NETWORKINTERFACES
#define __NETWORKINTERFACES

#include <sys/types.h>
#include <ifaddrs.h>
#include <string>
#include <netinet/in.h>

class NetworkInterfaces
{
	public:
		static void bindInterface(in_addr_t &s_addr,std::string interface, bool ipv6 = false);
		static struct ifaddrs * ifaddr;
	private:
		static void initIfaddr();
};
#endif
