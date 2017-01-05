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

#include "MetaDataClient.hpp"
#include <algorithm>

MetaDataClient::MetaDataClient()
{
	
}

void MetaDataClient::observe(int nr,int stream, bool dropable)
{
	observeList.insert( std::pair<int,int>(nr,stream) );
	dropableList.insert( std::pair<int,bool>(nr,dropable) );
}

void MetaDataClient::stopObserve(int nr,int& stream,bool& dropable)
{
	auto it = observeList.find( nr );
	if (it != observeList.end())
		stream = it->second;
	auto it2 = dropableList.find( nr );
	if (it2 != dropableList.end())
		dropable = it2->second;
	observeList.erase(nr);
	dropableList.erase(nr);
}

bool MetaDataClient::doesObserve(int nr,int& stream,bool& dropable)
{
	auto it = observeList.find( nr );
	if (it == observeList.end())
		return false;
	auto it2 = dropableList.find( nr );
	if (it2 == dropableList.end())
		return false;
	stream = it->second;
	dropable = it2->second;
	return true;
}
