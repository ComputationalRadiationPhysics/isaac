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

#include "MetaDataClient.hpp"
#include <algorithm>

MetaDataClient::MetaDataClient()
{
	
}

void MetaDataClient::observe(int nr)
{
	observeList.push_back(nr);
	observeList.unique();
}

void MetaDataClient::stopObserve(int nr)
{
	observeList.remove(nr);
}

bool MetaDataClient::doesObserve(int nr)
{
	return (std::find(observeList.begin(), observeList.end(), nr) != observeList.end());
}
