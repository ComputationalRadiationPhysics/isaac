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

Master::Master(std::string name)
{
	this->name = name;
}

errorCode Master::addDataConnector(MetaDataConnector *dataConnector)
{
	dataConnectorList.push_back(dataConnector);
	dataConnector->setMaster(this);
	return 0;
}

errorCode Master::remDataConnector(MetaDataConnector *dataConnector)
{
	//TODO
	return 0;
}

errorCode run()
{
	//TODO
	return 0;
}

Master::~Master()
{
	dataConnectorList.clear();
}
