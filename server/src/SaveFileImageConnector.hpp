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

#pragma once

#include "ImageConnector.hpp"

#include <string>
#include <map>

class SaveFileImageConnector : public ImageConnector
{
	public:
		SaveFileImageConnector(std::string dir);
		errorCode init(int minport,int maxport);
		errorCode run();
		std::string getName();
	private:
		std::string dir;
		std::map<InsituConnectorGroup*,std::string> groupDir;
		std::map<InsituConnectorGroup*,int> step;
};
