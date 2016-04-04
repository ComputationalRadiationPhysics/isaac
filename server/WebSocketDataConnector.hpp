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

#include "MetaDataConnector.hpp"

#include <libwebsockets.h>

/** This class is used for the connection between the isaac server and
 * some frontend. It defines and abstract interface isaac will use.*/
class WebSocketDataConnector : public MetaDataConnector
{
	public:
		WebSocketDataConnector();
		errorCode init(int port);
		errorCode run();
		std::string getName();
	private:
		struct lws_context *context;
};
