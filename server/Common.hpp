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
 
#pragma once

#include <string>
#include <string.h>
#include <jansson.h>

typedef int ClientRef;
typedef int ObserverRef;
typedef int errorCode;

#define MAX_RECEIVE 262144 //256kb

typedef enum
{
	FORCE_EXIT = -1,
	NONE = 0,
	MASTER_HELLO,
	REGISTER_PLUGIN,
	EXIT_PLUGIN,
	PERIOD_DATA,
	UNKNOWN
} MessageType;

class MessageContainer
{
	public:
		MessageContainer(MessageType type = NONE,json_t *json_root = NULL)
		{
			this->json_root = json_root;
			json_t *json_type;
			if (type == NONE && (json_type = json_object_get(json_root, "type")) && json_is_string(json_type))
			{
				const char* str = json_string_value(json_type);
				if (strcmp(str,"hello") == 0)
					this->type = MASTER_HELLO;
				else
				if (strcmp(str,"register") == 0)
					this->type = REGISTER_PLUGIN;
				else
				if (strcmp(str,"exit") == 0)
					this->type = EXIT_PLUGIN;
				else
				if (strcmp(str,"period") == 0)
					this->type = PERIOD_DATA;
				else
					this->type = UNKNOWN;
			}
			else
				this->type = type;
		}
		MessageType type;
		json_t *json_root;
};
