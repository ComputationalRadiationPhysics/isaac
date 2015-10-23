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
	FEEDBACK_ALL = 0,
	FEEDBACK_MASTER = 1,
	MASTER_HELLO,
	REGISTER_MASTER,
	REGISTER_SLAVE,
	TELL_PLUGIN,
	EXIT_PLUGIN,
	PERIOD_MERGE,
	PERIOD_MASTER,
	OBSERVE,
	STOP,
	CLOSED,
	NONE,
	UNKNOWN,
} MessageType;

class MessageContainer
{
	public:
		MessageContainer(MessageType type = NONE,json_t *json_root = NULL, bool keep_json = false)
		{
			this->json_root = json_root;
			if (keep_json && json_root)
				json_incref( json_root );
			json_t *json_type;
			if (type == NONE && (json_type = json_object_get(json_root, "type")) && json_is_string(json_type))
			{
				const char* str = json_string_value(json_type);
				if (strcmp(str,"hello") == 0)
					this->type = MASTER_HELLO;
				else
				if (strcmp(str,"register master") == 0)
					this->type = REGISTER_MASTER;
				else
				if (strcmp(str,"register slave") == 0)
					this->type = REGISTER_SLAVE;
				else
				if (strcmp(str,"exit") == 0)
					this->type = EXIT_PLUGIN;
				else
				if (strcmp(str,"period merge") == 0)
					this->type = PERIOD_MERGE;
				else
				if (strcmp(str,"period master") == 0)
					this->type = PERIOD_MASTER;
				else
				if (strcmp(str,"observe") == 0)
					this->type = OBSERVE;
				else
				if (strcmp(str,"stop") == 0)
					this->type = STOP;
				else
				if (strcmp(str,"closed") == 0)
					this->type = CLOSED;
				else
				if (strcmp(str,"feedback all") == 0)
					this->type = FEEDBACK_ALL;
				else
				if (strcmp(str,"feedback master") == 0)
					this->type = FEEDBACK_MASTER;
				else
					this->type = UNKNOWN;
			}
			else
				this->type = type;
		}
		~MessageContainer()
		{
			if (json_root)
				json_decref( json_root );
		}
		MessageType type;
		json_t *json_root;
};
