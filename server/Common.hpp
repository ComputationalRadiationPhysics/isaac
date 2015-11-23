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
	FEEDBACK = 0,
	FEEDBACK_NEIGHBOUR,
	MASTER_HELLO,
	REGISTER,
	REGISTER_VIDEO,
	EXIT_PLUGIN,
	PERIOD,
	PERIOD_VIDEO,
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
				if (strcmp(str,"feedback") == 0)
					this->type = FEEDBACK;
				else
				if (strcmp(str,"feedback neighbour") == 0)
					this->type = FEEDBACK_NEIGHBOUR;
				else
				if (strcmp(str,"hello") == 0)
					this->type = MASTER_HELLO;
				else
				if (strcmp(str,"register") == 0)
					this->type = REGISTER;
				else
				if (strcmp(str,"register video") == 0)
					this->type = REGISTER_VIDEO;
				else
				if (strcmp(str,"exit") == 0)
					this->type = EXIT_PLUGIN;
				else
				if (strcmp(str,"period video") == 0)
					this->type = PERIOD_VIDEO;
				else
				if (strcmp(str,"period") == 0)
					this->type = PERIOD;
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


typedef enum
{
	IMG_FORCE_EXIT = -1,
	UPDATE_BUFFER = 0,
	GROUP_ADDED,
	GROUP_FINISHED,
	REGISTER_STREAM,
	GROUP_OBSERVED,
	GROUP_OBSERVED_STOPPED
} ImageBufferType;

class InsituConnectorGroup;

class ImageBufferContainer
{
	public:
		ImageBufferContainer(ImageBufferType type,uint8_t* buffer,InsituConnectorGroup* group,int ref_count,std::string target = "",void* reference = NULL)
		{
			this->type = type;
			this->buffer = buffer;
			this->group = group;
			this->ref_count = ref_count;
			this->target = target;
			this->reference = reference;
			pthread_mutex_init (&ref_mutex, NULL);
		}
		~ImageBufferContainer()
		{
			pthread_mutex_destroy(&ref_mutex);
		}
		void incref()
		{
			pthread_mutex_lock (&ref_mutex);
			ref_count++;
			pthread_mutex_unlock (&ref_mutex);
		}
		void suicide()
		{
			pthread_mutex_lock (&ref_mutex);
			ref_count--;
			if (ref_count <= 0)
			{
				pthread_mutex_unlock (&ref_mutex);
				free(buffer);
				delete this;
			}
			else
				pthread_mutex_unlock (&ref_mutex);
		}
		
		ImageBufferType type;
		uint8_t* buffer;
		InsituConnectorGroup* group;
		std::string target;
		void* reference;
		int ref_count;
		pthread_mutex_t ref_mutex;
};
	
