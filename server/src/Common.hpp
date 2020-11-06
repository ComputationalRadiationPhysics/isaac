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

#include <string>
#include <string.h>
#include <jansson.h>

#include <iostream>

typedef int ClientRef;
typedef int ObserverRef;
typedef int errorCode;

typedef enum
{
	FORCE_EXIT = -1,
	FEEDBACK = 0,
	MASTER_HELLO,
	REGISTER,
	REGISTER_VIDEO,
	EXIT_PLUGIN,
	PERIOD,
	PERIOD_VIDEO,
	OBSERVE,
	STOP,
	CLOSED,
	UPDATE,
	NONE,
	UNKNOWN,
} MessageType;

class MessageContainer
{
	public:
		static pthread_mutex_t deep_copy_mutex;
		MessageContainer(MessageType type = NONE,json_t *json_root = NULL, bool keep_json = false, bool drop_able = false)
		: drop_able(drop_able)
		{
			if (keep_json && json_root)
			{
				pthread_mutex_lock(&deep_copy_mutex);
				//json_root = json_deep_copy( json_root );
				json_incref( json_root );
				pthread_mutex_unlock(&deep_copy_mutex);
			}
			this->json_root = json_root;
			json_t *json_type;
			if (type == NONE && (json_type = json_object_get(json_root, "type")) && json_is_string(json_type))
			{
				const char* str = json_string_value(json_type);
				if (strcmp(str,"feedback") == 0)
					this->type = FEEDBACK;
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
				if (strcmp(str,"update") == 0)
					this->type = UPDATE;
				else
					this->type = UNKNOWN;
			}
			else
				this->type = type;
		}
		void suicide()
		{
			delete this;
		}
		~MessageContainer()
		{
			if (json_root)
				json_decref( json_root );
		}
		MessageType type;
		json_t *json_root;
		bool drop_able;
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

class ImageBuffer
{
	public:
		ImageBuffer(uint8_t* buffer, int ref_count) :
			buffer( buffer ),
			ref_count (ref_count )
		{
			pthread_mutex_init (&ref_mutex, NULL);
		}	
		~ImageBuffer()
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
		uint8_t* buffer;
		int ref_count;
		pthread_mutex_t ref_mutex;
};

class ImageBufferContainer
{
	public:
		ImageBufferContainer(ImageBufferType type,uint8_t* buffer,InsituConnectorGroup* group,int ref_count,std::string target = "",void* reference = NULL,json_t* json = NULL,json_t* payload = NULL,int insitu_id = 0) :
			type( type ),
			group( group ),
			target( target ),
			reference( reference ),
			ref_count( ref_count ),
			json( json ),
			payload( payload ),
			insitu_id( insitu_id )
		{
			image = new ImageBuffer( buffer, ref_count );
			pthread_mutex_init (&json_mutex, NULL);
			pthread_mutex_init (&payload_mutex, NULL);
		}
		void suicide()
		{
			ref_count--;
			if (ref_count <= 0)
				delete this;
		}
		~ImageBufferContainer()
		{
			pthread_mutex_destroy(&json_mutex);
			pthread_mutex_destroy(&payload_mutex);
		}
		ImageBufferType type;
		InsituConnectorGroup* group;
		std::string target;
		void* reference;
		ImageBuffer* image;
		int ref_count;
		pthread_mutex_t json_mutex;
		pthread_mutex_t payload_mutex;
		json_t* json;
		json_t* payload;
		int insitu_id;
};

