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

#ifndef __BROKER
#define __BROKER

#include <string>
#include <vector>
#include "Common.hpp"
#include "MetaDataConnector.hpp"
class MetaDataConnector;

#include "ImageConnector.hpp"
class ImageConnector;

#include <signal.h>
#include "MetaDataClient.hpp"
#include "InsituConnectorMaster.hpp"
#include <memory>
#include <map>

typedef struct MetaDataConnectorContainer_struct
{
	MetaDataConnector* connector;
	pthread_t thread;
} MetaDataConnectorContainer;

typedef struct ImageConnectorContainer_struct
{
	ImageConnector* connector;
	pthread_t thread;
} ImageConnectorContainer;

#define MAX_NODES 999999

class InsituConnectorGroup
{
	friend class Broker;
	public:
		InsituConnectorGroup(std::string name) :
			master( NULL ),
			initData( NULL ),
			id( 0 ),
			name( name),
			nodes( MAX_NODES ),
			video_buffer_size( 0 )
		{
			pthread_mutex_init (&streams_mutex, NULL);
		}
		int getID()
		{
			return id;
		}
		int getVideoBufferSize()
		{
			return video_buffer_size;
		}
		int getFramebufferWidth()
		{
			return framebuffer_width;
		}
		int getFramebufferHeight()
		{
			return framebuffer_height;
		}
		std::string getName()
		{
			return name;
		}
		~InsituConnectorGroup()
		{
			pthread_mutex_destroy(&streams_mutex);
		}
	private:
		InsituConnectorContainer* master;
		int nodes;
		std::string name;
		json_t* initData;
		int id;
		int framebuffer_width;
		int framebuffer_height;
		size_t video_buffer_size;
		std::map< std::string, std::map< void* , std::string > > streams;
		pthread_mutex_t streams_mutex;
};

class Broker
{
	public:
		Broker(std::string name,int inner_port,std::string interface);
		~Broker();
		errorCode addDataConnector(MetaDataConnector *dataConnector);
		errorCode addImageConnector(ImageConnector *imageConnector);
		MetaDataClient* addDataClient();
		void receiveVideo(InsituConnectorGroup* group,uint8_t* video_buffer,char* payload);
		errorCode run();
		std::string getStream(std::string connector,std::string name,std::string ref);
		static volatile sig_atomic_t force_exit;
	private:
		InsituConnectorMaster insituMaster;
		json_t* masterHello;
		json_t* masterHelloConnectorList;
		std::string name;
		std::vector< MetaDataConnectorContainer > dataConnectorList;
		std::vector< ImageConnectorContainer > imageConnectorList;
		ThreadList< InsituConnectorGroup* > insituConnectorGroupList;
		ThreadList< MetaDataClient* > dataClientList;
		int inner_port;
		std::string inner_interface;
		pthread_t insituThread;
};

#endif
