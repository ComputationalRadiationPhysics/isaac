/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or(at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Lesser Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#include "URIImageConnector.hpp"

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/insert_linebreaks.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/archive/iterators/ostream_iterator.hpp>
#include <sstream>

URIImageConnector::URIImageConnector()
{
}

std::string URIImageConnector::getName()
{
	return "JPEG_URI_Stream";
}

errorCode URIImageConnector::init(int minport,int maxport)
{
	return 0;
}

void isaac_init_destination(j_compress_ptr cinfo)
{
}
boolean isaac_jpeg_empty_output_buffer(j_compress_ptr cinfo)
{
	return true;
}
void isaac_jpeg_term_destination(j_compress_ptr cinfo)
{
}

errorCode URIImageConnector::run()
{
	uint64_t finish = 0;
	while(finish == 0)
	{
		ImageBufferContainer* message;
		while(message = clientGetMessage())
		{
			if (message->type == GROUP_OBSERVED)
			{
				streams.insert( std::pair<void*,InsituConnectorGroup*>(message->reference,message->group));
				uint8_t* dummy = (uint8_t*)malloc(1);
				dummy[0] = 0;
				clientSendMessage(new ImageBufferContainer(REGISTER_STREAM,dummy,message->group,1,message->target,message->reference));
				printf("URIImageConnector: Openend URI JPEG Stream\n");
			}
			if (message->type == GROUP_OBSERVED_STOPPED || message->type == GROUP_FINISHED)
			{
				auto it = streams.begin();
				while (it != streams.end())
				{
					auto next_it = std::next(it);
					if (it->second == message->group)
					{
						streams.erase(it);
						printf("URIImageConnector: Closed Stream\n");
					}
					it = next_it;
				}
			}
			if (message->type == UPDATE_BUFFER)
			{
				for (auto it = streams.begin(); it != streams.end(); it++)
				{
					if (it->second == message->group)
					{
						pthread_mutex_lock (&(message->json_mutex));
						pthread_mutex_lock (&(message->payload_mutex));
						json_object_set( message->json, "payload", message->payload ); 
						pthread_mutex_unlock (&(message->payload_mutex));
						pthread_mutex_unlock (&(message->json_mutex));
					}
				}
			}
			if (message->type == IMG_FORCE_EXIT)
				finish = 1;
			clientSendMessage( message );
		}
		usleep(1000);
	}	
}

URIImageConnector::~URIImageConnector()
{
}

