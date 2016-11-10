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

#include "SaveFileImageConnector.hpp"
#include <chrono>
#include <sys/stat.h> 
#include <sys/types.h>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>

SaveFileImageConnector::SaveFileImageConnector(std::string dir)
{
	showClient = false;
	this->dir = dir;
}

std::string SaveFileImageConnector::getName()
{
	return "SaveFileImageConnector";
}

errorCode SaveFileImageConnector::init(int minport,int maxport)
{
	return 0;
}

errorCode SaveFileImageConnector::run()
{
	int finish = 0;
	while (finish == 0)
	{
		ImageBufferContainer* message;
		while (message = clientGetMessage())
		{
			if (message->type == IMG_FORCE_EXIT)
				finish = 1;
			if (message->type == GROUP_ADDED)
			{
				std::chrono::seconds s = std::chrono::duration_cast< std::chrono::seconds >(std::chrono::system_clock::now().time_since_epoch());
				std::string localDir = dir + std::string("/");
				mkdir(localDir.c_str(),0777);
				localDir += message->group->getName() + std::string("/");
				mkdir(localDir.c_str(),0777);
				localDir += std::to_string(s.count()) + std::string("/");
				mkdir(localDir.c_str(),0777);
				groupDir.insert(std::make_pair(message->group,localDir));
				step.insert(std::make_pair(message->group,0));
			}
			if (message->type == GROUP_FINISHED)
			{
				groupDir.erase(message->group);
				step.erase(message->group);
			}
			if (message->type == UPDATE_BUFFER)
			{
				pthread_mutex_lock (&(message->payload_mutex));
				std::string payload( json_string_value( message->payload ) );
				pthread_mutex_unlock (&(message->payload_mutex));
				//Some copy and pasted black "base64 -> jpeg" magic from Broker.cpp
				//Search for : in payload
				const char* colon = strchr(payload.c_str(), ':');
				if (colon != NULL)
				{
					colon++;
					//Search for ; in payload
					const char* semicolon = strchr(colon, ';');
					if (semicolon != NULL)
					{
						//Search for , in payload
						const char* comma = strchr(semicolon, ',');
						if (comma != NULL)
						{
							//After the comma the base64 stream starts
							comma++;
							int whole_length = strlen(comma);

							uint8_t* temp_buffer = (uint8_t*)malloc(strlen(comma)+4); //Should always be enough
							//base64 -> binary data
							using namespace boost::archive::iterators;
							typedef
								transform_width
								<
									binary_from_base64
									<
										const uint8_t *
									>,
									8,
									6
								>
								base64_dec;
							int i = 0;
							try
							{
								base64_dec src_it(comma);
								for(; i < (whole_length+1)*3/4; ++i)
								{
									temp_buffer[i] = *src_it;
									++src_it;
								}
							}
							catch (dataflow_exception&)
							{
							}
							std::string filename = groupDir.at(message->group) + std::to_string(step.at(message->group)++) + std::string(".jpg");
							FILE *fp = fopen(filename.c_str(),"wb");
							fwrite (temp_buffer , sizeof(char), i, fp);
							fclose (fp);
							free(temp_buffer);
						}
					}
				}
			}
			clientSendMessage( message );
		}
		usleep(1000);
	}
}
