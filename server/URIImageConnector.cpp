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
						streams.erase(it);
					it = next_it;
				}
			}
			if (message->type == UPDATE_BUFFER)
			{
				for (auto it = streams.begin(); it != streams.end(); it++)
				{
					if (it->second == message->group)
					{
						struct jpeg_compress_struct cinfo;
						struct jpeg_error_mgr jerr;
						jpeg_destination_mgr dest;
						dest.init_destination = &isaac_init_destination;
						dest.empty_output_buffer = &isaac_jpeg_empty_output_buffer;
						dest.term_destination = &isaac_jpeg_term_destination;
						cinfo.err = jpeg_std_error(&jerr);
						jpeg_create_compress(&cinfo);
						cinfo.dest = &dest;
						std::vector<unsigned char> jpeg_buffer;
						jpeg_buffer.resize( message->group->getVideoBufferSize() );
						cinfo.dest->next_output_byte = (JOCTET*)( jpeg_buffer.data() );
						cinfo.dest->free_in_buffer = message->group->getVideoBufferSize();
						cinfo.image_width = message->group->getFramebufferWidth();
						cinfo.image_height = message->group->getFramebufferHeight();
						cinfo.input_components = 4;
						cinfo.in_color_space = JCS_EXT_RGBX;
						jpeg_set_defaults(&cinfo);
						jpeg_set_quality(&cinfo, 100, false);
						jpeg_start_compress(&cinfo, TRUE);
						while (cinfo.next_scanline < cinfo.image_height)
						{
							JSAMPROW row_pointer[1];
							row_pointer[0] = & ((JSAMPROW)(message->buffer))[cinfo.next_scanline * cinfo.image_width * cinfo.input_components];
							(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
						}
						jpeg_finish_compress(&cinfo);
						int count = message->group->getVideoBufferSize() - cinfo.dest->free_in_buffer;
						jpeg_destroy_compress(&cinfo);
						
						
						using namespace boost::archive::iterators;
						std::stringstream os;
						typedef
							base64_from_binary
							<
								transform_width
								<
									const unsigned char *,
									6,
									8
								>
							> 
							base64_text; // compose all the above operations in to a new iterator

						std::copy(
							base64_text(jpeg_buffer.data()),
							base64_text(jpeg_buffer.data() + count),
							boost::archive::iterators::ostream_iterator<char>(os)
						);
						
						char header[] = "data:image/jpeg;base64,";
						int l = strlen(header);
						char* payload = (char*)malloc(os.str().length()+1+l);
						memcpy(payload,header,l);
						memcpy(&(payload[l]),os.str().c_str(),os.str().length()+1);
						ImageBufferContainer* answer = new ImageBufferContainer(SEND_JSON,(uint8_t*)payload,message->group,1,"",it->first);
						clientSendMessage( answer );
					}
				}
			}
			if (message->type == IMG_FORCE_EXIT)
				finish = 1;
			message->suicide();
		}
		usleep(1000);
	}	
}

URIImageConnector::~URIImageConnector()
{
}

