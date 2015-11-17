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

#include "RTPImageConnector.hpp"
#include <boost/preprocessor.hpp>
#include <pthread.h>
#include <inttypes.h>

RTPImageConnector::RTPImageConnector(std::string url,bool zerolatency)
{
	this->url = url;
	this->zerolatency = zerolatency;
}

std::string RTPImageConnector::getName()
{
	return "RTPImageConnector";
}

#define GST_LOAD_ELEMENT_OR_DIE(stream,element) \
	stream.element = gst_element_factory_make(BOOST_PP_STRINGIZE(element), NULL); \
	if(!stream.element) \
	{ \
		fprintf(stderr,"Could not open " BOOST_PP_STRINGIZE(element)"\n"); \
		return 1; \
	}

errorCode RTPImageConnector::init(int minport,int maxport)
{
	this->minport = minport;
	this->maxport = maxport;
	tStream defaultstream;
	defaultstream.is_used = false;
	streams.resize(maxport-minport,defaultstream);
	gst_init(NULL,NULL);
	return 0;
}

void suicideNotify(gpointer data)
{
	ImageBufferContainer* message = (ImageBufferContainer*)data;
	message->suicide();
}

errorCode RTPImageConnector::run()
{
	uint64_t finish = 0;
	while(finish == 0)
	{
		ImageBufferContainer* message;
		while(message = clientGetMessage())
		{
			if (message->type == GROUP_OBSERVED)
			{
				int nr;
				for (nr = 0; nr < streams.size(); nr++)
					if (!streams[nr].is_used)
						break;
				if (nr < streams.size())
				{
					//gst-launch-1.0 appsrc ! videoconf ! x264enc ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000
					GST_LOAD_ELEMENT_OR_DIE(streams[nr],appsrc)
					g_object_set (G_OBJECT (streams[nr].appsrc), "caps",
						gst_caps_new_simple ("video/x-raw",
						"format", G_TYPE_STRING, "RGBA",
						"width", G_TYPE_INT, message->group->getFramebufferWidth(),
						"height", G_TYPE_INT, message->group->getFramebufferHeight(),
						"framerate", GST_TYPE_FRACTION, 0, 1,
						NULL), NULL);
					g_object_set (G_OBJECT (streams[nr].appsrc),
						"do-timestamp", 1,
						"format", GST_FORMAT_TIME, NULL);
					GST_LOAD_ELEMENT_OR_DIE(streams[nr],videoconvert)
					GST_LOAD_ELEMENT_OR_DIE(streams[nr],x264enc)
					g_object_set (G_OBJECT (streams[nr].x264enc),
						"tune", zerolatency ? 0x00000004 : 0x00000000,
						"psy-tune", 2,
						"speed-preset", 1,
						"bitrate", 3000,
						"threads", 2,
						"byte-stream", 1,  NULL);
					GST_LOAD_ELEMENT_OR_DIE(streams[nr],rtph264pay)
					g_object_set(G_OBJECT(streams[nr].rtph264pay),
						"config-interval", 10,
						"pt", 96, NULL);
					GST_LOAD_ELEMENT_OR_DIE(streams[nr],udpsink)
					g_object_set(G_OBJECT(streams[nr].udpsink),
						"host", message->target.c_str(),
						"port", nr+minport, NULL);
					
					streams[nr].pipeline = gst_pipeline_new( NULL );
					streams[nr].bin = gst_bin_new( NULL );
					gst_bin_add_many(GST_BIN(streams[nr].bin), streams[nr].appsrc, streams[nr].videoconvert, streams[nr].x264enc, streams[nr].rtph264pay, streams[nr].udpsink, NULL);
					gst_bin_add(GST_BIN(streams[nr].pipeline), streams[nr].bin);

					if(!gst_element_link_many(streams[nr].appsrc, streams[nr].videoconvert, streams[nr].x264enc, streams[nr].rtph264pay, streams[nr].udpsink, NULL))
						fprintf(stderr,"RTPImageConnector: Could not link elements for rtp stream\n");
					else
					{
						streams[nr].is_used = true;
						streams[nr].group = message->group;
						streams[nr].ref = message->reference;
						streams[nr].url = message->target;
						streams[nr].last = 0;
						char* register_message = (char*)malloc(128);
						sprintf(register_message,"v=0\nm=video %i RTP/AVP 96\nc=IN IP4 %s\na=rtpmap:96 H264/90000\n",nr+minport,url.c_str());
						clientSendMessage(new ImageBufferContainer(REGISTER_STREAM,(uint8_t*)register_message,message->group,1,message->target,message->reference));
						printf("RTIPImageConnector: Openend Stream at port %i\n",minport+nr);
						gst_element_set_state(GST_ELEMENT(streams[nr].pipeline), GST_STATE_PLAYING);
					}
				}
				else
					fprintf(stderr,"RTPImageConnector: No free port!\n");
			}
			if (message->type == GROUP_OBSERVED_STOPPED || message->type == GROUP_FINISHED)
			{
				int nr;
				for (nr = 0; nr < streams.size(); nr++)
					if ((message->type == GROUP_OBSERVED_STOPPED && streams[nr].ref == message->reference) ||
						(message->type == GROUP_FINISHED && streams[nr].group == message->group))
						break;
				if (nr < streams.size() && streams[nr].is_used)
				{
					gst_app_src_end_of_stream( (GstAppSrc*)streams[nr].appsrc );
					gst_element_set_state(streams[nr].pipeline, GST_STATE_NULL);
					gst_object_unref(GST_OBJECT(streams[nr].pipeline));
					streams[nr].is_used = false;
					printf("RTIPImageConnector: Closed Stream\n");
				}
			}
			if (message->type == UPDATE_BUFFER)
			{
				uint64_t now = getTicksMs();
				int nr;
				for (nr = 0; nr < streams.size(); nr++)
					if (streams[nr].is_used && streams[nr].group == message->group && now - 30 > streams[nr].last)
					{
						message->incref();
						GstBuffer *buffer = gst_buffer_new_wrapped_full (GstMemoryFlags(0), message->buffer, streams[nr].group->getVideoBufferSize(), 0, streams[nr].group->getVideoBufferSize(), (gpointer)message, suicideNotify);
						gst_app_src_push_buffer( (GstAppSrc*)streams[nr].appsrc, buffer);
						streams[nr].last = now;
					}
			}
			if (message->type == IMG_FORCE_EXIT)
				finish = 1;
			message->suicide();
		}
		usleep(1000);
	}	
	int nr;
	for (nr = 0; nr < streams.size(); nr++)
		if (streams[nr].is_used)
		{
			gst_app_src_end_of_stream( (GstAppSrc*)streams[nr].appsrc );
			gst_element_set_state(streams[nr].pipeline, GST_STATE_NULL);
			gst_object_unref(GST_OBJECT(streams[nr].pipeline));
			streams[nr].is_used = false;
		}
}

uint64_t RTPImageConnector::getTicksMs()
{
	struct timespec ts;
	if (clock_gettime(CLOCK_MONOTONIC_RAW,&ts) == 0)
		return ts.tv_sec*1000 + ts.tv_nsec/1000000;
	return 0;
}


RTPImageConnector::~RTPImageConnector()
{
}

