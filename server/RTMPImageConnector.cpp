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

#include "RTMPImageConnector.hpp"
#include <boost/preprocessor.hpp>
#include <pthread.h>
#include <inttypes.h>

RTMPImageConnector::RTMPImageConnector(std::string name, std::string apikey, std::string base_url) :
	group(NULL),
	name(name),
	apikey(apikey),
	base_url(base_url),
	heartbeat_image(NULL)
{
	showClient = false;
}

std::string RTMPImageConnector::getName()
{
	return std::string("RTMPImageConnector(") + name + std::string(")");
}

errorCode RTMPImageConnector::init(int minport,int maxport)
{
	return 0;
}

#define RTMP_LOAD_ELEMENT_OR_DIE(element) \
	if (success) \
	{ \
		element = gst_element_factory_make(BOOST_PP_STRINGIZE(element), NULL); \
		if(!element) \
		{ \
			fprintf(stderr,"RTMPImageConnector: Could not open " BOOST_PP_STRINGIZE(element)"\n"); \
			success = 0; \
		} \
	}

uint64_t RTMPImageConnector::getTicksMs()
{
	struct timespec ts;
	if (clock_gettime(CLOCK_MONOTONIC_RAW,&ts) == 0)
		return ts.tv_sec*1000 + ts.tv_nsec/1000000;
	return 0;
}

void RTMPImageConnector::addFrame( ImageBuffer* image, GstAppSrc* appsrc, InsituConnectorGroup* group)
{
	uint64_t val = gst_app_src_get_current_level_bytes( appsrc );
	if ( val == 0)
	{
		image->incref();
		GstBuffer *buffer = gst_buffer_new_wrapped_full (GstMemoryFlags(0), image->buffer, group->getVideoBufferSize(), 0, group->getVideoBufferSize(), (gpointer)(image), suicideNotify);
		if (gst_app_src_push_buffer( appsrc, buffer) != GST_FLOW_OK)
			printf("RTMPImageConnector: Error while sending buffer\n");
	}
}

void* RTMPImageConnector::heartbeatFunction(void* ptr)
{
	RTMPImageConnector* myself = (RTMPImageConnector*)ptr;
	while (myself->heartbeat_finish == false)
	{
		uint64_t now = getTicksMs();
		if (myself->heartbeat_image)
		{
			pthread_mutex_lock(&myself->heartbeat_mutex);
			if ( now - myself->heartbeat > ISAAC_MAX_HEARTBEAT )
			{
				addFrame( myself->heartbeat_image, (GstAppSrc*)(myself->appsrc), myself->group );
				myself->heartbeat = now;
			}
			pthread_mutex_unlock(&myself->heartbeat_mutex);
		}
		usleep(10000);
	}
}

errorCode RTMPImageConnector::run()
{
	pthread_mutex_init (&heartbeat_mutex, NULL);
	heartbeat_finish = false;
	heartbeat = getTicksMs();
	pthread_create(&heartbeat_thread,NULL,RTMPImageConnector::heartbeatFunction,this);
	int finish = 0;
	while (finish == 0)
	{
		ImageBufferContainer* message;
		while (message = clientGetMessage())
		{
			if (message->type == IMG_FORCE_EXIT)
				finish = 1;
			if (message->type == GROUP_FINISHED)
			{
				if (group == message->group)
				{
					group = NULL;
					gst_app_src_end_of_stream( (GstAppSrc*)appsrc );
					gst_element_set_state(pipeline, GST_STATE_NULL);
					gst_object_unref(GST_OBJECT(pipeline));
					printf("RTMPImageConnector: Closed Stream\n");
				}
			}
			if (message->type == UPDATE_BUFFER)
			{
				if (group == NULL)
				{
					//gst-launch-1.0 appsrc ! video/x-raw,… ! videoconvert ! capsfilter ! video/x-raw,format=I420 ! videorate ! video/x-raw,framerate=15/1 ! x264enc threads=2 bitrate=400 tune=zerolatency ! flvmux ! rtmpsink location=rtmp://live-fra.twitch.tv/app/$APIKEY
					gboolean success = 1;
					RTMP_LOAD_ELEMENT_OR_DIE(appsrc)
					if (success)
						g_object_set (G_OBJECT (appsrc), "caps",
							gst_caps_new_simple ("video/x-raw",
							"format", G_TYPE_STRING, "RGBx",
							"bpp", G_TYPE_INT, 32,
							"depth", G_TYPE_INT, 32,
							"width", G_TYPE_INT, message->group->getFramebufferWidth(),
							"height", G_TYPE_INT, message->group->getFramebufferHeight(),
							"framerate", GST_TYPE_FRACTION, 0, 1,
							NULL), NULL);
					if (success)
						g_object_set (G_OBJECT (appsrc),
							"do-timestamp", 1,
							"min-percent", 0,
							"min-latency", 0,
							"emit-signals", 0,
							"is-live", 1,
							"format", GST_FORMAT_TIME,
							NULL);
					RTMP_LOAD_ELEMENT_OR_DIE(videorate)
					RTMP_LOAD_ELEMENT_OR_DIE(capsfilter)
					videorate_capsfilter = capsfilter;
					if (success)
						g_object_set (G_OBJECT (videorate_capsfilter), "caps",
							gst_caps_new_simple ("video/x-raw",
							"framerate", GST_TYPE_FRACTION, 15, 1,
							NULL), NULL);
					RTMP_LOAD_ELEMENT_OR_DIE(videoconvert)
					RTMP_LOAD_ELEMENT_OR_DIE(capsfilter)
					if (success)
						g_object_set (G_OBJECT (capsfilter), "caps",
							gst_caps_new_simple ("video/x-raw",
							"format", G_TYPE_STRING, "I420",
							NULL), NULL);
					RTMP_LOAD_ELEMENT_OR_DIE(x264enc)
					if (success)
						g_object_set (G_OBJECT (x264enc),
							"tune", 0x00000004,
							"psy-tune", 2,
							"speed-preset", 1,
							"bitrate", 400,
							"threads", 2,
							"byte-stream", 1,
							NULL);
					RTMP_LOAD_ELEMENT_OR_DIE(flvmux)
					if (success)
						g_object_set (G_OBJECT (flvmux),
							"streamable", 1,
							NULL);
					RTMP_LOAD_ELEMENT_OR_DIE(rtmpsink)
					char location[512];
					sprintf(location,"rtmp://%s/%s",base_url.c_str(),apikey.c_str());
					if (success)
						g_object_set(G_OBJECT(rtmpsink),
							"location", location, NULL);
					if (success)
					{
						pipeline = gst_pipeline_new( NULL );
						bin = gst_bin_new( NULL );
						gst_bin_add_many(GST_BIN(bin), appsrc, videoconvert, capsfilter, x264enc, flvmux, rtmpsink, NULL);
						gst_bin_add(GST_BIN(pipeline), bin);
						success = gst_element_link_many(appsrc, videoconvert, capsfilter, x264enc, flvmux, rtmpsink, NULL);
						if ( !success )
							fprintf(stderr,"RTMPImageConnector: Could not link elements for rtmp stream.\n");
						if (gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
							printf("RTMPImageConnector: Could not play stream!\n");
						else
							printf("RTMPImageConnector: Openend H264 Stream\n");
						group = message->group;
					}
				}
				if (group == message->group) //We show always the very first group
				{
					pthread_mutex_lock(&heartbeat_mutex);
					addFrame( message->image, (GstAppSrc*)appsrc, group );
					heartbeat = getTicksMs();
					if (heartbeat_image) //Releasing old frame
						heartbeat_image->suicide();
					heartbeat_image = message->image;
					heartbeat_image->incref();
					pthread_mutex_unlock(&heartbeat_mutex);
				}
			}
			clientSendMessage( message );
		}
		usleep(1000);
	}
	heartbeat_finish = true;
	pthread_join(heartbeat_thread,NULL);
	pthread_mutex_destroy(&heartbeat_mutex);
}
