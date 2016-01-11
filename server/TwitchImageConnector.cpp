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

#include "TwitchImageConnector.hpp"
#include <boost/preprocessor.hpp>
#include <pthread.h>
#include <inttypes.h>

TwitchImageConnector::TwitchImageConnector(std::string apikey) :
	group(NULL),
	apikey(apikey)
{
	showClient = false;
}

std::string TwitchImageConnector::getName()
{
	return "TwitchImageConnector";
}

errorCode TwitchImageConnector::init(int minport,int maxport)
{
	return 0;
}

#define TWI_LOAD_ELEMENT_OR_DIE(element) \
	element = gst_element_factory_make(BOOST_PP_STRINGIZE(element), NULL); \
	if(!element) \
	{ \
		fprintf(stderr,"Could not open " BOOST_PP_STRINGIZE(element)"\n"); \
		return 1; \
	}

errorCode TwitchImageConnector::run()
{
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
					printf("TwitchImageConnector: Closed Stream\n");
				}
			}
			if (message->type == UPDATE_BUFFER)
			{
				if (group == NULL)
				{
					group = message->group;
					//gst-launch-1.0 appsrc | video/x-raw,width=1024,height=768 ! videoconvert | x264enc threads=2 bitrate=400 tune=zerolatency ! flvmux ! rtmpsink location=rtmp://live-fra.twitch.tv/app/$APIKEY
					TWI_LOAD_ELEMENT_OR_DIE(appsrc)
					g_object_set (G_OBJECT (appsrc), "caps",
						gst_caps_new_simple ("video/x-raw",
						"format", G_TYPE_STRING, "RGBx",
						"bpp", G_TYPE_INT, 32,
						"depth", G_TYPE_INT, 32,
						"width", G_TYPE_INT, message->group->getFramebufferWidth(),
						"height", G_TYPE_INT, message->group->getFramebufferHeight(),
						"framerate", GST_TYPE_FRACTION, 0, 1,
						NULL), NULL);
					g_object_set (G_OBJECT (appsrc),
						"do-timestamp", 1,
						"min-percent", 0,
						"emit-signals", 0,
						"format", GST_FORMAT_TIME, NULL);
					TWI_LOAD_ELEMENT_OR_DIE(videoconvert)
					TWI_LOAD_ELEMENT_OR_DIE(capsfilter)
					g_object_set (G_OBJECT (capsfilter), "caps",
						gst_caps_new_simple ("video/x-raw",
						"format", G_TYPE_STRING, "I420",
						NULL), NULL);
					TWI_LOAD_ELEMENT_OR_DIE(x264enc)
					g_object_set (G_OBJECT (x264enc),
						"tune", 0x00000004,
						"psy-tune", 2,
						"speed-preset", 1,
						"bitrate", 200,
						"threads", 2,
						"byte-stream", 1,
						NULL);
					TWI_LOAD_ELEMENT_OR_DIE(flvmux)
					TWI_LOAD_ELEMENT_OR_DIE(rtmpsink)
					char location[512];
					sprintf(location,"rtmp://live-fra.twitch.tv/app/%s",apikey.c_str());
					g_object_set(G_OBJECT(rtmpsink),
						"location", location, NULL);
					pipeline = gst_pipeline_new( NULL );
					bin = gst_bin_new( NULL );
					gboolean success = 0;
					gst_bin_add_many(GST_BIN(bin), appsrc, videoconvert, capsfilter, x264enc, flvmux, rtmpsink, NULL);
					gst_bin_add(GST_BIN(pipeline), bin);
					success = gst_element_link_many(appsrc, videoconvert, capsfilter, x264enc, flvmux, rtmpsink, NULL);
					if ( !success )
						fprintf(stderr,"TwitchImageConnector: Could not link elements for twitch stream.\n");
					if (gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE)
						printf("TwitchImageConnector: Could not play stream!\n");
					else
						printf("TwitchImageConnector: Openend H264 Stream\n");
				}
				if (group == message->group) //We show always the very first group
				{
					uint64_t val = gst_app_src_get_current_level_bytes( (GstAppSrc*)appsrc );
					if ( val == 0)
					{
						message->incref();
						GstBuffer *buffer = gst_buffer_new_wrapped_full (GstMemoryFlags(0), message->buffer, group->getVideoBufferSize(), 0, group->getVideoBufferSize(), (gpointer)message, suicideNotify);
						if (gst_app_src_push_buffer( (GstAppSrc*)appsrc, buffer) != GST_FLOW_OK)
							printf("TwitchImageConnector: Error while sending buffer\n");
					}
				}
			}
			message->suicide();
		}
		usleep(1000);
	}
}
