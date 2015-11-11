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

RTPImageConnector::RTPImageConnector()
{
}

std::string RTPImageConnector::getName()
{
	return "RTPImageConnector(using gStreamer)";
}

#define GST_LOAD_ELEMENT_OR_DIE(element) \
	element = gst_element_factory_make(BOOST_PP_STRINGIZE(element), NULL); \
	if(!element) \
	{ \
		fprintf(stderr,"Could not open "BOOST_PP_STRINGIZE(element)"\n"); \
		return 1; \
	}

void* RTPImageConnector::run_gstreamer_wrapper(void* ptr)
{
	RTPImageConnector* myself =(RTPImageConnector*)ptr;
	myself->run_gstreamer();
}

errorCode RTPImageConnector::init(int port)
{
	gst_init(NULL,NULL);
	//gst-launch-1.0 videotestsrc ! x264enc ! rtph264pay config-interval=10 pt=96 ! udpsink host=127.0.0.1 port=5000
	GST_LOAD_ELEMENT_OR_DIE(videotestsrc)
	GST_LOAD_ELEMENT_OR_DIE(x264enc)
	GST_LOAD_ELEMENT_OR_DIE(rtph264pay)
	g_object_set(G_OBJECT(rtph264pay), "config-interval", 10, "pt", 96, NULL);
	GST_LOAD_ELEMENT_OR_DIE(udpsink)
	g_object_set(G_OBJECT(udpsink), "host", "127.0.0.1", "port", 5000, NULL);
	
	pipeline = gst_pipeline_new( NULL );
	bin = gst_bin_new( NULL );
	gst_bin_add_many(GST_BIN(bin), videotestsrc, x264enc, rtph264pay, udpsink, NULL);
	gst_bin_add(GST_BIN(pipeline), bin);

	if(!gst_element_link_many(videotestsrc, x264enc, rtph264pay, udpsink, NULL))
	{
		fprintf(stderr,"Could not link elements for rtp stream\n");
		return 1;
	}
	
	gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_PLAYING);
	loop = g_main_loop_new(NULL, FALSE);
	pthread_create(&loop_thread,NULL,RTPImageConnector::run_gstreamer_wrapper,this);
	return 0;
}

errorCode RTPImageConnector::run_gstreamer()
{
	g_main_loop_run(loop);
}

errorCode RTPImageConnector::run()
{
	int finish = 0;
	while(finish == 0)
	{
		ImageBufferContainer* message;
		while(message = clientGetMessage())
		{
			if(message->type == IMG_FORCE_EXIT)
				finish = 1;
			message->suicide();
		}
		usleep(1000);
	}	
}

RTPImageConnector::~RTPImageConnector()
{
	g_main_loop_quit(loop);
	pthread_join(loop_thread,NULL);
	gst_element_set_state(pipeline, GST_STATE_NULL);
	gst_object_unref(GST_OBJECT(pipeline));
	gst_object_unref(GST_OBJECT(bin));
	gst_object_unref(GST_OBJECT(udpsink));
	gst_object_unref(GST_OBJECT(rtph264pay));
	gst_object_unref(GST_OBJECT(x264enc));	
	gst_object_unref(GST_OBJECT(videotestsrc));
}

