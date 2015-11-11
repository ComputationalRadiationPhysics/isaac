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

#include "ImageConnector.hpp"
#include "Runable.hpp"
#include <gst/gst.h>

class RTPImageConnector : public ImageConnector
{
	public:
		RTPImageConnector();
		~RTPImageConnector();
		errorCode init(int port);
		errorCode run();
		errorCode run_gstreamer();
		std::string getName();
	private:
		static void* run_gstreamer_wrapper(void* ptr);
		GstElement *videotestsrc;
		GstElement *x264enc;
		GstElement *rtph264pay;
		GstElement *udpsink;
		GstElement *pipeline;
		GstElement *bin;
		GMainLoop *loop;
		pthread_t loop_thread;
};
