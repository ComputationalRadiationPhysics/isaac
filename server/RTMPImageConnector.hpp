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
#include "RTPImageConnector.hpp"
#include "Runable.hpp"
#include <gst/gst.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#define ISAAC_MAX_HEARTBEAT 200

class RTMPImageConnector : public ImageConnector
{
	public:
		RTMPImageConnector( std::string name, std::string apikey, std::string base_url );
		errorCode init(int minport,int maxport);
		errorCode run();
		std::string getName();
		static void* heartbeatFunction(void* ptr);
		static uint64_t getTicksMs();
		static void addFrame( ImageBuffer* image, GstAppSrc* appsrc, InsituConnectorGroup* group);
	private:
		InsituConnectorGroup* group;
		std::string name;
		std::string apikey;
		std::string base_url;
		GstElement *appsrc;
		GstElement *videoconvert;
		GstElement *capsfilter;
		GstElement *videorate_capsfilter;
		GstElement *videorate;
		GstElement *x264enc;
		GstElement *flvmux;
		GstElement *rtmpsink;
		GstElement *pipeline;
		GstElement *bin;
		pthread_mutex_t heartbeat_mutex;
		pthread_t heartbeat_thread;
		uint64_t heartbeat;
		ImageBuffer* heartbeat_image;
		volatile bool heartbeat_finish;
};
