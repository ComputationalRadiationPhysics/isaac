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

class TwitchImageConnector : public ImageConnector
{
	public:
		TwitchImageConnector(std::string apikey);
		errorCode init(int minport,int maxport);
		errorCode run();
		std::string getName();
	private:
		InsituConnectorGroup* group;
		std::string apikey;
		GstElement *appsrc;
		GstElement *videoconvert;
		GstElement *capsfilter;
		GstElement *x264enc;
		GstElement *flvmux;
		GstElement *rtmpsink;
		GstElement *pipeline;
		GstElement *bin;
};
