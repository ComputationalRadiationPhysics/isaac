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

#pragma once

#include "ImageConnector.hpp"
#include "Runable.hpp"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <vector>

class RTPImageConnector : public ImageConnector
{
public:
    RTPImageConnector(std::string url, bool zerolatency = false, bool raw = false);
    ~RTPImageConnector();
    errorCode init(int minport, int maxport);
    errorCode run();
    std::string getName();

private:
    typedef struct
    {
        bool is_used;
        GstElement* appsrc;
        GstElement* videoconvert;
        GstElement* x264enc;
        GstElement* rtph264pay;
        GstElement* capsfilter;
        GstElement* jpegenc;
        GstElement* rtpjpegpay;
        GstElement* udpsink;
        GstElement* pipeline;
        GstElement* bin;
        InsituConnectorGroup* group;
        std::string url;
        void* ref;
    } tStream;
    int minport;
    int maxport;
    std::vector<tStream> streams;
    std::string url;
    bool zerolatency;
    bool raw;
};

void suicideNotify(gpointer data);
