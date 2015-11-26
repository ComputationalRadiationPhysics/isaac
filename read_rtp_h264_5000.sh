#!/bin/sh
gst-launch-1.0 -v udpsrc port=5000 !  application/x-rtp, encoding-name=H264,payload=96 !  rtph264depay ! avdec_h264 !  videoconvert ! autovideosink
