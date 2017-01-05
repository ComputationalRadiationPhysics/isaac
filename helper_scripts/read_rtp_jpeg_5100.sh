#!/bin/sh
gst-launch-1.0 -v udpsrc port=5100 !  application/x-rtp, encoding-name=JPEG,payload=96 !  rtpjpegdepay !  jpegdec !  autovideosink
