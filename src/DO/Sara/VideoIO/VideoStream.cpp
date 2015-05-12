// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define _USE_MATH_DEFINES

extern "C" {
# include <libavcodec/avcodec.h>
# include <libavformat/avformat.h>
# include <libavformat/avio.h>
# include <libavutil/file.h>
}

#include "VideoStream.hpp"


namespace DO {

  inline static Yuv8 get_yuv_pixel(const AVFrame *frame, int x, int y)
  {
    Yuv8 yuv;
    yuv(0) = frame->data[0][y*frame->linesize[0] + x];
    yuv(1) = frame->data[1][y / 2 * frame->linesize[1] + x / 2];
    yuv(2) = frame->data[2][y / 2 * frame->linesize[2] + x / 2];
    return yuv;
  }

  inline static unsigned char clamp(int value)
  {
    if (value < 0)
      return 0;
    if (value > 255)
      return 255;
    return value;
  }

  inline static Rgb8 convert(const Yuv8& yuv)
  {
    Rgb8 rgb;
    int C = yuv(0) - 16;
    int D = yuv(1) - 128;
    int E = yuv(2) - 128;
    rgb(0) = clamp((298 * C + 409 * E + 128) >> 8);
    rgb(1) = clamp((298 * C - 100 * D - 208 * E + 128) >> 8);
    rgb(2) = clamp((298 * C + 516 * D + 128) >> 8);
    return rgb;
  }

} /* namespace DO */


namespace DO {

  bool VideoStream::_registered_all_codecs = false;

  VideoStream::VideoStream()
  {
    if (!_registered_all_codecs)
    {
      av_register_all();
      _registered_all_codecs = true;
    }
  }

  VideoStream::VideoStream(const std::string& file_path)
    : VideoStream()
  {
    open(file_path);
  }

  VideoStream::~VideoStream()
  {
    close();
  }

  void
  VideoStream::open(const std::string& file_path)
  {
    // Read the video file.
    if (avformat_open_input(&_video_format_context, file_path.c_str(),
                            nullptr, nullptr) != 0)
      throw std::runtime_error("Could not open video file!");

    // Read the video stream metadata.
    if (avformat_find_stream_info(_video_format_context, nullptr) != 0)
      throw std::runtime_error("Could not retrieve video stream info!");
    av_dump_format(_video_format_context, 0, file_path.c_str(), 0);

    // Retrieve video stream.
    _video_stream = -1;
    for (unsigned i = 0; i != _video_format_context->nb_streams; ++i)
    {
      if (_video_format_context->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
      {
        _video_stream = i;
        break;
      }
    }
    if (_video_stream == -1)
      throw std::runtime_error("Could not retrieve video stream!");

    // Retrieve the video codec context.
    _video_codec_context = _video_format_context->streams[_video_stream]->codec;

    // Retrieve the video codec.
    _video_codec = avcodec_find_decoder(_video_codec_context->codec_id);
    if (!_video_codec)
      throw std::runtime_error("Could not find supported codec!");

    // Open the video codec.
    if (avcodec_open2(_video_codec_context, _video_codec, nullptr) < 0)
      throw std::runtime_error("Could not open video codec!");

    // Allocate video frame.
    _video_frame = av_frame_alloc();
    if (!_video_frame)
      throw std::runtime_error("Could not allocate video frame!");

    _video_frame_pos = 0;
  }

  void
  VideoStream::close()
  {
    if (_video_frame)
    {
      av_frame_free(&_video_frame);
      _video_frame = nullptr;
      _video_frame_pos = std::numeric_limits<size_t>::max();
    }

    if (_video_codec_context)
    {
      avcodec_close(_video_codec_context);
      avcodec_free_context(&_video_codec_context);
      _video_codec_context = nullptr;
      _video_codec = nullptr;
    }

    if (_video_format_context)
      _video_format_context = nullptr;
  }

  void
  VideoStream::seek(std::size_t frame_pos)
  {
    av_seek_frame(_video_format_context, _video_stream, frame_pos,
                  AVSEEK_FLAG_BACKWARD);
  }

  bool
  VideoStream::read(Image<Rgb8>& video_frame)
  {
    AVPacket _video_packet;
    int length, got_video_frame;

    while (av_read_frame(_video_format_context, &_video_packet) >= 0)
    {
      length = avcodec_decode_video2(_video_codec_context, _video_frame,
        &got_video_frame, &_video_packet);

      if (length < 0)
        return false;

      if (got_video_frame)
      {
        int w = _video_codec_context->width;
        int h = _video_codec_context->height;

        if (video_frame.width() != w || video_frame.height() != h)
          video_frame.resize(w, h);

        for (int y = 0; y < h; ++y)
        {
          for (int x = 0; x < w; ++x)
          {
            Yuv8 yuv = get_yuv_pixel(_video_frame, x, y);
            video_frame(x, y) = DO::convert(yuv);
          }
        }

        ++_video_frame_pos;
        return true;
      }
    }

    return false;
  }


} /* namespace DO */