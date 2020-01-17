// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/file.h>
#include <libavutil/imgutils.h>
}

#include "VideoStream.hpp"
#include <DO/Sara/Core/DebugUtilities.hpp>

#include <cstdio>
#include <cstdlib>


namespace DO::Sara {

  inline static Yuv8 get_yuv_pixel(const AVFrame* frame, int x, int y)
  {
    Yuv8 yuv;
    yuv(0) = frame->data[0][y * frame->linesize[0] + x];
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

}  // namespace DO::Sara


namespace DO::Sara {

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

  auto VideoStream::open(const std::string& file_path) -> void
  {
    // Read the video file.
    if (avformat_open_input(&_video_format_context, file_path.c_str(), nullptr,
                            nullptr) < 0)
      throw std::runtime_error("Could not open video file!");

    // Read the video stream metadata.
    if (avformat_find_stream_info(_video_format_context, nullptr) != 0)
      throw std::runtime_error("Could not get video stream info!");

    // Find the video decoder.
    _video_stream_index = av_find_best_stream(
        _video_format_context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);

    _video_codec_params =
        _video_format_context->streams[_video_stream_index]->codecpar;

    _video_codec = avcodec_find_decoder(_video_codec_params->codec_id);
    if (_video_codec == nullptr)
      throw std::runtime_error{"Could not find video decoder!"};

    // Create a decoder context.
    _video_codec_context = avcodec_alloc_context3(_video_codec);
    if (_video_codec_context == nullptr)
      throw std::runtime_error{"Could not allocate video decoder context!"};

    if (avcodec_parameters_to_context(_video_codec_context,
                                      _video_codec_params))
      throw std::runtime_error{"Could not copy video decoder context!"};

    // Open it.
    if (avcodec_open2(_video_codec_context, _video_codec, nullptr) < 0)
      throw std::runtime_error{"Could not open video decoder!"};

    // Allocate buffer to read the video frame.
    if (_picture == nullptr)
      _picture = av_frame_alloc();
    if (_picture == nullptr)
      throw std::runtime_error{"Could not allocate video frame!"};

    // Initialize a video packet.
    if (_pkt == nullptr)
      _pkt = av_packet_alloc();
    if (_pkt == nullptr)
      throw std::runtime_error("Could not allocate video packet!");
    av_init_packet(_pkt);

    SARA_DEBUG << "#[VideoStream] sizes = " << sizes().transpose() << std::endl;
    SARA_DEBUG
        << "#[VideoStream] time base = " << _video_stream_index << ": "  //
        << _video_format_context->streams[_video_stream_index]->time_base.num
        << "/"
        << _video_format_context->streams[_video_stream_index]->time_base.den
        << std::endl;

    if (_video_codec_context->pix_fmt != AV_PIX_FMT_YUV420P)
      throw std::runtime_error{"VideoStream error: unsupported pixel format! "
                               "Extend the implementation please!"};

    _end_of_stream = false;
  }

  auto VideoStream::close() -> void
  {
    // Flush the decoder (draining mode.)
    this->decode(_video_codec_context, _picture, nullptr);

    // Free the data structures.
    av_packet_unref(_pkt);
    av_frame_free(&_picture);
    avcodec_close(_video_codec_context);
    avformat_close_input(&_video_format_context);
    avcodec_free_context(&_video_codec_context);
    av_packet_free(&_pkt);

    // No need to deallocate these.
    _video_stream_index = -1;
    _video_codec_params = nullptr;
    _video_codec = nullptr;

    _end_of_stream = true;
  }

  auto VideoStream::read(ImageView<Rgb8>& video_frame) -> bool
  {
    if (video_frame.sizes() != sizes())
      throw std::domain_error{
          "Video frame sizes and video stream sizes are not equal!"};
    do
    {
      if (!_end_of_stream)
        if (av_read_frame(_video_format_context, _pkt) < 0)
          _end_of_stream = true;

      if (_end_of_stream)
      {
        _pkt->data = nullptr;
        _pkt->size = 0;
        return false;
      }

      if (_pkt->stream_index == _video_stream_index || _end_of_stream)
      {
        _got_frame = 0;

        if (_pkt->pts == AV_NOPTS_VALUE)
          _pkt->pts = _pkt->dts = _i;

        // Decompress the video frame.
#ifdef OLD_WAY
        const auto bytes_used = avcodec_decode_video2(
            _video_codec_context, _picture, &_got_frame, _pkt);
        if (bytes_used < 0)
          throw std::runtime_error{"Could not decode video frame!"};

        if (_got_frame)
        {
          const auto w = width();
          const auto h = height();
          auto video_frame_data = video_frame.data();

          for (auto y = 0; y < h; ++y)
          {
            for (auto x = 0; x < w; ++x)
            {
              auto yuv = get_yuv_pixel(_picture, x, y);
              *video_frame_data = Sara::convert(yuv);
              ++video_frame_data;
            }
          }
          return true;
        }
#else
        _got_frame = decode(_video_codec_context, _picture, _pkt);

        if (_got_frame)
        {
          av_packet_unref(_pkt);
          av_init_packet(_pkt);

          const auto w = width();
          const auto h = height();
          auto video_frame_data = video_frame.data();

          for (auto y = 0; y < h; ++y)
          {
            for (auto x = 0; x < w; ++x)
            {
              auto yuv = get_yuv_pixel(_picture, x, y);
              *video_frame_data = Sara::convert(yuv);
              ++video_frame_data;
            }
          }

          return true;
        }
#endif
      }
      ++_i;
    } while (!_end_of_stream || _got_frame);

    return false;
  }

  auto VideoStream::seek(std::size_t frame_pos) -> void
  {
    av_seek_frame(_video_format_context, _video_stream_index, frame_pos,
                  AVSEEK_FLAG_BACKWARD);
  }

  auto VideoStream::decode(AVCodecContext* dec_ctx, AVFrame* frame,
                           AVPacket* pkt) -> bool
  {
    auto ret = int{};

    // Transfer raw compressed video data to the packet.
    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0)
      throw std::runtime_error{"Error sending a packet for decoding!"};

    // Decode the compressed video data into an uncompressed video frame.
    ret = avcodec_receive_frame(dec_ctx, frame);

    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return false;

    if (ret < 0)
      throw std::runtime_error{"Error during decoding!"};

    return true;
  }

  auto VideoStream::frame_rate() const -> float
  {
    return static_cast<float>(_video_codec_context->framerate.num) /
           _video_codec_context->framerate.den;
  }

  auto VideoStream::width() const -> int
  {
    return _video_codec_context->width;
  }

  auto VideoStream::height() const -> int
  {
    return _video_codec_context->height;
  }


}  // namespace DO::Sara
