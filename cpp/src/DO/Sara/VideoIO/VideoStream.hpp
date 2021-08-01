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

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image.hpp>


struct AVCodec;
struct AVCodecContext;
struct AVCodecParameters;
struct AVFormatContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;


namespace DO { namespace Sara {

  //! @defgroup VideoIO Video I/O
  //! @{

  class DO_SARA_EXPORT VideoStream
  {
  public:
    VideoStream();

    VideoStream(const VideoStream&) = delete;

    VideoStream(const std::string& file_path);

    ~VideoStream();

    auto open(const std::string& file_path) -> void;

    auto close() -> void;

    auto read() -> bool;

    auto frame() const -> ImageView<Rgb8>;

    auto seek(std::size_t frame_pos) -> void;

    auto frame_rate() const -> float;

    auto width() const -> int;

    auto height() const -> int;

    auto sizes() const -> Vector2i
    {
      return Vector2i{width(), height()};
    }

    friend inline VideoStream& operator>>(VideoStream& video_stream,
                                          ImageView<Rgb8>& video_frame)
    {
      if (!video_stream.read())
        video_frame = {};
      else
        video_frame = video_stream.frame();
      return video_stream;
    }

  private:
    auto decode(AVPacket* pkt) -> bool;

  private:
    static bool _registered_all_codecs;

    // Hardware acceleration type.
    static int _hw_device_type;

    std::vector<std::uint8_t> _buffer;

    // FFmpeg internals.
    int _video_stream_index = -1;
    const AVCodecParameters* _video_codec_params = nullptr;
    const AVCodec* _video_codec = nullptr;
    AVFormatContext* _video_format_context = nullptr;
    AVCodecContext* _video_codec_context = nullptr;
    AVFrame* _device_picture = nullptr;
    AVFrame* _picture = nullptr;
    AVPacket* _pkt = nullptr;

    SwsContext *_sws_context = nullptr;
    AVFrame* _picture_rgb = nullptr;

    bool _end_of_stream{true};
    int _got_frame{};
    int _i{};
  };

  //! @}

}}  // namespace DO::Sara
