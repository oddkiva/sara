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
struct AVFormatContext;
struct AVFrame;


namespace DO { namespace Sara {

  class DO_SARA_EXPORT VideoStream : public std::streambuf
  {
  public:
    VideoStream();

    VideoStream(const VideoStream&) = delete;

    VideoStream(const std::string& file_path);

    ~VideoStream();

    VideoStream& operator=(const VideoStream&) = delete;

    int width() const;

    int height() const;

    Vector2i sizes() const
    {
      return Vector2i{ width(), height() };
    }

    void open(const std::string& file_path);

    void close();

    void seek(std::size_t frame_pos);

    bool read(ImageView<Rgb8>& video_frame);

    friend inline VideoStream& operator>>(VideoStream& video_stream,
                                          ImageView<Rgb8>& video_frame)
    {
      if (!video_stream.read(video_frame))
        video_frame = Image<Rgb8>();
      return video_stream;
    }

  private:
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 9, 100)
    static bool _registered_all_codecs;
#endif

    AVFormatContext *_video_format_context = nullptr;
    int _video_stream = -1;
    AVCodec *_video_codec = nullptr;
    AVCodecContext *_video_codec_context = nullptr;
    AVFrame *_video_frame = nullptr;
    size_t _video_frame_pos = std::numeric_limits<size_t>::max();
  };

} /* namespace Sara */
} /* namespace DO */
