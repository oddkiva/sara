// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef VIDEOIO_VIDEOSTREAM_HPP
#define VIDEOIO_VIDEOSTREAM_HPP

#include <DO/Sara/Core/Image.hpp>


struct AVCodec;
struct AVCodecContext;
struct AVFormatContext;
struct AVFrame;


namespace DO {

  class VideoStream : public std::streambuf
  {
  public:
    VideoStream();

    VideoStream(const VideoStream&) = delete;

    VideoStream(const std::string& file_path);

    ~VideoStream();

    VideoStream& operator=(const VideoStream&) = delete;

    void open(const std::string& file_path);

    void close();

    void seek(std::size_t frame_pos);

    bool read(Image<Rgb8>& video_frame);

    friend inline VideoStream& operator>>(VideoStream& video_stream,
                                          Image<Rgb8>& video_frame)
    {
      if (!video_stream.read(video_frame))
        video_frame = Image<Rgb8>();
      return video_stream;
    }

  private:
    static bool _registered_all_codecs;

    AVFormatContext *_video_format_context = nullptr;
    int _video_stream = -1;
    AVCodec *_video_codec = nullptr;
    AVCodecContext *_video_codec_context = nullptr;
    AVFrame *_video_frame = nullptr;
    size_t _video_frame_pos = std::numeric_limits<size_t>::max();
  };

}


#endif /* VIDEOIO_VIDEOSTREAM_HPP */