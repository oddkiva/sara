#pragma once

#include <DO/Sara/Core/Image.hpp>

#include <array>
#include <cstdio>
#include <string>


struct AVCodec;
struct AVCodecContext;
struct AVDictionary;
struct AVFormatContext;
struct AVOutputFormat;
struct AVFrame;
struct AVPacket;
struct AVStream;
struct SwsContext;
struct SwrContext;


namespace DO::Sara {

  class VideoWriter
  {
  public:
    // a wrapper around a single output AVStream
    struct OutputStream
    {
      AVStream* stream = nullptr;
      AVCodecContext* encoding_context = nullptr;
      /* pts of the next frame that will be generated */
      int64_t next_pts = 0;
      int samples_count = 0;
      AVFrame* frame = nullptr;
      AVFrame* tmp_frame = nullptr;
      float t, tincr, tincr2;
      struct SwsContext* sws_ctx = nullptr;
      struct SwrContext* swr_ctx = nullptr;
    };

    /*!
     *  Possible preset quality:
     *  - ultrafast
     *  - superfast
     *  - veryfast
     *  - faster
     *  - fast
     *  - medium – default preset
     *  - slow
     *  - slower
     *  - veryslow
     */
    VideoWriter(const std::string& filepath, const Eigen::Vector2i& sizes,
                int frame_rate = 25,
                const std::string& preset_quality = "ultrafast");

    ~VideoWriter();

    auto write(const ImageView<Rgb8>& image) -> void;

    auto generate_dummy() -> void;

  private:
    OutputStream _video_stream;
    OutputStream _audio_stream;
    AVOutputFormat* _output_format;
    AVFormatContext* _format_context;
    AVCodec* _audio_codec;
    AVCodec* _video_codec;
    AVDictionary* _options = nullptr;

    int _have_audio = 0;
    int _have_video = 0;
    int _encode_audio = 0;
    int _encode_video = 0;
  };

}  // namespace DO::Sara
