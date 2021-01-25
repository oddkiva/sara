#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/VideoIO/VideoWriter.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/avassert.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libavutil/timestamp.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}


#ifdef av_ts2str
#undef av_ts2str
av_always_inline char* av_ts2str(int64_t ts)
{
  thread_local char str[AV_ERROR_MAX_STRING_SIZE];
  memset(str, 0, sizeof(str));
  return av_ts_make_string(str, ts);
}
#endif

#ifdef av_ts2timestr
#undef av_ts2timestr
av_always_inline char* av_ts2timestr(int64_t ts, AVRational *tb)
{
  thread_local char str[AV_ERROR_MAX_STRING_SIZE];
  memset(str, 0, sizeof(str));
  return av_ts_make_time_string(str, ts, tb);
}
#endif

#ifdef av_err2str
#undef av_err2str
av_always_inline char* av_err2str(int errnum)
{
  // static char str[AV_ERROR_MAX_STRING_SIZE];
  // thread_local may be better than static in multi-thread circumstance
  thread_local char str[AV_ERROR_MAX_STRING_SIZE];
  memset(str, 0, sizeof(str));
  return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}
#endif


constexpr auto STREAM_DURATION = 10.0;
constexpr auto STREAM_FRAME_RATE = 25;              /* 25 images/s */
constexpr auto STREAM_PIX_FMT = AV_PIX_FMT_YUV420P; /* default pix_fmt */
constexpr auto SCALE_FLAGS = SWS_BICUBIC;


namespace DO::Sara {

  using OutputStream = VideoWriter::OutputStream;


  static void log_packet(const AVFormatContext* format_context,
                         const AVPacket* packet)
  {
    AVRational* time_base =
        &format_context->streams[packet->stream_index]->time_base;
    printf("pts:%s pts_time:%s dts:%s dts_time:%s duration:%s duration_time:%s "
           "stream_index:%d\n",
           av_ts2str(packet->pts), av_ts2timestr(packet->pts, time_base),
           av_ts2str(packet->dts), av_ts2timestr(packet->dts, time_base),
           av_ts2str(packet->duration),
           av_ts2timestr(packet->duration, time_base), packet->stream_index);
  }

  static int write_frame(AVFormatContext* format_context, AVCodecContext* c,
                         AVStream* st, AVFrame* frame)
  {
    int ret;
    // send the frame to the encoder
    ret = avcodec_send_frame(c, frame);
    if (ret < 0)
      throw std::runtime_error{format(
          "Error sending a frame to the encoder: %s\n", av_err2str(ret))};

    while (ret >= 0)
    {
      AVPacket packet = {0};
      ret = avcodec_receive_packet(c, &packet);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        break;
      else if (ret < 0)
        throw std::runtime_error{
            format("Error encoding a frame: %s\n", av_err2str(ret))};

      // Rescale output packet timestamp values from codec to stream timebase.
      av_packet_rescale_ts(&packet, c->time_base, st->time_base);
      packet.stream_index = st->index;

      // Write the compressed frame to the media file.
      log_packet(format_context, &packet);
      ret = av_interleaved_write_frame(format_context, &packet);
      av_packet_unref(&packet);
      if (ret < 0)
        throw std::runtime_error{
            format("Error while writing output packet: %s", av_err2str(ret))};
    }

    return ret == AVERROR_EOF ? 1 : 0;
  }


  // ======================================================================== //
  // Add an output stream.
  static void add_stream(OutputStream* out_stream, AVFormatContext* out_context,
                         AVCodec** codec, enum AVCodecID codec_id)
  {
    AVCodecContext* c;
    /* find the encoder */
    *codec = avcodec_find_encoder(codec_id);
    if (!(*codec))
      throw std::runtime_error{format("Could not find encoder for '%s'",
                                      avcodec_get_name(codec_id))};
    out_stream->stream = avformat_new_stream(out_context, nullptr);
    if (!out_stream->stream)
      throw std::runtime_error{"Could not allocate stream"};

    out_stream->stream->id = out_context->nb_streams - 1;
    c = avcodec_alloc_context3(*codec);
    if (!c)
      throw std::runtime_error{"Could not alloc an encoding context"};

    out_stream->encoding_context = c;
    switch ((*codec)->type)
    {
    case AVMEDIA_TYPE_AUDIO:
      c->sample_fmt =
          (*codec)->sample_fmts ? (*codec)->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
      c->bit_rate = 64000;
      c->sample_rate = 44100;
      if ((*codec)->supported_samplerates)
      {
        c->sample_rate = (*codec)->supported_samplerates[0];
        for (int i = 0; (*codec)->supported_samplerates[i]; ++i)
        {
          if ((*codec)->supported_samplerates[i] == 44100)
            c->sample_rate = 44100;
        }
      }
      c->channels = av_get_channel_layout_nb_channels(c->channel_layout);
      c->channel_layout = AV_CH_LAYOUT_STEREO;
      if ((*codec)->channel_layouts)
      {
        c->channel_layout = (*codec)->channel_layouts[0];
        for (int i = 0; (*codec)->channel_layouts[i]; ++i)
        {
          if ((*codec)->channel_layouts[i] == AV_CH_LAYOUT_STEREO)
            c->channel_layout = AV_CH_LAYOUT_STEREO;
        }
      }
      c->channels = av_get_channel_layout_nb_channels(c->channel_layout);
      out_stream->stream->time_base = AVRational{1, c->sample_rate};
      break;
    case AVMEDIA_TYPE_VIDEO:
      c->codec_id = codec_id;
      c->bit_rate = 400000;
      /* Resolution must be a multiple of two. */
      c->width = 352;
      c->height = 288;
      /* timebase: This is the fundamental unit of time (in seconds) in terms
       * of which frame timestamps are represented. For fixed-fps content,
       * timebase should be 1/framerate and timestamp increments should be
       * identical to 1. */
      out_stream->stream->time_base = {1, STREAM_FRAME_RATE};
      c->time_base = out_stream->stream->time_base;
      c->gop_size = 12; /* emit one intra frame every twelve frames at most */
      c->pix_fmt = STREAM_PIX_FMT;
      if (c->codec_id == AV_CODEC_ID_MPEG2VIDEO)
      {
        /* just for testing, we also add B-frames */
        c->max_b_frames = 2;
      }
      if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO)
      {
        /* Needed to avoid using macroblocks in which some coeffs overflow.
         * This does not happen with normal video, it just happens here as
         * the motion of the chroma plane does not match the luma plane. */
        c->mb_decision = 2;
      }
      break;
    default:
      break;
    }
    /* Some formats want stream headers to be separate. */
    if (out_context->oformat->flags & AVFMT_GLOBALHEADER)
      c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  static void add_video_stream(OutputStream* ostream,
                               AVFormatContext* format_context, AVCodec** codec,
                               enum AVCodecID codec_id, int width, int height,
                               int frame_rate)
  {
    AVCodecContext* c;
    /* find the encoder */
    *codec = avcodec_find_encoder(codec_id);
    if (!(*codec))
      throw std::runtime_error{format("Could not find encoder for '%s'",
                                      avcodec_get_name(codec_id))};

    ostream->stream = avformat_new_stream(format_context, nullptr);
    if (!ostream->stream)
      throw std::runtime_error{"Could not allocate stream"};

    ostream->stream->id = format_context->nb_streams - 1;
    c = avcodec_alloc_context3(*codec);
    if (!c)
      throw std::runtime_error{"Could not allocate an encoding context"};

    ostream->encoding_context = c;

    c->codec_id = codec_id;
    // c->bit_rate = 400000;
    /* Resolution must be a multiple of two. */
    c->width = width;
    c->height = height;
    /* timebase: This is the fundamental unit of time (in seconds) in terms
     * of which frame timestamps are represented. For fixed-fps content,
     * timebase should be 1/framerate and timestamp increments should be
     * identical to 1. */
    ostream->stream->time_base = {1, frame_rate};
    c->time_base = ostream->stream->time_base;
    c->gop_size = 12; /* emit one intra frame every twelve frames at most */
    c->pix_fmt = STREAM_PIX_FMT;
    if (c->codec_id == AV_CODEC_ID_MPEG2VIDEO)
    {
      /* just for testing, we also add B-frames */
      c->max_b_frames = 2;
    }
    if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO)
    {
      /* Needed to avoid using macroblocks in which some coeffs overflow.
       * This does not happen with normal video, it just happens here as
       * the motion of the chroma plane does not match the luma plane. */
      c->mb_decision = 2;
    }

    /* Some formats want stream headers to be separate. */
    if (format_context->oformat->flags & AVFMT_GLOBALHEADER)
      c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }


  // ======================================================================== //
  // Audio output
  //
  static AVFrame* alloc_audio_frame(enum AVSampleFormat sample_fmt,
                                    uint64_t channel_layout, int sample_rate,
                                    int nb_samples)
  {
    AVFrame* frame = av_frame_alloc();
    if (!frame)
      throw std::runtime_error{"Error allocating an audio frame"};

    frame->format = sample_fmt;
    frame->channel_layout = channel_layout;
    frame->sample_rate = sample_rate;
    frame->nb_samples = nb_samples;
    if (nb_samples)
    {
      const auto ret = av_frame_get_buffer(frame, 0);
      if (ret < 0)
        throw std::runtime_error{"Error allocating an audio buffer"};
    }
    return frame;
  }

  static void open_audio(AVFormatContext*, AVCodec* codec, OutputStream* ost,
                         AVDictionary* opt_arg)
  {
    AVCodecContext* c;
    int nb_samples;
    int ret;
    AVDictionary* opt = nullptr;
    c = ost->encoding_context;
    /* open it */
    av_dict_copy(&opt, opt_arg, 0);
    ret = avcodec_open2(c, codec, &opt);
    av_dict_free(&opt);
    if (ret < 0)
      throw std::runtime_error{
          format("Could not open audio codec: %s", av_err2str(ret))};

    /* init signal generator */
    ost->t = 0;
    ost->tincr = 2 * M_PI * 110.0 / c->sample_rate;
    /* increment frequency by 110 Hz per second */
    ost->tincr2 = 2 * M_PI * 110.0 / c->sample_rate / c->sample_rate;
    if (c->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)
      nb_samples = 10000;
    else
      nb_samples = c->frame_size;
    ost->frame = alloc_audio_frame(c->sample_fmt, c->channel_layout,
                                   c->sample_rate, nb_samples);
    ost->tmp_frame = alloc_audio_frame(AV_SAMPLE_FMT_S16, c->channel_layout,
                                       c->sample_rate, nb_samples);
    /* copy the stream parameters to the muxer */
    ret = avcodec_parameters_from_context(ost->stream->codecpar, c);
    if (ret < 0)
      throw std::runtime_error{"Could not copy the stream parameters"};

    /* create resampler context */
    ost->swr_ctx = swr_alloc();
    if (!ost->swr_ctx)
      throw std::runtime_error{"Could not allocate resampler context"};

    /* set options */
    av_opt_set_int(ost->swr_ctx, "in_channel_count", c->channels, 0);
    av_opt_set_int(ost->swr_ctx, "in_sample_rate", c->sample_rate, 0);
    av_opt_set_sample_fmt(ost->swr_ctx, "in_sample_fmt", AV_SAMPLE_FMT_S16, 0);
    av_opt_set_int(ost->swr_ctx, "out_channel_count", c->channels, 0);
    av_opt_set_int(ost->swr_ctx, "out_sample_rate", c->sample_rate, 0);
    av_opt_set_sample_fmt(ost->swr_ctx, "out_sample_fmt", c->sample_fmt, 0);
    /* initialize the resampling context */
    if ((ret = swr_init(ost->swr_ctx)) < 0)
      throw std::runtime_error{"Failed to initialize the resampling context\n"};
  }


  /* Prepare a 16 bit dummy audio frame of 'frame_size' samples and
   * 'nb_channels' channels. */
  static AVFrame* get_audio_frame(OutputStream* ost)
  {
    AVFrame* frame = ost->tmp_frame;
    int j, i, v;
    int16_t* q = reinterpret_cast<int16_t*>(frame->data[0]);
    /* check if we want to generate more frames */
    if (av_compare_ts(ost->next_pts, ost->encoding_context->time_base,
                      STREAM_DURATION, {1, 1}) > 0)
      return nullptr;
    for (j = 0; j < frame->nb_samples; j++)
    {
      v = (int) (sin(ost->t) * 10000);
      for (i = 0; i < ost->encoding_context->channels; i++)
        *q++ = v;
      ost->t += ost->tincr;
      ost->tincr += ost->tincr2;
    }
    frame->pts = ost->next_pts;
    ost->next_pts += frame->nb_samples;
    return frame;
  }

  /*
   * encode one audio frame and send it to the muxer
   * return 1 when encoding is finished, 0 otherwise
   */
  static int write_audio_frame(AVFormatContext* oc, OutputStream* ost)
  {
    AVCodecContext* c;
    AVFrame* frame;
    int ret;
    int dst_nb_samples;
    c = ost->encoding_context;
    frame = get_audio_frame(ost);
    if (frame)
    {
      /* convert samples from native format to destination codec format, using
       * the resampler */
      /* compute destination number of samples */
      dst_nb_samples = av_rescale_rnd(
          swr_get_delay(ost->swr_ctx, c->sample_rate) + frame->nb_samples,
          c->sample_rate, c->sample_rate, AV_ROUND_UP);
      av_assert0(dst_nb_samples == frame->nb_samples);
      /* when we pass a frame to the encoder, it may keep a reference to it
       * internally;
       * make sure we do not overwrite it here
       */
      ret = av_frame_make_writable(ost->frame);
      if (ret < 0)
        throw std::runtime_error{"Could not make frame writable!"};

      /* convert to destination format */
      ret = swr_convert(ost->swr_ctx, ost->frame->data, dst_nb_samples,
                        (const uint8_t**) frame->data, frame->nb_samples);
      if (ret < 0)
        throw std::runtime_error{"Error while converting audio frame!"};

      frame = ost->frame;
      frame->pts = av_rescale_q(ost->samples_count,
                                {1, c->sample_rate}, c->time_base);
      ost->samples_count += dst_nb_samples;
    }
    return write_frame(oc, c, ost->stream, frame);
  }


  // ======================================================================== //
  // Video output
  //
  static AVFrame* allocate_picture(enum AVPixelFormat pix_fmt, int width,
                                   int height)
  {
    AVFrame* picture = av_frame_alloc();
    if (!picture)
      return nullptr;
    picture->format = pix_fmt;
    picture->width = width;
    picture->height = height;

    // Allocate the buffers for the frame data.
    const auto ret = av_frame_get_buffer(picture, 0);
    if (ret < 0)
      throw std::runtime_error{"Could not allocate video frame data!"};

    return picture;
  }

  static void open_video(AVFormatContext*, AVCodec* codec,
                         OutputStream* ostream, AVDictionary* opt_arg)
  {
    int ret;
    AVCodecContext* c = ostream->encoding_context;
    AVDictionary* opt = nullptr;
    av_dict_copy(&opt, opt_arg, 0);
    /* open the codec */
    ret = avcodec_open2(c, codec, &opt);
    av_dict_free(&opt);
    if (ret < 0)
      throw std::runtime_error{
          format("Could not open video codec: %s", av_err2str(ret))};

    /* allocate and init a re-usable frame */
    ostream->frame = allocate_picture(c->pix_fmt, c->width, c->height);
    if (!ostream->frame)
      throw std::runtime_error{"Could not allocate video frame!"};

    /* If the output format is not YUV420P, then a temporary YUV420P
     * picture is needed too. It is then converted to the required
     * output format. */
    ostream->tmp_frame = nullptr;
    if (c->pix_fmt != AV_PIX_FMT_YUV420P)
    {
      ostream->tmp_frame =
          allocate_picture(AV_PIX_FMT_YUV420P, c->width, c->height);
      if (!ostream->tmp_frame)
        throw std::runtime_error{"Could not allocate temporary picture!"};
    }
    /* copy the stream parameters to the muxer */
    ret = avcodec_parameters_from_context(ostream->stream->codecpar, c);
    if (ret < 0)
      throw std::runtime_error{"Could not copy the stream parameters!"};
  }

  /* Prepare a dummy image. */
  static void fill_yuv_image(AVFrame* pict, int frame_index, int width,
                             int height)
  {
    int x, y, i;
    i = frame_index;
    /* Y */
    for (y = 0; y < height; y++)
      for (x = 0; x < width; x++)
        pict->data[0][y * pict->linesize[0] + x] = x + y + i * 3;
    /* Cb and Cr */
    for (y = 0; y < height / 2; y++)
    {
      for (x = 0; x < width / 2; x++)
      {
        pict->data[1][y * pict->linesize[1] + x] = 128 + y + i * 2;
        pict->data[2][y * pict->linesize[2] + x] = 64 + x + i * 5;
      }
    }
  }

  static AVFrame* get_video_frame(OutputStream* ostream)
  {
    AVCodecContext* c = ostream->encoding_context;
    /* check if we want to generate more frames */
    if (av_compare_ts(ostream->next_pts, c->time_base, STREAM_DURATION,
                      {1, 1}) > 0)
      return nullptr;
    /* when we pass a frame to the encoder, it may keep a reference to it
     * internally; make sure we do not overwrite it here */
    if (av_frame_make_writable(ostream->frame) < 0)
      throw std::runtime_error{"Could not make frame writable!"};

    if (c->pix_fmt != AV_PIX_FMT_YUV420P)
    {
      /* as we only generate a YUV420P picture, we must convert it
       * to the codec pixel format if needed */
      if (!ostream->sws_ctx)
      {
        ostream->sws_ctx = sws_getContext(
            c->width, c->height, AV_PIX_FMT_YUV420P, c->width, c->height,
            c->pix_fmt, SCALE_FLAGS, nullptr, nullptr, nullptr);
        if (!ostream->sws_ctx)
          throw std::runtime_error{
              "Could not initialize the conversion context"};
      }
      fill_yuv_image(ostream->tmp_frame, ostream->next_pts, c->width,
                     c->height);
      sws_scale(ostream->sws_ctx,
                (const uint8_t* const*) ostream->tmp_frame->data,
                ostream->tmp_frame->linesize, 0, c->height,
                ostream->frame->data, ostream->frame->linesize);
    }
    else
      fill_yuv_image(ostream->frame, ostream->next_pts, c->width, c->height);

    ostream->frame->pts = ostream->next_pts++;

    return ostream->frame;
  }

  /*
   * encode one video frame and send it to the muxer
   * return 1 when encoding is finished, 0 otherwise
   */
  static int write_video_frame(AVFormatContext* format_context,
                               OutputStream* ostream)
  {
    return write_frame(format_context, ostream->encoding_context,
                       ostream->stream, get_video_frame(ostream));
  }

  static void close_stream(AVFormatContext*, OutputStream* os)
  {
    avcodec_free_context(&os->encoding_context);
    av_frame_free(&os->frame);
    av_frame_free(&os->tmp_frame);
    sws_freeContext(os->sws_ctx);
    swr_free(&os->swr_ctx);
  }


  VideoWriter::VideoWriter(const std::string& filepath,
                           const Eigen::Vector2i& sizes,  //
                           int frame_rate,
                           const std::string& preset_quality)
  {
    av_dict_set(&_options, "preset", preset_quality.c_str(), 0);

    /* allocate the output media context */
    avformat_alloc_output_context2(&_format_context, nullptr, nullptr,
                                   filepath.c_str());
    if (!_format_context)
    {
      std::cout << "Could not deduce output format from file extension: "
                   "using MPEG.\n"
                << std::endl;
      avformat_alloc_output_context2(&_format_context, nullptr, "mpeg",
                                     filepath.c_str());
    }
    if (!_format_context)
      throw std::runtime_error{"Could not allocate output media context!"};

    _output_format = _format_context->oformat;
    /* Add the audio and video streams using the default format codecs
     * and initialize the codecs. */
    if (_output_format->video_codec != AV_CODEC_ID_NONE)
    {
      // add_stream(&video_st, oc, &video_codec, fmt->video_codec);
      add_video_stream(&_video_stream, _format_context, &_video_codec,
                       _output_format->video_codec, sizes.x(), sizes.y(),
                       frame_rate);
      _have_video = 1;
      _encode_video = 1;
    }
    if (_output_format->audio_codec != AV_CODEC_ID_NONE)
    {
      add_stream(&_audio_stream, _format_context, &_audio_codec,
                 _output_format->audio_codec);
      _have_audio = 1;
      _encode_audio = 1;
    }

    /* Now that all the parameters are set, we can open the audio and
     * video codecs and allocate the necessary encode buffers. */
    if (_have_video)
      open_video(_format_context, _video_codec, &_video_stream, _options);
    if (_have_audio)
      open_audio(_format_context, _audio_codec, &_audio_stream, _options);
    av_dump_format(_format_context, 0, filepath.c_str(), 1);

    auto ret = int{};
    /* open the output file, if needed */
    if (!(_output_format->flags & AVFMT_NOFILE))
    {
      ret = avio_open(&_format_context->pb, filepath.c_str(), AVIO_FLAG_WRITE);
      if (ret < 0)
        throw std::runtime_error{format("Could not open '%s': %s!",
                                        filepath.c_str(), av_err2str(ret))};
    }

    /* Write the stream header, if any. */
    ret = avformat_write_header(_format_context, &_options);
    if (ret < 0)
      throw std::runtime_error{format(
          "Error occurred when opening output file: %s\n", av_err2str(ret))};
  }

  VideoWriter::~VideoWriter()
  {
    finish();
  }

  static AVFrame* get_video_frame(OutputStream* ostream,
                                  const ImageView<Rgb8>& image)
  {
    auto c = ostream->encoding_context;

    // Impose a hard constraint here on the input image sizes.
    if (image.width() != c->width || image.height() != c->height)
      throw std::runtime_error{
          "Input image sizes does not conform to input video stream sizes!"};

    // When we pass a frame to the encoder, it may keep a reference to it
    // internally; make sure we do not overwrite it here.
    if (av_frame_make_writable(ostream->frame) < 0)
      throw std::runtime_error{"Could not make frame writable!"};

    // Create a pixel conversion context from RGB24 to YUV420P.
    if (!ostream->sws_ctx)
      ostream->sws_ctx =
          sws_getContext(c->width, c->height, AV_PIX_FMT_RGB24, c->width,
                         c->height, AV_PIX_FMT_YUV420P, 0, 0, 0, 0);

    // Extract the image byte address pointer.
    auto image_rgb_ptr = const_cast<Rgb8*>(image.data());
    auto image_byte_ptr = reinterpret_cast<std::uint8_t*>(image_rgb_ptr);

    // A RGB24 image consists of only one image plane.
    std::uint8_t* image_planes[] = {image_byte_ptr};
    // The list of image strides (only one).
    std::int32_t image_linesizes[] = {3 * image.width()};

    // Perform the image format conversion.
    sws_scale(ostream->sws_ctx, image_planes, image_linesizes,
              0 /* beginning of the image slice */,
              image.height() /* number of rows in the image slice */,
              ostream->frame->data, ostream->frame->linesize);

    // Increment the image frame number;
    ostream->frame->pts = ostream->next_pts++;

    return ostream->frame;
  }

  auto VideoWriter::write(const ImageView<Rgb8>& image) -> void
  {
    const auto video_frame = get_video_frame(&_video_stream, image);
    write_frame(_format_context, _video_stream.encoding_context,
                _video_stream.stream, video_frame);
  }

  auto VideoWriter::finish() -> void
  {
    // Write the trailer, if any. The trailer must be written before you close
    // the CodecContexts open when you wrote the header; otherwise
    // av_write_trailer() may try to use memory that was freed on
    // av_codec_close().
    if (_format_context)
      av_write_trailer(_format_context);

    // Close the video codec.
    if (_have_video)
    {
      close_stream(_format_context, &_video_stream);
      _have_video = 0;
      _video_stream = {};
    }
    // Close the audio codec.
    if (_have_audio)
    {
      close_stream(_format_context, &_audio_stream);
      _have_audio = 0;
      _audio_stream = {};
    }

    // Close the output file.
    if (_output_format)
    {
      if (!(_output_format->flags & AVFMT_NOFILE))
        avio_closep(&_format_context->pb);
      _output_format = nullptr;
    }

    // Free the stream.
    if (_format_context)
    {
      avformat_free_context(_format_context);
      _format_context = nullptr;
    }
  }

  auto VideoWriter::generate_dummy() -> void
  {
    while (_encode_video || _encode_audio)
    {
      /* select the stream to encode */
      if (_encode_video &&
          (!_encode_audio ||
           av_compare_ts(_video_stream.next_pts,
                         _video_stream.encoding_context->time_base,
                         _audio_stream.next_pts,
                         _audio_stream.encoding_context->time_base) <= 0))
        _encode_video = !write_video_frame(_format_context, &_video_stream);
      else
        _encode_audio = !write_audio_frame(_format_context, &_audio_stream);
    }
  }

}  // namespace DO::Sara
