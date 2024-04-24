#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/VideoIO/VideoWriter.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
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
#  undef av_ts2str
av_always_inline char* av_ts2str(int64_t ts)
{
  thread_local char str[AV_ERROR_MAX_STRING_SIZE];
  memset(str, 0, sizeof(str));
  return av_ts_make_string(str, ts);
}
#endif

#ifdef av_ts2timestr
#  undef av_ts2timestr
av_always_inline char* av_ts2timestr(int64_t ts, AVRational* tb)
{
  thread_local char str[AV_ERROR_MAX_STRING_SIZE];
  memset(str, 0, sizeof(str));
  return av_ts_make_time_string(str, ts, tb);
}
#endif

#ifdef av_err2str
#  undef av_err2str
av_always_inline char* av_err2str(int errnum)
{
  // static char str[AV_ERROR_MAX_STRING_SIZE];
  // thread_local may be better than static in multi-thread circumstance
  thread_local char str[AV_ERROR_MAX_STRING_SIZE];
  memset(str, 0, sizeof(str));
  return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}
#endif

static constexpr auto STREAM_PIX_FMT = AV_PIX_FMT_YUV420P; /* default pix_fmt */


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
      AVPacket packet = {};
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
  static void add_video_stream(OutputStream* ostream,
                               AVFormatContext* format_context,
                               const AVCodec** codec,
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

  static void open_video(AVFormatContext*, const AVCodec* codec,
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
                           int frame_rate, const std::string& preset_quality)
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

    /* Now that all the parameters are set, we can open the audio and
     * video codecs and allocate the necessary encode buffers. */
    if (_have_video)
      open_video(_format_context, _video_codec, &_video_stream, _options);
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
    if (_options)
      av_dict_free(&_options);

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

}  // namespace DO::Sara
