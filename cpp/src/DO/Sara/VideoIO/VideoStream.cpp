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

#include <DO/Sara/VideoIO/VideoStream.hpp>

#include <DO/Sara/ImageProcessing/Rotate.hpp>
#include <DO/Sara/Logging/Logger.hpp>

// #define PROFILE_VIDEOSTREAM
#if defined(__APPLE__)
// #define HWACCEL
#endif

// Deactivate hardware acceleration for now. The CPU multi-threaded decoding is
// quite good in the meantime.
#if defined(HWACCEL)
#  undef HWACCEL
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#if defined(PROFILE_VIDEOSTREAM)
#  include <DO/Sara/Core/Timer.hpp>
#endif

#include <optional>
#include <thread>


namespace DO::Sara {

#if defined(PROFILE_VIDEOSTREAM)
  static Timer video_stream_profiler;
#  define VIDEO_STREAM_TIC video_stream_profiler.restart();
#  define VIDEO_STREAM_TOC(what)                                               \
    {                                                                          \
      const auto elapsed = video_stream_profiler.elapsed_ms();                 \
      std::cout << "[" << (what) << "] " << elapsed << " ms" << std::endl;     \
    }
#else
#  define VIDEO_STREAM_TIC
#  define VIDEO_STREAM_TOC(what)                                               \
    {                                                                          \
    }
#endif

  bool VideoStream::_registered_all_codecs = false;
#if defined(HWACCEL)
#  if defined(__APPLE__)
  int VideoStream::_hw_device_type = AV_HWDEVICE_TYPE_VIDEOTOOLBOX;
#  else
  int VideoStream::_hw_device_type = AV_HWDEVICE_TYPE_CUDA;
#  endif

  static AVBufferRef* hw_device_ctx = NULL;
  static enum AVPixelFormat hw_pix_fmt;

  static int hw_decoder_init(AVCodecContext* ctx,
                             const enum AVHWDeviceType type)
  {
    int err = 0;

    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type, NULL, NULL, 0)) < 0)
    {
      fprintf(stderr, "Failed to create specified hardware device context.\n");
      return err;
    }
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    return err;
  }

  static enum AVPixelFormat get_hw_format(AVCodecContext*,
                                          const enum AVPixelFormat* pix_fmts)
  {
    const enum AVPixelFormat* p;

    for (p = pix_fmts; *p != -1; p++)
    {
      if (*p == hw_pix_fmt)
        return *p;
    }

    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
  }
#endif

  // Since FFmpeg 5.x, copied from "cmdutils.c".
  static double get_rotation_degrees(const int32_t* displaymatrix)
  {
    double theta = 0;
    if (displaymatrix)
      theta = -std::round(av_display_rotation_get(displaymatrix));

    theta -= 360 * std::floor(theta / 360 + 0.9 / 360);

    if (std::fabs(theta - 90 * std::round(theta / 90)) > 2)
      av_log(NULL, AV_LOG_WARNING,
             "Odd rotation angle.\n"
             "If you want to help, upload a sample "
             "of this file to https://streams.videolan.org/upload/ "
             "and contact the ffmpeg-devel mailing list. "
             "(ffmpeg-devel@ffmpeg.org)");

    return theta;
  }

  VideoStream::VideoStream()
  {
    if (!_registered_all_codecs)
    {
      // av_register_all() got deprecated in lavf 58.9.100.
      // We don't need to use it anymore since FFmpeg 4.0.
#if (LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100))
      av_register_all();
#endif
      _registered_all_codecs = true;
    }
  }

  VideoStream::VideoStream(const std::string& file_path, const bool autorotate)
    : VideoStream()
  {
    open(file_path, autorotate);
  }

  VideoStream::~VideoStream()
  {
    close();
  }

  auto VideoStream::open(const std::string& file_path, const bool autorotate)
      -> void
  {
    auto& logger = Logger::get();

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

    auto& video_stream = _video_format_context->streams[_video_stream_index];
    _video_codec_params = video_stream->codecpar;

    _video_codec = avcodec_find_decoder(_video_codec_params->codec_id);
    if (_video_codec == nullptr)
      throw std::runtime_error{"Could not find video decoder!"};

#if defined(HWACCEL)
    // Initialize the audio-video codec hardware config.
    for (auto i = 0;; i++)
    {
      const AVCodecHWConfig* config = avcodec_get_hw_config(_video_codec, i);
      if (!config)
      {
        throw std::runtime_error{fmt::format(
            "Decoder %s does not support device type {}", _video_codec->name,
            av_hwdevice_get_type_name(AVHWDeviceType(_hw_device_type)))};
      }
      if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
          config->device_type == _hw_device_type)
      {
        SARA_LOGD(logger, "Successfully initialized hardware AV codec config!");
        hw_pix_fmt = config->pix_fmt;
        break;
      }
    }
#endif

    // Create a decoder context.
    _video_codec_context = avcodec_alloc_context3(_video_codec);
    if (_video_codec_context == nullptr)
      throw std::runtime_error{"Could not allocate video decoder context!"};

    // Enable multi-threaded decoding.
    _video_codec_context->thread_count = std::thread::hardware_concurrency();

    if (avcodec_parameters_to_context(_video_codec_context,
                                      _video_codec_params))
      throw std::runtime_error{"Could not copy video decoder context!"};

#if defined(HWACCEL)
    auto video_stream = _video_format_context->streams[_video_stream_index];
    if (avcodec_parameters_to_context(_video_codec_context,
                                      video_stream->codecpar) < 0)
      throw std::runtime_error{"Could not initialize the video codec context!"};

    _video_codec_context->get_format = get_hw_format;

    if (hw_decoder_init(_video_codec_context,  //
                        AVHWDeviceType(_hw_device_type)) < 0)
      throw std::runtime_error{"Could not initialize the video codec context"};
#endif

    // Open it.
    if (avcodec_open2(_video_codec_context, _video_codec, nullptr) < 0)
      throw std::runtime_error{"Could not open video decoder!"};


#if defined(HWACCEL)
    // Allocate the device picture buffer.
    if (_device_picture == nullptr)
      _device_picture = av_frame_alloc();
    if (_device_picture == nullptr)
      throw std::runtime_error{"Could not allocate device video frame!"};
#endif

    // Allocate buffer to read the video frame.
    if (_picture == nullptr)
      _picture = av_frame_alloc();
    if (_picture == nullptr)
      throw std::runtime_error{"Could not allocate host video frame!"};

    // Initialize a video packet.
    if (_pkt == nullptr)
      _pkt = av_packet_alloc();
    if (_pkt == nullptr)
      throw std::runtime_error("Could not allocate video packet!");

    const auto sizes_original = Eigen::Vector2i{
        _video_codec_context->width,  //
        _video_codec_context->height  //
    };
    SARA_LOGD(logger, "Image sizes  : {}", sizes_original.transpose().eval());
    SARA_LOGD(logger, "Pixel format : {}",
              av_get_pix_fmt_name(_video_codec_context->pix_fmt));
    SARA_LOGD(
        logger, "Time base    : {}: {}/{}",  //
        _video_stream_index,
        _video_format_context->streams[_video_stream_index]->time_base.num,
        _video_format_context->streams[_video_stream_index]->time_base.den);

    // Get video format converter to RGB24.
    _sws_context = sws_getContext(  //
        sizes_original.x(), sizes_original.y(),
#if defined(HWACCEL)
        AV_PIX_FMT_NV12,
#else
        _video_codec_context->pix_fmt,
#endif
        sizes_original.x(), sizes_original.y(),  //
        AV_PIX_FMT_RGB24, SWS_POINT,             //
        nullptr, nullptr, nullptr);
    if (_sws_context == nullptr)
      throw std::runtime_error{"Could not allocate SWS context!"};

    // The converted the video frame.
    if (_picture_rgb == nullptr)
      _picture_rgb = av_frame_alloc();
    if (_picture_rgb == nullptr)
      throw std::runtime_error{"Could not allocate video frame!"};

    // TODO: make it RGBA 32 bit to optimize for SSE/AVX2?
    constexpr auto alignment_size = 1;
    const auto byte_size = av_image_alloc(
        _picture_rgb->data, _picture_rgb->linesize, sizes_original.x(),
        sizes_original.y(), AV_PIX_FMT_RGB24, alignment_size);
    if (byte_size != sizes_original.x() * sizes_original.y() * 3)
      throw std::runtime_error{
          "Allocated memory size for the converted video frame is wrong!"};

    _end_of_stream = false;


    // Initialize the frame rotater.
    //
    // This worked until FFmpeg 5.x.
    const auto rotate_tag = av_dict_get(video_stream->metadata, "rotate",  //
                                        nullptr, 0);
    const auto rotation_angle_from_tag =
        rotate_tag == nullptr ? 0 : std::stoi(rotate_tag->value);
    // Since FFmpeg 5.x, we need to do this.
    const auto display_mat_ptr =
        reinterpret_cast<int32_t*>(av_stream_get_side_data(
            video_stream, AV_PKT_DATA_DISPLAYMATRIX, nullptr));
    const auto rotation_angle_from_display_matrix =
        static_cast<int>(get_rotation_degrees(display_mat_ptr));

    // Before FFmpeg 5.x
    auto rotation_angle = rotation_angle_from_tag;
    // After FFmpeg 5.x
    if (rotation_angle == 0 && rotation_angle_from_display_matrix != 0)
      rotation_angle = rotation_angle_from_display_matrix;
    if (!autorotate)
      rotation_angle = 0;

    const auto frame_original = ImageView<Rgb8>{
        reinterpret_cast<Rgb8*>(_picture_rgb->data[0]),
        sizes_original  //
    };
    _frame_rotater = FrameRotater{frame_original, rotation_angle};
  }

  auto VideoStream::close() -> void
  {
    // Flush the decoder (draining mode.)
    if (_video_format_context != nullptr)
      this->decode(nullptr);

    // Free the data structures.
    if (_pkt != nullptr)
      av_packet_unref(_pkt);
    av_frame_free(&_device_picture);
    av_frame_free(&_picture);
    avcodec_close(_video_codec_context);
    avformat_close_input(&_video_format_context);
    avcodec_free_context(&_video_codec_context);
    av_packet_free(&_pkt);

    // No need to deallocate these.
    _video_stream_index = -1;
    _video_codec_params = nullptr;
    _video_codec = nullptr;

    sws_freeContext(_sws_context);
    if (_picture_rgb != nullptr)
      av_freep(_picture_rgb->data);
    av_frame_free(&_picture_rgb);

    _end_of_stream = true;
  }

  auto VideoStream::read() -> bool
  {
    do
    {
      if (!_end_of_stream && av_read_frame(_video_format_context, _pkt) < 0)
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
        _got_frame = decode(_pkt);
        // N.B.: always unreference the packet after decoding even if the frame
        // is not complete.
        av_packet_unref(_pkt);

        if (_got_frame)
        {
          VIDEO_STREAM_TIC

          // Convert to RGB24 pixel format.
          sws_scale(_sws_context, _picture->data, _picture->linesize, 0,
                    _frame_rotater._src.height(), _picture_rgb->data,
                    _picture_rgb->linesize);

          // av_frame_unref(_picture);
          VIDEO_STREAM_TOC("To RGB24")

          VIDEO_STREAM_TIC
          _frame_rotater.update();
          VIDEO_STREAM_TOC("Unrotate");

          return true;
        }
      }
      ++_i;
    } while (!_end_of_stream || _got_frame);

    return false;
  }

  auto VideoStream::frame() const -> ImageView<Rgb8>
  {
    const auto& frame = _frame_rotater._dst;
    const auto frame_ptr = const_cast<Rgb8*>(frame.data());
    return {frame_ptr, frame.sizes()};
  }

  auto VideoStream::seek(std::size_t frame_pos) -> void
  {
    av_seek_frame(_video_format_context, _video_stream_index, frame_pos,
                  AVSEEK_FLAG_BACKWARD);
  }

  auto VideoStream::decode(AVPacket* pkt) -> bool
  {
    auto ret = int{};

    // Transfer raw compressed video data to the packet.
    VIDEO_STREAM_TIC
    ret = avcodec_send_packet(_video_codec_context, pkt);
    if (ret < 0)
      throw std::runtime_error{"Error sending a packet for decoding!"};
    VIDEO_STREAM_TOC("Send packet")

    // Decode the compressed video data into an uncompressed video frame.
    VIDEO_STREAM_TIC
#if defined(HWACCEL)
    ret = avcodec_receive_frame(_video_codec_context, _device_picture);
#else
    ret = avcodec_receive_frame(_video_codec_context, _picture);
#endif
    VIDEO_STREAM_TOC("Receive frame")

    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return false;

    if (ret < 0)
      throw std::runtime_error{"Error during decoding!"};

#if defined(HWACCEL)
    // Copy the data from the device buffer to the host buffer.
    if (_device_picture->format == hw_pix_fmt)
    {
      if (av_hwframe_transfer_data(_picture, _device_picture, 0) < 0)
        throw std::runtime_error{
            "Error transferring the data to system memory"};
    }
    else  // Otherwise do nothing.
      _picture = _device_picture;
#endif

    return true;
  }

  auto VideoStream::frame_rate() const -> float
  {
    return static_cast<float>(_video_codec_context->framerate.num) /
           _video_codec_context->framerate.den;
  }

  auto VideoStream::width() const -> int
  {
    return _frame_rotater._dst.width();
  }

  auto VideoStream::height() const -> int
  {
    return _frame_rotater._dst.height();
  }


  VideoStream::FrameRotater::FrameRotater(const ImageView<Rgb8>& src,
                                          const int rotation_angle)
    : _rotation_angle{rotation_angle}
  {
    _src.swap(ImageView<Rgb8>{const_cast<Rgb8*>(src.data()), src.sizes()});
    if (_rotation_angle == 0)
      _dst.swap(ImageView<Rgb8>{
          const_cast<Rgb8*>(_src.data()),  //
          _src.sizes()                     //
      });
    else if (_rotation_angle == 90)
    {
      _src_rotated.resize(_src.sizes().reverse());
      _dst.swap(ImageView<Rgb8>{_src_rotated.data(), _src_rotated.sizes()});
    }
    else
      throw std::runtime_error{"Unsupported rotation angle!"};
  }

  auto VideoStream::FrameRotater::update() -> void
  {
    if (_rotation_angle == 90)
      rotate_cw_90(_src, _dst);
    else if (_rotation_angle != 0)
      throw std::runtime_error{"Unsupported rotation angle!"};
  }

}  // namespace DO::Sara
