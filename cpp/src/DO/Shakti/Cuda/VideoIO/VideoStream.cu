// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Shakti/Cuda/VideoIO/VideoStream.hpp>

#include "nvidia-video-codec-sdk/NvCodec/NvDecoder/NvDecoder.h"
#include "nvidia-video-codec-sdk/Utils/ColorSpace.h"
#include "nvidia-video-codec-sdk/Utils/FFmpegDemuxer.h"
#include "nvidia-video-codec-sdk/Utils/NvCodecUtils.h"

#include <array>


namespace DriverApi {

  auto init() -> void
  {
    ck(cuInit(0));
  }

  auto get_device_count() -> int
  {
    auto num_gpus = 0;
    ck(cuDeviceGetCount(&num_gpus));
    return num_gpus;
  }

  CudaContext::CudaContext(int gpu_id_)
    : gpu_id{gpu_id_}
  {
    ck(cuDeviceGet(&cuda_device, gpu_id));

    std::array<char, 80> device_name;
    ck(cuDeviceGetName(device_name.data(), static_cast<int>(device_name.size()),
                       cuda_device));
    std::cout << "GPU in use: " << device_name.data() << std::endl;

    ck(cuCtxCreate(&cuda_context, CU_CTX_BLOCKING_SYNC, cuda_device));
  }

  CudaContext::CudaContext(CudaContext&& other)
  {
    std::swap(gpu_id, other.gpu_id);
    std::swap(cuda_context, other.cuda_context);
    std::swap(cuda_device, other.cuda_device);
  }

  CudaContext::~CudaContext()
  {
    if (cuda_context)
    {
      ck(cuCtxDestroy(cuda_context));
      cuda_context = 0;
      cuda_device = 0;
      gpu_id = -1;
    }
  }

  auto CudaContext::make_current() -> void
  {
    ck(cuCtxSetCurrent(cuda_context));
  }

  DeviceBgraBuffer::DeviceBgraBuffer(int width_, int height_)
    : width{width_}
    , height{height_}
  {
    ck(cuMemAlloc(&data, width * height * 4));
    ck(cuMemsetD8(data, 0, width * height * 4));
  }

  DeviceBgraBuffer::~DeviceBgraBuffer()
  {
    if (data)
      ck(cuMemFree(data));
  }

  auto
  DeviceBgraBuffer::to_host(DO::Sara::ImageView<DO::Sara::Bgra8>& image) const
      -> void
  {
    ck(cudaMemcpy(reinterpret_cast<void*>(image.data()),
                  reinterpret_cast<const void*>(data), width * height * 4,
                  cudaMemcpyDeviceToHost));
  }

}  // namespace DriverApi

simplelogger::Logger* logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();


namespace DO::Shakti {

  struct VideoStream::Impl
  {
    Impl(const std::string& video_filepath,
         const DriverApi::CudaContext& context)
      : demuxer{video_filepath.c_str()}
      , decoder{context.cuda_context, true,
                FFmpeg2NvCodecId(demuxer.GetVideoCodec())}
    {
    }

    auto
    read_decoded_frame_packet(DriverApi::DeviceBgraBuffer& bgra_frame_buffer)
        -> void
    {
      const auto raw_frame_packet = decoder.GetFrame();
      const auto iMatrix = decoder.GetVideoFormatInfo()
                               .video_signal_description.matrix_coefficients;

      // Launch CUDA kernels for colorspace conversion from raw video to raw
      // image formats which OpenGL textures can work with
      if (decoder.GetBitDepth() == 8)
      {
        if (decoder.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
          YUV444ToColor32<BGRA32>(
              raw_frame_packet,
              decoder.GetWidth(),
              reinterpret_cast<uint8_t*>(bgra_frame_buffer.data),
              bgra_frame_buffer.width * 4, decoder.GetWidth(),
              decoder.GetHeight(), iMatrix);

        else  // default assumed NV12
          Nv12ToColor32<BGRA32>(
              raw_frame_packet,
              decoder.GetWidth(),
              reinterpret_cast<uint8_t*>(bgra_frame_buffer.data),
              bgra_frame_buffer.width * 4, decoder.GetWidth(),
              decoder.GetHeight(), iMatrix);
      }
      else
      {
        if (decoder.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
          YUV444P16ToColor32<BGRA32>(
              raw_frame_packet,
              2 * decoder.GetWidth(),
              reinterpret_cast<uint8_t*>(bgra_frame_buffer.data),
              bgra_frame_buffer.width * 4, decoder.GetWidth(),
              decoder.GetHeight(), iMatrix);
        else  // default assumed P016
          P016ToColor32<BGRA32>(
              reinterpret_cast<uint8_t*>(raw_frame_packet[frame_index]),
              2 * decoder.GetWidth(),
              reinterpret_cast<uint8_t*>(bgra_frame_buffer.data),
              bgra_frame_buffer.width * 4, decoder.GetWidth(),
              decoder.GetHeight(), iMatrix);
      }

      ++frame_index;

      if (frame_index == num_frames_decoded)
        num_frames_decoded = frame_index = 0;
    }

    auto decode() -> bool
    {
      // Initialize the video stream.
      do
      {
        if (!demuxer.Demux(&frame_data_compressed.data,
                           &frame_data_compressed.size))
          return false;

        num_frames_decoded = decoder.Decode(frame_data_compressed.data,
                                            frame_data_compressed.size);
      } while (num_frames_decoded <= 0);
      return true;
    }

    auto read(DriverApi::DeviceBgraBuffer& bgra_frame_buffer) -> bool
    {
      if (num_frames_decoded == 0 && !decode())
        return false;

#ifdef DEBUG
      LOG(INFO) << decoder.GetVideoInfo();
#endif

      if (frame_index < num_frames_decoded)
        read_decoded_frame_packet(bgra_frame_buffer);

      return true;
    }

    mutable FFmpegDemuxer demuxer;
    NvDecoder decoder;

    std::int32_t num_frames_decoded = 0;
    std::int32_t frame_index = 0;

    struct EncodedVideoBuffer
    {
      std::uint8_t* data = nullptr;
      std::int32_t size = 0;
    } frame_data_compressed;
  };


  auto VideoStream::ImplDeleter::operator()(const VideoStream::Impl* p) const
      -> void
  {
    delete p;
  }


  VideoStream::VideoStream(const std::string& video_filepath,
                           const DriverApi::CudaContext& context)
    : _impl{new VideoStream::Impl{video_filepath, context}}
  {
  }

  auto VideoStream::width() const -> int
  {
    return _impl->demuxer.GetWidth();
  }

  auto VideoStream::height() const -> int
  {
    return _impl->demuxer.GetHeight();
  }

  auto VideoStream::decode() -> bool
  {
    return _impl->decode();
  }

  auto VideoStream::read(DriverApi::DeviceBgraBuffer& bgra_frame_buffer) -> bool
  {
    return _impl->read(bgra_frame_buffer);
  }

}  // namespace DO::Shakti


static auto get_output_format_names(unsigned short output_format_mask,
                                    char* OutputFormats) -> void
{
  if (output_format_mask == 0)
  {
    strcpy(OutputFormats, "N/A");
    return;
  }

  if (output_format_mask & (1U << cudaVideoSurfaceFormat_NV12))
    strcat(OutputFormats, "NV12 ");

  if (output_format_mask & (1U << cudaVideoSurfaceFormat_P016))
    strcat(OutputFormats, "P016 ");

  if (output_format_mask & (1U << cudaVideoSurfaceFormat_YUV444))
    strcat(OutputFormats, "YUV444 ");

  if (output_format_mask & (1U << cudaVideoSurfaceFormat_YUV444_16Bit))
    strcat(OutputFormats, "YUV444P16 ");
}

auto show_decoder_capability() -> void
{
  ck(cuInit(0));
  int num_gpus = 0;
  ck(cuDeviceGetCount(&num_gpus));
  std::cout << "Decoder Capability" << std::endl << std::endl;
  const char* codec_names[] = {
      "JPEG", "MPEG1", "MPEG2", "MPEG4", "H264", "HEVC", "HEVC", "HEVC",
      "HEVC", "HEVC",  "HEVC",  "VC1",   "VP8",  "VP9",  "VP9",  "VP9"};
  const char* chroma_format_strings[] = {"4:0:0", "4:2:0", "4:2:2", "4:4:4"};
  char output_formats[64];
  cudaVideoCodec codecs[] = {
      cudaVideoCodec_JPEG,  cudaVideoCodec_MPEG1, cudaVideoCodec_MPEG2,
      cudaVideoCodec_MPEG4, cudaVideoCodec_H264,  cudaVideoCodec_HEVC,
      cudaVideoCodec_HEVC,  cudaVideoCodec_HEVC,  cudaVideoCodec_HEVC,
      cudaVideoCodec_HEVC,  cudaVideoCodec_HEVC,  cudaVideoCodec_VC1,
      cudaVideoCodec_VP8,   cudaVideoCodec_VP9,   cudaVideoCodec_VP9,
      cudaVideoCodec_VP9};
  int bit_depth_minus_8[] = {0, 0, 0, 0, 0, 0, 2, 4, 0, 2, 4, 0, 0, 0, 2, 4};

  cudaVideoChromaFormat chroma_formats[] = {
      cudaVideoChromaFormat_420, cudaVideoChromaFormat_420,
      cudaVideoChromaFormat_420, cudaVideoChromaFormat_420,
      cudaVideoChromaFormat_420, cudaVideoChromaFormat_420,
      cudaVideoChromaFormat_420, cudaVideoChromaFormat_420,
      cudaVideoChromaFormat_444, cudaVideoChromaFormat_444,
      cudaVideoChromaFormat_444, cudaVideoChromaFormat_420,
      cudaVideoChromaFormat_420, cudaVideoChromaFormat_420,
      cudaVideoChromaFormat_420, cudaVideoChromaFormat_420};

  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id)
  {
    auto cuda_context = DriverApi::CudaContext{gpu_id};

    for (auto i = 0u; i < sizeof(codecs) / sizeof(codecs[0]); ++i)
    {
      CUVIDDECODECAPS decode_caps = {};
      decode_caps.eCodecType = codecs[i];
      decode_caps.eChromaFormat = chroma_formats[i];
      decode_caps.nBitDepthMinus8 = bit_depth_minus_8[i];

      cuvidGetDecoderCaps(&decode_caps);

      output_formats[0] = '\0';
      get_output_format_names(decode_caps.nOutputFormatMask, output_formats);

      // setw() width = maximum_width_of_string + 2 spaces
      std::cout << "Codec  " << std::left << std::setw(7) << codec_names[i]
                << "BitDepth  " << std::setw(4)
                << decode_caps.nBitDepthMinus8 + 8 << "ChromaFormat  "
                << std::setw(7)
                << chroma_format_strings[decode_caps.eChromaFormat]
                << "Supported  " << std::setw(3)
                << (int) decode_caps.bIsSupported << "MaxWidth  "
                << std::setw(7) << decode_caps.nMaxWidth << "MaxHeight  "
                << std::setw(7) << decode_caps.nMaxHeight << "MaxMBCount  "
                << std::setw(10) << decode_caps.nMaxMBCount << "MinWidth  "
                << std::setw(5) << decode_caps.nMinWidth << "MinHeight  "
                << std::setw(5) << decode_caps.nMinHeight << "SurfaceFormat  "
                << std::setw(11) << output_formats << std::endl;
    }

    std::cout << std::endl;
  }
}
