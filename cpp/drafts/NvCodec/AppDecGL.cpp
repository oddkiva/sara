/*
 * Copyright 2017-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//---------------------------------------------------------------------------
//! \file AppDecGL.cpp
//! \brief Source file for AppDecGL sample
//!
//! This sample application illustrates the decoding of media file and display
//! of decoded frames in a window. This is done by CUDA interop with OpenGL. For
//! a detailed list of supported codecs on your NVIDIA GPU, refer :
//! https://developer.nvidia.com/nvidia-video-codec-sdk#NVDECFeatures
//---------------------------------------------------------------------------

#include <iostream>
#include <thread>

#include <cuda.h>

#include "nvidia-video-codec-sdk-9.1.23/NvCodec/NvDecoder/NvDecoder.h"
#include "nvidia-video-codec-sdk-9.1.23/Utils/ColorSpace.h"
#include "nvidia-video-codec-sdk-9.1.23/Utils/FFmpegDemuxer.h"
#include "nvidia-video-codec-sdk-9.1.23/Utils/NvCodecUtils.h"

#include "AppDecUtils.h"
#include "FramePresenterGL.h"


simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

/**
 *   @brief  Function to decode media file pointed by "szInFilePath" parameter.
 *           The decoded frames are displayed by using the OpenGL-CUDA interop.
 *   @param  cuContext - Handle to CUDA context
 *   @param  szInFilePath - Path to file to be decoded
 *   @return 0 on failure
 *   @return 1 on success
 */
auto decode(CUcontext cuContext, char* szInFilePath) -> int
{

  FFmpegDemuxer demuxer(szInFilePath);
  NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
  FramePresenterGL presenter(
      cuContext, demuxer.GetWidth(), demuxer.GetHeight());

  // Check whether we have valid NVIDIA libraries installed
  if (!presenter.isVendorNvidia()) {
    std::cout << "\nFailed to find NVIDIA libraries\n";
    return 0;
  }

  // CUDA device memory pointer.
  auto frame_device_ptr = static_cast<uint8_t*>(nullptr);
  auto frame_pitch = int{};
  auto frame_packet_flags = static_cast<uint8_t**>(nullptr);

  auto video_data_buffer = static_cast<uint8_t*>(nullptr);
  auto video_byte_size = int{};

  auto num_frames_returned = int{};
  auto num_frames = int{};

  do {
    demuxer.Demux(&video_data_buffer, &video_byte_size);
    dec.Decode(video_data_buffer, video_byte_size, &frame_packet_flags, &num_frames_returned);

    if (!num_frames && num_frames_returned)
      LOG(INFO) << dec.GetVideoInfo();

    for (auto i = 0; i < num_frames_returned; ++i)
    {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(20ms);

      // Specify the frame data to render on OpenGL.
      presenter.GetDeviceFrameBuffer(&frame_device_ptr, &frame_pitch);

      // Launch cuda kernels for colorspace conversion from raw video to raw
      // image formats which OpenGL textures can work with
      if (dec.GetBitDepth() == 8) {
        if (dec.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
          YUV444ToColor32<BGRA32>((uint8_t*)frame_packet_flags[i], dec.GetWidth(),
              (uint8_t*)frame_device_ptr, frame_pitch, dec.GetWidth(), dec.GetHeight());
        else // default assumed NV12
          Nv12ToColor32<BGRA32>((uint8_t*)frame_packet_flags[i], dec.GetWidth(),
              (uint8_t*)frame_device_ptr, frame_pitch, dec.GetWidth(), dec.GetHeight());
      } else {
        if (dec.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
          YUV444P16ToColor32<BGRA32>((uint8_t*)frame_packet_flags[i], 2 * dec.GetWidth(),
              (uint8_t*)frame_device_ptr, frame_pitch, dec.GetWidth(), dec.GetHeight());
        else // default assumed P016
          P016ToColor32<BGRA32>((uint8_t*)frame_packet_flags[i], 2 * dec.GetWidth(),
              (uint8_t*)frame_device_ptr, frame_pitch, dec.GetWidth(), dec.GetHeight());
      }
    }
    num_frames += num_frames_returned;

  } while (video_byte_size);

  std::cout << "Total frame decoded: " << num_frames << std::endl;
  return 1;
}

int main(int argc, char** argv)
{
  char szInFilePath[256] = "";
  auto iGpu = 0;

  try {
    ParseCommandLine(argc, argv, szInFilePath, NULL, iGpu, NULL, NULL);
    CheckInputFile(szInFilePath);

    ck(cuInit(0));
    auto nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
      std::ostringstream err;
      err << "GPU ordinal out of range. Should be within [" << 0 << ", "
          << nGpu - 1 << "]" << std::endl;
      throw std::invalid_argument(err.str());
    }

    CUcontext cuContext = NULL;
    createCudaContext(&cuContext, iGpu, CU_CTX_SCHED_BLOCKING_SYNC);

    std::cout << "Decode with NvDecoder." << std::endl;
    decode(cuContext, szInFilePath);
  } catch (const std::exception& ex) {
    std::cout << ex.what();
    exit(1);
  }
  return 0;
}
