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
//! This sample application illustrates the decoding of media file and display of decoded frames in a window.
//! This is done by CUDA interop with OpenGL.
//! For a detailed list of supported codecs on your NVIDIA GPU, refer : https://developer.nvidia.com/nvidia-video-codec-sdk#NVDECFeatures
//---------------------------------------------------------------------------

#include <cuda.h>
#include <iostream>
#include "NvDecoder/NvDecoder.h"
#include "../Utils/NvCodecUtils.h"
#include "../Utils/FFmpegDemuxer.h"
#include "FramePresenterGL.h"
#include "../Common/AppDecUtils.h"
#include "../Utils/ColorSpace.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

/**
*   @brief  Function to decode media file pointed by "szInFilePath" parameter.
*           The decoded frames are displayed by using the OpenGL-CUDA interop.
*   @param  cuContext - Handle to CUDA context
*   @param  szInFilePath - Path to file to be decoded
*   @return 0 on failure
*   @return 1 on success
*/
int Decode(CUcontext cuContext, char *szInFilePath) {

    FFmpegDemuxer demuxer(szInFilePath);
    NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
    FramePresenterGL presenter(cuContext, demuxer.GetWidth(), demuxer.GetHeight());

    // Check whether we have valid NVIDIA libraries installed
    if (!presenter.isVendorNvidia()) {
        std::cout<<"\nFailed to find NVIDIA libraries\n";
        return 0;
    }

    uint8_t *dpFrame = 0;
    int nPitch = 0;
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t *pVideo = NULL;
    uint8_t **ppFrame;
    do {
        demuxer.Demux(&pVideo, &nVideoBytes);
        dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
        if (!nFrame && nFrameReturned)
            LOG(INFO) << dec.GetVideoInfo();

        for (int i = 0; i < nFrameReturned; i++) {
            presenter.GetDeviceFrameBuffer(&dpFrame, &nPitch);
            // Launch cuda kernels for colorspace conversion from raw video to raw image formats which OpenGL textures can work with
            if (dec.GetBitDepth() == 8) {
                if (dec.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
                    YUV444ToColor32<BGRA32>((uint8_t *)ppFrame[i], dec.GetWidth(), (uint8_t *)dpFrame, nPitch, dec.GetWidth(), dec.GetHeight());
                else // default assumed NV12
                    Nv12ToColor32<BGRA32>((uint8_t *)ppFrame[i], dec.GetWidth(), (uint8_t *)dpFrame, nPitch, dec.GetWidth(), dec.GetHeight());
            }
            else {
                if (dec.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
                    YUV444P16ToColor32<BGRA32>((uint8_t *)ppFrame[i], 2 * dec.GetWidth(), (uint8_t *)dpFrame, nPitch, dec.GetWidth(), dec.GetHeight());
                else // default assumed P016
                    P016ToColor32<BGRA32>((uint8_t *)ppFrame[i], 2 * dec.GetWidth(), (uint8_t *)dpFrame, nPitch, dec.GetWidth(), dec.GetHeight());
            }
        }
        nFrame += nFrameReturned;
    } while (nVideoBytes);
    std::cout << "Total frame decoded: " << nFrame << std::endl;
    return 1;
}

int main(int argc, char **argv) 
{
    char szInFilePath[256] = "";
    int iGpu = 0;
    try
    {
        ParseCommandLine(argc, argv, szInFilePath, NULL, iGpu, NULL, NULL);
        CheckInputFile(szInFilePath);

        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        if (iGpu < 0 || iGpu >= nGpu)
        {
            std::ostringstream err;
            err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
            throw std::invalid_argument(err.str());
        }

        CUcontext cuContext = NULL;
        createCudaContext(&cuContext, iGpu, CU_CTX_SCHED_BLOCKING_SYNC);

        std::cout << "Decode with NvDecoder." << std::endl;
        Decode(cuContext, szInFilePath);
    }
    catch(const std::exception& ex)
    {
        std::cout << ex.what();
        exit(1);
    }
    return 0;
}
