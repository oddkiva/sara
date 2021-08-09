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
#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
}
#include "NvCodecUtils.h"

//---------------------------------------------------------------------------
//! \file FFmpegDemuxer.h
//! \brief Provides functionality for stream demuxing
//!
//! This header file is used by Decode/Transcode apps to demux input video clips
//! before decoding frames from it.
//---------------------------------------------------------------------------

/**
 * @brief libavformat wrapper class. Retrieves the elementary encoded stream
 * from the container format.
 */
class FFmpegDemuxer
{
private:
  AVFormatContext* _fmtc = nullptr;
  AVIOContext* _avioc = nullptr;

  /*!< AVPacket stores compressed data typically exported
                        by demuxers and then passed as input to decoders */
  AVPacket* _pkt = nullptr;
  AVPacket* _pktFiltered = nullptr;
  AVBSFContext* _bsfc = nullptr;

  int _videoStreamIndex;
  bool _isMp4H264, _isMp4HEVC, _isMp4MPEG4;
  AVCodecID _videoCodec;
  AVPixelFormat _chromaFormat;
  int _width, _height, _bitDepth, _bytesPerPixel, _chromaHeight;
  double _timeBase = 0.0;

  uint8_t* _dataWithHeader = nullptr;

  unsigned int _frameCount = 0;

public:
  class DataProvider
  {
  public:
    virtual ~DataProvider()
    {
    }

    virtual int GetData(uint8_t* pBuf, int nBuf) = 0;
  };

private:
  /**
   *   @brief  Private constructor to initialize libavformat resources.
   *   @param  _fmtc - Pointer to AVFormatContext allocated inside
   * avformat_open_input()
   */
  FFmpegDemuxer(AVFormatContext* fmtc)
    : _fmtc{fmtc}
  {
    if (!_fmtc)
    {
      LOG(ERROR) << "No AVFormatContext provided.";
      return;
    }

    LOG(INFO) << "Media format: " << _fmtc->iformat->long_name << " ("
              << _fmtc->iformat->name << ")";

    ck(avformat_find_stream_info(_fmtc, nullptr));
    _videoStreamIndex =
        av_find_best_stream(_fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (_videoStreamIndex < 0)
    {
      LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " "
                 << "Could not find stream in input file";
      return;
    }

    // _fmtc->streams[_videoStreamIndex]->need_parsing = AVSTREAM_PARSE_NONE;
    _videoCodec = _fmtc->streams[_videoStreamIndex]->codecpar->codec_id;
    _width = _fmtc->streams[_videoStreamIndex]->codecpar->width;
    _height = _fmtc->streams[_videoStreamIndex]->codecpar->height;
    _chromaFormat =
        (AVPixelFormat) _fmtc->streams[_videoStreamIndex]->codecpar->format;
    AVRational rTimeBase = _fmtc->streams[_videoStreamIndex]->time_base;
    _timeBase = av_q2d(rTimeBase);

    // Set bit depth, chroma height, bits per pixel based on chroma format of
    // input
    switch (_chromaFormat)
    {
    case AV_PIX_FMT_YUV420P10LE:
      _bitDepth = 10;
      _chromaHeight = (_height + 1) >> 1;
      _bytesPerPixel = 2;
      break;
    case AV_PIX_FMT_YUV420P12LE:
      _bitDepth = 12;
      _chromaHeight = (_height + 1) >> 1;
      _bytesPerPixel = 2;
      break;
    case AV_PIX_FMT_YUV444P10LE:
      _bitDepth = 10;
      _chromaHeight = _height << 1;
      _bytesPerPixel = 2;
      break;
    case AV_PIX_FMT_YUV444P12LE:
      _bitDepth = 12;
      _chromaHeight = _height << 1;
      _bytesPerPixel = 2;
      break;
    case AV_PIX_FMT_YUV444P:
      _bitDepth = 8;
      _chromaHeight = _height << 1;
      _bytesPerPixel = 1;
      break;
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_YUVJ422P:  // jpeg decoder output is subsampled to NV12 for
                               // 422/444 so treat it as 420
    case AV_PIX_FMT_YUVJ444P:  // jpeg decoder output is subsampled to NV12 for
                               // 422/444 so treat it as 420
      _bitDepth = 8;
      _chromaHeight = (_height + 1) >> 1;
      _bytesPerPixel = 1;
      break;
    default:
      LOG(WARNING) << "ChromaFormat not recognized. Assuming 420";
      _bitDepth = 8;
      _chromaHeight = (_height + 1) >> 1;
      _bytesPerPixel = 1;
    }

    _isMp4H264 = _videoCodec == AV_CODEC_ID_H264 &&
                 (!strcmp(_fmtc->iformat->long_name, "QuickTime / MOV") ||
                  !strcmp(_fmtc->iformat->long_name, "FLV (Flash Video)") ||
                  !strcmp(_fmtc->iformat->long_name, "Matroska / WebM"));
    _isMp4HEVC = _videoCodec == AV_CODEC_ID_HEVC &&
                 (!strcmp(_fmtc->iformat->long_name, "QuickTime / MOV") ||
                  !strcmp(_fmtc->iformat->long_name, "FLV (Flash Video)") ||
                  !strcmp(_fmtc->iformat->long_name, "Matroska / WebM"));

    _isMp4MPEG4 = _videoCodec == AV_CODEC_ID_MPEG4 &&
                  (!strcmp(_fmtc->iformat->long_name, "QuickTime / MOV") ||
                   !strcmp(_fmtc->iformat->long_name, "FLV (Flash Video)") ||
                   !strcmp(_fmtc->iformat->long_name, "Matroska / WebM"));

    // Initialize packet fields with default values
    _pkt = av_packet_alloc();
    _pktFiltered = av_packet_alloc();

    // Initialize bitstream filter and its required resources
    if (_isMp4H264)
    {
      const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
      if (!bsf)
      {
        LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " "
                   << "av_bsf_get_by_name() failed";
        return;
      }
      ck(av_bsf_alloc(bsf, &_bsfc));
      avcodec_parameters_copy(_bsfc->par_in,
                              _fmtc->streams[_videoStreamIndex]->codecpar);
      ck(av_bsf_init(_bsfc));
    }
    if (_isMp4HEVC)
    {
      const AVBitStreamFilter* bsf = av_bsf_get_by_name("hevc_mp4toannexb");
      if (!bsf)
      {
        LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " "
                   << "av_bsf_get_by_name() failed";
        return;
      }
      ck(av_bsf_alloc(bsf, &_bsfc));
      avcodec_parameters_copy(_bsfc->par_in,
                              _fmtc->streams[_videoStreamIndex]->codecpar);
      ck(av_bsf_init(_bsfc));
    }
  }

  AVFormatContext* CreateFormatContext(DataProvider* pDataProvider)
  {

    AVFormatContext* ctx = nullptr;
    if (!(ctx = avformat_alloc_context()))
    {
      LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
      return nullptr;
    }

    uint8_t* _avioc_buffer = nullptr;
    int _avioc_buffer_size = 8 * 1024 * 1024;
    _avioc_buffer = (uint8_t*) av_malloc(_avioc_buffer_size);
    if (!_avioc_buffer)
    {
      LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
      return nullptr;
    }
    _avioc = avio_alloc_context(_avioc_buffer, _avioc_buffer_size, 0,
                                pDataProvider, &ReadPacket, nullptr, nullptr);
    if (!_avioc)
    {
      LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
      return nullptr;
    }
    ctx->pb = _avioc;

    ck(avformat_open_input(&ctx, nullptr, nullptr, nullptr));
    return ctx;
  }

  /**
   *   @brief  Allocate and return AVFormatContext*.
   *   @param  szFilePath - Filepath pointing to input stream.
   *   @return Pointer to AVFormatContext
   */
  AVFormatContext* CreateFormatContext(const char* szFilePath)
  {
    avformat_network_init();

    AVFormatContext* ctx = nullptr;
    ck(avformat_open_input(&ctx, szFilePath, nullptr, nullptr));
    return ctx;
  }

public:
  FFmpegDemuxer(const char* szFilePath)
    : FFmpegDemuxer(CreateFormatContext(szFilePath))
  {
  }

  FFmpegDemuxer(DataProvider* pDataProvider)
    : FFmpegDemuxer(CreateFormatContext(pDataProvider))
  {
    _avioc = _fmtc->pb;
  }

  ~FFmpegDemuxer()
  {

    if (!_fmtc)
    {
      return;
    }

    if (_pkt && _pkt->data)
    {
      av_packet_unref(_pkt);
      av_packet_free(&_pkt);
    }
    if (_pktFiltered && _pktFiltered->data)
    {
      av_packet_unref(_pktFiltered);
      av_packet_free(&_pktFiltered);
    }

    if (_bsfc)
    {
      av_bsf_free(&_bsfc);
    }

    avformat_close_input(&_fmtc);

    if (_avioc)
    {
      av_freep(&_avioc->buffer);
      av_freep(&_avioc);
    }

    if (_dataWithHeader)
    {
      av_free(_dataWithHeader);
    }
  }

  AVCodecID GetVideoCodec()
  {
    return _videoCodec;
  }

  AVPixelFormat GetChromaFormat()
  {
    return _chromaFormat;
  }

  int GetWidth()
  {
    return _width;
  }

  int GetHeight()
  {
    return _height;
  }

  int GetBitDepth()
  {
    return _bitDepth;
  }

  int GetFrameSize()
  {
    return _width * (_height + _chromaHeight) * _bytesPerPixel;
  }

  bool Demux(uint8_t** ppVideo, int* pnVideoBytes, int64_t* pts = nullptr)
  {
    if (!_fmtc)
      return false;

    *pnVideoBytes = 0;

    if (_pkt->data)
      av_packet_unref(_pkt);

    int e = 0;
    while ((e = av_read_frame(_fmtc, _pkt)) >= 0 &&
           _pkt->stream_index != _videoStreamIndex)
      av_packet_unref(_pkt);

    if (e < 0)
      return false;

    if (_isMp4H264 || _isMp4HEVC)
    {
      if (_pktFiltered->data)
        av_packet_unref(_pktFiltered);
      ck(av_bsf_send_packet(_bsfc, _pkt));
      ck(av_bsf_receive_packet(_bsfc, _pktFiltered));
      *ppVideo = _pktFiltered->data;
      *pnVideoBytes = _pktFiltered->size;
      if (pts)
        *pts = (int64_t)(_pktFiltered->pts * 1000 * _timeBase);
    }
    else
    {
      if (_isMp4MPEG4 && (_frameCount == 0))
      {
        int extraDataSize =
            _fmtc->streams[_videoStreamIndex]->codecpar->extradata_size;

        if (extraDataSize > 0)
        {

          // extradata contains start codes 00 00 01. Subtract its size
          _dataWithHeader = (uint8_t*) av_malloc(extraDataSize + _pkt->size -
                                                 3 * sizeof(uint8_t));

          if (!_dataWithHeader)
          {
            LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
            return false;
          }

          memcpy(_dataWithHeader,
                 _fmtc->streams[_videoStreamIndex]->codecpar->extradata,
                 extraDataSize);
          memcpy(_dataWithHeader + extraDataSize, _pkt->data + 3,
                 _pkt->size - 3 * sizeof(uint8_t));

          *ppVideo = _dataWithHeader;
          *pnVideoBytes = extraDataSize + _pkt->size - 3 * sizeof(uint8_t);
        }
      }
      else
      {
        *ppVideo = _pkt->data;
        *pnVideoBytes = _pkt->size;
      }

      if (pts)
        *pts = (int64_t)(_pkt->pts * 1000 * _timeBase);
    }

    _frameCount++;

    return true;
  }

  static int ReadPacket(void* opaque, uint8_t* pBuf, int nBuf)
  {
    return ((DataProvider*) opaque)->GetData(pBuf, nBuf);
  }
};


inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id)
{
  switch (id)
  {
  case AV_CODEC_ID_MPEG1VIDEO:
    return cudaVideoCodec_MPEG1;
  case AV_CODEC_ID_MPEG2VIDEO:
    return cudaVideoCodec_MPEG2;
  case AV_CODEC_ID_MPEG4:
    return cudaVideoCodec_MPEG4;
  case AV_CODEC_ID_VC1:
    return cudaVideoCodec_VC1;
  case AV_CODEC_ID_H264:
    return cudaVideoCodec_H264;
  case AV_CODEC_ID_HEVC:
    return cudaVideoCodec_HEVC;
  case AV_CODEC_ID_VP8:
    return cudaVideoCodec_VP8;
  case AV_CODEC_ID_VP9:
    return cudaVideoCodec_VP9;
  case AV_CODEC_ID_MJPEG:
    return cudaVideoCodec_JPEG;
  default:
    return cudaVideoCodec_NumCodecs;
  }
}
