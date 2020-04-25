#include "nvidia-video-codec-sdk-9.1.23/NvCodec/NvDecoder/NvDecoder.h"
#include "nvidia-video-codec-sdk-9.1.23/Utils/ColorSpace.h"
#include "nvidia-video-codec-sdk-9.1.23/Utils/FFmpegDemuxer.h"
#include "nvidia-video-codec-sdk-9.1.23/Utils/NvCodecUtils.h"

#include "FramePresenterGLV2.h"


simplelogger::Logger* logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

namespace driver_api {

  auto init()
  {
    ck(cuInit(0));
  }

  auto get_device_count()
  {
    auto num_gpus = 0;
    ck(cuDeviceGetCount(&num_gpus));
    return num_gpus;
  }

  struct CudaContext
  {
    CUcontext cuda_context{NULL};
    CUdevice cuda_device{NULL};
    int gpu_id{-1};

    CudaContext(int gpu_id_ = 0)
      : gpu_id{gpu_id_}
    {
      ck(cuDeviceGet(&cuda_device, gpu_id));

      char device_name[80];
      ck(cuDeviceGetName(device_name, sizeof(device_name), cuda_device));
      std::cout << "GPU in use: " << device_name << std::endl;

      ck(cuCtxCreate(&cuda_context, CU_CTX_BLOCKING_SYNC, cuda_device));
    }

    CudaContext(CudaContext&& other)
    {
      std::swap(gpu_id, other.gpu_id);
      std::swap(cuda_context, other.cuda_context);
      std::swap(cuda_device, other.cuda_device);
    }

    CudaContext(const CudaContext&) = delete;

    ~CudaContext()
    {
      if (cuda_context)
      {
        ck(cuCtxDestroy(cuda_context));
        cuda_context = NULL;
        cuda_device = NULL;
        gpu_id = -1;
      }
    }

    operator CUcontext() const
    {
      return cuda_context;
    }

    auto make_current()
    {
      ck(cuCtxSetCurrent(cuda_context));
    }
  };

}  // namespace driver_api


struct VideoStream
{
  FFmpegDemuxer demuxer;
  NvDecoder dec;

  std::uint8_t** video_frame_packet_flags{nullptr};

  std::int32_t num_frames_decoded{};
  std::int32_t frame_index{};

  struct EncodedVideoBuffer
  {
    std::uint8_t* data{nullptr};
    std::int32_t size{};
  } encoded_video_buffer;

  struct DeviceFrameBuffer
  {
    std::uint8_t* cuda_device_ptr{nullptr};
    std::int32_t pitch_size{};
  } device_frame_buffer;

  VideoStream(const std::string& video_filepath,
              const driver_api::CudaContext& context)
    : demuxer{video_filepath.c_str()}
    , dec{context.cuda_context, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec())}
  {
  }

  auto read() -> bool
  {
    demuxer.Demux(&encoded_video_buffer.data, &encoded_video_buffer.size);
    dec.Decode(encoded_video_buffer.data, encoded_video_buffer.size,
               &video_frame_packet_flags, &num_frames_decoded);

    // Let's check if we need to decode again the video again.
    if (frame_index > 0 && num_frames_decoded)
      LOG(INFO) << dec.GetVideoInfo();

    if (frame_index == num_frames_decoded)
  }
};
