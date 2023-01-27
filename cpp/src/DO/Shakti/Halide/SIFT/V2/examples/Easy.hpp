#pragma once

#include <DO/Sara/Core.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Shakti/Halide/Utilities.hpp>

#if defined(USE_SHAKTI_CUDA_VIDEOIO)
#  include <DO/Shakti/Cuda/VideoIO.hpp>
#else
#  include <DO/Sara/VideoIO.hpp>
#endif

#include <filesystem>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace halide = DO::Shakti::HalideBackend;


namespace easy {

  struct VideoStream
  {
    VideoStream(const std::filesystem::path& video_filepath,
                DriverApi::CudaContext& cuda_context)
      : _video_stream{
            new shakti::VideoStream{video_filepath.string(), cuda_context}}
    {
      if (_video_stream.get() == nullptr)
        throw std::runtime_error{
            "Failed to initialize GPU video stream object"};
      _host_frame.resize(_video_stream->sizes());
      //_device_frame.reset(new DriverApi::DeviceBgraBuffer{
      //    _video_stream->width(), _video_stream->height()});
      _device_frame = DriverApi::DeviceBgraBuffer{_video_stream->width(),
                                                  _video_stream->height()};
    }

    auto host_frame() -> const sara::Image<sara::Bgra8>&
    {
      return _host_frame;
    }

    auto width() const noexcept -> int
    {
      return _video_stream->width();
    }

    auto height() const noexcept -> int
    {
      return _video_stream->height();
    }

    auto sizes() const noexcept -> Eigen::Vector2i
    {
      return _video_stream->sizes();
    }

    auto read() -> bool
    {
#ifdef USE_SHAKTI_CUDA_VIDEOIO
      // const auto has_frame = _video_stream->read(*_device_frame);
      const auto has_frame = _video_stream->read(_device_frame);
      sara::tic();
      //_device_frame->to_host(_host_frame);
      _device_frame.to_host(_host_frame);
      sara::toc("Copy to host");
#else
      const auto has_frame = video_stream.read();
#endif

      return has_frame;
    }

#ifdef USE_SHAKTI_CUDA_VIDEOIO
    sara::Image<sara::Bgra8> _host_frame;
    // std::unique_ptr<DriverApi::DeviceBgraBuffer> _device_frame;
    DriverApi::DeviceBgraBuffer _device_frame;
    std::unique_ptr<shakti::VideoStream> _video_stream;
#else
    std::unique_ptr<sara::VideoStream> _video_stream;
#endif
  };

  struct ToGrayscaleColorConverter
  {
    ToGrayscaleColorConverter(const Eigen::Vector2i& sizes)
    {
      _frame_gray32f = sara::Image<float>{sizes};
      auto _frame_gray_tensor =
          sara::tensor_view(_frame_gray32f)
              .reshape(Eigen::Vector4i{1, 1, sizes.y(), sizes.x()});
      _buffer_gray_4d = halide::as_runtime_buffer(_frame_gray_tensor);
    }

    sara::Image<float> _frame_gray32f;
    ::Halide::Runtime::Buffer<float> _buffer_gray_4d;

    auto operator()(const sara::ImageView<sara::Bgra8>& frame) -> void
    {
      sara::from_bgra8_to_gray32f(frame, _frame_gray32f);
      // Prepare the GPU buffer.
      _buffer_gray_4d.set_host_dirty();
    }

    auto operator()(const sara::ImageView<sara::Rgb8>& frame) -> void
    {
      sara::from_rgb8_to_gray32f(frame, _frame_gray32f);
      // Prepare the GPU buffer.
      _buffer_gray_4d.set_host_dirty();
    }

    auto device_buffer() -> ::Halide::Runtime::Buffer<float>&
    {
      return _buffer_gray_4d;
    }
  };

}  // namespace easy