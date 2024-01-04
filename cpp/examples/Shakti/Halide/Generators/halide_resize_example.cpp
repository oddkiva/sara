#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Halide/Utilities.hpp>

#include "shakti_gray32f_to_rgb8u_cpu.h"
#include "shakti_rgb8u_to_gray32f_cpu.h"
#include "shakti_reduce_32f_gpu.h"


namespace halide = DO::Shakti::HalideBackend;


using namespace std;
using namespace DO::Sara;


auto reduce(ImageView<float>& src, ImageView<float>& dst)
{
  auto src_tensor_view =
      tensor_view(src).reshape(Vector4i{1, 1, src.height(), src.width()});
  auto dst_tensor_view =
      tensor_view(dst).reshape(Vector4i{1, 1, dst.height(), dst.width()});

  auto src_buffer = halide::as_runtime_buffer(src_tensor_view);
  auto dst_buffer = halide::as_runtime_buffer(dst_tensor_view);

  src_buffer.set_host_dirty();
  shakti_reduce_32f_gpu(src_buffer, dst.width(), dst.height(), dst_buffer);
  dst_buffer.copy_to_host();
}

auto halide_pipeline() -> void
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath = "C:/Users/David/Desktop/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath = "/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  VideoStream video_stream(video_filepath);

  // Input and output images.
  auto frame_rgb8 = video_stream.frame();
  auto frame_gray32f = Image<float>{video_stream.sizes()};
  auto frame_gray32f_reduced = Image<float>{video_stream.sizes() / 2};
  auto frame_gray32f_reduced_as_rgb = Image<Rgb8>{video_stream.sizes() / 2};

  // Halide input and output buffers.
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame_rgb8);
  auto buffer_gray32f = halide::as_runtime_buffer<float>(frame_gray32f);
  auto buffer_gray32f_reduced =
      halide::as_runtime_buffer<float>(frame_gray32f_reduced);
  auto buffer_gray8 =
      halide::as_interleaved_runtime_buffer(frame_gray32f_reduced_as_rgb);

  create_window(video_stream.sizes() / 2);
  while (true)
  {
    tic();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    toc("Video Decoding");

    tic();
    {
      // Use parallelisation and vectorization.
      shakti_rgb8u_to_gray32f_cpu(buffer_rgb, buffer_gray32f);
      ::reduce(frame_gray32f, frame_gray32f_reduced);
      shakti_gray32f_to_rgb8u_cpu(buffer_gray32f_reduced, buffer_gray8);
    }
    toc("Halide");

    display(frame_gray32f_reduced_as_rgb);
  }
}


GRAPHICS_MAIN()
{
  halide_pipeline();
  return 0;
}
