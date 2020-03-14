#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "shakti_halide_utilities.hpp"

#include "shakti_halide_rgb_to_gray.h"
#include "shakti_halide_gray32f_to_rgb.h"
#include "shakti_halide_gaussian_blur.h"


namespace halide = DO::Shakti::HalideBackend;


using namespace std;
using namespace DO::Sara;


auto halide_pipeline() -> void
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath =
      "/Users/david/GitLab/DO-CV/sara/cpp/examples/Sara/VideoIO/orion_1.mpg"s;
#else
  const auto video_filepath = "/home/david/Desktop/test.mp4"s;
  //const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  VideoStream video_stream(video_filepath);

  // Input and output images.
  auto frame_rgb8 = video_stream.frame();
  auto frame_gray32f = Image<float>{video_stream.sizes()};
  auto frame_gray32f_blurred = Image<float>{video_stream.sizes()};
  auto frame_gray_as_rgb = Image<Rgb8>{video_stream.sizes()};

  // Halide input and output buffers.
  auto buffer_rgb = halide::as_interleaved_rgb_runtime_buffer(frame_rgb8);
  auto buffer_gray32f = halide::as_runtime_buffer<float>(frame_gray32f);
  auto buffer_gray32f_blurred = halide::as_runtime_buffer<float>(frame_gray32f);
  auto buffer_gray8 = halide::as_interleaved_rgb_runtime_buffer(frame_gray_as_rgb);

  /* const auto target = */ halide::get_gpu_target();

  // shakti_halide_rgb_to_gray_argv(target);
  // shakti_halide_gray32f_to_rgb_argv(target);

  create_window(video_stream.sizes());
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
      buffer_rgb.set_host_dirty();

      shakti_halide_rgb_to_gray(buffer_rgb, buffer_gray32f);
      shakti_halide_gaussian_blur(buffer_gray32f, buffer_gray32f_blurred);
      shakti_halide_gray32f_to_rgb(buffer_gray32f_blurred, buffer_gray8);

      buffer_gray8.copy_to_host();
    }
    toc("Halide");

    display(frame_gray_as_rgb);
  }
}


GRAPHICS_MAIN()
{
  halide_pipeline();
  return 0;
}
