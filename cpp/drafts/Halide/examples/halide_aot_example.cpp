#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "../Utilities.hpp"

#include "shakti_halide_gaussian_blur.h"
#include "shakti_halide_gray32f_to_rgb.h"
#include "shakti_halide_rgb_to_gray.h"


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
      "/Users/david/Desktop/Datasets/humanising-autonomy/turn_bikes.mp4"s;
#else
  // const auto video_filepath = "/home/david/Desktop/test.mp4"s;
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  VideoStream video_stream(video_filepath);

  // Input and output images.
  auto frame_rgb8 = video_stream.frame();
  auto frame_gray32f = Image<float>{video_stream.sizes()};
  auto frame_gray32f_blurred = Image<float>{video_stream.sizes()};
  auto frame_gray_as_rgb = Image<Rgb8>{video_stream.sizes()};

  // Halide input and output buffers.
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame_rgb8);
  auto buffer_gray32f = halide::as_runtime_buffer<float>(frame_gray32f);
  auto buffer_gray32f_blurred =
      halide::as_runtime_buffer<float>(frame_gray32f_blurred);
  auto buffer_gray8 = halide::as_interleaved_runtime_buffer(frame_gray_as_rgb);

  const auto sigma = 3.f;
  const auto truncation_factor = 4;

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
      // Use parallelization and vectorization.
      shakti_halide_rgb_to_gray(buffer_rgb, buffer_gray32f);

#define USE_HALIDE_AOT_IMPLEMENTATION
#ifdef USE_HALIDE_AOT_IMPLEMENTATION
      // The strategy is to transpose the array and then convolve the rows. So
      // (1) we transpose the matrix and convolve the (transposed) columns.
      // (2) we transpose the matrix and convolve the rows.
      //
      // The timing are reported on a machine with:
      // - CPU Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz (cat /proc/cpuinfo)
      // - nVidia Titan X (Pascal) (nvidia-smi)
      //
      // On the CPU, using SSE instructions to vectorise the data divides by 4
      // the computation time:
      // - convolving a 1920x1080 image goes down from ~210ms to ~50ms.
      //   which is a dramatic improvement.
      // - Then parallelizing over the columns on the transposed data divides by
      // more
      //   than 3 the computation time (from ~50ms to ~15ms)
      //
      // The cumulated improvements are spectacular (14x faster) all along with
      // very little effort.
      //
      // On a CUDA-capable GPU, it takes ~7ms to process the same images. So the
      // CPU version is quite fast.
      // On the GPU, with sigma = 80.f, the processing time is about ~15ms!
      {
        buffer_gray32f.set_host_dirty();
        shakti_halide_gaussian_blur(buffer_gray32f, sigma, truncation_factor,
                                    buffer_gray32f_blurred);
        buffer_gray32f_blurred.copy_to_host();
      }
      shakti_halide_gray32f_to_rgb(buffer_gray32f_blurred, buffer_gray8);
#elif defined(USE_SARA_GAUSSIAN_BLUR_IMPLEMENTATION)
      // Sara's unoptimized code takes 240 ms to blur (no SSE instructions and
      // no column-based transposition)
      apply_gaussian_filter(frame_gray32f, frame_gray32f_blurred, sigma);
      shakti_halide_gray32f_to_rgb(buffer_gray32f_blurred, buffer_gray8);
#elif defined(USE_SARA_DERICHE_IMPLEMENTATION)
      // Without parallelization and anything, deriche filter is still running
      // reasonably fast (between 45 and 50ms).
      inplace_deriche_blur(frame_gray32f, sigma);
      shakti_halide_gray32f_to_rgb(buffer_gray32f, buffer_gray8);
#endif
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
