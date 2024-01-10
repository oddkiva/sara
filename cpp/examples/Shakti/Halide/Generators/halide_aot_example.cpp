#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Halide/GaussianConvolution.hpp>
#include <DO/Shakti/Halide/Utilities.hpp>

#include "shakti_gray32f_to_rgb8u_cpu.h"
#include "shakti_rgb8u_to_gray32f_cpu.h"


namespace halide = DO::Shakti::HalideBackend;


using namespace std;
using namespace DO::Sara;


auto halide_pipeline(int argc, char** argv) -> int
{
  using namespace std::string_literals;

  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " video_path" << std::endl;
    return 1;
  }

  const auto video_filepath = argv[1];
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

  const auto sigma = 5.f;
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
      shakti_rgb8u_to_gray32f_cpu(buffer_rgb, buffer_gray32f);

// #define USE_HALIDE_AOT_GPU_IMPL
#define USE_SARA_GAUSSIAN_BLUR_IMPLEMENTATION
// #define USE_SARA_DERICHE_IMPLEMENTATION
#if defined(USE_HALIDE_AOT_GPU_IMPL)
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
      //   more than 3 the computation time (from ~50ms to ~15ms)
      //
      // The cumulated improvements are spectacular (14x faster) all along with
      // very little effort.
      //
      // On a CUDA-capable GPU, it takes ~7ms to process the same images. So the
      // CPU version is quite fast.
      // On the GPU, with sigma = 80.f, the processing time is about ~15ms!
      halide::gaussian_convolution(frame_gray32f, frame_gray32f_blurred, sigma,
                                   truncation_factor);
      shakti_gray32f_to_rgb8u_cpu(buffer_gray32f_blurred, buffer_gray8);
      toc("Halide Gaussian");
#elif defined(USE_SARA_GAUSSIAN_BLUR_IMPLEMENTATION)
      // Sara's unoptimized code takes 240 ms to blur (no SSE instructions and
      // no column-based transposition)
      //
      // Parallelizing the implementation of the linear filtering with OpenMP,
      // we are then down to 25ms, not bad at all for a very minimal change!
      //
      // I have implemented a better schedule for CPU, it performs better than
      // Halide GPU implementation (OMG!).
      apply_gaussian_filter(frame_gray32f, frame_gray32f_blurred, sigma,
                            truncation_factor);
      shakti_gray32f_to_rgb8u_cpu(buffer_gray32f_blurred, buffer_gray8);
      toc("Sara Gaussian");
#elif defined(USE_SARA_DERICHE_IMPLEMENTATION)
      // Without parallelization and anything, deriche filter is still running
      // reasonably fast (between 45 and 50ms).
      inplace_deriche_blur(frame_gray32f, sigma);
      shakti_gray32f_to_rgb8u_cpu(buffer_gray32f, buffer_gray8);
      toc("Sara Deriche");
#endif
    }

    display(frame_gray_as_rgb);
  }

  return 0;
}


auto main(int argc, char** const argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(halide_pipeline);
  return app.exec();
}
