#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "Halide.h"
#include "shakti_halide_rgb_to_gray.h"
#include "shakti_halide_gray32f_to_rgb.h"
#include "shakti_halide_gaussian_blur.h"

#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif


using namespace std;
using namespace DO::Sara;


namespace DO::Sara::HalideBackend {

  auto find_non_cuda_gpu_target() -> Halide::Target
  {
    using namespace Halide;

    Target target = get_host_target();

#ifdef _WIN32
    if (LoadLibraryA("d3d12.dll") != nullptr)
      target.set_feature(Target::D3D12Compute);
    else if (LoadLibraryA("OpenCL.dll") != nullptr)
      target.set_feature(Target::OpenCL);
#elif __APPLE__
    // OS X doesn't update its OpenCL drivers, so they tend to be broken.
    // CUDA would also be a fine choice on machines with NVidia GPUs.
    if (dlopen(
            "/System/Library/Frameworks/Metal.framework/Versions/Current/Metal",
            RTLD_LAZY) != NULL)
      target.set_feature(Target::Metal);
#else
    if (dlopen("libOpenCL.so", RTLD_LAZY) != NULL)
      target.set_feature(Target::OpenCL);
#endif

    return target;
  }

  auto get_gpu_target() -> Halide::Target
  {
    // Configure Halide to use CUDA before we compile the pipeline.
    constexpr auto use_cuda =
#if defined(__APPLE__)
        false;
#else
        true;
#endif

    auto target = Halide::Target{};
    if constexpr (use_cuda)
    {
      target = Halide::get_host_target();
      target.set_feature(Halide::Target::CUDA);
    }
    else
    {
      // We Will try to use in this order:
      // - Microsoft DirectX
      // - Apple Metal Performance Shaders
      // - OpenCL
      target = find_non_cuda_gpu_target();
    }

    return target;
  }

  auto as_interleaved_rgb_buffer(ImageView<Rgb8>& image)
  {
    return Halide::Runtime::Buffer<uint8_t>::make_interleaved(
        reinterpret_cast<uint8_t*>(image.data()), image.width(), image.height(),
        3);
  }

  template <typename T>
  auto as_buffer(ImageView<T>& image)
  {
    return Halide::Runtime::Buffer<T>(image.data(), image.width(),
                                      image.height());
  }

}  // namespace DO::Sara::halide


struct TicToc {
  // Timer.
  Timer timer;
  double elapsed;
} tictoc;

void tic()
{
  tictoc.timer.restart();
}

void toc(const std::string& what)
{
  const auto elapsed = tictoc.timer.elapsed_ms();
  std::cout << "[" << what << "] " << elapsed <<  " ms" << std::endl;
}


auto halide_pipeline() -> void
{
  namespace halide = HalideBackend;

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
  auto buffer_rgb = halide::as_interleaved_rgb_buffer(frame_rgb8);
  auto buffer_gray32f = halide::as_buffer<float>(frame_gray32f);
  auto buffer_gray32f_blurred = halide::as_buffer<float>(frame_gray32f);
  auto buffer_gray8 = halide::as_interleaved_rgb_buffer(frame_gray_as_rgb);

  const auto target = halide::get_gpu_target();

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
