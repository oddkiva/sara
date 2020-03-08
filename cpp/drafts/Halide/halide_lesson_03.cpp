#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "Halide.h"

#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif


using namespace std;
using namespace DO::Sara;


namespace DO::Sara::halide {

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

  template <typename T = std::uint8_t>
  struct Pipeline
  {
    Halide::Var x{};
    Halide::Var y{};
    Halide::Var c{};

    Halide::Var xo{};
    Halide::Var yo{};
    Halide::Var xi{};
    Halide::Var yi{};

    Halide::Buffer<T> input_buffer;

    explicit Pipeline(const ImageView<T>& input_image)
      : input_buffer{Halide::Buffer<T>::make_interleaved(
            const_cast<T*>(input_image.data()), input_image.width(),
            input_image.height(), 3)}
    {
    }

    explicit Pipeline(const ImageView<Rgb8>& input_image)
      : input_buffer{Halide::Buffer<T>::make_interleaved(
            const_cast<std::uint8_t*>(
                reinterpret_cast<const std::uint8_t*>(input_image.data())),
            input_image.width(), input_image.height(), 3)}
    {
      static_assert(std::is_same_v<T, std::uint8_t>);
    }
  };

  struct CastToFloat : public Pipeline<std::uint8_t>
  {
    using base_type = Pipeline<std::uint8_t>

    Halide::Func cast{"Cast"};
    Halide::Buffer<float> output_buffer;

    explicit CastToFloat(const ImageView<Rgb8>& input_image,
                         ImageView<Rgb8>& output_image)
      : {Halide::Buffer<uint8_t>::make_interleaved(
            (uint8_t*) (input_image.data()), input_image.width(),
            input_image.height(), 3)}
      , output_buffer{Halide::Buffer<uint8_t>::make_interleaved(
            reinterpret_cast<uint8_t*>(output_image.data()),
            output_image.width(), output_image.height(), 3)}
    {
      // Halide uses the following indexing (x, y, c)
      // 0: x-coordinate
      // 1: y-coordinate
      // 2: c being the channel index.
      cast(x, y, c) = input_buffer(x, y, c) / 255.f;

      // The image is assumed to be in packed pixel format.
      //             0   1       w - 1
      // row 0     : RGB RGB ... RGB
      // row 1     : RGB RGB ... RGB
      // ...
      // row h - 1 : RGB RGB ... RGB
      cast.output_buffer()    //
          .dim(0)             // x
          .set_stride(3)      // because of the packed pixel format.
          .dim(2)             // c
          .set_stride(1)      // because of the packed pixel format
          .set_bounds(0, 3);  // RGB = [0, 1, 2] so 3 channels.

      // for y:
      //   for x:
      //     for c: <- unroll this loop
      cast.reorder(c, x, y).bound(c, 0, 3).unroll(c);

      cast.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
    };

    // auto operator()(const )
    // {
    //   input_buffer.set_host_dirty();
    //   cast.realize(output_buffer);
    //   output_buffer.copy_to_host();
    // }
  };

  struct Deriche : Pipeline<std::uint8_t>
  {
    Halide::Func deriche{"Deriche"};

    Deriche()
    {
      deriche(x, y, c) = input_buffer(x, y, c) / 255.f;

      deriche
          .output_buffer()    //
          .dim(0)             // x
          .set_stride(3)      // because of the packed pixel format.
          .dim(2)             // c
          .set_stride(1)      // because of the packed pixel format
          .set_bounds(0, 3);  // RGB = [0, 1, 2] so 3 channels.

      deriche.reorder(c, x, y).bound(c, 0, 3).unroll(c);

      deriche.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
    };
  };

}  // namespace DO::Sara::halide

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
  // const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  VideoStream video_stream(video_filepath);

  // Configure Halide to use the GPU.
  auto target = halide::get_gpu_target();

  // Input and output images.
  auto input_image = video_stream.frame();
  auto output_image = Image<Rgb8>{video_stream.sizes()};

  // Halide input and output buffers.
  auto input_buffer = Halide::Buffer<uint8_t>::make_interleaved(
      reinterpret_cast<uint8_t*>(input_image.data()), input_image.width(),
      input_image.height(), 3);

  auto output_buffer = Halide::Buffer<uint8_t>::make_interleaved(
      reinterpret_cast<uint8_t*>(output_image.data()), output_image.width(),
      output_image.height(), 3);

  // Halide pipeline.
  auto x = Halide::Var{};
  auto y = Halide::Var{};
  auto c = Halide::Var{};

  auto xo = Halide::Var{};
  auto yo = Halide::Var{};
  auto xi = Halide::Var{};
  auto yi = Halide::Var{};

  // Implicitly cast into RGB planar format.
  auto cast = Halide::Func{"cast"};
  cast(x, y, c) = input_buffer(x, y, c) / 255.f;

  auto padded = Halide::Func{"padded"};
  padded(x, y, c) = cast(clamp(x, 0, input_buffer.width() - 1),   //
                         clamp(y, 0, input_buffer.height() - 1),  //
                         c);

  // The filter
  auto filter = Halide::Func{"filter"};

  // Laplacian.
  // filter(x, y, c) = Halide::abs(
  //    padded(x, y, c) - (padded(x + 1, y + 0, c) + padded(x - 1, y + 0, c) +
  //                       padded(x + 0, y + 1, c) + padded(x + 0, y - 1, c)) /
  //                          4.f);

  // Blur.
  filter(x, y, c) = (padded(x - 1, y - 1, c) + padded(x - 0, y - 1, c) +
                     padded(x + 1, y - 1, c) + padded(x - 1, y + 0, c) +
                     padded(x - 0, y + 0, c) + padded(x + 1, y + 0, c) +
                     padded(x - 1, y + 1, c) + padded(x - 0, y + 1, c) +
                     padded(x + 1, y + 1, c)) /
                    9.f;

  // The output result to show on the screen.
  auto filter_rescaled = Halide::Func{"rescaled"};
  // filter_rescaled(x, y, c) =
  //     Halide::cast<uint8_t>((filter(x, y, c) / 2.f) * 255.f);
  filter_rescaled(x, y, c) = Halide::cast<uint8_t>(filter(x, y, c) * 255.f);

  // Specify that the output buffer is in interleaved RGB format.
  filter_rescaled.output_buffer()
      .dim(0)
      .set_stride(3)
      .dim(2)
      .set_stride(1)
      .set_bounds(0, 3);
  filter_rescaled.reorder(c, x, y).bound(c, 0, 3).unroll(c);
  filter_rescaled.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);

  // Calculate the padded image as an intermediate result for the laplacian
  // calculation in a shared memory.
  padded.compute_at(filter_rescaled, xo);
  padded.gpu_threads(x, y);

  filter_rescaled.compile_jit(target);

  // Timer.
  auto timer = Timer{};
  auto elapsed = double{};

  create_window(video_stream.sizes());
  while (true)
  {
    timer.restart();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    elapsed = timer.elapsed_ms();
    std::cout << "Video decoding time = " << elapsed << " ms" << std::endl;

    timer.restart();
    {
      input_buffer.set_host_dirty();
      filter_rescaled.realize(output_buffer);
      output_buffer.copy_to_host();
    }
    elapsed = timer.elapsed_ms();
    std::cout << "Halide computation time = " << elapsed << " ms" << std::endl;

    display(output_image);
  }
}


auto halide_deriche_pipeline(const std::string& video_filepath) -> void
{
  VideoStream video_stream(video_filepath);

  const auto gpu_target = halide::get_gpu_target();

  // Input and output images.
  auto input_image = video_stream.frame();
  auto output_image = Image<Rgb8>{video_stream.sizes()};

  // Halide input and output buffers.
  auto input_buffer = Halide::Buffer<uint8_t>::make_interleaved(
      reinterpret_cast<uint8_t*>(input_image.data()), input_image.width(),
      input_image.height(), 3);

  auto output_buffer = Halide::Buffer<uint8_t>::make_interleaved(
      reinterpret_cast<uint8_t*>(output_image.data()), output_image.width(),
      output_image.height(), 3);

  // Halide pipeline.
  auto x = Halide::Var{};
  auto y = Halide::Var{};
  auto c = Halide::Var{};

  auto xo = Halide::Var{};
  auto yo = Halide::Var{};
  auto xi = Halide::Var{};
  auto yi = Halide::Var{};

  // Implicitly cast into RGB planar format.
  auto cast = Halide::Func{"cast"};
  cast(x, y, c) = input_buffer(x, y, c) / 255.f;

  Halide::RDom ry(1, input_image.height() - 1);
  Halide::Expr flip_ry = input_image.height() - ry - 1;
  Halide::Expr alpha;

  // The filter
  auto deriche = Halide::Func{"filter"};
  deriche(x, ry, c) =
      (1 - alpha) * deriche(x, ry - 1, c) + alpha * cast(x, ry, c);

  // The output result to show on the screen.
  auto filter_rescaled = Halide::Func{"rescaled"};
  filter_rescaled(x, y, c) = Halide::cast<uint8_t>(deriche(x, y, c) * 255.f);

  // Specify that the output buffer is in interleaved RGB format.
  filter_rescaled.output_buffer()
      .dim(0)
      .set_stride(3)
      .dim(2)
      .set_stride(1)
      .set_bounds(0, 3);
  filter_rescaled.reorder(c, x, y).bound(c, 0, 3).unroll(c);
  filter_rescaled.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);

  // Calculate the padded image as an intermediate result for the laplacian
  // calculation in a shared memory.
  // padded.compute_at(filter_rescaled, xo);
  // padded.gpu_threads(x, y);

  filter_rescaled.compile_jit(gpu_target);

  // Timer.
  auto timer = Timer{};
  auto elapsed = double{};

  create_window(video_stream.sizes());
  while (true)
  {
    timer.restart();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    elapsed = timer.elapsed_ms();
    std::cout << "Video decoding time = " << elapsed << " ms" << std::endl;

    timer.restart();
    {
      input_buffer.set_host_dirty();
      filter_rescaled.realize(output_buffer);
      output_buffer.copy_to_host();
    }
    elapsed = timer.elapsed_ms();
    std::cout << "Halide computation time = " << elapsed << " ms" << std::endl;

    display(output_image);
  }
}


GRAPHICS_MAIN()
{
  halide_pipeline();
  return 0;
}
