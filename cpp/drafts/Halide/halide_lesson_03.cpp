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
    return Halide::Buffer<uint8_t>::make_interleaved(
        reinterpret_cast<uint8_t*>(image.data()), image.width(), image.height(),
        3);
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

    Halide::Target target{get_gpu_target()};

    Halide::Buffer<T>& input_buffer;

    Pipeline(Halide::Buffer<T>& input)
      : input_buffer{input}
    {
    }

    auto set_target(const Halide::Target& t) -> void
    {
      target = t;
    }

    auto set_output_format_to_packed_rgb(Halide::Func& f) -> void
    {
      f.output_buffer()       //
          .dim(0)             // x
          .set_stride(3)      // because of the packed pixel format.
          .dim(2)             // c
          .set_stride(1)      // because of the packed pixel format
          .set_bounds(0, 3);  // RGB = [0, 1, 2] so 3 channels.

      // for y:
      //   for x:
      //     for c: <- unroll this loop
      f.reorder(c, x, y).bound(c, 0, 3).unroll(c);

      f.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
    }
  };

  struct CastToFloat : public Pipeline<std::uint8_t>
  {
    using base_type = Pipeline<std::uint8_t>;

    mutable Halide::Func cast{"Cast"};

    CastToFloat(Halide::Buffer<std::uint8_t>& input)
      : base_type{input}
    {
      cast(x, y, c) = input_buffer(x, y, c) / 255.f;
    }

    auto operator()(Halide::Buffer<float>& output_buffer) const -> void
    {
      input_buffer.set_host_dirty();
      cast.realize(output_buffer);
      output_buffer.copy_to_host();
    }
  };

  struct Padded : public CastToFloat
  {
    using base_type = CastToFloat;

    mutable Halide::Func padded{"Padded"};

    Padded(Halide::Buffer<std::uint8_t>& input)
      : base_type{input}
    {
      padded(x, y, c) = cast(clamp(x, 0, input_buffer.width() - 1),   //
                             clamp(y, 0, input_buffer.height() - 1),  //
                             c);
    }
  };

  struct Blur3x3 : public Padded
  {
    using base_type = Padded;

    mutable Halide::Func blur{"Blur3x3"};

    Blur3x3(Halide::Buffer<std::uint8_t>& input)
      : base_type{input}
    {
      blur(x, y, c) = (padded(x - 1, y - 1, c) + padded(x - 0, y - 1, c) +
                       padded(x + 1, y - 1, c) + padded(x - 1, y + 0, c) +
                       padded(x - 0, y + 0, c) + padded(x + 1, y + 0, c) +
                       padded(x - 1, y + 1, c) + padded(x - 0, y + 1, c) +
                       padded(x + 1, y + 1, c)) /
                      9.f;
    }
  };

  struct Laplacian : public Padded
  {
    using base_type = Padded;

    mutable Halide::Func laplacian{"Laplacian"};

    Laplacian(Halide::Buffer<std::uint8_t>& input)
      : base_type{input}
    {
      laplacian(x, y, c) = padded(x, y, c) -
                           (padded(x + 1, y + 0, c) + padded(x - 1, y + 0, c) +
                            padded(x + 0, y + 1, c) + padded(x + 0, y - 1, c)) /
                               4.f;
    }
  };

  struct Blur3x3Vis : public Blur3x3
  {
    using base_type = Blur3x3;

    mutable Halide::Func rescaled{"Rescaled"};

    Blur3x3Vis(Halide::Buffer<std::uint8_t>& input)
      : base_type{input}
    {
      // The output result to show on the screen.
      rescaled(x, y, c) = Halide::cast<uint8_t>(blur(x, y, c) * 255.f);

      set_output_format_to_packed_rgb(rescaled);

      // Calculate the padded image as an intermediate result for the laplacian
      // calculation in a shared memory.
      padded.compute_at(rescaled, xo);
      padded.gpu_threads(x, y);

      rescaled.compile_jit(target);
    }

    auto operator()(Halide::Buffer<std::uint8_t>& output_buffer) const -> void
    {
      input_buffer.set_host_dirty();
      rescaled.realize(output_buffer);
      output_buffer.copy_to_host();
    }
  };

  struct LaplacianVis : public Laplacian
  {
    using base_type = Laplacian;

    mutable Halide::Func rescaled{"Rescaled"};

    LaplacianVis(Halide::Buffer<std::uint8_t>& input)
      : base_type{input}
    {
      // The output result to show on the screen.
      rescaled(x, y, c) = Halide::cast<uint8_t>(laplacian(x, y, c) * 255.f);

      set_output_format_to_packed_rgb(rescaled);

      // Calculate the padded image as an intermediate result for the laplacian
      // calculation in a shared memory.
      padded.compute_at(rescaled, xo);
      padded.gpu_threads(x, y);

      rescaled.compile_jit(target);
    }

    auto operator()(Halide::Buffer<std::uint8_t>& output_buffer) const -> void
    {
      input_buffer.set_host_dirty();
      rescaled.realize(output_buffer);
      output_buffer.copy_to_host();
    }
  };

  struct Deriche : Pipeline<std::uint8_t>
  {
    using base_type = Pipeline<std::uint8_t>;

    Halide::Func deriche{"Deriche"};

    Deriche(Halide::Buffer<std::uint8_t>& input)
      : base_type{input}
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
  // const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  VideoStream video_stream(video_filepath);

  // Input and output images.
  auto input_image = video_stream.frame();
  auto output_image = Image<Rgb8>{video_stream.sizes()};

  // Halide input and output buffers.
  auto input_buffer = halide::as_interleaved_rgb_buffer(input_image);
  auto output_buffer = halide::as_interleaved_rgb_buffer(output_image);

  // Blur pipeline.
  //const auto pipeline = halide::Blur3x3Vis{input_buffer};
  const auto pipeline = halide::LaplacianVis{input_buffer};


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
    pipeline(output_buffer);
    toc("Halide");

    display(output_image);
  }
}


GRAPHICS_MAIN()
{
  halide_pipeline();
  return 0;
}
