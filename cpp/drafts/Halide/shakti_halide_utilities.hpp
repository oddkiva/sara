#pragma once

#include <DO/Sara/Core.hpp>

#include "Halide.h"

#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif


namespace DO::Shakti::HalideBackend {

  namespace sara = DO::Sara;

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


  auto as_interleaved_rgb_buffer(sara::ImageView<sara::Rgb8>& image)
  {
    return Halide::Buffer<uint8_t>::make_interleaved(
        reinterpret_cast<uint8_t*>(image.data()), image.width(), image.height(),
        3);
  }

  template <typename T>
  auto as_buffer(sara::ImageView<T>& image)
  {
    return Halide::Buffer<T>(image.data(), image.width(),
                                      image.height());
  }


  auto as_interleaved_rgb_runtime_buffer(sara::ImageView<sara::Rgb8>& image)
  {
    return Halide::Runtime::Buffer<uint8_t>::make_interleaved(
        reinterpret_cast<uint8_t*>(image.data()), image.width(), image.height(),
        3);
  }

  template <typename T>
  auto as_runtime_buffer(sara::ImageView<T>& image)
  {
    return Halide::Runtime::Buffer<T>(image.data(), image.width(),
                                      image.height());
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

  struct RgbToGray32f : public Pipeline<std::uint8_t>
  {
    using base_type = Pipeline<std::uint8_t>;

    mutable Halide::Func to_gray{"rgb_to_gray32f"};

    RgbToGray32f(Halide::Buffer<std::uint8_t>& input)
      : base_type{input}
    {
      auto r = input(x, y, 0) / 255.f;
      auto g = input(x, y, 1) / 255.f;
      auto b = input(x, y, 2) / 255.f;

      to_gray(x, y) = 0.2125f * r + 0.7154f * g + 0.0721f * b;

      to_gray.compile_jit(target);
    }

    auto operator()(Halide::Buffer<float>& output_buffer) const -> void
    {
      input_buffer.set_host_dirty();
      to_gray.realize(output_buffer);
      output_buffer.copy_to_host();
    }
  };

  template <typename T>
  struct Padded : public Pipeline<T>
  {
    using base_type = Pipeline<T>;

    mutable Halide::Func padded{"Padded"};

    Padded(Halide::Buffer<T>& input)
      : base_type{input}
    {
      using base_type::x;
      using base_type::y;
      using base_type::c;
      using base_type::input_buffer;
      padded(x, y, c) = cast(clamp(x, 0, input_buffer.width() - 1),   //
                             clamp(y, 0, input_buffer.height() - 1),  //
                             c);
    }
  };

  struct Gaussian : public Pipeline<float>
  {
    using base_type = Pipeline<float>;

    mutable Halide::Func padded{"Padded"};
    mutable Halide::Func conv_x{"ConvX"};
    mutable Halide::Func conv_y{"ConvY"};
    mutable Halide::Func kernel{"Kernel"};

    Gaussian(Halide::Buffer<float>& input, float sigma,
             float gaussian_truncation_factor = 4.f)
      : base_type{input}
    {
      using namespace Halide;

      const auto w = input.width();
      const auto h = input.height();
      const auto radius = cast<int>(sigma / 2) * truncation_factor;

      // Define the unnormalized gaussian function.
      auto gaussian_unnormalized = Func{"gaussian_unnormalized"};
      gaussian_unnormalized(x) = exp(-(x * x) / (2 * sigma * sigma));

      // Define the summation variable `k` defined on a summation domain.
      auto k = RDom(-radius, 2 * radius + 1);
      // Calculate the normalization factor by summing with the summation
      // variable.
      auto normalization_factor = sum(gaussian_unnormalized(k));

      auto gaussian = Func{"gaussian"};
      gaussian(x) = gaussian_unnormalized(x) / normalization_factor;

      // 1st pass: transpose and convolve the columns.
      auto input_t = Func{"input_transposed"};
      input_t(y, x) = input(x, y);

      auto conv_y_t = Func{"conv_y_t"};
      conv_y_t(x, y) = sum(input_t(clamp(x + k, 0, h - 1), y) * gaussian(k));


      // 2nd pass: transpose and convolve the rows.
      auto conv_y = Func{"conv_y"};
      conv_y(x, y) = conv_y_t(y, x);

      auto& conv_x = output;
      conv_x(x, y) = sum(conv_y(clamp(x + k, 0, w - 1), y) * gaussian(k));

      // GPU schedule.
      if (target)
      {
        // Compute the gaussian first.
        gaussian.compute_root();

        // 1st pass: transpose and convolve the columns
        conv_y_t.compute_root();
        conv_y_t.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y);

        // 2nd pass: transpose and convolve the rows.
        conv_x.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        // Compute the gaussian first.
        gaussian.compute_root();

        // 1st pass: transpose and convolve the columns
        conv_y_t.compute_root();
        conv_y_t.hexagon()
            .prefetch(conv_y_t, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);

        // 2nd pass: transpose and convolve the rows.
        conv_y.compute_root();
        conv_x.hexagon()
            .prefetch(conv_y_t, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);
      }

      // CPU schedule.
      else
      {
        // Compute the gaussian first.
        gaussian.compute_root();

        // 1st pass: transpose and convolve the columns
        conv_y_t.compute_root();
        conv_y_t.split(y, yo, yi, 8).parallel(yo).vectorize(x, 8);

        // 2nd pass: transpose and convolve the rows.
        conv_y.compute_root();
        conv_x.split(y, yo, yi, 8).parallel(yo).vectorize(x, 8);
      }

      conv_x.compile_jit(target);
    }

    auto operator()(Halide::Buffer<float>& output_buffer) const -> void
    {
      input_buffer.set_host_dirty();
      cast.realize(output_buffer);
      output_buffer.copy_to_host();
    }
  };

}  // namespace DO::Shakti::HalideBackend


namespace DO::Sara {

  struct TicToc : public Timer
  {
    static auto instance() -> TicToc&
    {
      static TicToc _instance;
      return _instance;
    }
  };

  inline auto tic()
  {
    TicToc::instance().restart();
  }

  inline auto toc(const std::string& what)
  {
    const auto elapsed = TicToc::instance().elapsed_ms();
    std::cout << "[" << what << "] " << elapsed << " ms" << std::endl;
  }

}  // namespace DO::Sara
