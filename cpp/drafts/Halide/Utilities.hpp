#pragma once

#include <DO/Sara/Core.hpp>

#include <Halide.h>

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


  template <typename T, typename ColorSpace>
  inline auto
  as_interleaved_buffer(sara::ImageView<sara::Pixel<T, ColorSpace>>& image)
  {
    return Halide::Buffer<T>::make_interleaved(
        reinterpret_cast<T*>(image.data()), image.width(), image.height(),
        sara::Pixel<T, ColorSpace>::num_channels());
  }

  template <typename T, typename ColorSpace>
  auto as_interleaved_runtime_buffer(
      sara::ImageView<sara::Pixel<T, ColorSpace>>& image)
  {
    return Halide::Runtime::Buffer<T>::make_interleaved(
        reinterpret_cast<T*>(image.data()), image.width(), image.height(),
        sara::Pixel<T, ColorSpace>::num_channels());
  }

  template <typename T>
  inline auto as_buffer(sara::ImageView<T>& image)
  {
    return Halide::Buffer<T>(image.data(), image.width(), image.height());
  }

  template <typename T>
  inline auto as_buffer(sara::ImageView<T, 3>& image)
  {
    return Halide::Buffer<T>(image.data(), image.size(0), image.size(1),
                             image.size(2));
  }

  template <typename T>
  auto as_buffer_3d(sara::ImageView<T>& image)
  {
    static constexpr auto num_channels = sara::PixelTraits<T>::num_channels;
    return Halide::Buffer<T>(image.data(), image.width(), image.height(),
                             num_channels);
  }

  template <typename T>
  auto as_runtime_buffer(sara::ImageView<T>& image)
  {
    return Halide::Runtime::Buffer<T>(image.data(), image.width(),
                                      image.height());
  }

  template <typename T>
  inline auto as_runtime_buffer(sara::TensorView_<T, 3>& chw_tensor)
  {
    return Halide::Runtime::Buffer<T>(chw_tensor.data(), chw_tensor.size(2),
                                      chw_tensor.size(1), chw_tensor.size(0));
  }

  template <typename T>
  auto as_runtime_buffer_3d(sara::ImageView<T>& image)
  {
    static constexpr auto num_channels = sara::PixelTraits<T>::num_channels;
    return Halide::Runtime::Buffer<T>(image.data(), image.width(),
                                      image.height(), num_channels);
  }

  template <typename T>
  inline auto as_buffer(std::vector<T>& v)
  {
    return Halide::Buffer<T>(v.data(), v.size());
  }

  template <typename T>
  inline auto as_runtime_buffer(std::vector<T>& v)
  {
    return Halide::Runtime::Buffer<T>(v.data(), v.size());
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
