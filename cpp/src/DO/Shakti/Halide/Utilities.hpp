#pragma once

#include <DO/Sara/Core/TicToc.hpp>

#include <DO/Shakti/Halide/MyHalide.hpp>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif


namespace DO::Shakti::HalideBackend {

  namespace sara = DO::Sara;

  inline auto find_non_cuda_gpu_target() -> Halide::Target
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
    {
      target.set_feature(Target::Metal);
    }
#else
    if (dlopen("libOpenCL.so", RTLD_LAZY) != NULL)
      target.set_feature(Target::OpenCL);
#endif

    return target;
  }

  inline auto get_gpu_target() -> Halide::Target
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
  inline auto as_interleaved_runtime_buffer(
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
  inline auto as_buffer_3d(sara::ImageView<T>& image)
  {
    static constexpr auto num_channels = sara::PixelTraits<T>::num_channels;
    return Halide::Buffer<T>(image.data(), image.width(), image.height(),
                             num_channels);
  }

  template <typename T>
  inline auto as_runtime_buffer(sara::ImageView<T>& image)
  {
    return Halide::Runtime::Buffer<T>(image.data(), image.width(),
                                      image.height());
  }

  template <typename T>
  inline auto as_runtime_buffer(sara::TensorView_<T, 2>& hw_tensor)
  {
    return Halide::Runtime::Buffer<T>(hw_tensor.data(), hw_tensor.size(1),
                                      hw_tensor.size(0));
  }

  template <typename T>
  inline auto as_runtime_buffer(sara::TensorView_<T, 3>& chw_tensor)
  {
    return Halide::Runtime::Buffer<T>(chw_tensor.data(), chw_tensor.size(2),
                                      chw_tensor.size(1), chw_tensor.size(0));
  }

  template <typename T>
  inline auto as_runtime_buffer(sara::TensorView_<T, 4>& nchw_tensor)
  {
    return Halide::Runtime::Buffer<T>(nchw_tensor.data(), nchw_tensor.size(3),
                                      nchw_tensor.size(2), nchw_tensor.size(1),
                                      nchw_tensor.size(0));
  }

  template <typename T>
  inline auto as_runtime_buffer_3d(sara::ImageView<T>& image)
  {
    static constexpr auto num_channels = sara::PixelTraits<T>::num_channels;
    return Halide::Runtime::Buffer<T>(image.data(), image.width(),
                                      image.height(), num_channels);
  }

  template <typename T>
  inline auto as_buffer(std::vector<T>& v)
  {
    return Halide::Buffer<T>(v.data(), static_cast<int>(v.size()));
  }

  template <typename T>
  inline auto as_runtime_buffer(std::vector<T>& v)
  {
    return Halide::Runtime::Buffer<T>(v.data(), static_cast<int>(v.size()));
  }

}  // namespace DO::Shakti::HalideBackend
