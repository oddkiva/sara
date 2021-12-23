#pragma once

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>

#include <DO/Shakti/Halide/BinaryOperators.hpp>
#include <DO/Shakti/Halide/SIFT/V3/ExtremumDataStructures.hpp>
#include <DO/Shakti/Halide/GaussianConvolution.hpp>
#include <DO/Shakti/Halide/Resize.hpp>

#include "shakti_gray32f_to_rgb8u_cpu.h"

#include "shakti_convolve_batch_32f_gpu.h"
#include "shakti_count_extrema_gpu.h"
#include "shakti_dominant_gradient_orientations_gpu_v2.h"
#include "shakti_dominant_gradient_orientations_gpu_v3.h"
#include "shakti_forward_difference_32f_gpu.h"
#include "shakti_local_scale_space_extremum_32f_gpu_v2.h"
#include "shakti_local_scale_space_extremum_32f_gpu_v3.h"
#include "shakti_polar_gradient_2d_32f_gpu_v2.h"
#include "shakti_refine_scale_space_extrema_gpu_v2.h"
#include "shakti_refine_scale_space_extrema_gpu_v3.h"
#include "shakti_stream_compaction_gpu.h"


namespace DO::Shakti::HalideBackend::v3 {

  struct SiftOctaveParameters
  {
    bool profile = false;
    Sara::Timer timer;

    //! @brief Gaussian octave.
    //! @{
    float scale_camera = 1.f;
    float scale_initial = 1.6f;
    float scale_factor = std::pow(2.f, 1 / 3.f);
    int num_scales = 3;
    int gaussian_truncation_factor = 4;
    //! @}

    //! @brief Extremum detection thresholds.
    float edge_ratio = 10.0f;
    float extremum_thres = 0.01f;

    //! @brief Dominant gradient orientations.
    int num_orientation_bins = 36;
    float ori_gaussian_truncation_factor = 3.f;
    float scale_multiplying_factor = 1.5f;
    float peak_ratio_thres = 0.8f;

    std::vector<float> scales;
    std::vector<float> sigmas;
    sara::Tensor_<float, 2> kernels;

    Halide::Runtime::Buffer<float> kernel_x_buffer;
    Halide::Runtime::Buffer<float> kernel_y_buffer;

    inline auto tic()
    {
      if (profile)
        timer.restart();
    }

    inline auto toc(const std::string& what)
    {
      if (profile)
      {
        const auto elapsed = timer.elapsed_ms();
        SARA_DEBUG << "[" << what << "] " << elapsed << " ms" << std::endl;
      }
    }

    auto kernel_size(float sigma) const
    {
      return static_cast<int>(2 * gaussian_truncation_factor * sigma) + 1;
    }

    auto initialize_kernels() -> void
    {
      // Set up the list of scales in the discrete octave.
      scales = std::vector<float>(num_scales + 3);
      for (auto i = 0; i < num_scales + 3; ++i)
        scales[i] = scale_initial * std::pow(scale_factor, i);

      // Calculate the Gaussian smoothing values.
      sigmas = std::vector<float>(num_scales + 3);
      for (auto i = 0u; i < sigmas.size(); ++i)
        sigmas[i] =
            std::sqrt(std::pow(scales[i], 2) - std::pow(scale_camera, 2));

      // Fill the Gaussian kernels.
      const auto kernel_size_max = kernel_size(sigmas.back());
      const auto kernel_mid = kernel_size_max / 2;

      kernels.resize(num_scales + 3, kernel_size_max);
      kernels.flat_array().fill(0);

      for (auto n = 0; n < kernels.size(0); ++n)
      {
        const auto& sigma = sigmas[n];
        const auto ksize = kernel_size(sigma);
        const auto kradius = ksize / 2;
        const auto two_sigma_squared = 2 * sigma * sigma;

        for (auto k = 0; k < ksize; ++k)
          kernels(n, k + kernel_mid - kradius) =
              exp(-std::pow(k - kradius, 2) / two_sigma_squared);

        const auto kernel_sum =
            std::accumulate(&kernels(n, kernel_mid - kradius),
                            &kernels(n, kernel_mid - kradius) + ksize, 0.f);

        for (auto k = 0; k < ksize; ++k)
          kernels(n, k + kernel_mid - kradius) /= kernel_sum;
      }

      // Wrap the gaussian kernels as Halide buffers.
      kernel_x_buffer = Halide::Runtime::Buffer<float>(
          kernels.data(), kernel_size_max, 1, 1, num_scales + 3);
      kernel_x_buffer.set_min(-kernel_mid, 0, 0, 0);

      kernel_y_buffer = Halide::Runtime::Buffer<float>(
          kernels.data(), 1, kernel_size_max, 1, num_scales + 3);
      kernel_y_buffer.set_min(0, -kernel_mid, 0, 0);

      // Transfer the host data to the GPU device.
      kernel_x_buffer.set_host_dirty();
      kernel_y_buffer.set_host_dirty();
    }
  };


  struct SiftOctavePipeline
  {
    using GradientBuffer = std::array<Halide::Runtime::Buffer<float>, 2>;

    v3::SiftOctaveParameters params;

    // Gaussian octave.
    Halide::Runtime::Buffer<float> x_convolved;
    Halide::Runtime::Buffer<float> y_convolved;

    // Gradient octave.
    Halide::Runtime::Buffer<float> gradient_mag;
    Halide::Runtime::Buffer<float> gradient_ori;

    // DoG octave.
    Halide::Runtime::Buffer<float> dog;

    // Extrema map octave.
    Halide::Runtime::Buffer<std::int8_t> extrema_map;

    v3::QuantizedExtremumArray extrema_quantized;
    v3::ExtremumArray extrema;
    v2::OrientedExtremumArray extrema_oriented;

    v2::DominantOrientationDenseMap dominant_orientation_dense_map;
    v2::DominantOrientationSparseMap dominant_orientation_sparse_map;

    auto initialize(int w, int h) -> void
    {
      const auto& num_scales = params.num_scales;

      x_convolved = Halide::Runtime::Buffer<float>(w, h, num_scales + 3, 1);
      y_convolved = Halide::Runtime::Buffer<float>(w, h, num_scales + 3, 1);

      gradient_mag = Halide::Runtime::Buffer<float>(w, h, num_scales + 3, 1);
      gradient_ori = Halide::Runtime::Buffer<float>(w, h, num_scales + 3, 1);

      dog = Halide::Runtime::Buffer<float>(w, h, num_scales + 2, 1);

      extrema_map = Halide::Runtime::Buffer<std::int8_t>(w, h, num_scales, 1);
    }

    auto feed(Halide::Runtime::Buffer<float>& gray_image)
    {
      sara::tic();
      shakti_convolve_batch_32f_gpu(gray_image, params.kernel_x_buffer,
                                    x_convolved);
      sara::toc("Convolving on x-axis");

      sara::tic();
      shakti_convolve_batch_32f_gpu(x_convolved, params.kernel_y_buffer,
                                    y_convolved);
      sara::toc("Convolving on y-axis");

      auto& gaussian = y_convolved;

      sara::tic();
      shakti_forward_difference_32f_gpu(gaussian, 2, dog);
      sara::toc("DoG");

      sara::tic();
      shakti_polar_gradient_2d_32f_gpu_v2(gaussian, gradient_mag, gradient_ori);
      sara::toc("Gaussian gradients");

      sara::tic();
      shakti_local_scale_space_extremum_32f_gpu_v3(
          dog, params.edge_ratio, params.extremum_thres, extrema_map);
      sara::toc("Extrema maps");

      // 120-150 ms with the extremum count.
      // 206 ms to populate the list of extremas.
      compress_quantized_extrema_maps_cpu();

      // Super-slow.
      // compress_quantized_extrema_maps_gpu();

      refine_scale_space_extrema();
      compute_dominant_orientations();
      populate_oriented_extrema();
    }

    auto compress_quantized_extrema_maps_cpu() -> void
    {
      sara::tic();
      extrema_map.copy_to_host();
      sara::toc("Copy extrema map buffers to host");

      sara::tic();
      const auto num_extrema = std::count_if(      //
          extrema_map.begin(), extrema_map.end(),  //
          [](const auto& v) { return v != 0; }     //
      );
      sara::toc("Counting number of extrema");

      if (num_extrema == 0)
        return;

      sara::tic();
      extrema_quantized.resize(num_extrema);

      auto i = 0;
#pragma omp parallel for schedule(dynamic) reduction(+:i)
      for (auto n = 0; n < extrema_map.dim(3).extent(); ++n)
      {
        for (auto s = 0; s < extrema_map.dim(2).extent(); ++s)
        {
          for (auto y = 0; y < extrema_map.height(); ++y)
          {
            for (auto x = 0; x < extrema_map.width(); ++x)
            {
              if (extrema_map(x, y, s, n) == 0)
                continue;

              extrema_quantized[i].x = x;
              extrema_quantized[i].y = y;
              extrema_quantized[i].s = s + 1;  // Notice the +1.
              extrema_quantized[i].n = n;
              extrema_quantized[i].scale = params.scales[s + 1];
              extrema_quantized[i].type = extrema_map(x, y, s, n);
              ++i;
            }
          }
        }
      }
      sara::toc("Populating list of refined extrema");
    }

    auto compress_quantized_extrema_maps_gpu() -> void
    {
      sara::tic();
      auto extremum_count_host = std::int32_t{};
      auto extremum_count =
          Halide::Runtime::Buffer<std::int32_t>{&extremum_count_host, 1}.sliced(
              0);
      shakti_count_extrema_gpu(extrema_map, extremum_count);
      sara::toc("GPU Counting number of extrema");

      sara::tic();
      extremum_count.copy_to_host();
      sara::toc("GPU-CPU transfer extrema count");

      // SARA_CHECK(extremum_count_host);
      // SARA_CHECK(extremum_count());

      sara::tic();
      extrema_quantized.resize(extremum_count_host);
      shakti_stream_compaction_gpu(extrema_map,          //
                                   extrema_quantized.x,  //
                                   extrema_quantized.y,  //
                                   extrema_quantized.s,  //
                                   extrema_quantized.n,  //
                                   extrema_quantized.type);
      for (auto i = 0; i < extrema_quantized.s.dim(0).extent(); ++i)
        extrema_quantized.scale(i) = params.scales[extrema_quantized.s(i)];
      sara::toc("GPU stream compaction");
    }

    auto refine_scale_space_extrema() -> void
    {
      sara::tic();

      if (extrema_quantized.size() > 0)
      {
        extrema_quantized.x.set_host_dirty();
        extrema_quantized.y.set_host_dirty();
        extrema_quantized.s.set_host_dirty();
        extrema_quantized.n.set_host_dirty();
        extrema_quantized.scale.set_host_dirty();
        extrema.resize(extrema_quantized.size());

        shakti_refine_scale_space_extrema_gpu_v3(dog,                      //
                                                 extrema_quantized.x,      //
                                                 extrema_quantized.y,      //
                                                 extrema_quantized.s,      //
                                                 extrema_quantized.n,      //
                                                 extrema_quantized.scale,  //
                                                 params.scale_factor,      //
                                                 extrema.x,                //
                                                 extrema.y,                //
                                                 extrema.s,                //
                                                 extrema.value);

        // Copy these as well.
        extrema.n = extrema_quantized.n;
        extrema.type = extrema_quantized.type;
      }
      else
      {
        extrema.x.reset();
        extrema.y.reset();
        extrema.s.reset();
        extrema.n.reset();
        extrema.value.reset();
        extrema.type.reset();
      }

      sara::toc("Refining scale-space extrema");
    }

    auto compute_dominant_orientations() -> void
    {
      if (extrema.size() == 0)
      {
        dominant_orientation_dense_map.peak_map.reset();
        dominant_orientation_dense_map.peak_residuals.reset();
        return;
      }

      sara::tic();

      // Just copy the array of scales back to find the maximum scale value.
      //
      // We will copy the remaining data when populating the list of oriented
      // extrema.
      extrema.s.copy_to_host();
      const auto& scale_max =
          *std::max_element(extrema.s.begin(), extrema.s.end());

      //  Outputs.
      dominant_orientation_dense_map.resize(extrema.size(),
                                            params.num_orientation_bins);

      // Prepare data for the GPU.
      extrema_quantized.s.set_host_dirty();
      extrema.n.set_host_dirty();

      // Run the operation on the GPU.
      shakti_dominant_gradient_orientations_gpu_v3(
          gradient_mag, gradient_ori,               //
          extrema.x,                                //
          extrema.y,                                //
          extrema.s,                                //
          extrema_quantized.s,                      //
          extrema.n,                                //
          scale_max,                                //
          params.num_orientation_bins,              //
          params.ori_gaussian_truncation_factor,    //
          params.scale_multiplying_factor,          //
          params.peak_ratio_thres,                  //
          dominant_orientation_dense_map.peak_map,  //
          dominant_orientation_dense_map.peak_residuals);
      sara::toc("Compute dominant orientation dense maps");

      sara::tic();
      dominant_orientation_dense_map.copy_to_host();
      dominant_orientation_sparse_map = dominant_orientation_dense_map;
      sara::toc("Compressing dominant gradient orientations");
    }

    auto populate_oriented_extrema() -> void
    {
      auto& d = dominant_orientation_sparse_map;
      auto& e = extrema;
      auto& e_oriented = extrema_oriented;

      if (e.size() == 0)
      {
        e_oriented.x.reset();
        e_oriented.y.reset();
        e_oriented.s.reset();
        e_oriented.value.reset();
        e_oriented.type.reset();
        return;
      }

      sara::tic();

      // Copy the remaining data of the array of extrema to host.
      e.x.copy_to_host();
      e.y.copy_to_host();
      e.value.copy_to_host();
      // No need to copy e.type because it is already in the host memory.

      e_oriented.resize(d.orientation_map.size());

      auto k = 0;
      for (auto i = 0; i < e.size(); ++i)
      {
        const auto& thetas = d.dominant_orientations(i);

        for (const auto& theta : thetas)
        {
          e_oriented.x(k) = e[i].x;
          e_oriented.y(k) = e[i].y;
          e_oriented.s(k) = e[i].s;
          e_oriented.type(k) = extrema_quantized[i].type;
          e_oriented.value(k) = e[i].value;
          e_oriented.orientations(k) = theta;
          ++k;
        }
      }

      sara::toc("Populating list of oriented extrema");
    }

    auto gaussian(int s, int n) -> sara::ImageView<float>
    {
      return {y_convolved.sliced(3, n).sliced(2, s).data(),
              {y_convolved.width(), y_convolved.height()}};
    }

    auto difference_of_gaussians(int s, int n) -> sara::ImageView<float>
    {
      return {dog.sliced(3, n).sliced(2, s).data(),
              {dog.width(), dog.height()}};
    }

    auto gradient_magnitude(int s, int n) -> sara::ImageView<float>
    {
      return {gradient_mag.sliced(3, n).sliced(2, s).data(),
              {gradient_mag.width(), gradient_mag.height()}};
    }

    auto gradient_orientation(int s, int n) -> sara::ImageView<float>
    {
      return {gradient_ori.sliced(3, n).sliced(2, s).data(),
              {gradient_ori.width(), gradient_ori.height()}};
    }

    auto extrema_map_view(int s, int n) -> sara::ImageView<std::int8_t>
    {
      return {extrema_map.sliced(3, n).sliced(2, s).data(),
              {extrema_map.width(), extrema_map.height()}};
    }
  };

}  // namespace DO::Shakti::HalideBackend::v3
