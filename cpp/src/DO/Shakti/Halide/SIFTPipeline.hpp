#pragma once

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>

#include <DO/Shakti/Halide/BinaryOperators.hpp>
#include <DO/Shakti/Halide/ExtremumDataStructures.hpp>
#include <DO/Shakti/Halide/ExtremumDataStructuresV2.hpp>
#include <DO/Shakti/Halide/ExtremumDataStructuresV3.hpp>
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


namespace DO::Shakti::HalideBackend::v2 {

  struct SiftOctaveParameters
  {
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

    auto initialize_cached_scales()
    {
      scales = std::vector<float>(num_scales + 3);
      for (auto i = 0; i < num_scales + 3; ++i)
        scales[i] = scale_initial * std::pow(scale_factor, i);

      sigmas = std::vector<float>(num_scales + 3);
      for (auto i = 0u; i < sigmas.size(); ++i)
        sigmas[i] = i == 0 ? std::sqrt(std::pow(scale_initial, 2) -
                                       std::pow(scale_camera, 2))
                           : std::sqrt(std::pow(scales[i], 2) -
                                       std::pow(scales[i - 1], 2));
    }
  };


  struct SiftOctavePipeline
  {
    using GradientBuffer = std::array<Halide::Runtime::Buffer<float>, 2>;

    std::vector<Halide::Runtime::Buffer<float>> gaussians;
    std::vector<Halide::Runtime::Buffer<float>> dogs;
    std::vector<Halide::Runtime::Buffer<std::int8_t>> extrema_maps;
    std::vector<GradientBuffer> gradients;

    std::vector<v2::QuantizedExtremumArray> extrema_quantized;
    std::vector<v2::ExtremumArray> extrema;

    std::vector<v2::DominantOrientationDenseMap>
        dominant_orientation_dense_maps;
    std::vector<v2::DominantOrientationSparseMap>
        dominant_orientation_sparse_maps;
    std::vector<v2::OrientedExtremumArray> extrema_oriented;

    // Pipeline parameters.
    SiftOctaveParameters params;

    enum class FirstAction : std::uint8_t
    {
      Convolve = 0,
      Downscale = 1,
    };

    SiftOctavePipeline()
    {
      params.initialize_cached_scales();
    }

    SiftOctavePipeline(const SiftOctaveParameters& params_)
      : params{params_}
    {
      params.initialize_cached_scales();
    }

    auto initialize_buffers(std::int32_t w, std::int32_t h)
    {
      // Octave of Gaussians.
      gaussians.resize(params.num_scales + 3);
      for (auto& g : gaussians)
        g = Halide::Runtime::Buffer<float>(w, h, 1, 1);
      // Octave of Difference of Gaussians.
      dogs.resize(params.num_scales + 2);
      for (auto& dog : dogs)
        dog = Halide::Runtime::Buffer<float>(w, h, 1, 1);

      // Octave of DoG extrema maps.
      extrema_maps.resize(params.num_scales);
      for (auto& e : extrema_maps)
        e = Halide::Runtime::Buffer<std::int8_t>(w, h, 1, 1);
      extrema_quantized.resize(params.num_scales);
      extrema.resize(params.num_scales);

      // Octave of Gradients of Gaussians.
      gradients.resize(params.num_scales);
      for (auto& grad : gradients)
        grad = {Halide::Runtime::Buffer<float>(w, h, 1, 1),
                Halide::Runtime::Buffer<float>(w, h, 1, 1)};
      dominant_orientation_dense_maps.resize(params.num_scales);
      dominant_orientation_sparse_maps.resize(params.num_scales);

      // Final output: the list of oriented keypoints.
      extrema_oriented.resize(params.num_scales);
    }

    auto feed(Halide::Runtime::Buffer<float>& input,
              FirstAction first_action = FirstAction::Convolve)
    {
      // Compute the Gaussians.
      if (first_action == FirstAction::Convolve)
      {
        sara::tic();
        HalideBackend::gaussian_convolution(
            input,                               //
            gaussians[0],                        //
            params.sigmas[0],                    //
            params.gaussian_truncation_factor);  //
        sara::toc("Convolving for Gaussian 0: " +
                  std::to_string(params.sigmas[0]));
      }
      else if (first_action == FirstAction::Downscale)
      {
        sara::tic();
        HalideBackend::scale(input, gaussians[0]);
        sara::toc("Downsampling for Gaussian 0: " +
                  std::to_string(params.sigmas[0]));
      }
      else
      {
        throw std::runtime_error{"Not implemented"};
      }

      for (auto i = 1u; i < gaussians.size(); ++i)
      {
        sara::tic();
        HalideBackend::gaussian_convolution(  //
            gaussians[i - 1],                 //
            gaussians[i],                     //
            params.sigmas[i],                 //
            params.gaussian_truncation_factor);
        sara::toc("Convolving for Gaussian " + std::to_string(i) + ": " +
                  std::to_string(params.sigmas[i]));
      }

      // Compute the DoGs.
      for (auto i = 0u; i < dogs.size(); ++i)
      {
        sara::tic();
        HalideBackend::subtract(gaussians[i + 1], gaussians[i], dogs[i]);
        sara::toc("DoG " + std::to_string(i));
      }

      // Localize the extrema.
      for (auto i = 0u; i < extrema_maps.size(); ++i)
      {
        sara::tic();
        shakti_local_scale_space_extremum_32f_gpu_v2(
            dogs[i], dogs[i + 1], dogs[i + 2],         //
            params.edge_ratio, params.extremum_thres,  //
            extrema_maps[i]);
        sara::toc("DoG extremum localization " + std::to_string(i));
      }

      // Compute the gradients.
      for (auto i = 0u; i < gradients.size(); ++i)
      {
        sara::tic();
        shakti_polar_gradient_2d_32f_gpu_v2(gaussians[i + 1], gradients[i][0],
                                            gradients[i][1]);
        sara::toc("Gradients in polar coordinates " + std::to_string(i));
      }

      // Compress and refine the extrema.
      compress_quantized_extrema_maps();
      refine_extrema();

      // Compute dominant orientations.
      compute_dominant_orientations();
      compress_dominant_orientations();

      // The final result.
      populate_oriented_extrema();
    }

    auto compress_quantized_extrema_maps() -> void
    {
      sara::tic();
      for (auto& extrema_map : extrema_maps)
        extrema_map.copy_to_host();
      sara::toc("Copy extrema map buffers to host");

      sara::tic();
#pragma omp parallel for
      for (auto s = 0; s < static_cast<int>(extrema_maps.size()); ++s)
      {
        const auto& dog_ext_map = extrema_maps[s];
        const auto num_extrema = std::count_if(      //
            dog_ext_map.begin(), dog_ext_map.end(),  //
            [](const auto& v) { return v != 0; }     //
        );

        if (num_extrema == 0)
          continue;

        // Populate the list of extrema for the corresponding scale.
        extrema_quantized[s].resize(num_extrema);

        auto i = 0;
        for (auto y = 0; y < dog_ext_map.height(); ++y)
        {
          for (auto x = 0; x < dog_ext_map.width(); ++x)
          {
            if (dog_ext_map(x, y) == 0)
              continue;

            extrema_quantized[s].x(i) = x;
            extrema_quantized[s].y(i) = y;
            extrema_quantized[s].type(i) = dog_ext_map(x, y);
            ++i;
          }
        }
      }

      sara::toc("Populating list of extrema");
    }

    auto refine_extrema() -> void
    {
      sara::tic();
      for (auto s = 0u; s < extrema_quantized.size(); ++s)
      {
        auto& quantized = extrema_quantized[s];
        auto& refined = extrema[s];
        if (quantized.empty())
          continue;

        // Resize the buffers.
        refined.resize(quantized.size());

        // Copy the necessary data from host to device memory.
        quantized.x.set_host_dirty();
        quantized.y.set_host_dirty();
        refined.type = quantized.type;  // Should be cheap. (shallow copy).

        // Run the operation on the GPU.
        shakti_refine_scale_space_extrema_gpu_v2(
            dogs[s], dogs[s + 1], dogs[s + 2],  //
            quantized.x, quantized.y,           //
            dogs[s].width(), dogs[s].height(),  //
            params.scales[s + 1],               //
            params.scale_factor,                //
            refined.x,                          //
            refined.y,                          //
            refined.s,                          //
            refined.value);                     //
      }
      sara::toc("Refined extrema");
    }

    auto compute_dominant_orientations() -> void
    {
      sara::tic();
      for (auto s = 0u; s < dominant_orientation_dense_maps.size(); ++s)
      {
        // Inputs.
        auto& e = extrema[s];
        if (e.empty())
          continue;

        // Just copy the array of scales back to find the maximum scale value.
        //
        // We will copy the remaining data when populating the list of oriented
        // extrema.
        e.s.copy_to_host();
        const auto& scale_max = *std::max_element(e.s.begin(), e.s.end());

        //  Outputs.
        auto& d = dominant_orientation_dense_maps[s];
        d.resize(e.size(), params.num_orientation_bins);

        shakti_dominant_gradient_orientations_gpu_v2(
            gradients[s][0], gradients[s][1],       //
            e.x, e.y, e.s,                          //
            scale_max,                              //
            params.num_orientation_bins,            //
            params.ori_gaussian_truncation_factor,  //
            params.scale_multiplying_factor,        //
            params.peak_ratio_thres,                //
            d.peak_map,                             //
            d.peak_residuals);
      }
      sara::toc("Dense dominant gradient orientations");
    }

    auto compress_dominant_orientations() -> void
    {
      sara::tic();
      for (auto s = 0u; s < dominant_orientation_dense_maps.size(); ++s)
      {
        auto& dense = dominant_orientation_dense_maps[s];
        if (dense.empty())
          continue;
        auto& sparse = dominant_orientation_sparse_maps[s];
        dense.copy_to_host();
        sparse = dense;
      }
      sara::toc("Sparse dominant gradient orientations");
    }

    auto populate_oriented_extrema() -> void
    {
      sara::tic();
#pragma omp parallel for
      for (auto s = 0; s < static_cast<int>(extrema_oriented.size()); ++s)
      {
        auto& d = dominant_orientation_sparse_maps[s];
        auto& e = extrema[s];
        auto& e_oriented = extrema_oriented[s];
        if (e.empty())
          continue;

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
            e_oriented.type(k) = e[i].type;
            e_oriented.value(k) = e[i].value;
            e_oriented.orientations(k) = theta;
            ++k;
          }
        }
      }
      sara::toc("Populating oriented extrema");
    }


    auto gaussian_view(int i) -> sara::ImageView<float>
    {
      auto& g = gaussians[i];
      g.copy_to_host();
      return {g.data(), {g.width(), g.height()}};
    }

    auto dog_view(int i) -> sara::ImageView<float>
    {
      auto& dog = dogs[i];
      dog.copy_to_host();
      return {dog.data(), {dog.width(), dog.height()}};
    }

    auto extrema_map_view(int i) -> sara::ImageView<std::int8_t>
    {
      auto& extrema = extrema_maps[i];
      extrema.copy_to_host();
      return {extrema.data(), {extrema.width(), extrema.height()}};
    }
  };


  struct SiftPyramidPipeline
  {
    Sara::Timer timer;

    float scale_initial = 1.6f;
    int image_padding_size = 1;

    // Normal.
    float scale_camera = 1;
    int start_octave_index = 0;

    // Overkill but possible.
    // float scale_camera = 0.25;
    // int start_octave_index = -2;

    // Ultra options.
    // float scale_camera = 0.5;
    // int start_octave_index = -1;

    Halide::Runtime::Buffer<float> input_rescaled;
    std::vector<SiftOctavePipeline> octaves;

    auto initialize(int start_octave, int width, int height) -> void
    {
      start_octave_index = start_octave;

      // Adjust the scale of the photograph acquired by the camera.
      scale_camera = 1.f / std::pow(2, start_octave_index);

      // Deduce the maximum number of octaves.
      const auto l = std::min(width, height);  // l = min image image sizes.
      const auto b = 1;                        // b = image border size.

      /*
       * Calculation details:
       *
       * We must satisfy:
       *   l / 2^k > 2b
       *   2^k < l / (2b)
       *   k < log(l / (2b)) / log(2)
       *
       */
      // const auto num_octaves = 1;
      const auto num_octaves =
          static_cast<int>(std::log(l / (2.f * b)) / std::log(2.f)) -
          start_octave_index;

      if (num_octaves < 0)
        throw std::runtime_error{"Invalid start octave!"};

      if (start_octave_index < 0)
      {
        const auto w = width * std::pow(2, -start_octave_index);
        const auto h = height * std::pow(2, -start_octave_index);
        input_rescaled = Halide::Runtime::Buffer<float>(w, h, 1, 1);
      }
      else if (start_octave_index > 0)
      {
        input_rescaled = Halide::Runtime::Buffer<float>(width, height, 1, 1);
      }

      octaves.resize(num_octaves);
      for (auto o = start_octave_index; o < start_octave_index + num_octaves;
           ++o)
      {
        const auto w = o < 0  //
                           ? width * std::pow(2, -o)
                           : width / std::pow(2, o);
        const auto h = o < 0  //
                           ? height * std::pow(2, -o)
                           : height / std::pow(2, o);

        octaves[o - start_octave_index].initialize_buffers(w, h);
      }
    }

    auto feed(Halide::Runtime::Buffer<float>& input)
    {
      if (start_octave_index < 0)
      {
        sara::tic();
        HalideBackend::enlarge(input, input_rescaled);
        sara::toc("Upscaling the image");
      }
      else if (start_octave_index > 0)
      {
        const auto& scale =
            octaves.front().params.scales[octaves.front().params.num_scales];
        const auto& sigma = std::sqrt(scale * scale - 1);
        sara::tic();
        HalideBackend::gaussian_convolution(
            input, input_rescaled, sigma,
            octaves.front().params.gaussian_truncation_factor);
        sara::toc("Convolving for downscaling: sigma = " +
                  std::to_string(sigma));
      }

      auto& which_input = start_octave_index != 0 ? input_rescaled : input;

      for (auto o = 0u; o < octaves.size(); ++o)
      {
        timer.restart();

        if (o == 0)
        {
          if (start_octave_index <= 0)
            octaves[o].feed(which_input,
                            SiftOctavePipeline::FirstAction::Convolve);
          else
            octaves[o].feed(which_input,
                            SiftOctavePipeline::FirstAction::Downscale);
        }
        else
        {
          auto& prev_octave = octaves[o - 1];
          const auto& prev_num_scales = prev_octave.params.num_scales;
          auto& prev_g = prev_octave.gaussians[prev_num_scales];
          octaves[o].feed(prev_g, SiftOctavePipeline::FirstAction::Downscale);
        }

        const auto elapsed = timer.elapsed_ms();
        SARA_DEBUG << sara::format("SIFT Octave %d [%dx%d]: %f ms",
                                   start_octave_index + o,
                                   octaves[o].gaussians[0].width(),
                                   octaves[o].gaussians[0].height(), elapsed)
                   << std::endl;
      }
    }

    auto octave_scaling_factor(int o) const
    {
      return std::pow(2, o);
    }

    auto input_rescaled_view() -> sara::ImageView<float>
    {
      input_rescaled.copy_to_host();
      return {input_rescaled.data(),
              {input_rescaled.width(), input_rescaled.height()}};
    }
  };

}  // namespace DO::Shakti::HalideBackend::v2


namespace DO::Shakti::HalideBackend::v3 {

  struct SiftOctaveParameters
  {
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
