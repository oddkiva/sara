#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>

#include <DO/Shakti/Halide/SIFT/V2/Pipeline.hpp>

#include <DO/Shakti/Halide/BinaryOperators.hpp>
#include <DO/Shakti/Halide/GaussianConvolution.hpp>
#include <DO/Shakti/Halide/Resize.hpp>
#include <DO/Shakti/Halide/SeparableConvolution2d.hpp>

#include "shakti_gray32f_to_rgb8u_cpu.h"

#include "shakti_dominant_gradient_orientations_gpu_v2.h"
#include "shakti_local_scale_space_extremum_32f_gpu_v2.h"
#include "shakti_polar_gradient_2d_32f_gpu_v2.h"
#include "shakti_refine_scale_space_extrema_gpu_v2.h"

#include "shakti_sift_descriptor_gpu_v5.h"


namespace DO::Shakti::HalideBackend::v2 {

  auto SiftOctaveParameters::initialize_gaussian_kernels() -> void
  {
    scales = std::vector<float>(scale_count + 3);
    for (auto i = 0; i < scale_count + 3; ++i)
      scales[i] = scale_initial * std::pow(scale_factor, i);

    sigmas = std::vector<float>(scale_count + 3);
    for (auto i = 0u; i < sigmas.size(); ++i)
    {
      sigmas[i] =
          i == 0
              ? std::sqrt(std::pow(scale_initial, 2) -
                          std::pow(scale_camera, 2))
              : std::sqrt(std::pow(scales[i], 2) - std::pow(scales[i - 1], 2));
    }

    kernels = std::vector<::Halide::Runtime::Buffer<float>>{};
    std::transform(sigmas.begin(), sigmas.end(), std::back_inserter(kernels),
                   [](const auto& sigma) {
                     const auto k = Sara::make_gaussian_kernel(sigma);
                     auto k_buffer = ::Halide::Runtime::Buffer<float>(k.size());
                     std::copy_n(k.data(), k.size(), k_buffer.data());
                     k_buffer.set_host_dirty();
                     return k_buffer;
                   });
  }


  auto SiftOctavePipeline::initialize_buffers(std::int32_t scale_count,
                                              std::int32_t w, std::int32_t h)
      -> void
  {
    params.set_scale_count(scale_count);
    params.initialize_gaussian_kernels();

    // Octave of Gaussians.
    gaussians.resize(params.scale_count + 3);
    for (auto& g : gaussians)
      g = ::Halide::Runtime::Buffer<float>(w, h, 1, 1);
    // Octave of Difference of Gaussians.
    dogs.resize(params.scale_count + 2);
    for (auto& dog : dogs)
      dog = ::Halide::Runtime::Buffer<float>(w, h, 1, 1);

    // Octave of DoG extrema maps.
    extrema_maps.resize(params.scale_count);
    for (auto& e : extrema_maps)
      e = ::Halide::Runtime::Buffer<std::int8_t>(w, h, 1, 1);
    extrema_quantized.resize(params.scale_count);
    extrema.resize(params.scale_count);

    // Octave of Gradients of Gaussians.
    gradients.resize(params.scale_count);
    for (auto& grad : gradients)
      grad = {::Halide::Runtime::Buffer<float>(w, h, 1, 1),
              ::Halide::Runtime::Buffer<float>(w, h, 1, 1)};
    dominant_orientation_dense_maps.resize(params.scale_count);
    dominant_orientation_sparse_maps.resize(params.scale_count);

    // Final output: the list of oriented keypoints.
    extrema_oriented.resize(params.scale_count);

    // Initialize the list of SIFT descriptors.
    descriptors.resize(gradients.size());
  }

  auto SiftOctavePipeline::feed(::Halide::Runtime::Buffer<float>& input,
                                FirstAction first_action) -> void
  {
    // Compute the Gaussians.
    if (first_action == FirstAction::Convolve)
    {
      tic();
#ifdef IMPL_V1
      HalideBackend::gaussian_convolution(
          input,                               //
          gaussians[0],                        //
          params.sigmas[0],                    //
          params.gaussian_truncation_factor);  //
#else
      HalideBackend::separable_convolution_2d(  //
          input, params.kernels[0],
          gaussians[0],                       //
          params.kernels[0].dim(0).extent(),  //
          -params.kernels[0].dim(0).extent() / 2);
#endif
      toc("Convolving for Gaussian 0: " + std::to_string(params.sigmas[0]));
    }
    else if (first_action == FirstAction::Downscale)
    {
      tic();
      HalideBackend::scale(input, gaussians[0]);
      toc("Downsampling for Gaussian 0: " + std::to_string(params.sigmas[0]));
    }
    else
    {
      throw std::runtime_error{"Not implemented"};
    }

    for (auto i = 1u; i < gaussians.size(); ++i)
    {
      tic();
#ifdef IMPL_V1
      HalideBackend::gaussian_convolution(  //
          gaussians[i - 1],                 //
          gaussians[i],                     //
          params.sigmas[i],                 //
          params.gaussian_truncation_factor);
#else
      HalideBackend::separable_convolution_2d(  //
          gaussians[i - 1],                     //
          params.kernels[i],
          gaussians[i],                       //
          params.kernels[i].dim(0).extent(),  //
          -params.kernels[i].dim(0).extent() / 2);
#endif
      toc("Convolving for Gaussian " + std::to_string(i) + ": " +
          std::to_string(params.sigmas[i]));
    }

    // Compute the DoGs.
    for (auto i = 0u; i < dogs.size(); ++i)
    {
      tic();
      HalideBackend::subtract(gaussians[i + 1], gaussians[i], dogs[i]);
      toc("DoG " + std::to_string(i));
    }

    // Localize the extrema.
    for (auto i = 0u; i < extrema_maps.size(); ++i)
    {
      tic();
      shakti_local_scale_space_extremum_32f_gpu_v2(
          dogs[i], dogs[i + 1], dogs[i + 2],         //
          params.edge_ratio, params.extremum_thres,  //
          extrema_maps[i]);
      toc("DoG extremum localization " + std::to_string(i));
    }

    // Compute the gradients.
    for (auto i = 0u; i < gradients.size(); ++i)
    {
      tic();
      shakti_polar_gradient_2d_32f_gpu_v2(gaussians[i + 1], gradients[i][0],
                                          gradients[i][1]);
      toc("Gradients in polar coordinates " + std::to_string(i));
    }

    // Compress and refine the extrema.
    compress_quantized_extrema_maps();
    refine_extrema();

    // Compute dominant orientations.
    compute_dominant_orientations();
    compress_dominant_orientations();

    // The final result.
    populate_oriented_extrema();

    // Initialize the host and device descriptor buffers.
    tic();

    auto desc_buffers =
        std::vector<::Halide::Runtime::Buffer<float>>(gradients.size());
    static constexpr auto N = 4;
    static constexpr auto O = 8;
    for (auto i = 0u; i < gradients.size(); ++i)
    {
      descriptors[i].resize({extrema_oriented[i].size(), N * N, O});
      desc_buffers[i] = as_runtime_buffer(descriptors[i]);
      desc_buffers[i].set_host_dirty();
    }

    // Calculate SIFT descriptor for each oriented extrema
    for (auto i = 0u; i < gradients.size(); ++i)
    {
      // Gradient buffers.
      //
      // Already available in the GPU device memory.
      auto& mag = gradients[i][0];
      auto& ori = gradients[i][1];

      // Extremum data buffers.
      //
      // Already available in the GPU device memory.
      auto& x = extrema_oriented[i].x;
      auto& y = extrema_oriented[i].y;
      auto& s = extrema_oriented[i].s;
      auto& o = extrema_oriented[i].orientations;

      // Resize the descriptors.

      // Calculate on GPU and copy back to CPU host memory.
      shakti_sift_descriptor_gpu_v5(mag, ori,    //
                                    x, y, s, o,  //
                                    desc_buffers[i]);
      desc_buffers[i].copy_to_host();
    }

    toc("Descriptors");
  }

  auto SiftOctavePipeline::compress_quantized_extrema_maps() -> void
  {
    tic();
    for (auto& extrema_map : extrema_maps)
      extrema_map.copy_to_host();
    toc("Copy extrema map buffers to host");

    tic();
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

    toc("Populating list of extrema");
  }

  auto SiftOctavePipeline::refine_extrema() -> void
  {
    tic();
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
    toc("Refined extrema");
  }

  auto SiftOctavePipeline::compute_dominant_orientations() -> void
  {
    tic();
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
    toc("Dense dominant gradient orientations");
  }

  auto SiftOctavePipeline::compress_dominant_orientations() -> void
  {
    tic();
    for (auto s = 0u; s < dominant_orientation_dense_maps.size(); ++s)
    {
      auto& dense = dominant_orientation_dense_maps[s];
      if (dense.empty())
        continue;
      auto& sparse = dominant_orientation_sparse_maps[s];
      dense.copy_to_host();
      sparse = dense;
    }
    toc("Sparse dominant gradient orientations");
  }

  auto SiftOctavePipeline::populate_oriented_extrema() -> void
  {
    tic();
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
    toc("Populating oriented extrema");
  }

  auto SiftOctavePipeline::gaussian_view(int i) -> sara::ImageView<float>
  {
    auto& g = gaussians[i];
    g.copy_to_host();
    return {g.data(), {g.width(), g.height()}};
  }

  auto SiftOctavePipeline::dog_view(int i) -> sara::ImageView<float>
  {
    auto& dog = dogs[i];
    dog.copy_to_host();
    return {dog.data(), {dog.width(), dog.height()}};
  }

  auto SiftOctavePipeline::extrema_map_view(int i)
      -> sara::ImageView<std::int8_t>
  {
    auto& extrema = extrema_maps[i];
    extrema.copy_to_host();
    return {extrema.data(), {extrema.width(), extrema.height()}};
  }


  auto SiftPyramidPipeline::initialize(int start_octave,
                                       int scale_count_per_octave,  //
                                       int width, int height) -> void
  {
    start_octave_index = start_octave;

    // Deduce the maximum number of octaves.
    const auto l = std::min(width, height);  // l = min image image sizes.
    const auto b = 8;                        // b = image border size.

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
      input_rescaled = ::Halide::Runtime::Buffer<float>(w, h, 1, 1);
    }
    else if (start_octave_index > 0)
    {
      input_rescaled = ::Halide::Runtime::Buffer<float>(width, height, 1, 1);
    }

    octaves.resize(num_octaves);
    for (auto o = start_octave_index; o < start_octave_index + num_octaves; ++o)
    {
      const auto w = o < 0  //
                         ? width * std::pow(2, -o)
                         : width / std::pow(2, o);
      const auto h = o < 0  //
                         ? height * std::pow(2, -o)
                         : height / std::pow(2, o);

      octaves[o - start_octave_index].profile = profile;
      octaves[o - start_octave_index].initialize_buffers(scale_count_per_octave,
                                                         w, h);
    }
  }

  auto SiftPyramidPipeline::feed(::Halide::Runtime::Buffer<float>& input)
      -> void
  {
    if (start_octave_index < 0)
    {
      tic();
      HalideBackend::enlarge(input, input_rescaled);
      toc("Upscaling the image");
    }
    else if (start_octave_index > 0)
    {
      const auto& scale =
          octaves.front().params.scales[octaves.front().params.scale_count];
      const auto& sigma = std::sqrt(scale * scale - 1);
      tic();
      HalideBackend::gaussian_convolution(
          input, input_rescaled, sigma,
          octaves.front().params.gaussian_truncation_factor);
      toc("Convolving for downscaling: sigma = " + std::to_string(sigma));
    }

    auto& which_input = start_octave_index != 0 ? input_rescaled : input;

    for (auto o = 0u; o < octaves.size(); ++o)
    {
      if (profile)
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
        const auto& prev_scale_count = prev_octave.params.scale_count;
        auto& prev_g = prev_octave.gaussians[prev_scale_count];
        octaves[o].feed(prev_g, SiftOctavePipeline::FirstAction::Downscale);
      }

      if (profile)
      {
        const auto elapsed = timer.elapsed_ms();
        SARA_DEBUG << Sara::format("SIFT Octave %d [%dx%d]: %f ms",
                                   start_octave_index + o,
                                   octaves[o].gaussians[0].width(),
                                   octaves[o].gaussians[0].height(), elapsed)
                   << std::endl;
      }
    }
  }

  auto SiftPyramidPipeline::get_keypoints(Sara::KeypointList<Sara::OERegion, float>& keys) const
      -> void
  {
    // Count the number of features.
    auto num_features = 0;
    for (auto o = 0u; o < octaves.size(); ++o)
    {
      const auto& octave = octaves[o];
      num_features += std::accumulate(
          octave.extrema_oriented.begin(), octave.extrema_oriented.end(), 0,
          [](const auto& a, const auto& b) { return a + b.size(); });
    }

    // Populate the list of features.
    auto& features = Sara::features(keys);
    features.clear();
    features.reserve(num_features);
    for (auto o = 0u; o < octaves.size(); ++o)
    {
      const auto& octave = octaves[o];
      const auto oct_scale_factor = octave_scaling_factor(  //
          start_octave_index + o                            //
      );

      for (auto s = 0u; s < octave.extrema_oriented.size(); ++s)
      {
        const auto& e = octave.extrema_oriented[s];

        for (auto f = 0; f < e.size(); ++f)
        {
          const auto& type = e.type(f);
          const auto& x = e.x(f) * oct_scale_factor;
          const auto& y = e.y(f) * oct_scale_factor;
          const auto& scale = e.s(f) * oct_scale_factor;
          const auto& theta = e.orientations(f);

          auto feature = sara::OERegion{{x, y}, scale};
          feature.orientation = theta;
          feature.type = Sara::OERegion::Type::DoG;
          feature.extremum_type = type == 1  //
                                      ? sara::OERegion::ExtremumType::Max
                                      : sara::OERegion::ExtremumType::Min;

          features.push_back(feature);
        }
      }
    }

    // Populate the list of descriptors.
    auto& descriptors = Sara::descriptors(keys);
    descriptors.resize(num_features, 128);
    auto dmat = descriptors.matrix();
    auto current_row = 0;
    for (auto o = 0u; o < octaves.size(); ++o)
    {
      const auto& octave = octaves[o];
      for (auto s = 0u; s < octave.extrema_oriented.size(); ++s)
      {
        // Accumulate the list of descriptors.
        const auto& di = octave.descriptors[s];
        const auto dmat_i =
            di.reshape(Eigen::Vector2i{di.size(0), di.size(1) * di.size(2)})
                .matrix();

        dmat.block(current_row, 0, di.size(0), 128) = dmat_i;
        current_row += dmat_i.rows();
      }
    }
  }

  auto SiftPyramidPipeline::octave_scaling_factor(int o) const -> float
  {
    return std::pow(2.f, static_cast<float>(o));
  }

  auto SiftPyramidPipeline::input_rescaled_view() -> Sara::ImageView<float>
  {
    input_rescaled.copy_to_host();
    return {input_rescaled.data(),
            {input_rescaled.width(), input_rescaled.height()}};
  }

}  // namespace DO::Shakti::HalideBackend::v2
