// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Tensor.hpp>

#include <drafts/Halide/MyHalide.hpp>
#include <drafts/Halide/Utilities.hpp>


namespace DO::Shakti::HalideBackend::v2 {

  //! @brief List of extrema localized in the discretized scale-space.
  struct QuantizedExtremumArray
  {
    //! @brief Quantized localization.
    Halide::Runtime::Buffer<std::int32_t> x;
    Halide::Runtime::Buffer<std::int32_t> y;
    float scale;
    //! @}

    //! @brief Extremum type.
    Halide::Runtime::Buffer<std::int8_t> type;

    QuantizedExtremumArray() = default;

    QuantizedExtremumArray(std::size_t size)
    {
      resize(size);
    }

    auto resize(std::size_t size) -> void
    {
      x = Halide::Runtime::Buffer<std::int32_t>(size);
      y = Halide::Runtime::Buffer<std::int32_t>(size);
      type = Halide::Runtime::Buffer<std::int8_t>(size);
    }

    auto size() const noexcept
    {
      return x.dim(0).extent();
    }
  };

  //! @brief List of refined extrema in the continuous scale-space.
  struct ExtremumArray
  {
    //! @brief Coordinates of the extrema
    //! @{
    Halide::Runtime::Buffer<float> x;
    Halide::Runtime::Buffer<float> y;
    Halide::Runtime::Buffer<float> s;
    //! @}

    //! @brief Extremum values.
    Halide::Runtime::Buffer<float> value;

    //! @brief Extremum types.
    Halide::Runtime::Buffer<std::int8_t> type;

    struct View
    {
      float& x;
      float& y;
      float& s;
      float& value;
      std::int8_t& type;
    };

    struct ConstView
    {
      const float& x;
      const float& y;
      const float& s;
      const float& value;
      const std::int8_t& type;
    };

    auto operator[](int i) -> View
    {
      return {x(i), y(i), s(i), value(i), type(i)};
    }

    auto operator[](int i) const -> ConstView
    {
      return {x(i), y(i), s(i), value(i), type(i)};
    }

    auto size() const noexcept
    {
      return x.dim(0).extent();
    }

    auto resize(std::size_t size)
    {
      x = Halide::Runtime::Buffer<float>(size);
      y = Halide::Runtime::Buffer<float>(size);
      s = Halide::Runtime::Buffer<float>(size);
      value = Halide::Runtime::Buffer<float>(size);
      type = Halide::Runtime::Buffer<std::int8_t>(size);
    }
  };


  //! @brief Dominant gradient orientation map for a given list of extrema.
  //! @{
  struct DominantOrientationDenseMap
  {
    Halide::Runtime::Buffer<bool> peak_map;
    Halide::Runtime::Buffer<float> peak_residuals;

    DominantOrientationDenseMap() = default;

    DominantOrientationDenseMap(int num_keypoints,
                                int num_orientation_bins = 36)
    {
      resize(num_keypoints, num_orientation_bins);
    }

    auto resize(int num_keypoints, int num_orientation_bins = 36) -> void
    {
      peak_map =
          Halide::Runtime::Buffer<bool>{num_keypoints, num_orientation_bins};
      peak_residuals =
          Halide::Runtime::Buffer<float>{num_keypoints, num_orientation_bins};
    }

    auto num_keypoints() const noexcept
    {
      return peak_map.dim(0).extent();
    }

    auto num_orientation_bins() const noexcept
    {
      return peak_map.dim(1).extent();
    }

    auto orientations_count() const
    {
      return std::count(peak_map.begin(), peak_map.end(), true);
    }

    auto copy_to_host()
    {
      peak_map.copy_to_host();
      peak_residuals.copy_to_host();
    }
  };

  struct DominantOrientationSparseMap
  {
    using extremum_index_type = int;
    using angle_type = float;
    using sparse_map_type = std::multimap<extremum_index_type, angle_type>;

    sparse_map_type orientation_map;

    //! @brief Make sure the data is copied to host memory.
    DominantOrientationSparseMap(const DominantOrientationDenseMap& dense)
    {
      const auto peak_map_view =
          Sara::TensorView_<bool, 2>{
              dense.peak_map.data(),
              {dense.num_keypoints(), dense.num_orientation_bins()}}
              .matrix();
      const Eigen::VectorXi peak_count =
          peak_map_view.rowwise().count().cast<int>();

      for (auto k = 0; k < dense.num_keypoints(); ++k)
      {
        if (peak_count(k) == 0)
        {
          orientation_map.insert({k, 0});
          continue;
        }

        const auto N = dense.num_orientation_bins();
        constexpr auto two_pi = 2 * static_cast<float>(M_PI);
        for (auto o = 0; o < dense.num_orientation_bins(); ++o)
        {
          if (!dense.peak_map(k, o))
            continue;

          auto ori = o + dense.peak_residuals(k, o);

          // Make sure that the angle is in the interval [0, N[.
          if (ori < 0)
            ori += N;
          else if (ori > N)
            ori -= N;
          // Convert to radians.
          ori = ori * two_pi / N;

          orientation_map.insert({k, ori});
        }
      }
    }

    operator sparse_map_type&() noexcept
    {
      return orientation_map;
    }

    operator const sparse_map_type&() const noexcept
    {
      return orientation_map;
    }

    auto dominant_orientations(extremum_index_type i) const
    {
      auto orientations = std::vector<angle_type>{};
      const auto [o_begin, o_end] = orientation_map.equal_range(i);
      for (auto o = o_begin; o != o_end; ++o)
        orientations.push_back(o->second);
      return orientations;
    };
  };
  //! @}


  //! @brief List of oriented extrema.
  struct OrientedExtremumArray : ExtremumArray
  {
    Halide::Runtime::Buffer<float> orientations;

    auto resize(std::int32_t size)
    {
      ExtremumArray::resize(size);
      orientations = Halide::Runtime::Buffer<float>{size};
    }
  };


  struct SiftOctaveParameters
  {
    //! @brief Gaussian octave.
    //! @{
    float scale_camera = 1.f;
    float scale_initial = 1.6f;
    float scale_factor = std::pow(2.f, 1 / 3.f);
    int num_scales = 6;
    int gaussian_truncation_factor{4};
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
      Downscale = 0,
    };

    SiftOctavePipeline(const SiftOctaveParameters& params_ = {})
      : params{params_}
    {
      params.initialize_cached_scales();
    }

    auto initialize_buffers(std::int32_t w, std::int32_t h)
    {
      // Octave of Gaussians.
      gaussians = std::vector(params.num_scales + 3,
                              Halide::Runtime::Buffer<float>(w, h, 1, 1));
      // Octave of Difference of Gaussians.
      dogs = std::vector(params.num_scales + 2,
                         Halide::Runtime::Buffer<float>(w, h, 1, 1));
      // Octave of DoG extrema maps.
      extrema_maps = std::vector(
          params.num_scales, Halide::Runtime::Buffer<std::int8_t>(w, h, 1, 1));

      // Octave of Gradients of Gaussians.
      gradients = std::vector(
          params.num_scales,
          GradientBuffer{Halide::Runtime::Buffer<float>(w, h, 1, 1),
                         Halide::Runtime::Buffer<float>(w, h, 1, 1)});

      extrema_quantized.resize(params.num_scales);
      extrema.resize(params.num_scales);

      dominant_orientation_dense_maps.resize(params.num_scales);
      dominant_orientation_sparse_maps.resize(params.num_scales);
      extrema_oriented.resize(params.num_scales);
    }

    auto feed(Halide::Runtime::Buffer<float>& input,
              FirstAction first_action = FirstAction::Convolve)
    {
      // Compute the Gaussians.
      if (first_action == FirstAction::Convolve)
        shakti_gaussian_convolution_v2(input, gaussians[0], params.sigmas[0],
                                       params.gaussian_truncation_factor);
      else if (first_action == FirstAction::Downscale)
      {
        if (input.width() != gaussians[0].width() * 2 ||
            input.height() != gaussians[0].height() * 2)
          throw std::runtime_error{"Invalid input sizes!"};
        shakti_scale_32f(input, gaussians[0]);
      }
      else
        throw std::runtime_error{"Not implemented"};

      for (auto i = 1u; i < gaussians.size(); ++i)
      {
        sara::tic();
        shakti_gaussian_convolution_v2(gaussians[i - 1], gaussians[i],
                                       params.sigmas[i], 4);
        sara::toc("Gaussian convolution " + std::to_string(i) + ": " +
                  std::to_string(params.sigmas[i]));
      }

      // Compute the DoGs.
      for (auto i = 0u; i < dogs.size(); ++i)
      {
        sara::tic();
        shakti_subtract_32f(gaussians[i + 1], gaussians[i], dogs[i]);
        sara::toc("DoG " + std::to_string(i));
      }

      // Compute the gradients.
      for (auto i = 0u; i < gradients.size(); ++i)
      {
        sara::tic();
        shakti_polar_gradient_2d_32f_v2(gaussians[i + 1], gradients[i][0],
                                        gradients[i][1]);
        sara::toc("Gradients in polar coordinates " + std::to_string(i));
      }

      // Localize the extrema.
      for (auto i = 0u; i < extrema_maps.size(); ++i)
      {
        sara::tic();
        shakti_local_scale_space_extremum_32f_v2(
            dogs[i], dogs[i + 1], dogs[i + 2],         //
            params.edge_ratio, params.extremum_thres,  //
            extrema_maps[i]);
        sara::toc("DoG extremum localization " + std::to_string(i));
      }

      compress_quantized_extrema_maps();
    }

    auto compress_quantized_extrema_maps() -> void
    {
      sara::tic();
      for (auto& extrema_map : extrema_maps)
        extrema_map.copy_to_host();
      sara::toc("Copy extrema map buffers to host");

      sara::tic();
      auto extrema_quantized =
          std::vector<v2::QuantizedExtremumArray>(extrema_maps.size());
#pragma omp parallel for
      for (auto s = 0u; s < extrema_maps.size(); ++s)
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
        if (quantized.size() == 0)
          continue;

        // Resize the buffers.
        refined.resize(quantized.size());

        // Copy the necessary data from host to device memory.
        quantized.x.set_host_dirty();
        quantized.y.set_host_dirty();

        // Run the operation on the GPU.
        shakti_refine_scale_space_extrema_v2(
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
        if (e.size() == 0)
          continue;

        // Only copy the array of scales back to find the maximum scale value.
        e.s.copy_to_host();
        const auto& scale_max = *std::max_element(e.s.begin(), e.s.end());

        //  Outputs.
        auto& d = dominant_orientation_dense_maps[s];
        d.resize(e.size(), params.num_orientation_bins);

        shakti_dominant_gradient_orientations_v2(
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
#pragma omp parallel for
      for (auto s = 0u; s < dominant_orientation_dense_maps.size(); ++s)
      {
        auto& dense = dominant_orientation_dense_maps[s];
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
      for (auto s = 0u; s < extrema_oriented.size(); ++s)
      {
        auto& d = dominant_orientation_sparse_maps[s];
        auto& e = extrema[s];
        auto& e_oriented = extrema_oriented[s];
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
  };


}  // namespace DO::Shakti::HalideBackend::v2
