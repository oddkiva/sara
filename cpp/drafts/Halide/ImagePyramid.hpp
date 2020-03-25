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


namespace DO::Shakti::HalideBackend {

  /*!
   *  The image pyramid is regular, i.e., it has:
   *  - the same number of scales in each octave
   *  - the same geometric progression factor in the scale in each octave
   */
  template <typename T, int N = 2>
  class ImagePyramid
  {
  public: /* member functions */
    //! Convenient typedefs.
    //! @{
    using scalar_type = T;

    using array_view_type = Sara::TensorView_<T, N>;
    using array_sizes_type = typename array_view_type::vector_type;

    using octave_view_type = Sara::TensorView_<T, N + 1>;
    using octave_type = Sara::Tensor_<T, N + 1>;
    using octave_sizes_type = typename octave_type::vector_type;
    //! @}

    //! @brief Default constructor.
    inline ImagePyramid() = default;

    //! @brief Reset image pyramid with the following parameters.
    void reset(const array_sizes_type& image_sizes, int num_octaves,
               int num_scales_per_octave, scalar_type initial_scale,
               scalar_type scale_geometric_factor)
    {
      _octaves.clear();
      _oct_scaling_factors.clear();

      _octaves.resize(num_octaves);
      _oct_scaling_factors.resize(num_octaves);

      for (int o = 0; o < num_octaves; ++o)
      {
        auto octave_sizes = octave_sizes_type{};
        octave_sizes << num_scales_per_octave, image_sizes / (1 << o);

        _octaves[o].resize(octave_sizes);
      }

      _scale_initial = initial_scale;
      _scale_geometric_factor = scale_geometric_factor;
    }

    //! @brief Mutable octave getter.
    auto operator()(int o) -> octave_type&
    {
      return _octaves[o];
    }

    //! @brief Immutable octave getter.
    auto operator()(int o) const -> const octave_type&
    {
      return _octaves[o];
    }

    //! @brief Mutable image getter.
    auto operator()(int o, int s) -> array_view_type&
    {
      return _octaves[o][s];
    }

    //! @brief Immutable image getter.
    auto operator()(int o, int s) const -> const array_view_type&
    {
      return _octaves[o][s];
    }

    //! @brief Mutable pixel getter.
    auto operator()(int o, int s, int y, int x) -> scalar_type&
    {
      static_assert(N == 2);
      return _octaves[o](s, y, x);
    }

    //! @brief Immutable pixel getter.
    auto operator()(int o, int s, int y, int x) const -> const scalar_type&
    {
      static_assert(N == 2);
      return _octaves[o](s, y, x);
    }

    //! @brief Immutable getter of the octave scaling factor.
    auto octave_scaling_factor(int o) const
    {
      return _oct_scaling_factors[o];
    }

    //! @brief Immutable getter of the number of octaves.
    auto num_octaves() const
    {
      return static_cast<int>(_octaves.size());
    }

    //! @brief Immutable getter of the number of scales per octave.
    auto num_scales_per_octave() const
    {
      return static_cast<int>(_octaves.front().size(0));
    }

    //! @brief Immutable getter of the initial scale.
    auto scale_initial() const
    {
      return _scale_initial;
    }

    //! @brief Immutable getter of the scale geometric factor.
    auto scale_geometric_factor() const
    {
      return _scale_geometric_factor;
    }

    //! @brief Immutable getter of the relative scale w.r.t. an octave.
    auto scale_relative_to_octave(int s) const
    {
      return pow(_scale_geometric_factor, s) * _scale_initial;
    }

    //! @brief Immutable getter of the scale relative to an octave.
    auto scale(int s, int o) const
    {
      return _oct_scaling_factors[o] * scale_relative_to_octave(s);
    }

  protected: /* data members */
    //! @{
    //! @brief Parameters.
    scalar_type _scale_initial;
    scalar_type _scale_geometric_factor;
    //! @}

    //! @{
    //! @brief Image data.
    std::vector<octave_type> _octaves;
    std::vector<scalar_type> _oct_scaling_factors;
    //! @}
  };

  //! @}

}  // namespace DO::Shakti::HalideBackend
