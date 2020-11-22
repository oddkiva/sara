// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/FeatureDescriptors/SIFT.hpp>


namespace DO::Sara {

  /*!
    @addtogroup FeatureDescriptors
    @{
   */

  //! @brief Functor class used to compute the SIFT Descriptor at some location.
  template <int N=4, int O=8>
  class ComputeRootSIFTDescriptor : ComputeSIFTDescriptor<N, O>
  {
    using base_type = ComputeSIFTDescriptor<N, O>;

  public: /* interface. */
    using descriptor_type = typename base_type::descriptor_type;

    //! @brief Constructor.
    inline ComputeRootSIFTDescriptor(float bin_scale_unit_length = 3.f,
                                     float max_bin_value = 0.2f)
      : base_type{bin_scale_unit_length, max_bin_value}
    {
    }

    //! @brief Computes the SIFT descriptor for keypoint @f$(x,y,\sigma,\theta)@f$.
    auto operator()(float x, float y, float sigma, float theta,
                    const ImageView<Vector2f>& grad_polar_coords) const
        -> descriptor_type
    {
      auto h = base_type::operator()(x, y, sigma, theta, grad_polar_coords);
      h /= h.template lpNorm<1>();
      h = h.array().cwise().sqrt().matrix();
      return h;
    }

    //! @brief Computes the **upright** SIFT descriptor for keypoint @f$(x,y,\sigma)@f$.
    auto operator()(float x, float y, float sigma,
                    const ImageView<Vector2f>& grad_polar_coords) const
        -> descriptor_type
    {
      return this->operator()(x, y, sigma, 0.f, grad_polar_coords);
    }

    //! Helper member function.
    auto operator()(const OERegion& f,
                    const ImageView<Vector2f>& grad_polar_coords) const
        -> descriptor_type
    {
      return this->operator()(f.x(), f.y(), f.scale(), f.orientation,
                              grad_polar_coords);
    }

    //! Helper member function.
    auto operator()(const std::vector<OERegion>& features,
                    const std::vector<Point2i>& scale_octave_pairs,
                    const ImagePyramid<Vector2f>& gradient_polar_coords) const
        -> Tensor_<float, 2>
    {
      auto sifts = Tensor_<float, 2>{{int(features.size()), base_type::Dim}};
      for (size_t i = 0; i < features.size(); ++i)
      {
        sifts.matrix().row(i) =
            this->operator()(features[i],
                             gradient_polar_coords(scale_octave_pairs[i](0),
                                                   scale_octave_pairs[i](1)))
                .transpose();
      }
      return sifts;
    }
  };

  //! @}


}
