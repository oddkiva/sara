// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_FEATUREDESCRIPTORS_DENSEFEATURE_HPP
#define DO_SARA_FEATUREDESCRIPTORS_DENSEFEATURE_HPP

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Image.hpp>

#include <DO/Sara/FeatureDescriptors/Orientation.hpp>
#include <DO/Sara/FeatureDescriptors/SIFT.hpp>

#include <DO/Sara/ImageProcessing/Deriche.hpp>


namespace DO { namespace Sara {

  //! \brief Dense Feature Computer class.
  template <typename BaseFeatureComputer = ComputeSIFTDescriptor>
  class DenseFeatureComputer
  {
  public:
    using descriptor_type = typename BaseFeatureComputer::descriptor_type;

    //! \brief Default Constructor.
    inline DenseFeatureComputer()
      : _compute_feature()
    {
    }

    //! \brief Operator.
    Image<descriptor_type> operator()(const Image<float>& image, int patch_size = 8)
    {
      // Blur the image a little bit.
      // By default, sigma is '1.6' and is justified in A-SIFT paper [Morel, Yu].
      auto blurred_image = image.compute<DericheBlur>(1.6f);

      // Compute the image gradients in polar coordinates.
      auto gradients = gradient_polar_coordinates(blurred_image);

      // Compute the feature vector in each pixel.
      auto patch_radius = patch_size / 2;
      auto features = Image<descriptor_type> { image.sizes() };
      features.array().fill(descriptor_type::Zero());
      for (auto y = patch_radius; y < image.height() - patch_radius; ++y)
        for (auto x = patch_radius; x < image.width() - patch_radius; ++x)
          features(x, y) = _compute_feature(float(x), float(y), float(patch_radius),
                                            gradients);

      return features;
    }

  private:
    BaseFeatureComputer _compute_feature;
  };


  /*!
    Helper function that computes the SIFT descriptor at each image pixel
    with scale equal to 'local_patch_size / 2'.
   */
  DO_EXPORT
  Image<ComputeSIFTDescriptor<>::descriptor_type>
  compute_dense_sift(const Image<float>& image, int local_patch_size = 8);

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATUREDESCRIPTORS_DENSEFEATURE_HPP */