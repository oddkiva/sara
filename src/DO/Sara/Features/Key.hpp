// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_FEATURES_KEY_HPP
#define DO_SARA_FEATURES_KEY_HPP

#include <Eigen/StdVector>

#include <DO/Sara/Core/StdVectorHelpers.hpp>

#include <DO/Sara/Features/DescriptorMatrix.hpp>
#include <DO/Sara/Features/Feature.hpp>


namespace DO { namespace Sara {

  /*!
    \ingroup feature_types
    @{
  */

  enum DescriptorType
  {
    RealDescriptor,
    BinaryDescriptor
  };

  template <DescriptorType> struct Bin;

  template <>
  struct Bin<RealDescriptor> { typedef float Type; };

  template <>
  struct Bin<BinaryDescriptor> { typedef unsigned char Type; };


  template <typename F, DescriptorType D>
  class Set
  {
  public:
    using bin_type = typename Bin<D>::Type;
    using feature_type = F;
    using descriptor_type =
      typename DescriptorMatrix<bin_type>::descriptor_type;
    using const_descriptor_type =
      typename DescriptorMatrix<bin_type>::const_descriptor_type;

    void resize(size_t num_keypoints, size_t descriptor_dimension)
    {
      features.resize(num_keypoints);
      descriptors.resize(num_keypoints, descriptor_dimension);
    }

    inline size_t size() const
    {
      if (features.size() != descriptors.size())
        throw std::runtime_error{
          "Number of features and number of descriptors don't match!"
        };
      return features.size();
    }

    inline void swap(Set& set)
    {
      std::swap(features, set.features);
      std::swap(descriptors, set.descriptors);
    }

    inline void append(const Set& other)
    {
      ::append(features, other.features);
      descriptors.append(other.descriptors);
    }

    std::vector<feature_type> features;
    DescriptorMatrix<bin_type> descriptors;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATURES_KEY_HPP */
