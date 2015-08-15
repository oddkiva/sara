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

  template <typename F, typename D>
  class KeyRef
  {
  public:
    using feature_type = F;
    using Descriptor = D;

    inline KeyRef(feature_type& f, Descriptor& d)
      : _f(&f)
      , _d(&d)
    {
    }

    inline feature_type& feature() const
    {
      return _f;
    }
    inline Descriptor& descriptor() const
    {
      return _d;
    }

    KeyRef operator=(KeyRef key) const
    {
      _f = key._f;
      _d = key._d;
      return *this;
    }

  private:
    feature_type& _f;
    Descriptor& _d;
  };


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

    using key_type = KeyRef<const feature_type, const_descriptor_type>;
    using const_key_type = KeyRef<const feature_type, const_descriptor_type>;

    inline key_type operator[](size_t i)
    {
      return KeyRef<feature_type, descriptor_type>(
        features[i], descriptors[i]);
    }

    inline const_key_type operator[](size_t i) const
    {
      return KeyRef<const feature_type, const_descriptor_type>(
        features[i], descriptors[i]);
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
