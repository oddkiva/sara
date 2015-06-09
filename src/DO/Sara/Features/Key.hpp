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

#include <DO/Sara/Features/Feature.hpp>


namespace DO { namespace Sara {

  /*!
    \ingroup Features
    @{
  */

  template <typename F, typename D>
  class KeyRef
  {
  public:
    typedef F Feature;
    typedef D Descriptor;
    inline KeyRef(Feature& f, Descriptor& d) : f_(&f), d_(&d) {}
    inline Feature& feature() const { return f_; }
    inline Descriptor& descriptor() const { return d_; }
    KeyRef operator=(KeyRef key) const
    { f_ = key.f_; d_ = key.d_; return *this; }

  private:
    Feature& f_;
    Descriptor& d_;
  };

  enum DescriptorType { RealDescriptor, BinaryDescriptor };
  template <DescriptorType> struct Bin;
  template <> struct Bin<RealDescriptor> { typedef float Type; };
  template <> struct Bin<BinaryDescriptor> { typedef unsigned char Type; };

  template <typename F, DescriptorType D>
  class Set
  {
  public:
    typedef typename Bin<D>::Type BinType;
    typedef F Feature;
    typedef typename DescriptorMatrix<BinType>::descriptor_type
      Descriptor;
    typedef typename DescriptorMatrix<BinType>::const_descriptor_type
      ConstDescriptor;

    typedef KeyRef<const Feature, ConstDescriptor> Key;
    typedef KeyRef<const Feature, ConstDescriptor> ConstKey;

    inline Key operator[](int i)
    {
      return KeyRef<Feature, Descriptor>(features[i], descriptors[i]);
    }

    inline ConstKey operator[](int i) const
    {
      return KeyRef<const Feature, ConstDescriptor>(features[i], descriptors[i]);
    }

    inline size_t size() const
    {
      if (features.size() != descriptors.size())
      {
        std::string msg("Number of features and number of descriptors don't match!");
        throw std::runtime_error(msg.c_str());
      }
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

    std::vector<F> features;
    DescriptorMatrix<BinType> descriptors;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATURES_KEY_HPP */
