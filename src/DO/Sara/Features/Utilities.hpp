// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_FEATURES_UTILITIES_HPP
#define DO_FEATURES_UTILITIES_HPP


#include <DO/Features/Feature.hpp>


namespace DO {

  /*!
    \ingroup Features
    @{
  */

  template <typename T>
  struct EqualDescriptor
  {
    EqualDescriptor(const DescriptorMatrix<T>& descriptors)
      : _descriptors(descriptors)
    {
    }

    inline bool operator()(int i1, int i2) const
    {
       return _descriptors[i1] == _descriptors[i2];
    }

    const DescriptorMatrix<T>& _descriptors;
  };

  template <>
  struct EqualDescriptor<float>
  {
    EqualDescriptor(const DescriptorMatrix<float>& descriptors)
      : _descriptors(descriptors)
    {
    }

    inline bool operator()(int i1, int i2) const
    {
       return (_descriptors[i1] - _descriptors[i2]).squaredNorm() < 1e-3;
    }

    const DescriptorMatrix<float>& _descriptors;
  };

  template <typename T>
  struct CompareFeatures
  {
    CompareFeatures(const std::vector<OERegion>& features,
                    const DescriptorMatrix<T>& descriptors)
      : _features(features)
      , _descriptors(descriptors)
      , _equal_descriptors(descriptors)
    {
    }

    inline bool operator()(int i1, int i2) const
    {
      if (DO::lexicographical_compare(_descriptors[i1], _descriptors[i2]))
        return true;
      if (_equal_descriptors(i1, i2) &&
          _features[i1].extremum_value() > _features[i2].extremum_value())
        return true;
      return false;
    }

    const std::vector<OERegion>& _features;
    const DescriptorMatrix<T>& _descriptors;
    EqualDescriptor<T> _equal_descriptors;
  };

  template<typename T>
  inline bool isfinite(T arg)
  {
    return
      arg == arg &&
      arg != std::numeric_limits<T>::infinity() &&
      arg != -std::numeric_limits<T>::infinity();
  }

  void remove_redundancies(std::vector<OERegion>& features,
                          DescriptorMatrix<float>& descriptors);

  inline void remove_redundancies(Set<OERegion, RealDescriptor>& keys)
  {
     remove_redundancies(keys.features, keys.descriptors);
  }

  //! @}

} /* namespace DO */


#endif /* DO_FEATURES_UTILITIES_HPP */
