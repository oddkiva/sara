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

namespace DO {

  /*!
    \ingroup Features
    @{
  */

  template <typename T>
  struct EqualDescriptor
  {
    EqualDescriptor(const DescriptorMatrix<T>& descriptors)
      : descriptors_(descriptors) {}
    inline bool operator()(int i1, int i2) const
    { return descriptors_[i1] == descriptors_[i2]; }
    const DescriptorMatrix<T>& descriptors_;
  };

  template <>
  struct EqualDescriptor<float>
  {
    EqualDescriptor(const DescriptorMatrix<float>& descriptors)
      : descriptors_(descriptors) {}
    inline bool operator()(int i1, int i2) const
    { return (descriptors_[i1] - descriptors_[i2]).squaredNorm() < 1e-3; }
    const DescriptorMatrix<float>& descriptors_;
  };

  template <typename T>
  struct CompareFeatures
  {
    CompareFeatures(const std::vector<OERegion>& features,
                    const DescriptorMatrix<T>& descriptors)
      : features_(features)
      , descriptors_(descriptors)
      , equal_descriptors_(descriptors) {}
    inline bool operator()(int i1, int i2) const
    { 
      if (lexCompare(descriptors_[i1], descriptors_[i2]))
        return true;
      if (equal_descriptors_(i1, i2) &&
          features_[i1].extremumValue() > features_[i2].extremumValue())
        return true;
      return false;
    }
    const std::vector<OERegion>& features_;
    const DescriptorMatrix<T>& descriptors_;
    EqualDescriptor<T> equal_descriptors_;
  };

  template<typename T> bool isfinite(T arg)
  {
    return arg == arg && 
      arg != std::numeric_limits<T>::infinity() &&
      arg != -std::numeric_limits<T>::infinity();
  }

  void removeRedundancies(std::vector<OERegion>& features,
                          DescriptorMatrix<float>& descriptors);

  //! @}

} /* namespace DO */



#endif /* DO_FEATURES_UTILITIES_HPP */