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

#ifndef DO_FEATURES_KEY_HPP
#define DO_FEATURES_KEY_HPP

#include <Eigen/StdVector>

namespace DO {

  /*!
    \ingroup Features
    @{
  */
  
  template <typename T>
  class Descriptors : private Matrix<T, Dynamic, Dynamic>
  {
  public:
    typedef Matrix<T, Dynamic, Dynamic> MatrixType;
    typedef Descriptor<T> DescriptorType;

  public:
    Descriptors() {}

  private:
    std::vector<DescriptorType > descriptors_;

  };

  class Key
  {
    Feature *pf_;
    DescriptorBase *pd_;

  public:
    inline Key() : pf_(0), pd_(0) {}
    inline Key(Feature& f, DescriptorBase& d) : pf_(&f), pd_(&d) {}

    //! Constant accessors.
    inline const Feature& feat() const { return *pf_; }
    inline const DescriptorBase& desc() const { return *pd_; }
    
    //! Non constant accessors.
    inline Feature& feat() { return *pf_; }
    inline DescriptorBase& desc() { return *pd_; }
    bool operator==(const Key& k) const
    { return pf_ == k.pf_ && pd_ == k.pd_; }
  };

  //! @}

} /* namespace DO */

#endif /* DO_FEATURES_KEY_HPP */