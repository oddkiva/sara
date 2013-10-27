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

  class Key
  {
    VisualFeature *pf_;
    DescriptorBase *pd_;

  public:
    inline Key() : pf_(0), pd_(0) {}
    inline Key(VisualFeature& f, DescriptorBase& d) : pf_(&f), pd_(&d) {}

    //! Constant accessors.
    inline const VisualFeature& feat() const { return *pf_; }
    inline const DescriptorBase& desc() const { return *pd_; }
    
    //! Non constant accessors.
    inline VisualFeature& feat() { return *pf_; }
    inline DescriptorBase& desc() { return *pd_; }
  };

  //typedef Matrix<float, 128, 1> Desc128f;
  //
  ////! Deprecated...
  //class Keypoint
  //{
  //public:
  //  Keypoint() {}
  //  Keypoint(const OERegion& f, Desc128f& d) : f_(f), d_(d) {}

  //  //! Constant accessors.
  //  inline const OERegion& feat() const { return f_; }
  //  inline const Desc128f& desc() const { return d_; }

  //  //! Non constant accessors.
  //  inline OERegion& feat() { return f_; }
  //  inline Desc128f& desc() { return d_; }
  //private:
  //  OERegion f_;
  //  Desc128f d_;
  //};

  //! @}

} /* namespace DO */

#endif /* DO_FEATURES_KEY_HPP */