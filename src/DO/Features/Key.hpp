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

  template <typename F, typename D>
  class Key
  {
  public:
    typedef F Feature;
    typedef D Descriptor;

    inline Key() {}
    inline Key(const Feature& f, const Descriptor& d) : pf_(&f), pd_(&d) {}

    //! Constant accessors.
    inline const Feature& feat() const { return pf_; }
    inline const Descriptor& desc() const { return pd_; }
    
    //! Non constant accessors.
    inline Feature& feat() { return pf_; }
    inline Descriptor& desc() { return pd_; }
    bool operator==(const Key& k) const
    { return feat() == k.feat() && desc() == k.desc(); }

  private:
    Feature *pf_;
    Descriptor *pd_;
  };

  struct KeyRef
  {
    Feature *f;
    void *d;
    int N;
    int type;
    int dScalarType;
  };

  typedef Key<OERegion, Desc128f> Keypoint;

  //! @}

} /* namespace DO */

#ifndef __APPLE__
  EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(DO::Key<DO::OERegion, DO::Desc128ub>)
#endif

#endif /* DO_FEATURES_KEY_HPP */