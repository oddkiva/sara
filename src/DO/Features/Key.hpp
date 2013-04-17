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

#ifndef DO_FEATURES_KEY_HPP
#define DO_FEATURES_KEY_HPP

#include <Eigen/StdVector>

namespace DO {

  template <typename F, typename D>
  class Key
  {
  public:
    typedef F Feature;
    typedef D Descriptor;

    inline Key() {}
    inline Key(const Feature& f, const Descriptor& d) : f_(f), d_(d) {}

    //! Constant accessors.
    inline const Feature& feat() const { return f_; }
    inline const Descriptor& desc() const { return d_; }
    
    //! Non constant accessors.
    inline Feature& feat() { return f_; }
    inline Descriptor& desc() { return d_; }
    bool operator==(const Key& k) const
    { return feat() == k.feat() && desc() == k.desc(); }

  private:
    Feature f_;
    Descriptor d_;
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

} /* namespace DO */

#ifndef __APPLE__
  EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(DO::Key<DO::OERegion, DO::Desc128ub>)
#endif

#endif /* DO_FEATURES_KEY_HPP */