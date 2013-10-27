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

  //! Deprecated...
  template <typename F, typename D>
  class Key
  {
    typedef F Feature;
    typedef D Descriptor;

    Feature *pf_;
    Descriptor *pd_;

  public:
    inline Key() : pf_(0), pd_(0) {}
    inline Key(Feature& f, Descriptor& d) : pf_(&f), pd_(&d) {}

    //! Constant accessors.
    inline const Feature& feat() const { return *pf_; }
    inline const Descriptor& desc() const { return *pd_; }
    
    //! Non constant accessors.
    inline Feature& feat() { return *pf_; }
    inline Descriptor& desc() { return *pd_; }

    inline void swap(const Key& k)
    {
      std::swap(pf_, k.pf_);
      std::swap(pd_, k.pd_);
    }
  };

  //! @}

} /* namespace DO */

namespace std
{
  template <typename F, typename D>
  inline void swap(DO::Key<F,D>& a, DO::Key<F,D>& b)
  { a.swap(b); }
}

#endif /* DO_FEATURES_KEY_HPP */