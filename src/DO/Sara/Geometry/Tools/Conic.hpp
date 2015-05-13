// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_GEOMETRY_TOOLS_CONIC_HPP
#define DO_SARA_GEOMETRY_TOOLS_CONIC_HPP

#include <DO/Sara/Geometry/Tools/MatrixBasedObject.hpp>


namespace DO { namespace Projective {

  template <typename T, int N>
  class Conic : public MatrixBasedObject<T,N>
  {
    typedef MatrixBasedObject<T,N> Base;
    using Base::mat_;
  public:
    using Base::Dimension;
    typedef typename Base::Mat  Mat;
    typedef typename Base::HVec HVec;
    typedef typename Base::Vec  Vec;
    //! Common constructors
    inline Conic() : Base() {}
    inline Conic(const Base& other) : Base(other) {}
    inline Conic(const Mat& data) : Base(data) {}
    //! Evaluation at point 'x'
    inline T operator()(const HVec& x) const
    { return x.transpose()*mat_*x; }
    //! Evaluation at point 'x'
    inline T operator()(const Vec& x) const
    { return (*this)((HVec() << x, 1).finished()); }
  };

  //! I/O
  template <typename T, int N>
  std::ostream& operator<<(std::ostream& os, const Conic<T, N>& P);

} /* namespace Projective */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_TOOLS_CONIC_HPP */
