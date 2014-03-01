// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_GEOMETRY_TOOLS_CONIC_HPP
#define DO_GEOMETRY_TOOLS_CONIC_HPP

#include <DO/Geometry/Tools/MatrixBasedObject.hpp>

namespace DO { namespace Projective {

  //! Rudimentary polynomial class.
  template <typename T, int N>
  class MatrixBasedObject
  {
  public:
    enum { Dimension = N };
    typedef Matrix<T, N+1, N+1> Mat;
    typedef Matrix<T, N+1, 1> HVec; // in projective space
    typedef Matrix<T, N  , 1> Vec;  // in Euclidean space
    //! Common constructors
    inline MatrixBasedObject() {}
    inline MatrixBasedObject(const MatrixBasedObject& other) { copy(other); }
    inline MatrixBasedObject(const Mat& data) : mat_(data) {}
    //! Assignment operator
    MatrixBasedObject& operator=(const MatrixBasedObject& other)
    { copy(other); return *this; }
    //! Matrix accessor
    inline Mat& matrix() { return mat_; }
    inline const Mat& matrix() const { return mat_; }
    //! Coefficient accessor
    inline T& operator()(int i, int j)
    { return mat_(i,j); }
    inline T operator()(int i, int j) const
    { return mat_(i,j); }
    //! Comparison operator
    inline bool operator==(const MatrixBasedObject& other) const
    { return mat_ == other.mat_; }
    inline bool operator!=(const MatrixBasedObject& other) const
    { return !operator=(other); }
  private:
    inline void copy(const MatrixBasedObject& other)
    { mat_ = other.mat_; }
  protected:
    Mat mat_;
  };

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
    //! I/O
    friend std::ostream& operator<<(std::ostream& os,const Conic& P);
  };
  
  template <typename T, int N>
  class Homography : public MatrixBasedObject<T,N>
  {
    typedef MatrixBasedObject<T,N> Base;
    using Base::mat_;
  public:
    using Base::Dimension;
    typedef typename Base::Mat  Mat;
    typedef typename Base::HVec HVec;
    typedef typename Base::Vec  Vec;
    //! Common constructors
    inline Homography() : Base() {}
    inline Homography(const Base& other) : Base(other) {}
    inline Homography(const Mat& data) : Base(data) {}
    //! Evaluation at point 'x'
    inline T operator()(const HVec& x) const
    { return x.transpose()*mat_*x; }
    //! Evaluation at point 'x'
    inline T operator()(const Vec& x) const
    { return (*this)((HVec() << x, 1).finished()); }
  };

} /* namespace Projective */
} /* namespace DO */

#endif /* DO_GEOMETRY_TOOLS_CONIC_HPP */
