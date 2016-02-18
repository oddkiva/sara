// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_GEOMETRY_TOOLS_MATRIXBASEDOBJECT_HPP
#define DO_SARA_GEOMETRY_TOOLS_MATRIXBASEDOBJECT_HPP

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara { namespace Projective {

  //! Rudimentary polynomial class.
  template <typename T, int N>
  class MatrixBasedObject
  {
  public:
    enum { Dimension = N };
    using Mat = Matrix<T, N+1, N+1>;
    using HVec = Matrix<T, N+1, 1>; // in projective space
    using Vec = Matrix<T, N  , 1>;  // in Euclidean space

    //! @{
    //! @brief Common constructors
    MatrixBasedObject() = default;

    inline MatrixBasedObject(const MatrixBasedObject& other)
    {
      copy(other);
    }

    inline MatrixBasedObject(const Mat& data)
      : _mat(data)
    {
    }
    //! @}

    //! @brief Assignment operator.
    MatrixBasedObject& operator=(const MatrixBasedObject& other)
    {
      copy(other);
      return *this;
    }

    //! @{
    //! @brief Matrix accessor.
    inline Mat& matrix()
    {
      return _mat;
    }

    inline const Mat& matrix() const { return _mat; }
    //! @}

    //! @{
    //! @brief Coefficient accessor.
    inline T& operator()(int i, int j)
    {
      return _mat(i,j);
    }

    inline T operator()(int i, int j) const
    {
      return _mat(i,j);
    }
    //! @}

    //! @{
    //! @brief Comparison operator.
    inline bool operator==(const MatrixBasedObject& other) const
    {
      return _mat == other._mat;
    }

    inline bool operator!=(const MatrixBasedObject& other) const
    {
      return !operator=(other);
    }
    //! @}

  private:
    inline void copy(const MatrixBasedObject& other)
    {
      _mat = other._mat;
    }

  protected:
    Mat _mat;
  };


  template <typename T, int N>
  class Homography : public MatrixBasedObject<T,N>
  {
    using Base = MatrixBasedObject<T, N>;
    using Base::_mat;

  public:
    using Base::Dimension;
    using Mat = typename Base::Mat;
    using HVec = typename Base::HVec;
    using Vec = typename Base::Vec;

    //! @{
    //! @brief Common constructors
    Homography() = default;

    inline Homography(const Base& other)
      : Base(other)
    {
    }

    inline Homography(const Mat& data)
      : Base(data)
    {
    }
    //! @}

    //! @{
    //! @brief Evaluation at point 'x'.
    inline T operator()(const HVec& x) const
    {
      return x.transpose()*_mat*x;
    }

    inline T operator()(const Vec& x) const
    {
      return (*this)((HVec() << x, 1).finished());
    }
    //! @}
  };


} /* namespace Sara */
} /* namespace Projective */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_TOOLS_MATRIXBASEDOBJECT_HPP */
