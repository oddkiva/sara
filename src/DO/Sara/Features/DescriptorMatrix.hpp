// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_FEATUREDESCRIPTORS_DESCRIPTORMATRIX_HPP
#define DO_SARA_FEATUREDESCRIPTORS_DESCRIPTORMATRIX_HPP

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  /*!
    \ingroup Features
    @{
  */

  template <typename T>
  class DescriptorMatrix : private Matrix<T, Dynamic, Dynamic>
  {
  public:
    typedef T bin_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef typename matrix_type::ColXpr descriptor_type;
    typedef typename matrix_type::ConstColXpr const_descriptor_type;

  public:
    DescriptorMatrix() = default;

    DescriptorMatrix(int num_descriptors, int dimension)
    {
      resize(num_descriptors, dimension);
    }

    inline void resize(int num_descriptors, int dimension)
    {
      matrix_type::resize(dimension, num_descriptors);
    }

    inline matrix_type& matrix()
    {
      return *this;
    }

    inline const matrix_type& matrix() const
    {
      return *this;
    }

    inline int size() const
    {
      return static_cast<int>(matrix_type::cols());
    }

    inline int dimension() const
    {
      return static_cast<int>(matrix_type::rows());
    }

    inline descriptor_type operator[](int i)
    {
      return this->col(i);
    }

    inline const_descriptor_type operator[](int i) const
    {
      return this->col(i);
    }

    inline void swap(DescriptorMatrix& other)
    {
      matrix_type::swap(other);
    }

    void append(const DescriptorMatrix& other)
    {
      if (dimension() != other.dimension() && matrix_type::size() != 0)
      {
        std::cerr << "Fatal: other descriptor matrix does not have same dimension" << std::endl;
        CHECK(dimension());
        CHECK(other.dimension());
        throw std::runtime_error{
          "Fatal: other descriptor matrix does not have same dimension"
        };
      }

      int dim = other.dimension();

      matrix_type tmp(dim, size() + other.size());
      tmp.block(0, 0, dim, size()) = *this;
      tmp.block(0, size(), dim, other.size()) = other;
      matrix_type::swap(tmp);
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATUREDESCRIPTORS_DESCRIPTORMATRIX_HPP */