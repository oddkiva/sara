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
    using bin_type = T;
    using matrix_type = Matrix<T, Dynamic, Dynamic>;
    using descriptor_type = typename matrix_type::ColXpr;
    using const_descriptor_type = typename matrix_type::ConstColXpr;

  public:
    //! @{
    //! \brief Constructor.
    DescriptorMatrix() = default;

    DescriptorMatrix(size_t num_descriptors, size_t dimension)
    {
      resize(num_descriptors, dimension);
    }
    //! @}

    //! @{
    //! \brief Accessors.
    inline matrix_type& matrix()
    {
      return *this;
    }

    inline const matrix_type& matrix() const
    {
      return *this;
    }

    inline size_t size() const
    {
      return matrix_type::cols();
    }

    inline size_t dimension() const
    {
      return matrix_type::rows();
    }

    inline descriptor_type operator[](size_t i)
    {
      return this->col(i);
    }

    inline const_descriptor_type operator[](size_t i) const
    {
      return this->col(i);
    }
    //! @}

    //! \brief Resize the descriptor matrix.
    inline void resize(size_t num_descriptors, size_t dimension)
    {
      matrix_type::resize(dimension, num_descriptors);
    }

    //! \brief Swap data between `DescriptorMatrix` objects.
    inline void swap(DescriptorMatrix& other)
    {
      matrix_type::swap(other);
    }

    //! \brief Append data from another `DescriptorMatrix` object.
    void append(const DescriptorMatrix& other)
    {
      if (dimension() != other.dimension() && matrix_type::size() != 0)
        throw std::runtime_error{
          "Fatal: other descriptor matrix does not have same dimension"
        };

      size_t dim = other.dimension();

      matrix_type data{ dim, size() + other.size() };
      data.block(0, 0, dim, size()) = *this;
      data.block(0, size(), dim, other.size()) = other;
      matrix_type::swap(data);
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATUREDESCRIPTORS_DESCRIPTORMATRIX_HPP */
