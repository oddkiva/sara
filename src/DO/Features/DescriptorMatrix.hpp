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

#ifndef DO_FEATUREDESCRIPTORS_DESCRIPTORMATRIX_HPP
#define DO_FEATUREDESCRIPTORS_DESCRIPTORMATRIX_HPP

namespace DO {

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
    DescriptorMatrix() {}
    DescriptorMatrix(int num_descriptors, int dimension)
    { resize(num_descriptors, dimension); }

    void resize(int num_descriptors, int dimension)
    { matrix_type::resize(dimension, num_descriptors); }

    matrix_type& matrix() { return *this; }
    const matrix_type& matrix() const { return *this; }

    int size() const { return static_cast<int>(cols()); }
    int dimension() const { return static_cast<int>(rows()); }

    typename descriptor_type operator[](int i)
    { return this->col(i); }

    typename const_descriptor_type operator[](int i) const
    { return this->col(i); }
  };

  //! @}

} /* namespace DO */

#endif /* DO_FEATUREDESCRIPTORS_DESCRIPTORMATRIX_HPP */