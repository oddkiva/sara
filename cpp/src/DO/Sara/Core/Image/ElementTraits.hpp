// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== /

//! @file

#ifndef DO_SARA_CORE_IMAGE_ELEMENTTRAITS_HPP
#define DO_SARA_CORE_IMAGE_ELEMENTTRAITS_HPP


#include <DO/Sara/Core/MultiArray/ElementTraits.hpp>
#include <DO/Sara/Core/Pixel/Pixel.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup Image Image
    @{
   */

  //! @brief The specialized element traits class when the entry is a color.
  template <typename T, typename ColorSpace>
  struct ElementTraits<Pixel<T, ColorSpace> >
  {
    //! @{
    //! @brief STL-like typedefs.
    using value_type = Array<T, ColorSpace::size, 1>;
    using size_type = size_t;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = value_type *;
    using const_iterator = const value_type *;
    static const bool is_scalar = false;
    //! @}
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_IMAGE_ELEMENTTRAITS_HPP */
