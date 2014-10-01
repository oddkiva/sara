// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== /

//! @file

#ifndef DO_CORE_IMAGE_ELEMENTTRAITS_HPP
#define DO_CORE_IMAGE_ELEMENTTRAITS_HPP


#include <DO/Core/MultiArray/ElementTraits.hpp>
#include <DO/Core/Pixel/Pixel.hpp>


namespace DO {

  /*!
    \ingroup Image Image
    @{
   */

  //! \brief The specialized element traits class when the entry is a color.
  template <typename T, typename Layout>
  struct ElementTraits<Pixel<T, Layout> >
  {
    typedef Array<T, Layout::size, 1> value_type; //!< STL-like typedef.
    typedef size_t size_type; //!< STL-like typedef.
    typedef value_type * pointer; //!< STL-like typedef.
    typedef const value_type * const_pointer; //!< STL-like typedef.
    typedef value_type& reference; //!< STL-like typedef.
    typedef const value_type& const_reference; //!< STL-like typedef.
    typedef value_type * iterator; //!< STL-like typedef.
    typedef const value_type * const_iterator; //!< STL-like typedef.
    static const bool is_scalar = false; //!< STL-like typedef.
  };

  //! @}

}


#endif /* DO_CORE_IMAGE_ELEMENTTRAITS_HPP */