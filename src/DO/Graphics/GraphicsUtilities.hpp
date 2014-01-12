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

#ifndef DO_GRAPHICS_GRAPHICSUTILITIES_HPP
#define DO_GRAPHICS_GRAPHICSUTILITIES_HPP

#include "DerivedQObjects/GraphicsApplicationImpl.hpp"

namespace DO {

  /*!
    \addtogroup GraphicsInternal

    @{
   */

  // ======================================================================== //
  // Convenience functions
  inline GraphicsApplication::Impl * getGuiApp()
  { return qobject_cast<GraphicsApplication::Impl *>(qApp); }

  inline UserThread& getUserThread()
  { return getGuiApp()->userThread; }

  inline bool activeWindowIsVisible()
  { return getGuiApp()->activeWindowIsVisible(); }

  //!@}

} /* namespace DO */

//! @}

#endif /* DO_GRAPHICS_GRAPHICSUTILITIES_HPP */