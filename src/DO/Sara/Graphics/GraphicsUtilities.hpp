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

#ifndef DO_SARA_GRAPHICS_GRAPHICSUTILITIES_HPP
#define DO_SARA_GRAPHICS_GRAPHICSUTILITIES_HPP

#include "DerivedQObjects/GraphicsApplicationImpl.hpp"


namespace DO { namespace Sara {

  /*!
    \addtogroup GraphicsInternal

    @{
   */

  //! @{
  //! Convenience functions.
  inline GraphicsApplication::Impl * gui_app()
  {
    return qobject_cast<GraphicsApplication::Impl *>(qApp);
  }

  inline UserThread& get_user_thread()
  {
    return gui_app()->userThread;
  }

  inline bool active_window_is_visible()
  {
    return gui_app()->activeWindowIsVisible();
  }
  //! @}

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GRAPHICS_GRAPHICSUTILITIES_HPP */
