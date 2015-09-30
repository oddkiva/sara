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

#ifndef DO_SARA_GRAPHICS_DRAW3D_HPP
#define DO_SARA_GRAPHICS_DRAW3D_HPP

//! @file

namespace DO { namespace Sara {

  /*!
    @ingroup Graphics
    @defgroup Draw3D Drawing 3D

    @{
   */

  //! @brief Display a mesh in the active OpenGLWindow window.
  DO_SARA_EXPORT
  void display_mesh(const SimpleTriangleMesh3f& mesh);

  //! @}

} /* namespace Sara */
} /* namespace DO */

#endif /* DO_SARA_GRAPHICS_DRAW3D_HPP */
