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
//! \brief Master header file of the Graphics module.

#ifndef DO_GRAPHICS_HPP
#define DO_GRAPHICS_HPP

#include <DO/Defines.hpp>
#include <DO/Core/Image.hpp>

#include "Graphics/Events.hpp"
#include "Graphics/GraphicsApplication.hpp"
#include "Graphics/WindowManagement.hpp"
#include "Graphics/PaintingCommands.hpp"
#include "Graphics/ImageIO.hpp"
#include "Graphics/ImageDraw.hpp"

// The following are still experimental but can be extended very easily.

// 3D viewer (still experimental and little features.)
// \todo: see if we need to separate the Mesh from the Graphics module.
#include "Graphics/Mesh.hpp"
#include "Graphics/Draw3D.hpp"

// Graphics view for interactive viewing (very little features for now.)
#include "Graphics/GraphicsViewCommands.hpp"

/*!
  \defgroup Graphics Graphics
  \brief The Graphics module relies on the Qt 5 framework. 
  It covers basic graphical features such as:
  - opening, closing a window,
  - draw 2D things,
  - display a 3D mesh,
  - interactive viewing of images.
 */


#endif /* DO_GRAPHICS_HPP */