// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
//! @brief Master header file of the Graphics module.

#pragma once

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Core/Image.hpp>

#include <DO/Sara/Graphics/Events.hpp>
#include <DO/Sara/Graphics/GraphicsApplication.hpp>
#include <DO/Sara/Graphics/WindowManagement.hpp>
#include <DO/Sara/Graphics/PaintingCommands.hpp>
#include <DO/Sara/Graphics/ImageIO.hpp>
#include <DO/Sara/Graphics/ImageDraw.hpp>

// The following are still experimental but can be extended very easily.

// 2D Geometry drawing utilities.
#include <DO/Sara/Graphics/Features/Draw.hpp>
#include <DO/Sara/Graphics/Geometry/DrawPolygon.hpp>
#include <DO/Sara/Graphics/Match/Draw.hpp>
#include <DO/Sara/Graphics/Match/PairWiseDrawer.hpp>

// 3D viewer (still experimental and little features.)
// \todo: see if we need to separate the Mesh from the Graphics module.
#include <DO/Sara/Graphics/Draw3D.hpp>

// Graphics view for interactive viewing (very little features for now.)
#include <DO/Sara/Graphics/GraphicsViewCommands.hpp>

/*!
  @defgroup Graphics Graphics
  @brief The Graphics module relies on the Qt 5 framework.
  It covers basic graphical features such as:
  - opening, closing a window,
  - draw 2D things,
  - display a 3D mesh,
  - interactive viewing of images.
 */
