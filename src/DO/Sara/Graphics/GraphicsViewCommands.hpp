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

#ifndef DO_SARA_GRAPHICS_GRAPHICSVIEWCOMMANDS_HPP
#define DO_SARA_GRAPHICS_GRAPHICSVIEWCOMMANDS_HPP

class QGraphicsItem;
class QGraphicsPixmapItem;

namespace DO {

  typedef QGraphicsItem * Item;
  typedef QGraphicsPixmapItem * ImageItem;

  /*!
    \ingroup Graphics
    \defgroup GraphicsView Graphics View
    \brief This submodule is based Qt Graphics View technology allowing for
    interactivity with added elements in a DO::GraphicsView window.
    @{
   */

  //! \brief Add image \b I to the active GraphicsView window.
  //! The added window can be:
  //! - rescaled by hitting key \b + or key \b -,
  //! - selected and moved in the GraphicsView window using the mouse.
  DO_EXPORT
  ImageItem add_image(const Image<Rgb8>& I, bool randomPos = false);

  //! \todo Not yet implemented I think. Check that.
  DO_EXPORT
  void draw_point(ImageItem pixItem, int x, int y, const Rgb8& c);

  //! @}

} /* namespace DO */

#endif /* DO_SARA_GRAPHICS_GRAPHICSVIEWCOMMANDS_HPP */