// ========================================================================= //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================= //

//! @file

#pragma once

#include <DO/Sara/Graphics/Events.hpp>


class QWidget;


namespace DO { namespace Sara {

  using Window = QWidget *;

  /*!
    @ingroup Graphics

    @defgroup WindowManagement Window management functions
    @brief Open, close windows, wait for a click or a key, listen to events.
    @{
   */

  //! @brief Return the active window.
  DO_SARA_EXPORT
  Window active_window();

  //! @brief Return the window sizes.
  DO_SARA_EXPORT
  Vector2i get_sizes(Window w);

  //! @brief Return the window width.
  DO_SARA_EXPORT
  inline int get_width(Window w)
  {
    return get_sizes(w)(0);
  }

  //! @brief Return the window height.
  DO_SARA_EXPORT
  inline int get_height(Window w)
  {
    return get_sizes(w)(1);
  }

  //! @{
  //! @brief Open a PaintingWindow for 2D drawing.
  DO_SARA_EXPORT
  Window create_window(int w, int h, const std::string& windowTitle = "Sara",
                       int x = -1, int y = -1);

  inline Window create_window(const Vector2i& sizes,
                              const std::string& windowTitle = "Sara",
                              int x = -1, int y = -1)
  {
    return create_window(sizes(0), sizes(1), windowTitle, x, y);
  }
  //! @}

  //! @{
  //! @brief Open a OpenGLWindow for 3D drawing.
  DO_SARA_EXPORT
  Window create_gl_window(int w, int h, const std::string& windowTitle = "Sara",
                      int x = -1, int y = -1);

  inline Window create_gl_window(const Vector2i& sizes,
                             const std::string& windowTitle = "Sara",
                             int x = -1, int y = -1)
  {
    return create_gl_window(sizes(0), sizes(1), windowTitle, x, y);
  }
  //! @}

  //! @{
  //! Open a GraphicsView for interactive viewing.
  DO_SARA_EXPORT
  Window create_graphics_view(int w, int h,
                              const std::string& windowTitle = "Sara",
                              int x = -1, int y = -1);

  DO_SARA_EXPORT
  inline Window create_graphics_view(const Vector2i& sizes,
                                 const std::string& windowTitle = "Sara",
                                 int x = -1, int y = -1)
  {
    return create_graphics_view(sizes(0), sizes(1), windowTitle, x, y);
  }
  //! @}

  //! @brief Close the window **w** (regardless of its type).
  //! By default, the active window is closed.
  DO_SARA_EXPORT
  void close_window(Window w = active_window());

  //! Set the chosen window **w** as the current active window (regardless of
  //! its type).
  DO_SARA_EXPORT
  void set_active_window(Window w);

  //! @{
  //! Resize the specified window **w** with the following parameters.
  DO_SARA_EXPORT
  void resize_window(int width, int height, Window w = active_window());

  inline void resize_window(const Vector2i& sizes, Window w = active_window())
  {
    resize_window(sizes(0), sizes(1), w);
  }
  //! @}

  //! Wait **msec** milliseconds before the window resumes its drawing.
  DO_SARA_EXPORT
  void millisleep(int msec);

  //! Wait **usec** microseconds before the window resumes its drawing.
  DO_SARA_EXPORT
  void microsleep(int usec);

  //! @brief Wait for a click from the user.
  //! - Works only on the *active* window.
  //! - Returns the clicked mouse button
  //! - store the click coordinates \f$(x,y)\f$.
  DO_SARA_EXPORT
  int get_mouse(int& x, int& y);

  //! @brief Wait for a click from the user (only on the *active* window)
  //! - Returns the clicked mouse button and
  //! - Stores the click coordinates \f$p\f$.
  inline int get_mouse(Point2i& p)
  {
    return get_mouse(p.x(), p.y());
  }

  //! @brief Wait for a click from the user (only on the *active* window)
  //! - The user can click on any opened windows.
  //! - Returns the clicked mouse button
  //! - store the click coordinates \f$p\f$.
  DO_SARA_EXPORT
  int any_get_mouse(Point2i& p);

  //! Wait for a click from the user only on the *active* window.
  inline void click()
  {
    Point2i p; get_mouse(p);
  }

  //! Wait for a click from the user on any opened windows.
  inline void any_click()
  {
    Point2i p; any_get_mouse(p);
  }

  //! @brief Wait for a hit key from the user.
  //! - Works only on the *active* window.
  //! - Returns the hit key.
  DO_SARA_EXPORT
  int get_key();

  //! @brief Wait for a hit key from the user.
  //! - Works on any opened windows.
  //! - Returns the hit key.
  DO_SARA_EXPORT
  int any_get_key();

  //! @Brief Listen to events sent from the active window.
  DO_SARA_EXPORT
  void get_event(int ms, Event& e);

  //! @}

} /* namespace Sara */
} /* namespace DO */
