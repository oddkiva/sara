// ========================================================================= //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================= //

//! @file

#ifndef DO_GRAPHICS_WINDOWCONTROL_HPP
#define DO_GRAPHICS_WINDOWCONTROL_HPP

#include "Events.hpp"

class QWidget;

namespace DO {

  typedef QWidget * Window;

  /*!
    \ingroup Graphics

    \defgroup WindowManagement Window management functions
    \brief Open, close windows, wait for a click or a key, listen to events.
    @{
   */

  //! Window getter
  DO_EXPORT
  Window active_window();
  //! Window size getter.
  DO_EXPORT
  Vector2i get_sizes(Window w);
  //! Window width getter.
  DO_EXPORT
  inline int get_width(Window w) { return get_sizes(w)(0); }
  //! Window height getter.
  DO_EXPORT
  inline int get_height(Window w) { return get_sizes(w)(1); }

  // ======================================================================= //
  // Windows handling function
  //! Open a PaintingWindow for 2D drawing.
  DO_EXPORT
  Window create_window(int w, int h, const std::string& windowTitle = "DO++",
                       int x = -1, int y = -1);
  inline Window create_window(const Vector2i& sizes,
                              const std::string& windowTitle = "DO++",
                              int x = -1, int y = -1)
  { return create_window(sizes(0), sizes(1), windowTitle, x, y); }
  //! Open a OpenGLWindow for 3D drawing.
  DO_EXPORT
  Window create_gl_window(int w, int h, const std::string& windowTitle = "DO++",
                      int x = -1, int y = -1);
  inline Window create_gl_window(const Vector2i& sizes,
                             const std::string& windowTitle = "DO++",
                             int x = -1, int y = -1)
  { return create_gl_window(sizes(0), sizes(1), windowTitle, x, y); }
  //! Open a GraphicsView for interactive viewing.
  DO_EXPORT
  Window create_graphics_view(int w, int h,
                              const std::string& windowTitle = "DO++",
                              int x = -1, int y = -1);
  DO_EXPORT
  inline Window create_graphics_view(const Vector2i& sizes,
                                 const std::string& windowTitle = "DO++",
                                 int x = -1, int y = -1)
  { return create_graphics_view(sizes(0), sizes(1), windowTitle, x, y); }
  //! \brief Close the window **w** (regardless of its type). 
  //! By default, the active window is closed.
  DO_EXPORT
  void close_window(Window w = active_window());
  //! Set the chosen window **w** as the current active window (regardless of
  //! its type).
  DO_EXPORT
  void set_active_window(Window w);
  //! Resize the specified window **w** with the following parameters.
  DO_EXPORT
  void resize_window(int width, int height, Window w = active_window());
  //! Resize the specified window **w** with the following parameters.
  inline void resize_window(const Vector2i& sizes, Window w = active_window())
  { resize_window(sizes(0), sizes(1), w); }

  // ======================================================================= //
  // Temporizing functions
  //! Wait **msec** milliseconds before the window resumes its drawing.
  DO_EXPORT
  void millisleep(int msec);
  //! Wait **usec** microseconds before the window resumes its drawing.
  DO_EXPORT
  void microsleep(int usec);

  // ======================================================================= //
  // I/O control functions
  //! \brief Wait for a click from the user. 
  //! - Works only on the *active* window.
  //! - Returns the clicked mouse button
  //! - store the click coordinates \f$(x,y)\f$.
  DO_EXPORT
  int get_mouse(int& x, int& y);
  //! \brief Wait for a click from the user (only on the *active* window)
  //! - Returns the clicked mouse button and 
  //! - Stores the click coordinates \f$p\f$.
  inline int get_mouse(Point2i& p)
  { return get_mouse(p.x(), p.y()); }
  //! \brief Wait for a click from the user (only on the *active* window)
  //! - The user can click on any opened windows.
  //! - Returns the clicked mouse button
  //! - store the click coordinates \f$p\f$.
  DO_EXPORT
  int any_get_mouse(Point2i& p);
  //! Wait for a click from the user only on the *active* window.
  inline void click()
  { Point2i p; get_mouse(p); }
  //! Wait for a click from the user on any opened windows.
  inline void any_click()
  { Point2i p; any_get_mouse(p); }
  //! \brief Wait for a hit key from the user.
  //! - Works only on the *active* window.
  //! - Returns the hit key.
  DO_EXPORT
  int get_key();
  //! \brief Wait for a hit key from the user.
  //! - Works on any opened windows.
  //! - Returns the hit key.
  DO_EXPORT
  int any_get_key();

  // ======================================================================= //
  // Window event management
  //! Listens to events sent from the active window.
  DO_EXPORT
  void get_event(int ms, Event& e);

  //! @}

} /* namespace DO */

#endif /* DO_GRAPHICS_WINDOWCONTROL_HPP */
