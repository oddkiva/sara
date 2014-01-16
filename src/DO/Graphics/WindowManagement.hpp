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

#ifndef DO_GRAPHICS_WINDOWCONTROL_HPP
#define DO_GRAPHICS_WINDOWCONTROL_HPP

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
  inline Window getActiveWindow();
  //! Window size getter.
  DO_EXPORT
  Vector2i getWindowSizes(Window w);
  //! Window width getter.
  DO_EXPORT
  inline int getWindowWidth(Window w) { return getWindowSizes(w)(0); }
  //! Window height getter.
  DO_EXPORT
  inline int getWindowHeight(Window w) { return getWindowSizes(w)(1); }

  // ====================================================================== //
  // Windows handling function
  //! Open a PaintingWindow for 2D drawing.
  DO_EXPORT
  Window openWindow(int w, int h, const std::string& windowTitle = "DO++",
                    int x = -1, int y = -1);
  inline Window openWindow(const Vector2i& sizes,
                           const std::string& windowTitle = "DO++",
                           int x = -1, int y = -1)
  { return openWindow(sizes(0), sizes(1), windowTitle, x, y); }
  //! Open a OpenGLWindow for 3D drawing.
  DO_EXPORT
  Window openGLWindow(int w, int h, const std::string& windowTitle = "DO++",
                      int x = -1, int y = -1);
  inline Window openGLWindow(const Vector2i& sizes,
                             const std::string& windowTitle = "DO++",
                             int x = -1, int y = -1)
  { return openGLWindow(sizes(0), sizes(1), windowTitle, x, y); }
  //! Open a GraphicsView for interactive viewing.
  DO_EXPORT
  Window openGraphicsView(int w, int h, const std::string& windowTitle = "DO++",
                          int x = -1, int y = -1);
  DO_EXPORT
  inline Window openGraphicsView(const Vector2i& sizes,
                                 const std::string& windowTitle = "DO++",
                                 int x = -1, int y = -1)
  { return openGraphicsView(sizes(0), sizes(1), windowTitle, x, y); }
  //! \brief Close the window **w** (regardless of its type). 
  //! By default, the active window is closed.
  DO_EXPORT
  void closeWindow(Window w = getActiveWindow());
  //! Set the chosen window **w** as the current active window (regardless of its type).
  DO_EXPORT
  void setActiveWindow(Window w);
  //! Resize the specified window **w** with the following parameters.
  DO_EXPORT
  void resizeWindow(Window w, int width, int height);

  // ====================================================================== //
  // Temporizing functions
  //! Wait **msec** milliseconds before the window resumes its drawing.
  DO_EXPORT
  void milliSleep(int msec);
  //! Wait **usec** microseconds before the window resumes its drawing.
  DO_EXPORT
  void microSleep(int usec);

  // ====================================================================== //
  // I/O control functions
  //! \brief Wait for a click from the user. 
  //! - Works only on the *active* window.
  //! - Returns the clicked mouse button
  //! - store the click coordinates \f$(x,y)\f$.
  DO_EXPORT
  int getMouse(int& x, int& y);
  //! \brief Wait for a click from the user (only on the *active* window)
  //! - Returns the clicked mouse button and 
  //! - Stores the click coordinates \f$p\f$.
  inline int getMouse(Point2i& p)
  { return getMouse(p.x(), p.y()); }
  //! \brief Wait for a click from the user (only on the *active* window)
  //! - The user can click on any opened windows.
  //! - Returns the clicked mouse button
  //! - store the click coordinates \f$p\f$.
  DO_EXPORT
  int anyGetMouse(Point2i& p);
  //! Wait for a click from the user only on the *active* window.
  inline void click()
  { Point2i p; getMouse(p); }
  //! Wait for a click from the user on any opened windows.
  inline void anyClick()
  { Point2i p; anyGetMouse(p); }
  //! \brief Wait for a hit key from the user.
  //! - Works only on the *active* window.
  //! - Returns the hit key.
  DO_EXPORT
  int getKey();
  //! \brief Wait for a hit key from the user.
  //! - Works on any opened windows.
  //! - Returns the hit key.
  DO_EXPORT
  int anyGetKey();

  // ====================================================================== //
  // Window event management
  //! Listens to events sent from the active window.
  DO_EXPORT
  void getEvent(int ms, Event& e);

  //! @}

} /* namespace DO */

#endif /* DO_GRAPHICS_WINDOWCONTROL_HPP */