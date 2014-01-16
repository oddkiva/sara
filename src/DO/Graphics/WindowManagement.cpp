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

#include <DO/Graphics.hpp>
#include "GraphicsUtilities.hpp"

namespace DO {

  QWidget * getActiveWindow()
  { 
    return getGuiApp()->activeWindow;
  }

  Vector2i getWindowSizes(Window w)
  {
    return Point2i(w->width(), w->height());
  }

  // ====================================================================== //
  //! Windows handling function
  Window openWindow(int w, int h, const std::string& windowTitle,
                    int x, int y)
  {
    QMetaObject::invokeMethod(getGuiApp(), "createPaintingWindow",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(int, w), Q_ARG(int, h),
                              Q_ARG(const QString&,
                                QString(windowTitle.c_str())),
                              Q_ARG(int, x), Q_ARG(int, y));
    return getGuiApp()->createdWindows.back();
  }

  Window openGLWindow(int w, int h, const std::string& windowTitle,
                      int x, int y)
  {
    QMetaObject::invokeMethod(getGuiApp(), "createOpenGLWindow",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(int, w), Q_ARG(int, h),
                              Q_ARG(const QString&,
                                QString(windowTitle.c_str())),
                              Q_ARG(int, x), Q_ARG(int, y));
    return getGuiApp()->createdWindows.back();
  }

  void closeWindow(Window w)
  {
    QMetaObject::invokeMethod(getGuiApp(), "closeWindow",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(QWidget *, w));
  }

  void setActiveWindow(Window w)
  {
    QMetaObject::invokeMethod(getGuiApp(), "setActiveWindow",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(QWidget *, w));
  }

  void resizeWindow(Window w, int width, int height)
  {
    QMetaObject::invokeMethod(w, "resizeScreen", 
                              Qt::BlockingQueuedConnection,
                              Q_ARG(int, width), Q_ARG(int, height));
  }

  // ====================================================================== //
  //! Temporizing functions
  void milliSleep(int msec)
  {
    getUserThread().milliSleep(msec);
  }

  void microSleep(int usec)
  {
    getUserThread().microSleep(usec);
  }

  // ====================================================================== //
  //! I/O control functions
  int getMouse(int& x, int& y)
  {
    if (!getActiveWindow())
      return -1;
    return getUserThread().getMouse(x, y);
  }

  int anyGetMouse(Point2i &p)
  {
    int c;
    getGuiApp()->connectAllWindowsIOEventsToUserThread();
    c = getMouse(p);
    getGuiApp()->disconnectAllWindowsIOEventsToUserThread();
    setActiveWindow(getActiveWindow());
    return c;
  }

  int getKey()
  {
    if (!getActiveWindow())
      return -1;
    return getUserThread().getKey();
  }

  int anyGetKey()
  {
    int k;
    getGuiApp()->connectAllWindowsIOEventsToUserThread();
    k = getKey();
    getGuiApp()->disconnectAllWindowsIOEventsToUserThread();
    if (getActiveWindow())
      setActiveWindow(getActiveWindow());
    return k;
  }

  // ====================================================================== //
  //! Window event management
  //! From active window only! I am lazy.
  void getEvent(int ms, Event& e)
  {
    QMetaObject::invokeMethod(getActiveWindow(), "waitForEvent", 
                              Qt::QueuedConnection,
                              Q_ARG(int, ms));
    getUserThread().listenToWindowEvents();
    getUserThread().getEvent(e);
  }

} /* namespace DO */