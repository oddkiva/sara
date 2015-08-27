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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>


namespace DO { namespace Sara {

  QWidget * active_window()
  {
    return gui_app()->m_activeWindow;
  }

  Vector2i get_sizes(Window w)
  {
    return Point2i(w->width(), w->height());
  }

  Window create_window(int w, int h, const std::string& windowTitle,
                    int x, int y)
  {
    QMetaObject::invokeMethod(gui_app(), "createWindow",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(int, 0),
                              Q_ARG(int, w), Q_ARG(int, h),
                              Q_ARG(const QString&,
                                    QString(windowTitle.c_str())),
                              Q_ARG(int, x), Q_ARG(int, y));
    return gui_app()->m_createdWindows.back();
  }

  Window create_gl_window(int w, int h, const std::string& windowTitle,
                      int x, int y)
  {
    QMetaObject::invokeMethod(gui_app(), "createWindow",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(int, 1),
                              Q_ARG(int, w), Q_ARG(int, h),
                              Q_ARG(const QString&,
                                    QString(windowTitle.c_str())),
                              Q_ARG(int, x), Q_ARG(int, y));
    return gui_app()->m_createdWindows.back();
  }

  void close_window(Window w)
  {
    QMetaObject::invokeMethod(gui_app(), "closeWindow",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(QWidget *, w));
  }

  void set_active_window(Window w)
  {
    QMetaObject::invokeMethod(gui_app(), "setActiveWindow",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(QWidget *, w));
  }

  void resize_window(int width, int height, Window w)
  {
    QMetaObject::invokeMethod(w, "resizeScreen",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(int, width), Q_ARG(int, height));
  }

  void millisleep(int msec)
  {
    get_user_thread().milliSleep(msec);
  }

  void microsleep(int usec)
  {
    get_user_thread().microSleep(usec);
  }

  int get_mouse(int& x, int& y)
  {
    if (!active_window())
      return -1;
    return get_user_thread().getMouse(x, y);
  }

  int any_get_mouse(Point2i &p)
  {
    int c;
    gui_app()->connectAllWindowsIOEventsToUserThread();
    c = get_mouse(p);
    gui_app()->disconnectAllWindowsIOEventsToUserThread();
    set_active_window(active_window());
    return c;
  }

  int get_key()
  {
    if (!active_window())
      return -1;
    return get_user_thread().getKey();
  }

  int any_get_key()
  {
    int k;
    gui_app()->connectAllWindowsIOEventsToUserThread();
    k = get_key();
    gui_app()->disconnectAllWindowsIOEventsToUserThread();
    if (active_window())
      set_active_window(active_window());
    return k;
  }

  void get_event(int ms, Event& e)
  {
    QMetaObject::invokeMethod(active_window(), "waitForEvent",
                              Qt::QueuedConnection,
                              Q_ARG(int, ms));
    get_user_thread().listenToWindowEvents();
    get_user_thread().getEvent(e);
  }

} /* namespace Sara */
} /* namespace DO */
