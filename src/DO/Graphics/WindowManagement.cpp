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
#include <QDebug>

namespace DO {

	// ====================================================================== //
	//! Windows handling function
	Window openWindow(int w, int h, const std::string& windowTitle,
					          int x, int y)
  {
		QMetaObject::invokeMethod(guiApp(), "createPaintingWindow",
								              Qt::BlockingQueuedConnection,
								              Q_ARG(int, w), Q_ARG(int, h),
								              Q_ARG(const QString&,
										            QString(windowTitle.c_str())),
								              Q_ARG(int, x), Q_ARG(int, y));
		return guiApp()->createdWindows.back();
	}

  Window openGLWindow(int w, int h, const std::string& windowTitle,
					            int x, int y)
  {
		QMetaObject::invokeMethod(guiApp(), "createOpenGLWindow",
								              Qt::BlockingQueuedConnection,
								              Q_ARG(int, w), Q_ARG(int, h),
								              Q_ARG(const QString&,
										            QString(windowTitle.c_str())),
								              Q_ARG(int, x), Q_ARG(int, y));
		return guiApp()->createdWindows.back();
	}

	void closeWindow(Window w)
  {
		QMetaObject::invokeMethod(guiApp(), "closeWindow",
								              Qt::BlockingQueuedConnection,
								              Q_ARG(QWidget *, w));
	}

	void setActiveWindow(Window w)
	{
		QMetaObject::invokeMethod(guiApp(), "setActiveWindow",
								              Qt::BlockingQueuedConnection,
								              Q_ARG(QWidget *, w));
	}
    
	// ====================================================================== //
	//! Temporizing functions
	void milliSleep(int msec)
	{
		userThread().milliSleep(msec);
	}

	void microSleep(int usec)
	{
		userThread().microSleep(usec);
	}

  // ====================================================================== //
  //! I/O control functions
  int getMouse(int& x, int& y)
  {
    if (!activeWindow())
      return -1;
    return userThread().getMouse(x, y);
  }

	int anyGetMouse(Point2i &p)
	{
		int c;
		guiApp()->connectAllWindowsIOEventsToUserThread();
		c = getMouse(p);
		guiApp()->disconnectAllWindowsIOEventsToUserThread();
		setActiveWindow(activeWindow());
		return c;
	}
       
	int getKey()
	{
		if (!activeWindow())
      return -1;
		return userThread().getKey();
	}
    
	int anyGetKey()
	{
		int k;
		guiApp()->connectAllWindowsIOEventsToUserThread();
		k = getKey();
		guiApp()->disconnectAllWindowsIOEventsToUserThread();
		if (activeWindow())
			setActiveWindow(activeWindow());
		return k;
	}
    
	// ====================================================================== //
	//! Window event management
	//! From active window only! I am lazy.
  void getEvent(int ms, Event& e)
  {
		QMetaObject::invokeMethod(activeWindow(), "waitForEvent", 
								              Qt::QueuedConnection,
								              Q_ARG(int, ms));
		userThread().listenToWindowEvents();
		userThread().getEvent(e);
	}

} /* namespace DO */