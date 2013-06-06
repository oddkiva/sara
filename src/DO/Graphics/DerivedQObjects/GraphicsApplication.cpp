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

#include "GraphicsApplication.hpp"
#include "PaintingWindow.hpp"
#include "OpenGLWindow.hpp"
#include "GraphicsView.hpp"
#include <QtWidgets>

namespace DO {

  GraphicsApplication::GraphicsApplication(int argc, char **argv) 
    : QApplication(argc, argv)
    , activeWindow(0)
    //, graphicsView(0)
    , mutex(QMutex::NonRecursive)
  {
    // Register painting data types.
    qRegisterMetaType<PaintingWindow *>("PaintingWindow *");
    qRegisterMetaType<QPolygonF>("QPolygonF");
    
    // Register mesh data structure.
    qRegisterMetaType<SimpleTriangleMesh3f>("SimpleTriangleMesh3f");
    
    // Register graphics view data types.
    //qRegisterMetaType<QGraphicsPixmapItem *>("QGraphicsPixmapItem *");
    //qRegisterMetaType<QGraphicsItem *>("QGraphicsItem *");
    
    // Register Event data types
    qRegisterMetaType<Event>("Event");

    // Make sure you quit after the user thread is finished.
    connect(&userThread, SIGNAL(finished()), this, SLOT(quit()));
    setQuitOnLastWindowClosed(false);
  }

  void GraphicsApplication::createPaintingWindow(int w, int h,
                                                 const QString& windowTitle,
                                                 int x, int y)
  {
    createdWindows << new PaintingWindow(w, h, windowTitle, x, y);
    if (createdWindows.size() == 1)
    {
      activeWindow = createdWindows.front();
      setActiveWindow(activeWindow);
    }
  }

  void GraphicsApplication::createOpenGLWindow(int w, int h,
                                               const QString& windowTitle,
                                               int x, int y)
  {
    createdWindows << new OpenGLWindow(w, h, windowTitle, x, y);
    if (createdWindows.size() == 1)
    {
      activeWindow = createdWindows.front();
      setActiveWindow(activeWindow);
    }
  }

  void GraphicsApplication::createGraphicsView(int w, int h,
                                               const QString& windowTitle,
                                               int x, int y)
  {
    createdWindows << new GraphicsView(w, h, windowTitle, x, y);
    if (createdWindows.size() == 1)
    {
      activeWindow = createdWindows.front();
      setActiveWindow(activeWindow);
    }
  }

  void GraphicsApplication::setActiveWindow(QWidget *w)
  {
    if (w == 0)
    {
      qWarning() << "I can't make a null window active!";
      return;
    }
    
    // Disconnect the communication between:
    // All windows and user thread in both directions.
    // Why? I don't want the user thread to listens to communicate from any
    // window but the requested window 'w'.
    disconnectAllWindowsIOEventsToUserThread();
    
    // This is now our current active window
    activeWindow = w;
    connectWindowIOEventsToUserThread(w);
  }

  void GraphicsApplication::closeWindow(QWidget *w)
  {
    QList<QPointer<QWidget> >::iterator wi = qFind(
      createdWindows.begin(), createdWindows.end(), w);
    if (wi == createdWindows.end())
    {
      qFatal("Could not find window!");
      quit();
    }

    // Store closing result here.
    bool closed = false;
    // Close the painting window if it is one.
    PaintingWindow *paintingWindow = qobject_cast<PaintingWindow *>(*wi);
    if (paintingWindow)
      closed = paintingWindow->scrollArea()->close();
    else
      closed = (*wi)->close();
    // Check the closing is successful.
    if (!closed)
    {
      qFatal("Could not close window!");
      quit();
    }

    if (!(*wi).isNull())
      delete *wi;
    createdWindows.erase(wi);
  }
  
  void GraphicsApplication::getFileFromDialogBox()
  {
    interactiveBox.filename = 
    QFileDialog::getOpenFileName(0, "Open File", "/home", 
                                 "Images (*.png *.xpm *.jpg)");
  }
  
  bool GraphicsApplication::activeWindowIsVisible()
  {
    mutex.lock();
    if (activeWindow.isNull())
    {
      mutex.unlock();
      qWarning() << "No active window!";
      return false;
    }
    if (activeWindow->isHidden())
    {
      mutex.unlock();
      qWarning() << "Active window is hidden!";
      return false;
    }
    mutex.unlock();
    return true;
  }

  void GraphicsApplication::connectWindowIOEventsToUserThread(QWidget *w)
  {
    // User thread listens to mouse events.
    connect(w, SIGNAL(releasedMouseButtons(int, int, Qt::MouseButtons)),
            &userThread, SLOT(pressedMouseButtons(int, int, Qt::MouseButtons)));
    // User thread listens to keyboard events.
    connect(w, SIGNAL(pressedKey(int)),
            &userThread, SLOT(pressedKey(int)));
    // User thread listens to a generic event.
    connect(w, SIGNAL(sendEvent(Event)),
            &userThread, SLOT(receivedEvent(Event)));
  }
  
  void GraphicsApplication::connectAllWindowsIOEventsToUserThread()
  {
    QList<QPointer<QWidget> >::iterator w = createdWindows.begin();
    for ( ; w != createdWindows.end(); )
    {
      if (w->isNull())
      {
        w = createdWindows.erase(w);
        continue;
      }
      connectWindowIOEventsToUserThread(*w);
      ++w;
    }
  }

  void GraphicsApplication::disconnectAllWindowsIOEventsToUserThread()
  {
    QList<QPointer<QWidget> >::iterator w = createdWindows.begin();
    for ( ; w != createdWindows.end(); )
    {
      if (w->isNull())
      {
        w = createdWindows.erase(w);
        continue;
      }
      disconnect(&userThread, 0, *w, 0);
      disconnect(*w, 0, &userThread, 0);
      ++w;
    }
  }
    
} /* namespace DO */