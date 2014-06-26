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

#include "GraphicsApplicationImpl.hpp"
#include <QFileDialog>
#include <QDebug>

namespace DO {

  GraphicsApplication::Impl::
  Impl(int argc_, char **argv_)
    : QApplication(argc_, argv_)
    , argc(argc_)
    , argv(argv_)
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
    qRegisterMetaType<QGraphicsPixmapItem *>("QGraphicsPixmapItem *");
    //qRegisterMetaType<QGraphicsItem *>("QGraphicsItem *");

    // Register Event data types
    qRegisterMetaType<Event>("Event");

    // Make sure you quit after the user thread is finished.
    connect(&userThread, SIGNAL(finished()), this, SLOT(quit()));
    setQuitOnLastWindowClosed(false);
  }

  GraphicsApplication::Impl::
  ~Impl()
  {
    QList<QPointer<QWidget> >::iterator w = createdWindows.begin();
    for ( ; w != createdWindows.end(); ++w)
    {
      if (!w->isNull())
      {
        PaintingWindow *paintingWindow = qobject_cast<PaintingWindow *>(*w);
        if (paintingWindow)
          delete paintingWindow->scrollArea();
        else
          delete *w;
      }
    }
  }

  void
  GraphicsApplication::Impl::
  createWindow(int windowType, int w, int h,
               const QString& windowTitle, int x, int y)
  {
    if (windowType == PAINTING_WINDOW)
      createdWindows << new PaintingWindow(w, h, windowTitle, x, y);
    if (windowType == OPENGL_WINDOW)
      createdWindows << new OpenGLWindow(w, h, windowTitle, x, y);
    if (windowType == GRAPHICS_VIEW)
      createdWindows << new GraphicsView(w, h, windowTitle, x, y);

    if (createdWindows.size() == 1)
    {
      activeWindow = createdWindows.front();
      setActiveWindow(activeWindow);
    }
  }

   void 
   GraphicsApplication::Impl::
   setActiveWindow(QWidget *w)
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

   void
   GraphicsApplication::Impl::
   closeWindow(QWidget *w) 
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

     createdWindows.erase(wi);
   }

   void
   GraphicsApplication::Impl::
   getFileFromDialogBox()
   {
     dialogBoxInfo.filename = 
       QFileDialog::getOpenFileName(0, "Open File", "/home", 
       "Images (*.png *.xpm *.jpg)");
   }

  bool
  GraphicsApplication::Impl::
  activeWindowIsVisible()
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

  void
  GraphicsApplication::Impl::
  connectWindowIOEventsToUserThread(QWidget *w)
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

  void 
  GraphicsApplication::Impl::
  connectAllWindowsIOEventsToUserThread()
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

  void
  GraphicsApplication::Impl::
  disconnectAllWindowsIOEventsToUserThread()
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
