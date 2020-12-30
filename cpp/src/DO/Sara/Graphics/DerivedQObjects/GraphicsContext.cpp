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

#include <QApplication>
#include <QDebug>
#include <QFileDialog>

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsContext.hpp>


namespace DO::Sara {

  auto GraphicsContext::instance() -> GraphicsContext&
  {
    static GraphicsContext singleton{};
    return singleton;
  }

  auto GraphicsContext::registerUserMain(int (*user_main)(int, char**))
      -> void
  {
    m_userThread.registerUserMain(user_main);
  }

  auto GraphicsContext::registerUserMain(std::function<int(int, char **)> user_main)
      -> void
  {
    m_userThread.registerUserMain(user_main);
  }

  GraphicsContext::GraphicsContext()
    : m_mutex(QMutex::NonRecursive)
  {
    // Register painting data types.
    qRegisterMetaType<PaintingWindow*>("PaintingWindow *");
    qRegisterMetaType<QPolygonF>("QPolygonF");

    // Register mesh data structure.
    qRegisterMetaType<SimpleTriangleMesh3f>("SimpleTriangleMesh3f");

    // Register graphics view data types.
    qRegisterMetaType<QGraphicsPixmapItem*>("QGraphicsPixmapItem *");

    // Register Event data types
    qRegisterMetaType<Event>("Event");

    // Make sure you quit after the user thread is finished.
    QApplication* app = qobject_cast<QApplication*>(qApp);
    if (app == nullptr)
      qFatal("Invalid Application!");
    connect(&m_userThread, SIGNAL(finished()), app, SLOT(quit()));
    app->setQuitOnLastWindowClosed(false);
  }

  auto GraphicsContext::activeWindow() -> QWidget *
  {
    return m_widgetList->m_activeWindow;
  }

  void GraphicsContext::createWindow(int windowType, int w, int h,
                                     const QString& windowTitle, int x, int y)
  {
    auto& m_createdWindows = m_widgetList->m_createdWindows;
    auto& m_activeWindow = m_widgetList->m_activeWindow;

    if (windowType == PAINTING_WINDOW)
      m_createdWindows << new PaintingWindow(w, h, windowTitle, x, y);
    if (windowType == OPENGL_WINDOW)
      m_createdWindows << new OpenGLWindow(w, h, windowTitle, x, y);
    if (windowType == GRAPHICS_VIEW)
      m_createdWindows << new GraphicsView(w, h, windowTitle, x, y);

    if (m_createdWindows.size() == 1)
    {
      m_activeWindow = m_createdWindows.front();
      setActiveWindow(m_activeWindow);
    }
  }

  void GraphicsContext::setActiveWindow(QWidget* w)
  {
    if (w == nullptr)
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
    auto& m_activeWindow = m_widgetList->m_activeWindow;
    m_activeWindow = w;
    connectWindowIOEventsToUserThread(w);
  }

  void GraphicsContext::closeWindow(QWidget* w)
  {
    auto& m_createdWindows = m_widgetList->m_createdWindows;
    auto wi = std::find(m_createdWindows.begin(), m_createdWindows.end(), w);
    if (wi == m_createdWindows.end())
    {
      qFatal("Could not find window!");
      qApp->quit();
    }

    // Store closing result here.
    auto closed = false;
    // Close the painting window if it is one.
    auto paintingWindow = qobject_cast<PaintingWindow*>(*wi);
    if (paintingWindow)
      closed = paintingWindow->scrollArea()->close();
    else
      closed = (*wi)->close();
    // Check the closing is successful.
    if (!closed)
    {
      qFatal("Could not close window!");
      qApp->quit();
    }

    m_createdWindows.erase(wi);
  }

  void GraphicsContext::getFileFromDialogBox()
  {
    m_dialogBoxInfo.filename = QFileDialog::getOpenFileName(
        0, "Open File", "/home", "Images (*.png *.xpm *.jpg)");
  }

  bool GraphicsContext::activeWindowIsVisible()
  {
    auto& m_activeWindow = m_widgetList->m_activeWindow;
    m_mutex.lock();
    if (m_activeWindow.isNull())
    {
      m_mutex.unlock();
      qWarning() << "No active window!";
      return false;
    }
    if (m_activeWindow->isHidden())
    {
      m_mutex.unlock();
      qWarning() << "Active window is hidden!";
      return false;
    }
    m_mutex.unlock();
    return true;
  }

  void GraphicsContext::connectWindowIOEventsToUserThread(QWidget* w)
  {
    // User thread listens to mouse events.
    if (qobject_cast<PaintingWindow*>(w))
      connect(w, SIGNAL(releasedMouseButtons(int, int, Qt::MouseButtons)),
              &m_userThread,
              SLOT(pressedMouseButtons(int, int, Qt::MouseButtons)));
    // User thread listens to keyboard events.
    connect(w, SIGNAL(pressedKey(int)), &m_userThread, SLOT(pressedKey(int)));
    // User thread listens to a generic event.
    connect(w, SIGNAL(sendEvent(Event)), &m_userThread,
            SLOT(receivedEvent(Event)));
  }

  void GraphicsContext::connectAllWindowsIOEventsToUserThread()
  {
    auto& m_createdWindows = m_widgetList->m_createdWindows;
    for (auto w = m_createdWindows.begin(); w != m_createdWindows.end();)
    {
      if (w->isNull())
      {
        w = m_createdWindows.erase(w);
        continue;
      }
      connectWindowIOEventsToUserThread(*w);
      ++w;
    }
  }

  void GraphicsContext::disconnectAllWindowsIOEventsToUserThread()
  {
    auto& m_createdWindows = m_widgetList->m_createdWindows;
    for (auto w = m_createdWindows.begin(); w != m_createdWindows.end();)
    {
      if (w->isNull())
      {
        w = m_createdWindows.erase(w);
        continue;
      }
      disconnect(&m_userThread, 0, *w, 0);
      disconnect(*w, 0, &m_userThread, 0);
      ++w;
    }
  }

}  // namespace DO::Sara
