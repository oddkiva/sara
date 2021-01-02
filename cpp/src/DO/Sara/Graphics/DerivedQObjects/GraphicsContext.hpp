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

//! @file

#pragma once

#include <QObject>
#include <QPixmap>
#include <QPointer>
#include <QString>
#include <QWidget>

#include <DO/Sara/Core/Pixel/Typedefs.hpp>

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsView.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/OpenGLWindow.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/PaintingWindow.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/UserThread.hpp>


namespace DO { namespace Sara {

  class WidgetList
  {
  public:
    QList<QPointer<QWidget>> m_createdWindows;
    QPointer<QWidget> m_activeWindow;

    ~WidgetList()
    {
      for (auto w = m_createdWindows.begin(); w != m_createdWindows.end(); ++w)
      {
        if (!w->isNull())
        {
          auto paintingWindow = qobject_cast<PaintingWindow*>(*w);
          if (paintingWindow)
            delete paintingWindow->scrollArea();
          else
            delete *w;
        }
      }
    }
  };

  //! Private implementation of the class GraphicsApplication
  class DO_SARA_EXPORT GraphicsContext : public QObject
  {
    Q_OBJECT

    //! @brief quick-and-dirty thing to read file from dialog box.
    //! @todo See if it can be done in a more elegant way.
    struct DialogBoxInfo
    {
      QPixmap pixmap;
      QString filename;
    };

  public: /* enum */
    enum WindowType
    {
      PAINTING_WINDOW = 0,
      OPENGL_WINDOW = 1,
      GRAPHICS_VIEW = 2
    };

  public: /* methods */
    GraphicsContext();
    auto registerUserMain(int (*userMain)(int, char**)) -> void;
    auto registerUserMain(std::function<int(int, char**)>) -> void;
    auto userThread() -> UserThread&
    {
      return m_userThread;
    }

    auto setWidgetList(WidgetList*) -> void;
    auto activeWindow() -> QWidget*;

    auto makeCurrent() -> void;
    static auto current() -> GraphicsContext*;

  public slots:
    void createWindow(int windowType, int w, int h, const QString& windowTitle,
                      int x, int y);
    void setActiveWindow(QWidget* w);
    void closeWindow(QWidget* w);
    void getFileFromDialogBox();

  public: /* connection methods for the keyboard and mouse handling. */
    bool activeWindowIsVisible();
    void connectWindowIOEventsToUserThread(QWidget* w);
    void connectAllWindowsIOEventsToUserThread();
    void disconnectAllWindowsIOEventsToUserThread();

  private:
    static GraphicsContext* m_current;

    QMutex m_mutex{QMutex::NonRecursive};
    UserThread m_userThread;
    WidgetList* m_widgetList = nullptr;

    DialogBoxInfo m_dialogBoxInfo;
  };


  namespace v2 {

    inline auto create_window(int h, int w, int x, int y)
    {
      auto ctx = GraphicsContext::current();
      QMetaObject::invokeMethod(
          ctx, "createWindow", Qt::BlockingQueuedConnection,
          Q_ARG(int, GraphicsContext::PAINTING_WINDOW), Q_ARG(int, w),
          Q_ARG(int, h), Q_ARG(const QString&, QString("TEST")), Q_ARG(int, x),
          Q_ARG(int, y));
      return ctx->activeWindow();
    }

    inline auto close_window(QWidget* w)
    {
      auto ctx = GraphicsContext::current();
      QMetaObject::invokeMethod(
          ctx, "closeWindow", Qt::BlockingQueuedConnection, Q_ARG(QWidget*, w));
    }

    inline auto active_window()
    {
      auto ctx = GraphicsContext::current();
      return ctx->activeWindow();
    }

    inline auto draw_point(int x, int y, const Rgb8& color) -> void
    {
      auto ctx = GraphicsContext::current();
      QMetaObject::invokeMethod(
          ctx->activeWindow(), "drawPoint", Qt::QueuedConnection, Q_ARG(int, x),
          Q_ARG(int, y),
          Q_ARG(const QColor&, QColor(color[0], color[1], color[2])));
    }

    inline auto draw_line(const Eigen::Vector2f& p1, const Eigen::Vector2f& p2,
                          const Rgb8& color, int pen_width = 1)
    {
      auto ctx = GraphicsContext::current();
      QMetaObject::invokeMethod(
          ctx->activeWindow(), "drawLine", Qt::QueuedConnection,
          Q_ARG(const QPointF&, QPointF(p1.x(), p1.y())),
          Q_ARG(const QPointF&, QPointF(p2.x(), p2.y())),
          Q_ARG(const QColor&, QColor(color[0], color[1], color[2])),
          Q_ARG(int, pen_width));
    }

    inline auto set_antialiasing(bool on = true)
    {
      auto ctx = GraphicsContext::current();
      QMetaObject::invokeMethod(ctx->activeWindow(), "setAntialiasing",
                                Qt::QueuedConnection, Q_ARG(bool, on));
    }

    inline auto get_key()
    {
      GraphicsContext::current()->userThread().getKey();
    }

    inline auto get_mouse(int& x, int& y)
    {
      auto ctx = GraphicsContext::current();
      ctx->userThread().getMouse(x, y);
    }

  }  // namespace v2

}}  // namespace DO::Sara
