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

#include <DO/Sara/Graphics/DerivedQObjects/UserThread.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/PaintingWindow.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/OpenGLWindow.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/GraphicsView.hpp>


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
    enum WindowType {
      PAINTING_WINDOW = 0,
      OPENGL_WINDOW = 1,
      GRAPHICS_VIEW = 2
    };

  public:
    static auto instance() -> GraphicsContext&;
    auto registerUserMain(int (*userMain)(int, char**)) -> void;
    auto registerUserMain(std::function<int(int, char **)>) -> void;
    auto userThread() -> UserThread& { return m_userThread; }
    auto activeWindow() -> QWidget *;

  private: /* methods */
    GraphicsContext();
    virtual ~GraphicsContext();

  public slots:
    void createWindow(int windowType, int w, int h,
                      const QString& windowTitle, int x, int y);
    void setActiveWindow(QWidget *w);
    void closeWindow(QWidget *w);
    void getFileFromDialogBox();

  public: /* connection methods for the keyboard and mouse handling. */
    bool activeWindowIsVisible();
    void connectWindowIOEventsToUserThread(QWidget *w);
    void connectAllWindowsIOEventsToUserThread();
    void disconnectAllWindowsIOEventsToUserThread();

  public:
    UserThread m_userThread;
    WidgetList *m_widgetList = nullptr;

    DialogBoxInfo m_dialogBoxInfo;

    QMutex m_mutex;
  };

} /* namespace Sara */
} /* namespace DO */
