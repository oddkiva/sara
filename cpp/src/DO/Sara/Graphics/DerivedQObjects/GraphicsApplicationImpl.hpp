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

#include <QApplication>
#include <QPixmap>
#include <QPointer>
#include <QString>
#include <QWidget>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/UserThread.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/PaintingWindow.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/OpenGLWindow.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/GraphicsView.hpp>


namespace DO { namespace Sara {

  //! @brief quick-and-dirty thing to read file from dialog box.
  //! @todo See if it can be done in a more elegant way.
  struct DialogBoxInfo
  {
    QPixmap pixmap;
    QString filename;
  };

  //! Private implementation of the class GraphicsApplication
  class DO_SARA_EXPORT GraphicsApplication::Impl : public QApplication
  {
    Q_OBJECT

  public: /* enum */
    enum WindowType {
      PAINTING_WINDOW = 0,
      OPENGL_WINDOW = 1,
      GRAPHICS_VIEW = 2
    };

  public: /* methods */
    Impl(int& argc, char **argv);

    virtual ~Impl();

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
    QList<QPointer<QWidget> > m_createdWindows;
    QWidget* m_activeWindow = nullptr;

    DialogBoxInfo m_dialogBoxInfo;

    QMutex m_mutex;
  };

} /* namespace Sara */
} /* namespace DO */
