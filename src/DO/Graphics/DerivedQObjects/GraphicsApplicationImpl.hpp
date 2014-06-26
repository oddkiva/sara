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

#ifndef DO_GRAPHICS_GRAPHICSAPPLICATIONIMPL_HPP
#define DO_GRAPHICS_GRAPHICSAPPLICATIONIMPL_HPP

#include <DO/Graphics.hpp>
#include <QApplication>
#include <QPointer>
#include <QPixmap>
#include <QString>
#include <QWidget>
#include "UserThread.hpp"
#include "PaintingWindow.hpp"
#include "OpenGLWindow.hpp"
#include "GraphicsView.hpp"

namespace DO {

  //! \brief quick-and-dirty thing to read file from dialog box.
  //! \todo See if it can be done in a more elegant way.
  struct DialogBoxInfo
  {
    QPixmap pixmap;
    QString filename;
  };

  //! Private implementation of the class GraphicsApplication
  class GraphicsApplication::Impl : public QApplication
  {
    Q_OBJECT
  public:
    Impl(int argc, char **argv);
    virtual ~Impl();

  public slots:
    void createPaintingWindow(int w, int h, const QString& windowTitle,
                              int x, int y);
    void createOpenGLWindow(int w, int h, const QString& windowTitle,
                            int x, int y);
    void createGraphicsView(int w, int h, const QString& windowTitle,
                            int x, int y);
    void setActiveWindow(QWidget *w);
    void closeWindow(QWidget *w);
    void getFileFromDialogBox();

  public: /* connection methods for the keyboard and mouse handling. */
    bool activeWindowIsVisible();
    void connectWindowIOEventsToUserThread(QWidget *w);
    void connectAllWindowsIOEventsToUserThread();
    void disconnectAllWindowsIOEventsToUserThread();

  public:
    int argc;
    char **argv;
    UserThread userThread;
    QList<QPointer<QWidget> > createdWindows;
    QPointer<QWidget> activeWindow;

    //GraphicsView *graphicsView;
    DialogBoxInfo dialogBoxInfo;

    QMutex mutex;
  };
    
} /* namespace DO */

#endif /* DO_GRAPHICS_GRAPHICSAPPLICATIONIMPL_HPP */