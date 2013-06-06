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

//! WTF: QString::fromStdString(s) makes DO.Graphics crash in RelWithDebInfo 
//! with Qt 5.0.1!!!!
//! I am not the only one who experienced this problem:
//! https://www.assembla.com/spaces/plus/messages/1093143
//! https://www.assembla.com/spaces/plus/tickets/162#/activity/ticket:


#ifndef DO_GRAPHICS_GRAPHICSAPPLICATION_HPP
#define DO_GRAPHICS_GRAPHICSAPPLICATION_HPP

#ifdef WIN32
#  define NOMINMAX
#endif

#include <DO/Defines.hpp>
#include <QApplication>
#include <QPointer>
#include <QPixmap>
#include <QString>
#include <QWidget>
#include "UserThread.hpp"

namespace DO {

  /*!
  \ingroup Graphics
  \defgroup GraphicsInternal Graphics Internals
  \brief This contains the Qt-based internal implementation of the Graphics 
  module.
  @{
 */

  //! \brief quick-and-dirty thing to read file from dialog box.
  //! \todo See if it can be done in a more elegant way.
  struct InteractiveBox
  {
    QPixmap pixmap;
    QString filename;
  };
  
  //! \brief QApplication-derived class
  //! This graphic application establishes communication between the user 
  //! drawing commands and the windows.
  class DO_EXPORT GraphicsApplication : public QApplication
  {
  Q_OBJECT
  public:
    GraphicsApplication(int argc, char **argv);

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
    UserThread userThread;
    QList<QPointer<QWidget> > createdWindows;
    QPointer<QWidget> activeWindow;
    
    //GraphicsView *graphicsView;
    InteractiveBox interactiveBox;
    
    QMutex mutex;
  };
  
  //! @}

} /* namespace DO */

#endif /* DO_GRAPHICS_GRAPHICSAPPLICATION_HPP */