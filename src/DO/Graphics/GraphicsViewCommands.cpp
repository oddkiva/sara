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
#include <QGraphicsItem>
#include "GraphicsUtilities.hpp"

namespace DO {

  static GraphicsView *view()
  { return qobject_cast<GraphicsView *>(getActiveWindow()); }  
  
  // ====================================================================== //
  //! Graphics view window control functions
  Window openGraphicsView(int w, int h, const std::string& windowTitle,
                          int x, int y)
  {
    QMetaObject::invokeMethod(getGuiApp(), "createWindow",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(int, 2),
                              Q_ARG(int, w), Q_ARG(int, h),
                              Q_ARG(const QString&,
                                    QString(windowTitle.c_str())),
                              Q_ARG(int, x), Q_ARG(int, y));
    return getGuiApp()->createdWindows.back();
  }

  // ====================================================================== //
  //! Convenience graphics scene functions
  QImage toQImage(const Image<Rgb8>& I)
  {
    return QImage(reinterpret_cast<const unsigned char*>(I.data()),
                  I.width(), I.height(), I.width()*3,
                  QImage::Format_RGB888);
  }

  QGraphicsPixmapItem *addImage(const Image<Rgb8>& I, bool randomPos)
  {
    QImage tmp(toQImage(I));
    QMetaObject::invokeMethod(view(), "addImageItem",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(const QImage&, tmp),
                              Q_ARG(bool, randomPos));
    return qgraphicsitem_cast<QGraphicsPixmapItem *>(view()->lastAddedItem());
  }

  void drawPoint(ImageItem pixItem, int x, int y, const Rgb8& c)
  {
    QMetaObject::invokeMethod(view(), "drawPoint",
                              Qt::QueuedConnection,
                              Q_ARG(int, x),
                              Q_ARG(int, y),
                              Q_ARG(const QColor&, 
                                    QColor(c[0], c[1], c[2])),
                              Q_ARG(QGraphicsPixmapItem *, pixItem));
  }

} /* namespace DO */