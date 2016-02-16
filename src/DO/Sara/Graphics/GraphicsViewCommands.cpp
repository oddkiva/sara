// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <QGraphicsItem>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>


namespace DO { namespace Sara {

  static GraphicsView *view()
  {
    return qobject_cast<GraphicsView *>(active_window());
  }

  // Graphics view window control functions.
  Window create_graphics_view(int w, int h, const std::string& window_title,
                              int x, int y)
  {
    QMetaObject::invokeMethod(gui_app(), "createWindow",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(int, 2),
                              Q_ARG(int, w), Q_ARG(int, h),
                              Q_ARG(const QString&,
                                    QString(window_title.c_str())),
                              Q_ARG(int, x), Q_ARG(int, y));
    return gui_app()->m_createdWindows.back();
  }

  QGraphicsPixmapItem *add_pixmap(const Image<Rgb8>& image, bool random_pos)
  {
    QImage qimage{ as_QImage(image) };
    QMetaObject::invokeMethod(view(), "addPixmapItem",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(const QImage&, qimage),
                              Q_ARG(bool, random_pos));
    return qgraphicsitem_cast<QGraphicsPixmapItem *>(view()->lastAddedItem());
  }

} /* namespace Sara */
} /* namespace DO */
