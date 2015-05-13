// ========================================================================= //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================= //

//! @file

#ifndef DO_SARA_GRAPHICS_PAINTINGWINDOW_HPP
#define DO_SARA_GRAPHICS_PAINTINGWINDOW_HPP


#include <QScrollArea>
#include <QWidget>
#include <QPainter>
#include <QTimer>

#include "../Events.hpp"


class QPixmap;

namespace DO {

  /*!
    \addtogroup GraphicsInternal

    @{
   */

  //! \brief QScrollArea-derived class on which we embed the PaintingWindow
  //! class in order to scroll the contents of the window properly.
  class ScrollArea : public QScrollArea
  {
    Q_OBJECT

  public:
    ScrollArea(QWidget *parent = 0);

  protected:
    void closeEvent(QCloseEvent *event);
  };

  //! \brief QWidget-derived class on which we draw things.
  //! I choose not to use QGLWidget because of some weird viewing artifacts...
  //! Maybe later...
  class PaintingWindow : public QWidget
  {
    Q_OBJECT

  public:
    PaintingWindow(int width, int height,
                   const QString& windowTitle = "DO-CV",
                   int x = -1, int y = -1,
                   QWidget* parent = 0);
    QScrollArea *scrollArea() { return scroll_area_; }
    QString windowTitle() const;
    int x() const;
    int y() const;

  public slots: /* drawing slots */
    // drawXXX
    void drawPoint(int x, int y, const QColor& c);
    void drawPoint(const QPointF& p, const QColor& c);
    void drawLine(int x1, int y1, int x2, int y2, const QColor& c,
                  int penWidth = 1);
    void drawLine(const QPointF& p1, const QPointF& p2, const QColor& c,
                  int penWidth = 1);
    void drawCircle(int xc, int yc, int r, const QColor& c, int penWidth = 1);
    void drawCircle(const QPointF& center, qreal r, const QColor& c,
                    int penWidth = 1);
    void drawEllipse(int x, int y, int w, int h, const QColor& c,
                     int penWidth = 1);
    void drawEllipse(const QPointF& center, qreal r1, qreal r2, qreal degree,
                     const QColor& c, int penWidth = 1);
    void drawRect(int x, int y, int w, int h, const QColor& c,
                  int penWidth = 1);
    void drawPoly(const QPolygonF& polygon, const QColor& c, int width);
    void drawText(int x,int y, const QString& s,const QColor& c, int fontSize,
                  qreal orientation, bool italic, bool bold, bool underline);
    void drawArrow(int x1, int y1, int x2, int y2, const QColor&  color,
                   int arrowWidth, int arrowHeight, int style, int width);
    // Display image
    void display(const QImage& image, int xoff = 0, int yoff = 0,
                 double fact = 1.);
    // fillXXX
    void fillCircle(int x, int y, int r, const QColor& c);
    void fillCircle(const QPointF& p, qreal r, const QColor& c);
    void fillEllipse(int x, int y, int w, int h, const QColor& c);
    void fillEllipse(const QPointF& p, qreal rx, qreal ry, qreal degree,
                     const QColor& c);
    void fillPoly(const QPolygonF& polygon, const QColor& c);
    void fillRect(int x, int y, int w, int h, const QColor& c);
    // Clear window.
    void clear();
    // Painting modes.
    void setAntialiasing(bool on = true);
    void setTransparency(bool on = true);
    // Save screen.
    void saveScreen(const QString& filename);
    // Resize screen.
    void resizeScreen(int width, int height);

  public slots: /* event management slots */
    void waitForEvent(int ms);
    void eventListeningTimerStopped();

  protected:
    void mouseMoveEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
    void paintEvent(QPaintEvent *event);

  signals:
    void movedMouse(int x, int y, Qt::MouseButtons buttons);
    void pressedMouseButtons(int x, int y, Qt::MouseButtons buttons);
    void releasedMouseButtons(int x, int y, Qt::MouseButtons buttons);
    void pressedKey(int key);
    void releasedKey(int key);
    void sendEvent(Event e);

  private:
    QScrollArea *scroll_area_;
    QPixmap  pixmap_;
    QPainter painter_;
    QTimer event_listening_timer_;
  };

  //! @}

} /* namespace DO */

#endif /* DO_SARA_GRAPHICS_PAINTINGWINDOW_HPP */
