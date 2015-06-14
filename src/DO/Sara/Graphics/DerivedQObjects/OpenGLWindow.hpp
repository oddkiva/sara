// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_GRAPHICS_OPENGLWINDOW_HPP
#define DO_SARA_GRAPHICS_OPENGLWINDOW_HPP

#include <QGLWidget>
#include <QQuaternion>
#include <QTime>
#include <QTimer>
#include <QVector>
#include <QVector3D>

#include "../Events.hpp"
#include "../Frame.hpp"
#include "../Mesh.hpp"


namespace DO { namespace Sara {

  /*!
    \addtogroup GraphicsInternal

    @{
   */

  //! \brief The TrackBall class is used the OpenGLWindow class to allow the
  //! user to view the 3D scene interactively.
  class TrackBall
  {
  public:
    TrackBall();
    // coordinates in [-1,1]x[-1,1]
    void push(const QPointF& p, const QQuaternion& transformation);
    void move(const QPointF& p);
    void release(const QPointF& p);
    QQuaternion rotation() const;
  private:
    QQuaternion rotation_;
    QVector3D axis_;
    QPointF lastPos_;
    bool pressed_;
  };

  //! \brief QGLWidget-derived class used to view 3D scenes.
  class OpenGLWindow : public QGLWidget
  {
    Q_OBJECT

  public:
    OpenGLWindow(int width, int height,
                 const QString& windowTitle = "DO-CV",
                 int x = -1, int y = -1,
                 QWidget* parent = 0);

  public slots:
    void setMesh(const SimpleTriangleMesh3f& mesh_);
    void displayMesh();
    void waitForEvent(int ms);
    void eventListeningTimerStopped();

  protected:
    void initializeGL();
    void paintEvent(QPaintEvent *event);
    void resizeGL(int width, int height);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
    void closeEvent(QCloseEvent *event);

  signals:
    void movedMouse(int x, int y, Qt::MouseButtons buttons);
    void pressedMouseButtons(int x, int y, Qt::MouseButtons buttons);
    void releasedMouseButtons(int x, int y, Qt::MouseButtons buttons);
    void pressedKey(int key);
    void releasedKey(int key);
    void sendEvent(Event e);

  protected:
    QPointF normalizePos(const QPointF& localPos) const;

  private:
    GLfloat scale_;
    Point3f center_;
    GLObject::Frame frame_;
    TrackBall trackball_;

    SimpleTriangleMesh3f mesh_;

    QPoint last_pos_;
    QColor background_color_, color_;

    bool display_frame_;

    QTimer event_listening_timer_;

  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GRAPHICS_OPENGLWINDOW_HPP */
