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

#include <QOpenGLWidget>
#include <QQuaternion>
#include <QTime>
#include <QTimer>
#include <QVector>
#include <QVector3D>

#include <DO/Sara/Defines.hpp>

#include "../Events.hpp"
#include "../Frame.hpp"
#include "../Mesh.hpp"


namespace DO { namespace Sara {

  /*!
    \addtogroup GraphicsInternal

    @{
   */

  //! @brief The TrackBall class is used the OpenGLWindow class to allow the
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

  //! @brief QGLWidget-derived class used to view 3D scenes.
  class DO_SARA_EXPORT OpenGLWindow : public QOpenGLWidget
  {
    Q_OBJECT

  public:
    OpenGLWindow(int width, int height,
                 const QString& windowTitle = "Sara",
                 int x = -1, int y = -1,
                 QWidget* parent = nullptr,
                 bool deleteOnClose = false);

  public slots:
    void setMesh(const SimpleTriangleMesh3f& mesh_);
    void setEulerAngles(int yaw, int pitch, int roll);
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
    void pressedKey(int key);
    void releasedKey(int key);
    void sendEvent(Event e);

  protected:
    QPointF normalizePos(const QPointF& localPos) const;

  private:
    // Model view matrix.
    GLfloat m_scale;
    Point3f m_center;
    GLfloat m_frameScale = 1;
    GL::Frame m_frame;
    TrackBall m_trackball;

    // Euler rotation of the object w.r.t. the OpenGL axes.
    Eigen::Matrix4f m_eulerRotation = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f m_axisPermutation = Eigen::Matrix3f::Identity();

    SimpleTriangleMesh3f m_mesh;

    QPoint m_lastPos;
    QColor m_backgroundColor;
    QColor m_color;;

    bool m_displayFrame;

    QTimer m_eventListeningTimer;

  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
