// ========================================================================== //
// This file is part of Kalpana.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Kalpana/Qt/3D/Frame.hpp>
#include <DO/Kalpana/Qt/3D/Scene.hpp>
#include <DO/Kalpana/Qt/3D/SceneItem.hpp>
#include <DO/Kalpana/Qt/3D/TrackBall.hpp>

#include <QOpenGLFunctions>
#include <QOpenGLWidget>

#include <Eigen/Core>


namespace DO { namespace Kalpana {

  using namespace Eigen;

  //! @brief Class derived from QOpenGLWidget to view 3D scenes.
  class DO_SARA_EXPORT Canvas3D : public QOpenGLWidget,
                                  protected QOpenGLFunctions
  {
  public:
    Canvas3D(Scene* scene, QWidget* parent = 0);

    void render3DScene();
    void renderTextOverlay();

  protected:
    void initializeGL();
    void resizeGL(int width, int height);

    void paintEvent(QPaintEvent* event);

    void mousePressEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);
    void mouseMoveEvent(QMouseEvent* event);

    void wheelEvent(QWheelEvent* event);

    void keyPressEvent(QKeyEvent* event);

  protected:
    QPointF normalizePos(const QPointF& localPos) const;

  private:
    //! Model-view parameters.
    float m_scale{1.0f};
    Vector3f m_center;

    //! World coordinate frame.
    Frame m_frame;
    bool m_displayFrame{false};

    //! The scene
    Scene* m_scene;

    //! Trackball parameters for mouse-based navigation.
    TrackBall m_trackball;
    QPoint m_lastPos;

    //! Rendering parameters.
    Vector4f m_backgroundColor{0.72f, 0.655f, 0.886f, .5f};
  };

}}  // namespace DO::Kalpana
