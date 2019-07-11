// ========================================================================== //
// This file is part of Kalpana.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <iostream>
#include <stdexcept>

#include <QtOpenGL>
#include <QtOpenGLExtensions>

#include <DO/Kalpana/3D.hpp>
#include <DO/Kalpana/Math/Projection.hpp>


using namespace std;


namespace DO { namespace Kalpana {

  Canvas3D::Canvas3D(Scene *scene, QWidget *parent)
    : QOpenGLWidget{ parent }
    , m_scene{ scene }
  {
    setAttribute(Qt::WA_DeleteOnClose);
    setAutoFillBackground(false);
  }

  void Canvas3D::initializeGL()
  {
    // TODO!
    //connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &GLWidget::cleanup);

    initializeOpenGLFunctions();

    // Set background color
    glClearColor(m_backgroundColor[0], m_backgroundColor[1],
                 m_backgroundColor[2], m_backgroundColor[3]);

    glShadeModel(GL_SMOOTH);  // Enable smooth shading

    // @TODO: create ambient light class.
    // Set up the cosmic background radiation.
    glEnable(GL_LIGHTING);
    GLfloat ambient_light[] = { 0.2f, 0.2f, 0.2f, 1.0f };
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient_light);

    // @TODO: create light class.
    // Set up light source 0.
    GLfloat light0_pos[]      = { 0.0f, 0.0f, 10.0f, 1.0f };
    GLfloat light0_spot_dir[]  = { 0.0f, 0.0f,-1.0f, 1.0f };
    GLfloat light0_diffuse[]  = { 0.8f, 0.5f, 0.5f, 0.8f };
    GLfloat light0_specular[] = { 1.0f, 1.0f, 0.0f, 1.0f };
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light0_pos);
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, light0_spot_dir);
    glEnable(GL_LIGHT0);

    // Set up color material.
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColorMaterial(GL_FRONT_AND_BACK, GL_SPECULAR);
    glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 100);
    glEnable(GL_COLOR_MATERIAL);

    glEnable(GL_MULTISAMPLE);

    // Normalize the vector for the lighting
    glEnable(GL_NORMALIZE);

    for (auto& scene_item : m_scene->_objects)
      scene_item->initialize();
  }

  static void multMatrix(const QMatrix4x4& m)
  {
    // static to prevent glMultMatrixf to fail on certain drivers
    static GLfloat mat[16];
    const float *data = m.constData();
    for (int index = 0; index < 16; ++index)
      mat[index] = data[index];
    glMultMatrixf(mat);
  }

  void Canvas3D::render3DScene()
  {
    makeCurrent();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Setup the viewing mode for the mesh
    glPolygonMode(GL_FRONT, GL_FILL);
    glPolygonMode(GL_BACK, GL_LINE);
    glEnable(GL_DEPTH_TEST);

    // Model-view transform.
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -15.0f);

    // The world frame is at z=-15 w.r.t. the camera frame.
    //
    // Scale the model
    glScalef(m_scale, m_scale, m_scale);
    // Rotate the model with the trackball.
    auto m = QMatrix4x4{};
    m.rotate(m_trackball.rotation());
    multMatrix(m);
    // Display the mesh.
    glPushMatrix();
    {
      // Center the model
      glTranslatef(-m_center.x(), -m_center.y(), -m_center.z());

      // Draw the model
      for (const auto& object : m_scene->_objects)
        object->draw();
    }
    glPopMatrix();

    // Object-centered frame.
    if (m_displayFrame)
      m_frame.draw(5, 0.1);

    // Disable the following to properly display the drawing with QPainter.
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

  void Canvas3D::renderTextOverlay()
  {
    QPainter p{ this };
    p.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);

    QString text = tr(
      "Use the mouse wheel to zoom and the mouse left button to rotate the "
      "scene.\nHit 'F' to toggle object-centered frame display");

    // Set the font style
    setFont(QFont("Helvetica [Cronyx]", 10, QFont::Bold));

    // Draw the bounding box within which the text will be drawn.
    QFontMetrics metrics = QFontMetrics(font());
    const auto border = qMax(4, metrics.leading());
    QRect rect = metrics.boundingRect(
      0, 0, width() - 2*border, int(height()*0.125),
      Qt::AlignCenter | Qt::TextWordWrap, text);
    p.fillRect(QRect(0, 0, width(), rect.height() + 2*border),
               QColor(0, 0, 0, 127));

    // Draw the text.
    p.setPen(Qt::white);
    p.fillRect(QRect(0, 0, width(), rect.height() + 2*border),
               QColor(0, 0, 0, 127));
    p.setFont(font());
    p.drawText((width() - rect.width())/2, border,
               rect.width(), rect.height(),
               Qt::AlignCenter | Qt::TextWordWrap, text);
    p.end();
  }

  void Canvas3D::paintEvent(QPaintEvent *)
  {
    render3DScene();
    renderTextOverlay();
  }

  void Canvas3D::resizeGL(int w, int h)
  {
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    const auto ratio = double(w) / h;
    const auto proj_matrix = perspective(60., ratio, 1., 100.);
    glLoadMatrixd(proj_matrix.data());

    glMatrixMode(GL_MODELVIEW);
  }

  QPointF Canvas3D::normalizePos(const QPointF& localPos) const
  {
    auto pos = localPos;
    pos.rx() -=  width()/2.; pos.rx() /= width()/2.;
    pos.ry() -= height()/2.; pos.ry() /= height()/2.; pos.ry() *= -1;
    return pos;
  }

  void Canvas3D::mousePressEvent(QMouseEvent *event)
  {
    QOpenGLWidget::mousePressEvent(event);
    if (event->isAccepted())
      return;

    auto pos = normalizePos(event->localPos());
    if (event->buttons() & Qt::LeftButton)
    {
      m_trackball.push(pos, m_trackball.rotation());
      event->accept();
    }
    update();
  }

  void Canvas3D::mouseReleaseEvent(QMouseEvent *event)
  {
    QOpenGLWidget::mouseReleaseEvent(event);
    if (event->isAccepted())
      return;

    auto pos = normalizePos(event->localPos());
    if (event->button() == Qt::LeftButton)
    {
      m_trackball.release(pos);
      event->accept();
    }
    update();
  }

  void Canvas3D::mouseMoveEvent(QMouseEvent *event)
  {
    QOpenGLWidget::mouseMoveEvent(event);
    if (event->isAccepted())
    {
      qDebug() << "mouse move event already accepted";
      return;
    }

    auto pos = normalizePos(event->localPos());
    if (event->buttons() & Qt::LeftButton)
    {
      m_trackball.move(pos);
      event->accept();
    }
    else
      m_trackball.release(pos);
    update();
  }

  void Canvas3D::wheelEvent(QWheelEvent* event)
  {
    QOpenGLWidget::wheelEvent(event);

    if (!event->isAccepted())
    {
      event->delta() > 0 ? m_scale += 0.05f*m_scale : m_scale -= 0.05f*m_scale;
      update();
    }
  }

  void Canvas3D::keyPressEvent(QKeyEvent *event)
  {
    if (event->key() == Qt::Key_F)
    {
      m_displayFrame=!m_displayFrame;
      update();
    }
  }

} /* namespace Kalpana */
} /* namespace DO */
