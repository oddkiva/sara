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

#include <DO/Sara/Graphics/Frame.hpp>


namespace DO::Sara::GL {

  Frame::Frame()
  {
    pQuadObj = gluNewQuadric();
    gluQuadricDrawStyle(pQuadObj, GLU_FILL);
    gluQuadricNormals(pQuadObj, GLU_SMOOTH);
  }

  Frame::~Frame()
  {
    gluDeleteQuadric(pQuadObj);
  }

  void Frame::draw(double axisLength, double axisRadius)
  {
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    {
      glColor3f(.4f, .4f, .4f);
      glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
      glMaterialfv(GL_FRONT, GL_AMBIENT, white);
      glMaterialfv(GL_FRONT, GL_DIFFUSE, white);
      gluSphere(pQuadObj, 3*axisRadius, 36, 18);
      glPushMatrix();
      {
        glColor3f(1.0f, 0.0f, 0.0f);
        glMaterialfv(GL_FRONT, GL_AMBIENT, red);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, red);
        glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
        gluCylinder(pQuadObj, axisRadius, axisRadius, axisLength,
          36, 18);
        glTranslatef(0.0f, 0.0f, axisLength);
        gluCylinder(pQuadObj, 2*axisRadius, 0.0, 3*axisRadius, 36, 10);
      }
      glPopMatrix();
      glPushMatrix();
      {
        glColor3f(0.0f, 1.0f, 0.0f);
        glMaterialfv(GL_FRONT, GL_AMBIENT, green);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, green);
        glRotatef(-90.0f, 1.0f, 0.0f, 0.0f);
        gluCylinder(pQuadObj, axisRadius, axisRadius, axisLength,
          36, 18);
        glTranslatef(0.0f, 0.0f, axisLength);
        gluCylinder(pQuadObj, 2*axisRadius, 0.0, 3*axisRadius, 36, 10);
      }
      glPopMatrix();
      glPushMatrix();
      {
        glColor3f(0.0f, 0.0f, 1.0f);
        glMaterialfv(GL_FRONT, GL_AMBIENT, blue);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, blue);
        gluCylinder(pQuadObj, axisRadius, axisRadius, axisLength,
          36, 10);
        glTranslatef(0.0f, 0.0f, axisLength);
        gluCylinder(pQuadObj, 2*axisRadius, 0.0, 3*axisRadius, 36, 10);
      }
      glPopMatrix();
    }
    glPopAttrib();
  }

} /* namespace GLObject */
