// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <QtTest>
#include <DO/Graphics/DerivedQObjects/OpenGLWindow.hpp>

using namespace DO;

class TestOpenGLWindow: public QObject
{
  Q_OBJECT
private slots:
  void test_OpenGLWindow_construction()
  {
    int width = 50;
    int height = 50;
    QString windowName = "painting window";
    int x = 200;
    int y = 300;

    OpenGLWindow *window = new OpenGLWindow(width, height,
                                            windowName,
                                            x, y);


    delete window;
  }

};

QTEST_MAIN(TestOpenGLWindow)
#include "test_opengl_window.moc"
