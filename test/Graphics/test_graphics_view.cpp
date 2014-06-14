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
#include <DO/Graphics/DerivedQObjects/GraphicsView.hpp>

using namespace DO;

class TestGraphicsView: public QObject
{
  Q_OBJECT
private slots:
  void test_construction_of_GraphicsView()
  {
  }
};

QTEST_MAIN(TestGraphicsView)
#include "test_graphics_view.moc"
