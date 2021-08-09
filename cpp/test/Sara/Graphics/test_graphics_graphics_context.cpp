// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_MODULE "Graphics/Graphics View Commands"

#include <boost/test/unit_test.hpp>

#include <QApplication>

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsContext.hpp>

#include "event_scheduler.hpp"


using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_graphics_context_drawing_scenario)
{
  // Open a 300x200 window.
  auto w = 300;
  auto h = 200;
  auto x = 100;
  auto y = 100;
  v2::create_window(h, w, x, y);

  for (auto y = 0; y < 200; ++y)
    for (auto x = 0; x < 300; ++x)
      v2::draw_point(x, y, Red8);

  v2::set_antialiasing();
  v2::draw_line({10.5f, 10.5f}, {20.8f, 52.8132f}, Blue8, 5);
  v2::draw_line({10.5f, 10.5f}, {20.8f, 52.8132f}, Magenta8, 2);

  v2::close_window(v2::active_window());
}

int worker_thread(int argc, char **argv)
{
  return boost::unit_test::unit_test_main([]() { return true; }, argc, argv);
}

int main(int argc, char **argv)
{
  // Create Qt Application.
  QApplication app{argc, argv};

  // Run the worker thread
  auto widgetList = WidgetList{};

  auto ctx = GraphicsContext{};
  ctx.makeCurrent();
  ctx.setWidgetList(&widgetList);
  ctx.registerUserMain(worker_thread);
  ctx.userThread().start();

  return app.exec();
}
