// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <QApplication>

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsContext.hpp>


using namespace std;
using namespace DO::Sara;


int worker_thread(int, char**);

int main(int argc, char** argv)
{
  QApplication app(argc, argv);

  auto widgetList = WidgetList{};

  auto ctx = GraphicsContext{};
  ctx.makeCurrent();
  ctx.setWidgetList(&widgetList);
  ctx.registerUserMain(worker_thread);
  ctx.userThread().start();

  return app.exec();
}

int worker_thread(int, char**)
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

  v2::get_key();

  v2::set_antialiasing();
  v2::draw_line({10.5f, 10.5f}, {20.8f, 52.8132f}, Blue8, 5);
  v2::draw_line({10.5f, 10.5f}, {20.8f, 52.8132f}, Magenta8, 2);

  v2::get_mouse(x, y);
  std::cout << x << " " << y << std::endl;

  return 0;
}
