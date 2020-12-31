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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/GraphicsContext.hpp>


using namespace std;
using namespace DO::Sara;


int main(int argc, char** argv)
{
  QApplication app(argc, argv);

  auto widgetList = WidgetList{};

  auto ctx = GraphicsContext{};
  ctx.makeCurrent();
  ctx.setWidgetList(&widgetList);
  ctx.registerUserMain(__main);
  ctx.userThread().start();

  return app.exec();
}


static Window create_window(int h, int w, int x, int y) {
  auto ctx = GraphicsContext::current();
  QMetaObject::invokeMethod(ctx, "createWindow",
                            Qt::BlockingQueuedConnection,
                            Q_ARG(int, GraphicsContext::PAINTING_WINDOW),
                            Q_ARG(int, w), Q_ARG(int, h),
                            Q_ARG(const QString&, QString("TEST")),
                            Q_ARG(int, x), Q_ARG(int, y));
  return ctx->activeWindow();
}

static void draw_point(int x, int y, const Rgb8& color) {
  auto ctx = GraphicsContext::current();
  QMetaObject::invokeMethod(ctx->activeWindow(),
                            "drawPoint",
                            Qt::QueuedConnection,
                            Q_ARG(int, x), Q_ARG(int, y),
                            Q_ARG(const QColor&, QColor(color[0], color[1], color[2])));
}

static void get_key() {
  GraphicsContext::current()->userThread().getKey();
}

static void set_antialiasing(bool on = true) {
  auto ctx = GraphicsContext::current();
  QMetaObject::invokeMethod(ctx->activeWindow(),
                            "setAntialiasing",
                            Qt::QueuedConnection,
                            Q_ARG(bool, on));
}

int __main(int, char**)
{
  // Open a 300x200 window.
  auto w = 300;
  auto h = 200;
  auto x = 100;
  auto y = 100;
  ::create_window(h, w, x, y);

  for (auto y = 0; y < 200; ++y)
    for (auto x = 0; x < 300; ++x)
      ::draw_point(x, y, Red8);

  ::get_key();

  ::set_antialiasing();

  auto ctx = GraphicsContext::current();
  QMetaObject::invokeMethod(ctx->activeWindow(),
                            "drawLine",
                            Qt::QueuedConnection,
                            Q_ARG(const QPointF&, QPointF(10.5f, 10.5f)),
                            Q_ARG(const QPointF&, QPointF(20.8f, 52.8132f)),
                            Q_ARG(const QColor&, QColor(Blue8[0], Blue8[1], Blue8[2])),
                            Q_ARG(int, 5));
  QMetaObject::invokeMethod(ctx->activeWindow(),
                            "drawLine",
                            Qt::QueuedConnection,
                            Q_ARG(const QPointF&, QPointF(10.5f, 10.5f)),
                            Q_ARG(const QPointF&, QPointF(20.8f, 52.8132f)),
                            Q_ARG(const QColor&, QColor(Magenta8[0], Magenta8[1], Magenta8[2])),
                            Q_ARG(int, 2));

  ctx->userThread().getMouse(x, y);
  std::cout << x << " " << y << std::endl;

  return 0;
}
