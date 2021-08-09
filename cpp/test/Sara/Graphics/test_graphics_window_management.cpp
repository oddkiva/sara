// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_MODULE "Graphics/Window Management Functions"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>

#include "event_scheduler.hpp"


using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestWindow)

BOOST_AUTO_TEST_CASE(test_open_and_close_window)
{
  auto window = qobject_cast<PaintingWindow*>(
      create_window(300, 300, "My Window", 50, 50));
  auto scroll_area = window->scrollArea();

  // Check window dimensions.
  BOOST_CHECK_EQUAL(get_width(window), window->width());
  BOOST_CHECK_EQUAL(get_height(window), window->height());
  BOOST_CHECK_EQUAL(get_sizes(window),
                    Vector2i(window->width(), window->height()));

  // Check window title.
  BOOST_CHECK(scroll_area->windowTitle() == QString("My Window"));

  // Check window position.
  BOOST_CHECK_EQUAL(window->x(), 50);
  BOOST_CHECK_EQUAL(window->y(), 50);

  // Check that the widget gets destroyed when we close the window.
  close_window(window);
  millisleep(50);
}

BOOST_AUTO_TEST_CASE(test_open_and_close_gl_window)
{
  auto window = create_gl_window(300, 300, "My Window", 50, 50);

  BOOST_CHECK_EQUAL(get_width(window), window->width());
  BOOST_CHECK_EQUAL(get_height(window), window->height());
  BOOST_CHECK_EQUAL(get_sizes(window),
                    Vector2i(window->width(), window->height()));
  BOOST_CHECK(window->windowTitle() == QString("My Window"));
  BOOST_CHECK(window->pos() == QPoint(50, 50));

  close_window(window);
  millisleep(50);
}

BOOST_AUTO_TEST_CASE(test_open_and_close_graphics_view)
{
  auto window = create_graphics_view(300, 300, "My Window", 50, 50);

  BOOST_CHECK_EQUAL(get_width(window), window->width());
  BOOST_CHECK_EQUAL(get_height(window), window->height());
  BOOST_CHECK_EQUAL(get_sizes(window),
                    Vector2i(window->width(), window->height()));
  BOOST_CHECK(window->windowTitle() == QString("My Window"));
  BOOST_CHECK(window->pos() == QPoint(50, 50));

  close_window(window);
  millisleep(50);
}

BOOST_AUTO_TEST_CASE(test_set_active_window)
{
  Window w1 = create_window(300, 300, "My Window", 10, 10);
  Window w2 = create_gl_window(300, 300, "My GL Window", 10, 10);
  Window w3 = create_graphics_view(300, 300, "My Graphics View", 10, 10);

  BOOST_CHECK(w1);
  BOOST_CHECK(w2);
  BOOST_CHECK(w3);

  BOOST_CHECK_EQUAL(w1, active_window());

  set_active_window(w2);
  BOOST_CHECK_EQUAL(w2, active_window());

  set_active_window(w3);
  BOOST_CHECK_EQUAL(w3, active_window());

  close_window(w1);
  close_window(w2);
  close_window(w3);
}

BOOST_AUTO_TEST_CASE(test_resize_window)
{
  Window w = create_window(300, 300, "My Window", 10, 10);
  BOOST_CHECK_EQUAL(w, active_window());
  BOOST_CHECK_EQUAL(get_sizes(w), Vector2i(300, 300));

  fill_circle(100, 100, 30, Red8);

  resize_window(500, 500);
  BOOST_CHECK_EQUAL(get_sizes(w), Vector2i(500, 500));

  fill_circle(100, 100, 30, Red8);
}

BOOST_AUTO_TEST_SUITE_END()


int worker_thread(int argc, char** argv)
{
  return boost::unit_test::unit_test_main([]() { return true; }, argc, argv);
}

int main(int argc, char** argv)
{
  // Create Qt Application.
  GraphicsApplication gui_app_(argc, argv);

  // Run the worker thread
  gui_app_.register_user_main(worker_thread);
  int return_code = gui_app_.exec();

  // Cleanup and terminate.
  return return_code;
}
