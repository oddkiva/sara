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
#define BOOST_TEST_MODULE "Graphics/Painting Commands"

#include <vector>

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>

#include "event_scheduler.cpp"


using namespace DO::Sara;


class TestFixtureForPaintingCommands
{
protected:
  Window _test_window;

public:
  TestFixtureForPaintingCommands()
  {
    _test_window = create_window(300, 300);
  }

  ~TestFixtureForPaintingCommands()
  {
    close_window(_test_window);
  }
};

BOOST_FIXTURE_TEST_SUITE(TestPaintingCommands, TestFixtureForPaintingCommands)

BOOST_AUTO_TEST_CASE(test_draw_point)
{
  BOOST_CHECK(draw_point(10, 10, Black8));
  BOOST_CHECK(draw_point(10, 10, Color4ub(255, 0, 0, 0)));
  BOOST_CHECK(draw_point(Point2f(10.f, 10.f), Red8));
}

BOOST_AUTO_TEST_CASE(test_draw_circle)
{
  BOOST_CHECK(draw_circle(10, 10, 5, Black8, 2));
  BOOST_CHECK(draw_circle(Point2i(10, 10), 5, Black8, 2));
  BOOST_CHECK(draw_circle(Point2f(10.f, 10.f), 5.f, Black8, 2));
  BOOST_CHECK(draw_circle(Point2d(10., 10.), 5., Black8, 2));
}

BOOST_AUTO_TEST_CASE(test_draw_ellipse)
{
  BOOST_CHECK(draw_ellipse(10, 10, 50, 100, Black8, 2));
  BOOST_CHECK(draw_ellipse(Point2f(10.f, 10.f), 5.f, 10.f, 40.f, Black8, 2));
  BOOST_CHECK(draw_ellipse(Point2d(10., 10.), 5., 10., 40., Black8, 2));
}

BOOST_AUTO_TEST_CASE(test_draw_line)
{
  BOOST_CHECK(draw_line(10, 10, 50, 100, Black8, 2));
  BOOST_CHECK(draw_line(Point2i(10, 10), Point2i(50, 100), Black8, 2));
  BOOST_CHECK(draw_line(Point2f(10, 10), Point2f(50, 100), Black8, 2));
  BOOST_CHECK(draw_line(Point2d(10, 10), Point2d(50, 100), Black8, 2));
}

BOOST_AUTO_TEST_CASE(test_draw_rect)
{
  BOOST_CHECK(draw_rect(10, 10, 50, 100, Black8, 2));
}

BOOST_AUTO_TEST_CASE(test_draw_poly)
{
  int x[] = {10, 20};
  int y[] = {10, 20};
  BOOST_CHECK(draw_poly(x, y, 2, Black8, 2));

  Point2i poly[] = {Point2i(10, 10), Point2i(20, 20)};
  BOOST_CHECK(draw_poly(poly, 2, Red8, 1));
}

BOOST_AUTO_TEST_CASE(test_draw_text)
{
  BOOST_CHECK(
      draw_text(10, 10, "example string", Red8, 14, 0.5, true, false, false));
}

BOOST_AUTO_TEST_CASE(test_draw_arrow)
{
  BOOST_CHECK(draw_arrow(10, 10, 30, 30, Red8));
  BOOST_CHECK(draw_arrow(10, 10, 30, 30, Red8, 1., 1., 1, 2));
  BOOST_CHECK(draw_arrow(Point2f(10, 10), Point2f(30, 30), Red8, 3));
}

BOOST_AUTO_TEST_CASE(test_fill_ellipse)
{
  BOOST_CHECK(fill_ellipse(10, 10, 50, 100, Black8));
  BOOST_CHECK(fill_ellipse(Point2i(10, 10), 5, 10, Black8));
}

BOOST_AUTO_TEST_CASE(test_fill_rect)
{
  BOOST_CHECK(fill_rect(10, 10, 50, 100, Black8));
  BOOST_CHECK(fill_rect(Point2i(10, 10), 50, 100, Black8));
}

BOOST_AUTO_TEST_CASE(test_fill_circle)
{
  BOOST_CHECK(fill_circle(10, 10, 2, Red8));
  BOOST_CHECK(fill_circle(Point2i(10, 10), 2, Red8));
  BOOST_CHECK(fill_circle(Point2f(10.f, 10.f), 2.f, Red8));
}

BOOST_AUTO_TEST_CASE(test_fill_poly)
{
  int x[] = {10, 20};
  int y[] = {10, 20};
  BOOST_CHECK(fill_poly(x, y, 2, Black8));

  Point2i poly[] = {Point2i(10, 10), Point2i(20, 20)};
  BOOST_CHECK(fill_poly(poly, 2, Red8));
}

BOOST_AUTO_TEST_CASE(test_put_color_image)
{
  const auto w = 50, h = 50;
  const auto data = std::vector<Color3ub>(w * h, Red8);
  BOOST_CHECK(put_color_image(10, 10, data.data(), w, h, 2.));
  BOOST_CHECK(put_color_image(Point2i(10, 10), data.data(), w, h, 2.));
}

BOOST_AUTO_TEST_CASE(test_put_grey_image)
{
  const auto w = 50, h = 50;
  const auto data = std::vector<unsigned char>(w * h, 0);
  BOOST_CHECK(put_grey_image(10, 10, data.data(), w, h, 2.));
  BOOST_CHECK(put_grey_image(Point2i(10, 10), data.data(), w, h, 2.));
}

BOOST_AUTO_TEST_CASE(test_display)
{
  auto image = Image<Color3ub>{50, 50};
  image.flat_array().fill(Red8);
  BOOST_CHECK(display(image, 0, 0, 1.4));
  BOOST_CHECK(display(image, Point2i(0, 10), 1.4));

  auto rgb_image = Image<Rgb8>{50, 50};
  rgb_image.flat_array().fill(Red8);
  BOOST_CHECK(display(rgb_image, 0, 0, 1.4));
  BOOST_CHECK(display(rgb_image, Point2i(0, 10), 1.4));
}

BOOST_AUTO_TEST_CASE(test_clear_window)
{
  BOOST_CHECK(clear_window());
}

BOOST_AUTO_TEST_CASE(test_set_antialiasing)
{
  BOOST_CHECK(set_antialiasing());
}

BOOST_AUTO_TEST_CASE(test_set_transparency)
{
  BOOST_CHECK(set_transparency());
}

BOOST_AUTO_TEST_CASE(test_save_screen)
{
  auto image = Image<Color3ub>(50, 50);
  image.flat_array().fill(Red8);
  BOOST_CHECK(display(image, 0, 0, 1.4));
  BOOST_CHECK(save_screen(active_window(), "test.png"));
}


using ColorTypes = boost::mpl::list<unsigned char, unsigned short, unsigned int,
                                    char, short, int, float, double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_generic_image_display, ColorType, ColorTypes)
{
  auto image = Image<ColorType>{20, 20};
  image.flat_array().fill(1);
  BOOST_CHECK(display(image, 0, 0, 1.4));
  BOOST_CHECK(display(image, 0, 0, 1.4));
}

BOOST_AUTO_TEST_SUITE_END()


int worker_thread(int argc, char **argv)
{
  return boost::unit_test::unit_test_main([]() { return true; }, argc, argv);
}

int main(int argc, char **argv)
{
  // Create Qt Application.
  GraphicsApplication gui_app(argc, argv);

  // Run the worker thread
  gui_app.register_user_main(worker_thread);
  int return_code = gui_app.exec();

  // Cleanup and terminate.
  return return_code;
}
