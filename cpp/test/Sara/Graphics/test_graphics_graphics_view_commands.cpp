// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_MODULE "Graphics/Graphics View Commands"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>

#include "event_scheduler.hpp"


using namespace DO::Sara;


class TestFixtureForGraphicsViewCommands
{
protected:
  Window _test_window;

public:
  TestFixtureForGraphicsViewCommands()
  {
    _test_window = create_graphics_view(300, 300);
  }

  virtual ~TestFixtureForGraphicsViewCommands()
  {
    close_window(_test_window);
  }
};

BOOST_FIXTURE_TEST_SUITE(TestGraphicsViewCommands, TestFixtureForGraphicsViewCommands)

BOOST_AUTO_TEST_CASE(test_view)
{
  BOOST_CHECK_EQUAL(active_window(), _test_window);
}

BOOST_AUTO_TEST_CASE(test_pixmap_item)
{
  auto image = Image<Rgb8>{ 3, 3 };
  image.matrix().fill(Black8);
  auto pixmap = add_pixmap(image);
  BOOST_CHECK(pixmap != nullptr);
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
  return gui_app.exec();
}
