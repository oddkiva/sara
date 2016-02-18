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

#include <gtest/gtest.h>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>

#include "event_scheduler.cpp"


using namespace DO::Sara;


class TestGraphicsViewCommands : public testing::Test
{
protected:
  Window _test_window;

  TestGraphicsViewCommands()
  {
    _test_window = create_graphics_view(300, 300);
  }

  virtual ~TestGraphicsViewCommands()
  {
    close_window(_test_window);
  }
};

TEST_F(TestGraphicsViewCommands, test_view)
{
  EXPECT_EQ(active_window(), _test_window);
}

TEST_F(TestGraphicsViewCommands, test_pixmap_item)
{
  auto image = Image<Rgb8>{ 3, 3 };
  image.matrix().fill(Black8);
  auto pixmap = add_pixmap(image);
  EXPECT_NE(pixmap, nullptr);
}

int worker_thread(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

int main(int argc, char **argv)
{
  // Create Qt Application.
  GraphicsApplication gui_app(argc, argv);

  // Run the worker thread
  gui_app.register_user_main(worker_thread);
  return gui_app.exec();
}