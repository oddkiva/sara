// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageIO/Database/ImageDatabase.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestImageDatabase, test_image_database_iterator)
{
  auto db_dir = string{src_path("../../../data/")};

  auto image_db = vector<string>{
    db_dir + "/" + "All.tif",
    db_dir + "/" + "ksmall.jpg",
    db_dir + "/" + "stinkbug.png",
  };

  auto image_it = begin_image_db(image_db);
  auto image_end = end_image_db(image_db);

  ASSERT_TRUE(image_it->sizes() != Vector2i::Zero());

  size_t i = 0;
  for (; image_it != image_end; ++image_it)
    ++i;
  ASSERT_EQ(i, 3);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
