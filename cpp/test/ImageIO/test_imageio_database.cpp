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
#include <DO/Sara/ImageIO/Database/ImageDataSet.hpp>
#include <DO/Sara/ImageIO/Database/TrainingDataSet.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestImageDatabase, test_image_database_iterator)
{
  const auto db_dir = string{src_path("../../../data/")};

  auto image_list = vector<string>{
    db_dir + "/" + "All.tif",
    db_dir + "/" + "ksmall.jpg",
    db_dir + "/" + "stinkbug.png",
  };

  const auto image_db = ImageDataSet{image_list};
  auto image_it = image_db.begin();
  auto image_end = image_db.end();

  EXPECT_TRUE(image_it->sizes() != Vector2i::Zero());

  size_t i = 0;
  for (; image_it != image_end; ++image_it)
    ++i;
  EXPECT_EQ(i, 3);
}

TEST(TestTrainingDataSet, test_training_data_set_initialization)
{
  const auto db_dir = string{src_path("../../../data/")};

  auto training_data_set = ImageClassificationTrainingDataSet{};
  {
    auto images = vector<string>{
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png"
    };
    training_data_set.set_image_data_set(std::move(images));
    EXPECT_EQ(images.size(), 0);
  }

  {
    auto labels = vector<int>{0, 0, 1};
    training_data_set.set_label_set(std::move(labels));
    EXPECT_EQ(labels.size(), 0);
  }

  auto sample_i = training_data_set.begin();

  EXPECT_NE(sample_i.x().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y(), 0);

  ++sample_i;
  EXPECT_NE(sample_i.x().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y(), 0);

  ++sample_i;
  EXPECT_NE(sample_i.x().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y(), 1);
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
