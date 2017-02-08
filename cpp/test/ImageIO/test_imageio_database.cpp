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
#include <DO/Sara/ImageIO/Database/TransformedTrainingDataSet.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestImageDatabase, test_image_database_iterator)
{
  const auto db_dir = string{src_path("../../../data/")};

  const auto image_db = ImageDataSet<Image<Rgb8>>{{
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  }};
  auto image_it = image_db.begin();
  auto image_end = image_db.end();

  // Check we could read something.
  EXPECT_NO_THROW(image_it.operator*());
  EXPECT_NO_THROW(image_it.operator->());
  EXPECT_NE(image_it->data(), nullptr);
  EXPECT_TRUE(image_it->sizes() != Vector2i::Zero());
  EXPECT_EQ(image_it.path(), db_dir + "/" + "All.tif");

  // Check iterator arithmetics.
  EXPECT_EQ(image_end - image_it, 3);

  size_t i = 0;
  for (; image_it != image_end; ++image_it)
    ++i;
  EXPECT_EQ(i, 3);

  for (; image_it != image_db.begin(); --image_it)
    --i;
  EXPECT_EQ(i, 0);
  EXPECT_EQ(image_it, image_db.begin());
}

TEST(TestTrainingDataSet,
     test_image_classification_training_data_set_initialization)
{
  const auto db_dir = string{src_path("../../../data/")};

  auto training_data_set = ImageClassificationTrainingDataSet{};

  training_data_set.set_image_data_set({
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  });

  training_data_set.set_label_set({0, 0, 1});

  auto sample_i = training_data_set.begin();

  EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y_ref(), 0);

  ++sample_i;
  EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y_ref(), 0);

  ++sample_i;
  EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y_ref(), 1);

  write_to_csv(training_data_set, "classification_dataset.csv");

  auto training_data_set2 = ImageClassificationTrainingDataSet{};
  read_from_csv(training_data_set2, "classification_dataset.csv");
  EXPECT_EQ(training_data_set, training_data_set2);
}

TEST(TestTrainingDataSet, test_image_segmentation_training_data_set_initialization)
{
  const auto db_dir = string{src_path("../../../data/")};

  auto training_data_set = ImageSegmentationTrainingDataSet{};

  training_data_set.set_image_data_set({
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  });

  training_data_set.set_label_set({
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  });

  for (auto s = training_data_set.begin(), s_end = training_data_set.end();
       s != s_end; ++s)
  {
    EXPECT_NE(s.x_ref().sizes(), Vector2i::Zero());
    EXPECT_EQ(s.y_ref().sizes(), s.x_ref().sizes());
  }

  for (auto s : training_data_set)
  {
    EXPECT_NE(s.first.sizes(), Vector2i::Zero());
    EXPECT_EQ(s.first.sizes(), s.second.sizes());
  }

  write_to_csv(training_data_set, "segmentation_dataset.csv");

  auto training_data_set2 = ImageSegmentationTrainingDataSet{};
  read_from_csv(training_data_set2, "segmentation_dataset.csv");
  EXPECT_EQ(training_data_set, training_data_set2);
}


TEST(TestTransformedTrainingDataSet,
     test_transformed_image_classification_training_data_set_initialization)
{
  const auto db_dir = string{src_path("../../../data/")};

  auto training_data_set = TransformedImageClassificationTrainingDataSet{};

  training_data_set.set_image_data_set({
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  });

  training_data_set.set_label_set({0, 0, 1});

  training_data_set.set_data_transform_set(
      {ImageDataTransform{}, ImageDataTransform{}, ImageDataTransform{}});

  auto sample_i = training_data_set.begin();

  EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y_ref(), 0);
  EXPECT_TRUE(sample_i.t_ref().use_original);

  ++sample_i;
  EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y_ref(), 0);
  EXPECT_TRUE(sample_i.t_ref().use_original);

  ++sample_i;
  EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y_ref(), 1);
  EXPECT_TRUE(sample_i.t_ref().use_original);

  write_to_csv(training_data_set, "transformed_classification_dataset.csv");

  auto training_data_set2 = TransformedImageClassificationTrainingDataSet{};
  read_from_csv(training_data_set2, "transformed_classification_dataset.csv");
  EXPECT_EQ(training_data_set, training_data_set2);
}

TEST(TestTransformedTrainingDataSet,
     test_transformed_image_segmentation_training_data_set_initialization)
{
  const auto db_dir = string{src_path("../../../data/")};

  auto training_data_set = TransformedImageSegmentationTrainingDataSet{};

  training_data_set.set_image_data_set({
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  });

  training_data_set.set_label_set({
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  });

  training_data_set.set_data_transform_set(
      {ImageDataTransform{}, ImageDataTransform{}, ImageDataTransform{}});

  for (auto s = training_data_set.begin(), s_end = training_data_set.end();
       s != s_end; ++s)
  {
    EXPECT_NE(s.x_ref().sizes(), Vector2i::Zero());
    EXPECT_EQ(s.y_ref().sizes(), s.x_ref().sizes());
    EXPECT_TRUE(s.t_ref().use_original);
  }

  write_to_csv(training_data_set, "transformed_segmentation_dataset.csv");

  auto training_data_set2 = TransformedImageSegmentationTrainingDataSet{};
  read_from_csv(training_data_set2, "transformed_segmentation_dataset.csv");
  EXPECT_EQ(training_data_set, training_data_set2);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
