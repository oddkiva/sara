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
  EXPECT_EQ(i, 3u);

  for (; image_it != image_db.begin(); --image_it)
    --i;
  EXPECT_EQ(i, 0u);
  EXPECT_EQ(image_it, image_db.begin());
}

TEST(TestTrainingDataSet,
     test_image_classification_training_data_set_initialization)
{
  const auto db_dir = string{src_path("../../../data/")};

  auto training_data_set = ImageClassificationTrainingDataSet{};

  training_data_set.x = {
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  };

  training_data_set.y = {0, 0, 1};

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

  training_data_set.x = {
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  };

  training_data_set.y = {
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  };

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

  training_data_set.x = {
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  };

  training_data_set.y = {0, 0, 1};

  auto data_transforms = vector<ImageDataTransform>(3);
  {
    for (auto& t : data_transforms)
      t.out_sizes = Vector2i{320, 240};

    data_transforms[0].set_zoom(1.3f);
    data_transforms[0].set_shift(Vector2i::Ones() * 5);
    data_transforms[0].set_fancy_pca(Vector3f::Ones());

    data_transforms[1].set_zoom(0.8f);
    data_transforms[1].set_flip(ImageDataTransform::Horizontal);
  }

  training_data_set.t = std::move(data_transforms);

  auto sample_i = training_data_set.begin();

  EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y_ref(), 0);
  EXPECT_FALSE(sample_i.t_ref().use_original);

  ++sample_i;
  EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y_ref(), 0);
  EXPECT_FALSE(sample_i.t_ref().use_original);

  ++sample_i;
  EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
  EXPECT_EQ(sample_i.y_ref(), 1);
  EXPECT_TRUE(sample_i.t_ref().use_original);

  write_to_csv(training_data_set, "transformed_classification_dataset.csv");

  auto training_data_set2 = TransformedImageClassificationTrainingDataSet{};
  read_from_csv(training_data_set2, "transformed_classification_dataset.csv");

  // Manual checks data member by data member.
  {
    auto sample_i = training_data_set2.begin();

    EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
    EXPECT_EQ(sample_i.y_ref(), 0);
    EXPECT_EQ(Vector2i(320, 240), sample_i.t_ref().out_sizes);
    EXPECT_FALSE(sample_i.t_ref().use_original);
    EXPECT_TRUE(sample_i.t_ref().apply_transform[ImageDataTransform::Zoom]);
    EXPECT_TRUE(sample_i.t_ref().z == 1.3f);
    EXPECT_TRUE(sample_i.t_ref().apply_transform[ImageDataTransform::Shift]);
    EXPECT_TRUE(sample_i.t_ref().t == Vector2i(5, 5));
    EXPECT_FALSE(sample_i.t_ref().apply_transform[ImageDataTransform::Flip]);
    EXPECT_TRUE(sample_i.t_ref().flip_type == ImageDataTransform::None);
    EXPECT_TRUE(sample_i.t_ref().apply_transform[ImageDataTransform::FancyPCA]);
    EXPECT_MATRIX_EQ(sample_i.t_ref().alpha, Vector3f::Ones());

    ++sample_i;
    EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
    EXPECT_EQ(sample_i.y_ref(), 0);
    EXPECT_EQ(Vector2i(320, 240), sample_i.t_ref().out_sizes);
    EXPECT_FALSE(sample_i.t_ref().use_original);
    EXPECT_TRUE(sample_i.t_ref().apply_transform[ImageDataTransform::Zoom]);
    EXPECT_TRUE(sample_i.t_ref().z == 0.8f);
    EXPECT_FALSE(sample_i.t_ref().apply_transform[ImageDataTransform::Shift]);
    EXPECT_TRUE(sample_i.t_ref().t == Vector2i::Zero());
    EXPECT_TRUE(sample_i.t_ref().apply_transform[ImageDataTransform::Flip]);
    EXPECT_TRUE(sample_i.t_ref().flip_type == ImageDataTransform::Horizontal);
    EXPECT_FALSE(sample_i.t_ref().apply_transform[ImageDataTransform::FancyPCA]);
    EXPECT_MATRIX_EQ(sample_i.t_ref().alpha, Vector3f::Zero());

    ++sample_i;
    EXPECT_NE(sample_i.x_ref().sizes(), Vector2i::Zero());
    EXPECT_EQ(sample_i.y_ref(), 1);
    EXPECT_EQ(Vector2i(320, 240), sample_i.t_ref().out_sizes);
    EXPECT_TRUE(sample_i.t_ref().use_original);
    EXPECT_FALSE(sample_i.t_ref().apply_transform[ImageDataTransform::Zoom]);
    EXPECT_TRUE(sample_i.t_ref().z == 1.f);
    EXPECT_FALSE(sample_i.t_ref().apply_transform[ImageDataTransform::Shift]);
    EXPECT_TRUE(sample_i.t_ref().t == Vector2i::Zero());
    EXPECT_FALSE(sample_i.t_ref().apply_transform[ImageDataTransform::Flip]);
    EXPECT_TRUE(sample_i.t_ref().flip_type == ImageDataTransform::None);
    EXPECT_FALSE(sample_i.t_ref().apply_transform[ImageDataTransform::FancyPCA]);
    EXPECT_MATRIX_EQ(sample_i.t_ref().alpha, Vector3f::Zero());
  }
  EXPECT_EQ(training_data_set, training_data_set2);
}

TEST(TestTransformedTrainingDataSet,
     test_transformed_image_segmentation_training_data_set_initialization)
{
  const auto db_dir = string{src_path("../../../data/")};

  auto training_data_set = TransformedImageSegmentationTrainingDataSet{};

  training_data_set.x = {
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  };

  training_data_set.y = {
      db_dir + "/" + "All.tif",
      db_dir + "/" + "ksmall.jpg",
      db_dir + "/" + "stinkbug.png",
  };

  auto data_transforms = vector<ImageDataTransform>(3);
  {
    for (auto& t : data_transforms)
      t.out_sizes = Vector2i{320, 240};

    data_transforms[0].set_zoom(1.3f);
    data_transforms[0].set_shift(Vector2i::Ones() * 5);
    data_transforms[0].set_fancy_pca(Vector3f::Ones());

    data_transforms[1].set_zoom(2.f);
    data_transforms[1].set_flip(ImageDataTransform::Horizontal);

    data_transforms[2].set_zoom(2.f);
    data_transforms[2].set_flip(ImageDataTransform::Horizontal);
  }
  training_data_set.t = data_transforms;

  for (auto s = training_data_set.begin(), s_end = training_data_set.end();
       s != s_end; ++s)
  {
    EXPECT_NE(s.x_ref().sizes(), Vector2i::Zero());
    EXPECT_EQ(s.y_ref().sizes(), s.x_ref().sizes());
    EXPECT_FALSE(s.t_ref().use_original);
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
