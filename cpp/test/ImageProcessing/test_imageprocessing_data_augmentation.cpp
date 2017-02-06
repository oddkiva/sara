// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //


#include <gtest/gtest.h>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageProcessing/DataAugmentation.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestDataAugmentation, test_zoom)
{
  auto t = ImageDataTransform{};
  t.set_zoom(2.f);
  t.out_sizes = Vector2i::Ones() * 2;

  auto in = Image<float>{3, 3};
  in.matrix() <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8;

  const auto out = t.extract_patch(in);
  EXPECT_MATRIX_EQ(out.sizes(), t.out_sizes);
}

TEST(TestDataAugmentation, test_shift)
{
  auto t = ImageDataTransform{};
  t.set_shift(Vector2i::Ones());
  t.out_sizes = Vector2i::Ones() * 2;

  auto in = Image<int>{3, 3};
  in.matrix() <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8;

  const auto out = t.extract_patch(in);

  auto true_out = Image<int>{2, 2};
  true_out.matrix() <<
    4, 5,
    7, 8;

  EXPECT_MATRIX_EQ(true_out.matrix(), out.matrix());
}

TEST(TestDataAugmentation, test_flip)
{
  auto t = ImageDataTransform{};
  t.set_flip(ImageDataTransform::Horizontal);
  t.out_sizes = Vector2i::Ones() * 3;

  auto in = Image<int>{3, 3};
  in.matrix() <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8;

  const auto out = t.extract_patch(in);

  auto true_out = Image<int>{3, 3};
  true_out.matrix() <<
    2, 1, 0,
    5, 4, 3,
    8, 7, 6;

  EXPECT_MATRIX_EQ(true_out.matrix(), out.matrix());
}

TEST(TestDataAugmentation, test_fancy_pca)
{
  auto t = ImageDataTransform{};
  t.set_fancy_pca(Vector3f::Zero());
  t.out_sizes = Vector2i::Ones() * 3;

  auto in = Image<Rgb32f>{3, 3};
  in(0, 0) = Vector3f::Ones() * 0; in(1, 0) = Vector3f::Ones() * 1; in(2, 0) = Vector3f::Ones() * 2;
  in(0, 1) = Vector3f::Ones() * 3; in(1, 1) = Vector3f::Ones() * 4; in(2, 1) = Vector3f::Ones() * 5;
  in(0, 2) = Vector3f::Ones() * 6; in(1, 2) = Vector3f::Ones() * 7; in(2, 2) = Vector3f::Ones() * 8;

  const auto out = t(in);
  const auto out_tensor = to_cwh_tensor(out);

  auto true_out_r = Image<float>{3, 3};
  true_out_r.matrix() <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8;

  EXPECT_MATRIX_EQ(true_out_r.matrix(), out_tensor[0].matrix());
}


TEST(TestDataAugmentation, test_compose_with_zooms)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{448, 238};

  const auto t = ImageDataTransform{};
  const auto z_ts = compose_with_zooms(in_sizes, out_sizes, 1 / 1.3f, 1.3f, 10, t);
  EXPECT_EQ(z_ts.size(), 10);
}

TEST(TestDataAugmentation, test_compose_with_shifts)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{448, 238};

  const auto t = ImageDataTransform{};
  const auto z_ts = compose_with_shifts(in_sizes, out_sizes, Vector2i::Ones(), t);
  EXPECT_EQ(z_ts.size(), 32*32);
}

TEST(TestDataAugmentation, test_compose_with_horizontal_flip)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{448, 238};

  const auto t = ImageDataTransform{};
  const auto z_ts = compose_with_horizontal_flip(t);
  EXPECT_EQ(z_ts.size(), 1);
}

TEST(TestDataAugmentation, test_compose_with_random_pca)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{448, 238};

  const auto std_dev = 0.5f;
  const auto num_samples = 10;

  const auto t = ImageDataTransform{};
  const auto z_ts = compose_with_random_fancy_pca(t, num_samples, std_dev);
  EXPECT_EQ(z_ts.size(), 10);
}

TEST(TestDataAugmentation, test_enumerate_image_data_transforms)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{448, 238};

  const auto t = ImageDataTransform{};
  const auto z_ts = enumerate_image_data_transforms(
      in_sizes, out_sizes, 1 / 1.3f, 1.3f, 10, Vector2i::Ones(), 10, 0.5f);
}


TEST(TestDataAugmentation, test_save_database_to_csv)
{
  EXPECT_TRUE(false);
}



int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
