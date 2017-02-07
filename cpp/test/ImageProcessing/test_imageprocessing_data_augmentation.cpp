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


TEST(TestImageDataTransform, test_zoom)
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

TEST(TestImageDataTransform, test_shift)
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

TEST(TestImageDataTransform, test_flip)
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

TEST(TestImageDataTransform, test_fancy_pca)
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


TEST(TestDataTransformEnumeration, test_compose_with_zooms)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{480 - 32, 270 -32};

  const auto parent_t = ImageDataTransform{};

  const auto ts =
      compose_with_zooms(in_sizes, out_sizes, 1 / 1.3f, 1.3f, 10, parent_t);

  EXPECT_EQ(ts.size(), 6); // counted manually.

  for (const auto& t : ts)
  {
    const auto rescaled_image_sizes = (in_sizes.cast<float>() * t.z).eval();
    const auto out_sizes_f = out_sizes.cast<float>();

    ASSERT_MATRIX_EQ(rescaled_image_sizes.cwiseMax(out_sizes_f),
                     rescaled_image_sizes);
  }
}

TEST(TestDataTransformEnumeration, test_compose_with_shifts)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{480 - 32, 270 - 32};

  const auto parent_t = ImageDataTransform{};
  const auto ts = compose_with_shifts(in_sizes, out_sizes, Vector2i::Ones(), parent_t);

  EXPECT_EQ(ts.size(), 32 * 32);
  for (int y = 0; y < 32; ++y)
    for (int x = 0; x < 32; ++x)
      ASSERT_MATRIX_EQ(Vector2i(x, y), ts[y * 32 + x].t);
}

TEST(TestDataTransformEnumeration, test_compose_with_horizontal_flip)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{480 - 32, 270 - 32};

  const auto parent_t = ImageDataTransform{};
  const auto ts = compose_with_horizontal_flip(parent_t);
  EXPECT_EQ(ts.size(), 1);
  EXPECT_TRUE(ts[0].apply_transform[ImageDataTransform::Flip]);
  EXPECT_EQ(ts[0].flip_type, ImageDataTransform::Horizontal);
}

TEST(TestDataTransformEnumeration, test_compose_with_random_pca)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{448, 238};

  const auto std_dev = 0.5f;
  const auto num_samples = 10;

  const auto parent_t = ImageDataTransform{};
  const auto dist = NormalDistribution{};
  const auto ts =
      compose_with_random_fancy_pca(parent_t, num_samples, std_dev, dist);
  EXPECT_EQ(ts.size(), 10);
}


TEST(TestDataTransformEnumeration,
     test_single_scale_no_shift_no_flip_no_fancy_pca)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{480, 270};

  const struct
  {
    float min{1}, max{1};
    int size{1};
  } z_range;

  const struct
  {
    int num_samples{0};
    float std_dev{0.5f};
  } fancy_pca_params;

  const auto shift_delta = Vector2i::Ones();

  const auto randn = NormalDistribution{};

  const auto ts = enumerate_image_data_transforms(
      in_sizes, out_sizes,
      z_range.min, z_range.max, z_range.size,
      shift_delta,
      false,
      fancy_pca_params.num_samples, fancy_pca_params.std_dev, randn);

  EXPECT_EQ(1, ts.size());
  EXPECT_TRUE(ts[0].use_original);
  EXPECT_MATRIX_EQ(out_sizes, ts[0].out_sizes);
}


TEST(TestDataAugmentation, test_augment_database)
{
  const auto in_sizes = Vector2i{480, 270};
  const auto out_sizes = Vector2i{478, 268};

  const auto data_indices = [](int a = 0, int b = 10) {
    auto out = vector<int>(b - a);
    std::iota(out.begin(), out.end(), a);
    return out;
  }();

  const struct
  {
    float min{1 / 1.3f}, max{1.3f};
    int size{10};
  } z_range;

  const struct
  {
    int num_samples{0};
    float std_dev{0.5f};
  } fancy_pca_params;

  const auto shift_delta = Vector2i::Ones();

  const auto randn = NormalDistribution{};

  const auto augmented_data_set = augment_dataset(
      data_indices, in_sizes, out_sizes,
      z_range.min, z_range.max, z_range.size,
      Vector2i::Ones(),
      false,
      0, 0.5f,
      randn);

  EXPECT_EQ(augmented_data_set.size(), 1);
}



int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
