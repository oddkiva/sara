// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/StdVectorHelpers.hpp>
#include <DO/Sara/ImageProcessing/DataAugmentation.hpp>


using namespace std;


namespace DO { namespace Sara {

  Image<Rgb32f> ImageDataTransform::operator()(const Image<Rgb32f>& in) const
  {
    auto out = extract_patch(in);

    if (apply_transform[FancyPCA])
      ColorFancyPCA{U, S}(out, alpha);

    return out;
  }


  VectorXf linspace(float a, float b, int num_samples)
  {
    auto range = VectorXf(num_samples);
    for (int i = 0; i < num_samples; ++i)
      range[i] = a + (b - a) * i / (num_samples - 1);
    return range;
  }

  VectorXf logspace(float a, float b, int num_samples)
  {
    return linspace(log(a), log(b), num_samples).array().exp().matrix();
  }

  auto compose_with_zooms(const Vector2i& in_image_sizes,
                          const Vector2i& out_image_sizes,
                          float zmin, float zmax, int num_scales,
                          const ImageDataTransform& parent_t,
                          bool ignore_zoom_factor_one)
      -> std::vector<ImageDataTransform>
  {
    const auto zs = logspace(zmin, zmax, num_scales);
    const auto z_image_sizes =
        (in_image_sizes.cast<float>() * zs.transpose());

    auto ts = vector<ImageDataTransform>{};
    for (int j = 0; j < num_scales; ++j)
      if (z_image_sizes.col(j).x() >= out_image_sizes.x() &&
          z_image_sizes.col(j).y() >= out_image_sizes.y())
      {
        if (j == num_scales / 2 && num_scales % 2 == 1 &&
            ignore_zoom_factor_one)
          continue;
        auto child_t = parent_t;
        child_t.set_zoom(zs[j]);
        ts.push_back(child_t);
      }

    return ts;
  }

  auto compose_with_shifts(const Vector2i& in_image_sizes,
                           const Vector2i& out_image_sizes,
                           const Vector2i& delta,
                           const ImageDataTransform& parent_t)
      -> std::vector<ImageDataTransform>
  {
    auto ts = vector<ImageDataTransform>{};
    for (int y = 0; y + out_image_sizes.y() < in_image_sizes.y();
         y += delta.y())
    {
      for (int x = 0; x + out_image_sizes.x() < in_image_sizes.y();
           x += delta.x())
      {
        auto child_t = parent_t;
        child_t.set_shift(Vector2i{x, y});
        ts.push_back(child_t);
      }
    }

    return ts;
  }

  auto compose_with_horizontal_flip(const ImageDataTransform& parent_t)
      -> std::vector<ImageDataTransform>
  {
    auto child_t = parent_t;
    child_t.set_flip(ImageDataTransform::Horizontal);
    return {child_t};
  };

  auto compose_with_random_fancy_pca(const ImageDataTransform& parent_t,
                                     int num_fancy_pca, float fancy_pca_std_dev)
      -> std::vector<ImageDataTransform>
  {
    for (int i = 0; i < num_fancy_pca; ++i)
    {
      auto alpha = Vector3f::Random();
    }
    return std::vector<ImageDataTransform>{};
  }

  auto enumerate_image_data_transforms(const Vector2i& in_sz,
                                       const Vector2i& out_sz,
                                       float zmin, float zmax, int num_scales,
                                       const Vector2i& delta,
                                       int num_fancy_pca_alpha,
                                       float fancy_pca_std_dev)
      -> std::vector<ImageDataTransform>
  {
    auto ts = std::vector<ImageDataTransform>{};

    // Put the identity data transform.
    auto t0 = ImageDataTransform{};
    t0.out_sizes = out_sz;
    ts.push_back(t0);

    // Fancy PCAs.
    const auto t1 =
        compose_with_random_fancy_pca(t0, num_fancy_pca_alpha, fancy_pca_std_dev);
    append(ts, t1);

    for (const auto& t1_i : t1)
    {
      // Zooms.
      const auto t2 = compose_with_zooms(in_sz, out_sz, zmin, zmax, num_scales, t1_i);
      append(ts, t2);

      // Shifts.
      for (const auto& t2_i : t2)
      {
        const auto t3 = compose_with_shifts(in_sz, out_sz, Vector2i::Ones(), t2_i);
        append(ts, t3);

        // Horiontal flip.
        for (const auto& t3_i : t3)
        {
          const auto t4 = compose_with_horizontal_flip(t3_i);
          append(ts, t4);
        }
      }
    }

    return ts;
  }

  auto augment_dataset(
      const std::vector<int>& data_indices,
      const Vector2i& in_sz, const Vector2i& out_sz,
      float zmin, float zmax, int num_scales,
      const Vector2i& delta,
      int num_fancy_pca, float fancy_pca_std_dev)
      -> std::vector<std::pair<int, ImageDataTransform>>
  {
    auto augmented_data = std::vector<std::pair<int, ImageDataTransform>>{};

    for (const auto i : data_indices)
    {
      const auto data_transforms = enumerate_image_data_transforms(
          in_sz, out_sz, zmin, zmax, num_scales, delta, num_fancy_pca,
          fancy_pca_std_dev);

      for (const auto t : data_transforms)
        augmented_data.push_back(make_pair(i, t));
    }
    return augmented_data;
  }

} /* namespace Sara */
} /* namespace DO */
