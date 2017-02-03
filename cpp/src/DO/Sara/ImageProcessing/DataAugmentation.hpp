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

//! @file

#pragma once

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/ImageProcessing/ColorFancyPCA.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>


namespace DO { namespace Sara {

  struct ImageDataTransform
  {
  public:
    //! @{
    //! @brief Transform types.
    enum FlipType
    {
      Horizontal,
      Vertical
    };

    enum TransformType
    {
      Zoom = 0,
      Shift = 1,
      Flip = 2,
      FancyPCA = 3,
      NumTransformTypes = 4
    };
    //! @}


    // ===================================================================== //
    //! @{
    //! @brief Transform setters.
    void set_zoom(float z)
    {
      use_original = false;
      this->apply_transform[Zoom] = true;
      this->z = z;
    }

    void unset_zoom()
    {
      this->apply_transform[Zoom] = false;
    }

    void set_shift(Vector2i t)
    {
      use_original = false;
      this->apply_transform[Shift] = true;
      this->t = t;
    }

    void set_flip(FlipType flip_type)
    {
      use_original = false;
      this->apply_transform[Flip] = true;
      this->flip_type = flip_type;
    }

    void unset_flip()
    {
      this->apply_transform[Flip] = false;
    }

    void set_fancy_pca(Vector3f alpha)
    {
      use_original = false;
      this->apply_transform[FancyPCA] = true;
      this->alpha = alpha;
    }

    void unset_fancy_pca()
    {
      this->apply_transform[FancyPCA] = false;
    }
    //! @}


    // ===================================================================== //
    //! @{
    //! @brief Transform the image data.
    template <typename T>
    Image<T> extract_patch(const Image<T>& in) const
    {
      if (use_original)
        return reduce(in, out_sizes);

      auto out = in;
      if (apply_transform[Zoom])
      {
        if (z < 1)
          out = reduce(in, 1 / z);
        else
          out = enlarge(in, z);
      }

      out = crop(out, t, t + out_sizes);

      if (apply_transform[Flip])
      {
        if (flip_type == Horizontal)
          flip_horizontally(out);
        else
          flip_vertically(out);
      }

      return out;
    }

    Image<Rgb32f> operator()(const Image<Rgb32f>& in) const
    {
      auto out = extract_patch(in);

      if (apply_transform[FancyPCA])
        ColorFancyPCA{U, S}(out, alpha);

      return out;
    }
    //! @}


    // ===================================================================== //
    // Parameters
    //
    //! @{ Final size.
    Vector2i out_sizes;

    //! @{ Use the original image.
    bool use_original{true};
    //! If not use,
    std::array<bool, NumTransformTypes> apply_transform{
        {false, false, false, false}};

    //! @{
    //! @brief Zoom factor.
    float z{1.f};
    //! @brief Rotation angle.
    float theta{0.f};
    //! Translation vector.
    Vector2i t{Vector2i::Zero()};
    //! @brief Flip type.
    FlipType flip_type{Horizontal};
    //! @}

    //! @{
    //! @brief Color fancy PCA parameters.
    Matrix3f U{Matrix3f::Identity()};
    Vector3f S{Vector3f::Ones()};
    Vector3f alpha{Vector3f::Zero()};
    //! @}
  };


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


  auto expand_zoom_transforms(const Vector2i& in_image_sizes,
                              const Vector2i& out_image_sizes, float zmin,
                              float zmax, int num_samples)
      -> std::vector<ImageDataTransform>;

  auto expand_crop_transforms(const Vector2i& in_image_sizes,
                              const Vector2i& out_image_sizes,
                              int delta_x, int delta_y)
      -> std::vector<ImageDataTransform>;

} /* namespace Sara */
} /* namespace DO */
