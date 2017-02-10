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

#include <array>
#include <vector>

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/ImageProcessing/ColorFancyPCA.hpp>
#include <DO/Sara/ImageProcessing/ColorJitter.hpp>
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
      None,
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
    // ! @{
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

    void set_shift(const Vector2i& t)
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

    void set_fancy_pca(const Vector3f& alpha)
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

    DO_SARA_EXPORT
    Image<Rgb32f> operator()(const Image<Rgb32f>& in) const;
    //! @}

    inline bool operator==(const ImageDataTransform& other) const
    {
      return out_sizes == other.out_sizes &&
        use_original == other.use_original &&
        apply_transform == other.apply_transform &&
        z == other.z &&
        theta && other.theta &&
        t == other.t &&
        flip_type == other.flip_type &&
        U == other.U &&
        S == other.S &&
        alpha == other.alpha;
    }

    inline bool operator!=(const ImageDataTransform& other) const
    {
      return !(*this == other);
    }

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


  DO_SARA_EXPORT
  auto compose_with_zooms(const Vector2i& in_image_sizes,
                          const Vector2i& out_image_sizes,
                          float zmin, float zmax, int num_scales,
                          const ImageDataTransform& parent_t,
                          bool ignore_zoom_factor_one = true)
      -> std::vector<ImageDataTransform>;

  DO_SARA_EXPORT
  auto compose_with_shifts(const Vector2i& in_image_sizes,
                           const Vector2i& out_image_sizes,
                           const Vector2i& delta,
                           const ImageDataTransform& parent_t)
      -> std::vector<ImageDataTransform>;

  DO_SARA_EXPORT
  auto compose_with_horizontal_flip(const ImageDataTransform& parent_t)
      -> std::vector<ImageDataTransform>;

  DO_SARA_EXPORT
  auto compose_with_random_fancy_pca(const ImageDataTransform& parent_t,
                                     int num_fancy_pca, float fancy_pca_std_dev,
                                     const NormalDistribution& randn)
      -> std::vector<ImageDataTransform>;

  DO_SARA_EXPORT
  auto enumerate_image_data_transforms(
      const Vector2i& in_image_sizes, const Vector2i& out_image_sizes,
      bool zoom, float zmin, float zmax, int num_z,
      bool shift, const Vector2i& delta,
      bool flip,
      bool fancy_pca, int num_fancy_pca, float fancy_pca_std_dev,
      const NormalDistribution& dist)
      -> std::vector<ImageDataTransform>;

  DO_SARA_EXPORT
  auto augment_data(const std::vector<int>& data_indices,
                    const Vector2i& in_sz, const Vector2i& out_sz,
                    bool zoom, float zmin, float zmax, int num_scales,
                    bool shift, const Vector2i& delta,
                    bool flip,
                    bool fancy_pca, int num_fancy_pca, float fancy_pca_std_dev,
                    const NormalDistribution& randn)
      -> std::vector<std::pair<int, ImageDataTransform>>;

} /* namespace Sara */
} /* namespace DO */
