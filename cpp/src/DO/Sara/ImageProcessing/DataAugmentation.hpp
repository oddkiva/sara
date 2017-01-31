#pragma once

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/ImageProcessing/Scaling.hpp>


namespace DO { namespace Sara {

  struct ImageDataTransform
  {
    enum TransformType
    {
      Zoom = 0,
      Shift = 1,
      Flip = 2,
      FancyPCA = 3,
      NumTransformTypes = 4
    };

    Image<Rgb32f> operator()(const Image<Rgb32f>& in) const
    {
      if (use_original)
        return reduce(in, out_sizes);

      // 1. Zoom
      auto out = in; 
      if (apply_transform[Zoom])
      {
        if (z < 1)
          out = reduce(in, 1 / z);
        else
          out = enlarge(in, z);
      }

      // 2. Shift
      if (apply_transform[Shift])
        out = crop(out, t, out_sizes);

      if (apply_transform[Flip])
        flip(out, flip_type);

      if (apply_transform[FancyPCA])
        out = ColorFancyPCA{U, S}(out, alpha);

      return out;
    }

    //! Final size.
    Vector2i out_sizes;

    //! Use the original image.
    bool use_original;
    //! If not use, 
    std::array<bool, NumTransformTypes> apply_transform;

    //! @{
    //! \brief Geometric transformation.
    //! Zoom factor.
    float z;
    //! Rotation angle.
    float theta;
    //! Translation vector.
    Vector2i t;
    //! Flip type.
    FlipType flip_type;
    //! @}

    //! @{
    //! Color perturbation.
    Matrix3f U;
    Matrix3f S;
    Vector3f alpha; 
    //! @}
  };


  VectorXf linspace(float a, float b, int num_samples)
  {
    auto range = VectorXf{num_samples};
    for (int i = 0; i <= num_samples; ++i)
      range[i] = a + i / (b - a) * (num_samples - 1);
    return range;
  }

  VectorXf logspace(float a, float b, int num_samples)
  {
    return linspace(log(a), log(b), num_samples).array().exp().matrix();
  }

  auto augment(const VectorXf& zs,
               const VectorXf& thetas,
               std::pair<int, int> offset_delta,
               std::array<bool, 2> flip)
      -> std::vector<ImageDataTransform>
  {
    return std::vector<ImageDataTransform>{};
  }


} /* namespace Sara */
} /* namespace DO */
