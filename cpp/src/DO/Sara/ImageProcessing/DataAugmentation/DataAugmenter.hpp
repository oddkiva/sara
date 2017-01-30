#pragma once

#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  struct TransformParameter
  {
    Image<Rgb32f> operator()(const Image<Rgb32f>&) const;

    void set_cache(Image<Rgb32f> *cache);

    Image<Rgb32f> *cache();

  private:
    Image<Rgb32f> *_cache = nullptr;

  private:
    enum FlipType { Vertical, Horizontal };

    bool use_original;

    float z;

    float theta;

    FlipType flip_type;

    Vector2i crop_offset;
    Vector2i crop_sizes;

    Vector3f fancy_pca_alpha; 

    std::array<bool, 5> apply_transform;
  };


  VectorXf linspace(float a, float b, int num_samples)
  {
    auto range = VectorXf{num_samples};
    for (int i = 0; i < num_samples; ++i)
      range[i] = a + i / (b-a) * num_samples;
  }

  VectorXf log_linspace(float max, int num_samples)
  {
    return linspace(log(-max), log(max), num_samples).exp();
  }

  auto augment(const VectorXf& zs,
               const VectorXf& thetas,
               std::pair<int, int> offset_delta,
               std::array<bool, 2> flip)
      -> std::vector<TransformParameter>
  {
    return std::vector<TransformParameter>{};
  }


} /* namespace Sara */
} /* namespace DO */
