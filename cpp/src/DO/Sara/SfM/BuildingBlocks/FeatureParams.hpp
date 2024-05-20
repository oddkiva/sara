#pragma once

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>


namespace DO::Sara {

  struct FeatureParams
  {
    ImagePyramidParams image_pyr_params = ImagePyramidParams(-1);
    float sift_nn_ratio = 0.6f;
    std::size_t num_matches_max = 1000u;
    int num_inliers_min = 100;
  };

}  // namespace DO::Sara
