#pragma once

#include <DO/Sara/Core/Image.hpp>

#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>


namespace DO::Sara {

  struct FeatureTracker
  {
    float sift_nn_ratio = 0.6f;
    ImagePyramidParams image_pyr_params = ImagePyramidParams(0);

    const ImageView<float>& _frame;
    std::array<KeypointList<OERegion, float>, 2> keys;
    std::vector<Match> matches;

    FeatureTracker(const ImageView<float>& frame)
      : _frame{frame}
    {
    }

    auto detect_features() -> void
    {
      print_stage("Computing keypoints...");
      std::swap(keys[0], keys[1]);
      keys[1] = compute_sift_keypoints(_frame, image_pyr_params);
    }

    auto match_features() -> void
    {
      print_stage("Matching keypoints...");
      if (features(keys[0]).empty() || features(keys[1]).empty())
      {
        SARA_DEBUG << "Skipping...\n";
        return;
      }

      matches = match(keys[0], keys[1], sift_nn_ratio);
      // Put a hard limit of 1000 matches to scale.
      if (matches.size() > 1000)
        matches.resize(1000);
    }
  };

}  // namespace DO::Sara
