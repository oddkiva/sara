#pragma once

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Features/Feature.hpp>

#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>


namespace DO::Sara::v2 {

  struct FeatureTracker
  {
    ImagePyramidParams image_pyr_params = ImagePyramidParams(0);
    float sift_nn_ratio = 0.6f;
    std::size_t num_matches_max = 1000u;

    auto detect_features(const ImageView<float>& image,
                         KeypointList<OERegion, float>& keypoints) const -> void
    {
      keypoints = compute_sift_keypoints(image, image_pyr_params);
    }

    auto match_features(const KeypointList<OERegion, float>& src_keys,
                        const KeypointList<OERegion, float>& dst_keys) const
        -> std::vector<Match>
    {
      if (features(src_keys).empty() || features(dst_keys).empty())
        return {};

      auto matches = match(src_keys, dst_keys, sift_nn_ratio);
      if (matches.size() > num_matches_max)
        matches.resize(num_matches_max);

      return matches;
    }
  };

}  // namespace DO::Sara::v2
