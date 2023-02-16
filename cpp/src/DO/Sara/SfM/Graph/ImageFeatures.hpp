#pragma once

#include <DO/Sara/Features/Feature.hpp>

#include <span>


namespace DO::Sara {

  struct ImageKeypoints
  {
    auto num_images() const -> int
    {
      return _num_images;
    };

    auto descriptor_dimension() const -> int
    {
      return _descriptor_dimension;
    }

    auto features(const int camera_view_id) const -> std::span<const OERegion>;

    auto descriptors(const int camera_view_id) const
        -> TensorView_<const float, 2>;

    //! @brief Feature indexing data in the form of (begin, end).
    std::vector<std::pair<int, int>> _feature_range_per_image;

    //! @brief Feature data.
    //! @{
    std::vector<std::vector<OERegion>> _features;
    std::vector<Eigen::MatrixXf> _descriptors;
    //! @}

    //! @brief Additional metadata per feature.
    std::vector<int> _camera_view_id;
    std::vector<int> _feature_index;

    int _descriptor_dimension = -1;
    int _num_images = -1;
  };

}  // namespace DO::Sara
