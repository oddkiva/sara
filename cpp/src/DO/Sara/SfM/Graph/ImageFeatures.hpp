#pragma once

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Features/Feature.hpp>
#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>

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

    auto features(const std::size_t camera_vertex) const
        -> std::span<const OERegion>;

    auto descriptors(const std::size_t camera_vertex) const
        -> TensorView_<const float, 2>;

    //! @brief Feature data.
    //! @{
    std::vector<std::vector<OERegion>> _features;
    std::vector<Eigen::MatrixXf> _descriptors;
    //! @}

    int _descriptor_dimension = -1;
    int _num_images = -1;
  };

}  // namespace DO::Sara
