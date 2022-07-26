#pragma once

#include <DO/Sara/ImageProcessing.hpp>


template <typename T>
struct Corner
{
  Eigen::Vector2<T> coords;
  float score;

  inline auto position() const -> const Eigen::Vector2<T>&
  {
    return coords;
  }

  inline auto operator<(const Corner& other) const -> bool
  {
    return score < other.score;
  }
};

// Select the local maxima of the cornerness functions.
auto select(const DO::Sara::ImageView<float>& cornerness,
            const float cornerness_adaptive_thres, const int border)
    -> std::vector<Corner<int>>;
