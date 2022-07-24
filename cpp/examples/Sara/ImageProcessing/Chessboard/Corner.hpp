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
inline auto select(const DO::Sara::ImageView<float>& cornerness,
                   const float cornerness_adaptive_thres, const int border)
    -> std::vector<Corner<int>>
{
  namespace sara = DO::Sara;

  const auto extrema = sara::local_maxima(cornerness);

  const auto cornerness_max = cornerness.flat_array().maxCoeff();
  const auto cornerness_thres = cornerness_adaptive_thres * cornerness_max;

  auto extrema_filtered = std::vector<Corner<int>>{};
  extrema_filtered.reserve(extrema.size());
  for (const auto& p : extrema)
  {
    const auto in_image_domain =
        border <= p.x() && p.x() < cornerness.width() - border &&  //
        border <= p.y() && p.y() < cornerness.height() - border;
    if (in_image_domain && cornerness(p) > cornerness_thres)
      extrema_filtered.push_back({p, cornerness(p)});
  }

  return extrema_filtered;
};
