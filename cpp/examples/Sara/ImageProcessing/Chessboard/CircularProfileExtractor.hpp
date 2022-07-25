#pragma once

#include <cmath>
#include <vector>

#include <DO/Sara/Core.hpp>


namespace sara = DO::Sara;


struct CircularProfileExtractor
{
  CircularProfileExtractor();

  // Sample a unit circle centered at the origin.
  auto initialize_circle_sample_points() -> void;

  auto operator()(const sara::ImageView<float>& image,
                  const Eigen::Vector2d& center) const -> Eigen::ArrayXf;

  int num_circle_sample_points = 36;
  double circle_radius = 10.;
  std::vector<Eigen::Vector2d> circle_sample_points;
};


inline auto dir(const float angle) -> Eigen::Vector2f
{
  return Eigen::Vector2f{std::cos(angle), std::sin(angle)};
};

auto localize_zero_crossings(const Eigen::ArrayXf& profile, int num_bins)
    -> std::vector<float>;
