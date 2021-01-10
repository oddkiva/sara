// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/LevelSets/FiniteDifferences.hpp>

#include "Utilities.hpp"

#include <iostream>
#include <numeric>
#include <set>


namespace sara = DO::Sara;

auto radial_distance(sara::Image<float>& phi, const Eigen::Vector2f& center)
{
  for (auto y = 0; y < phi.height(); ++y)
  {
    for (auto x = 0; x < phi.width(); ++x)
    {
      const auto xy = Eigen::Vector2f(x, y);
      phi(x, y) = (xy - center).norm();
    }
  }
}


enum State : std::uint8_t
{
  Alive = 0,
  Trial = 1,
  Far = 2,
  Forbidden = 3
};

struct CoordsValue
{
  Eigen::Vector2i coords;
  float val;

  inline auto operator<(const CoordsValue& other) const
  {
    return val < other.val;
  }
};


template <typename T>
auto solve_eikonal_equation_2d(const Eigen::Vector2i& x, const T fx,
                               const sara::Image<T>& u)
{
  const auto u0 = std::min(u(x - Eigen::Vector2i::Unit(0)),
                           u(x + Eigen::Vector2i::Unit(0)));
  const auto u1 = std::min(u(x - Eigen::Vector2i::Unit(1)),
                           u(x + Eigen::Vector2i::Unit(1)));

  const auto fx_inverse = 1 / (fx * fx);

  const auto a = 2.f;
  const auto b = -2 * (u0 + u1);
  const auto c = u0 * u0 + u1 * u1 - fx_inverse * fx_inverse;

  const auto delta = b * b - 4 * a * c;

  auto r0 = float{};
  if (delta >= 0)
    r0 = (-b + std::sqrt(delta)) / (2 * a);
  else
    r0 = std::min(u0, u1) + fx_inverse;

  return r0;
}


GRAPHICS_MAIN()
{
#define REAL_IMAGE
#ifdef REAL_IMAGE
  const auto image = sara::imread<float>(                       //
#ifdef __APPLE__
      "/Users/david/GitLab/DO-CV/sara/data/stinkbug.png"        //
#else
      "/home/david/GitLab/DO-CV/sara/data/stinkbug.png"        //
#endif
  );

  constexpr auto sigma = 3.f;
  const auto image_blurred = sara::gaussian(image, sigma);
  const auto laplacian = sara::laplacian(image_blurred);

  const auto grad = sara::gradient(image_blurred);
  const auto grad_mag =
      grad.cwise_transform([](const auto& v) { return v.squaredNorm(); });

  static_assert(std::is_same_v<decltype(image_blurred), decltype(grad_mag)>);

  // Create the speed function from the gradient magnitude.
  auto speed_times_dt = grad_mag.cwise_transform(
      [](const auto& v) { return std::exp(-v); }  //
  );

  // Extract the zero level set.
  const auto zeros = sara::extract_zero_level_set(laplacian);

  sara::create_window(image.sizes());
  sara::display(sara::color_rescale(laplacian));
#else
  const auto w = 512;
  const auto h = 512;
  auto image = sara::Image<float, 2>{w, h};
  radial_distance(image, Eigen::Vector2f(w, h) / 2);

  const auto grad = sara::gradient(image);
  const auto grad_mag =
      grad.cwise_transform([](const auto& v) { return v.norm(); });

  auto speed_times_dt = sara::Image<float, 2>{w, h};
  speed_times_dt.flat_array().fill(1);

  const auto zeros = std::vector{Eigen::Vector2i(w/2, h/2)};

  sara::create_window(image.sizes());
  sara::display(sara::color_rescale(image));
#endif


  for (const auto& p : zeros)
    sara::draw_point(p.x(), p.y(), sara::Red8);

  auto to_index = [&image](const Eigen::Vector2i& p) {
    return p.y() * image.width() + p.x();
  };
  // auto to_coords = [&image](const std::int32_t i) -> Eigen::Vector2i {
  //   const auto y = i / image.width();
  //   const auto x = i - y * image.width();
  //   return {x, y};
  // };


  // Fast marching.
  sara::tic();
  auto states = sara::Image<State>{image.sizes()};
  auto distances = sara::Image<float>{image.sizes()};
  auto predecessors = sara::Image<std::int32_t>{image.sizes()};
  states.flat_array().fill(State::Far);
  distances.flat_array().fill(std::numeric_limits<float>::max());
  predecessors.flat_array().fill(-1);

  const auto deltas = std::array<Eigen::Vector2i, 8>{
      Eigen::Vector2i{-1, 0},  //
      Eigen::Vector2i{+1, 0},  //
      Eigen::Vector2i{0, -1},  //
      Eigen::Vector2i{0, +1},   //
      Eigen::Vector2i{-1, -1},  //
      Eigen::Vector2i{-1, +1},  //
      Eigen::Vector2i{+1, -1},  //
      Eigen::Vector2i{+1, +1}   //
  };

  // Initialize the alive set from the zero level set.
  for (const auto& p : zeros)
  {
    states(p) = State::Alive;
    distances(p) = 0;
  }

  // Initialize the trial set.
  auto trial_set = std::multiset<CoordsValue>{};
  for (const auto& p : zeros)
  {
    for (const auto& delta : deltas)
    {
      const Eigen::Vector2i n = p + delta;
      if (n.x() < 0 || n.x() >= image.width() ||  //
          n.y() < 0 || n.y() >= image.height())
        continue;

      if (states(n) == State::Alive || states(n) == State::Forbidden)
        continue;

      states(n) = State::Trial;
      distances(n) = speed_times_dt(n);
      predecessors(n) = to_index(p);

      trial_set.insert({n, distances(n)});
    }
  }

  auto increase_priority = [&trial_set, &distances](const Eigen::Vector2i& p,
                                                    float val) {
    if (val < distances(p))
    {
      const auto p_it = trial_set.find({p, distances(p)});
      if (p_it != trial_set.end() && p_it->coords == p)
        trial_set.erase(p_it);
      trial_set.insert({p, val});
    }
  };


  // Propagate the wavefront.
  while (!trial_set.empty())
  {
    // Extract the closest trial point.
    const auto p = trial_set.begin()->coords;
    trial_set.erase(trial_set.begin());

    //  sara::draw_point(p.x(), p.y(), sara::Green8);
    if (states(p) == State::Alive)
    {
      std::cout << "OOPS!!!" << std::endl;
      continue;
    }

    // Update the neighbors.
    for (const auto& delta : deltas)
    {
      const Eigen::Vector2i n = p + delta;
      if (n.x() < 1 || n.x() >= image.width() -1 ||  //
          n.y() < 1 || n.y() >= image.height() -1)
        continue;

      if (states(n) == State::Alive || states(n) == State::Forbidden)
        continue;

      // At this point, a neighbor is either `Far` or `Trial` now.
      //
      // Update its distance value in both cases.
      const auto new_dist_n = solve_eikonal_equation_2d(n, speed_times_dt(n), distances);
      if (new_dist_n < distances(n))
      {
        distances(n) = new_dist_n;
        predecessors(n) = to_index(p);
      }

      if (states(n) == State::Far)
      {
        // Update its state.
        states(n) = State::Trial;

        // Insert it into the list of trial points.
        trial_set.insert({n, distances(n)});
      }

      // Increase the priority of the point if necessary.
      if (states(n) == State::Trial)
        increase_priority(n, distances(n));
    }
  }
  sara::toc("Fast Marching 2D");

  for (auto y = 0; y < distances.height(); ++y)
    for (auto x = 0; x < distances.width(); ++x)
      if (distances(x, y) == std::numeric_limits<float>::max())
        distances(x, y) = 0;

  auto d = distances.flat_array();
  const auto dmax = d.maxCoeff();
  d /= dmax;
  SARA_CHECK(dmax);
  sara::display(distances);
  sara::get_key();

  return 0;
}
