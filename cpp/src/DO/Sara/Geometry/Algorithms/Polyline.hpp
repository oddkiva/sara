// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Image.hpp>


namespace DO::Sara {

  //! @brief Calculate the length of a polyline.
  template <typename T, int N>
  auto length(const std::vector<Point<T, N>>& polylines)
  {
    if (polylines.size() < 2)
      throw std::runtime_error{"Ill-formed polyline!"};

    auto sum = T{};
    for (auto i = 0u; i < polylines.size() - 1; ++i)
      sum += (polylines[i + 1] - polylines[i]).norm();
    return sum;
  }

  //! @brief Calculate the linear directional mean of the polyline.
  /*!
   *  The linear directional mean is the mean orientation reweighted by the
   *  length of each line segment in the polyline.
   */
  template <typename T, int N>
  auto linear_directional_mean(const std::vector<Point<T, N>>& polylines)
  {
    if (polylines.size() < 2)
      throw std::runtime_error{"Ill-formed polyline!"};

    auto sine = float{};
    auto cosine = float{};
    for (auto i = 0u; i < polylines.size() - 1; ++i)
    {
      sine += (polylines[i + 1] - polylines[i]).y();
      cosine += (polylines[i + 1] - polylines[i]).x();
    }

    return std::atan2(sine, cosine);
  }

  //! @brief Calculate the center of mass of a polyline.
  template <typename T, int N>
  auto center_of_mass(const std::vector<Point<T, N>>& p)
  {
    if (p.size() < 2)
      throw std::runtime_error{"Ill-formed polyline!"};

    auto length = T{};
    Point<T, N> center = Point<T, N>::Zero();


    for (auto i = 0u; i < p.size() - 1; ++i)
    {
      const auto& a = p[i];
      const auto& b = p[i + 1];
      const auto li = (b - a).norm();
      const auto ci = (a + b) / 2;

      length += li;
      center += ci * li;
    }
    center /= length;

    return center;
  }

  //! @brief Calculate the matrix of inertia of a polyline.
  template <typename T>
  auto matrix_of_inertia(const std::vector<Point<T, 2>>& p, const Point<T, 2>& center_of_mass)
  {
    if (p.size() < 2)
      throw std::runtime_error{"Ill-formed polyline!"};

    auto length = T{};
    Eigen::Matrix<T, 2, 2> m = Eigen::Matrix<T, 2, 2>::Zero();

    const auto& c = center_of_mass;
    const auto& x2 = c.x() * c.x();
    const auto& y2 = c.y() * c.y();
    const auto& xy = c.x() * c.y();

    for (auto i = 0u; i < p.size() - 1; ++i)
    {
      const auto& a = p[i];
      const auto& b = p[i + 1];
      const auto& li = (b - a).norm();

      const auto& xi_2 = a.x() * a.x() + b.x() * b.x();
      const auto& yi_2 = a.y() * a.y() + b.y() * b.y();
      const auto& xi_yi = a.x() * a.y() + b.x() * b.y();

      // First row.
      m(0, 0) += (xi_2 - 2 * x2) * li;
      m(0, 1) += (xi_yi - 2 * xy) * li;

      // Second row.
      m(1, 0) += (xi_yi - 2 * xy) * li;
      m(1, 1) += (yi_2 - 2 * y2) * li;

      length += li;
    }

    m /= 2 * length;

    return m;
  }


  //! @brief Simplify a polyline by reducing each point clusters to a single
  //! point.
  template <typename T, int N>
  auto collapse(const std::vector<Point<T, N>>& p,  //
                const ImageView<float>& gradient,   //
                float threshold = 5e-2f,            //
                bool adaptive = true)               //
      -> std::vector<Eigen::Matrix<T, N, 1>>
  {
    if (p.size() < 2)
      throw std::runtime_error{"Invalid polyline!"};

    using Point = Eigen::Matrix<T, N, 1>;

    auto deltas = std::vector<Point>(p.size() - 1);
    for (auto i = 0u; i < deltas.size(); ++i)
      deltas[i] = p[i + 1] - p[i];

    auto lengths = std::vector<T>(deltas.size());
    std::transform(deltas.begin(), deltas.end(), lengths.begin(),
                   [](const auto& d) { return d.norm(); });

    if (adaptive)
    {
      const auto total_length =
          std::accumulate(lengths.begin(), lengths.end(), T{});
      for (auto& l : lengths)
        l /= total_length;
    }

    // Find the cuts.
    auto collapse_state = std::vector<std::uint8_t>(p.size(), 0);
    for (auto i = 0u; i < lengths.size(); ++i)
    {
      if (lengths[i] < threshold)
      {
        collapse_state[i] = 1;
        collapse_state[i + 1] = 1;
      }
    }

    auto p_collapsed = std::vector<Point>{};
    p_collapsed.reserve(p.size());
    for (auto i = 0u; i < p.size();)
    {
      if (collapse_state[i] == 0)
      {
        p_collapsed.push_back(p[i]);
        ++i;
        continue;
      }

      const auto& a = i;
      const auto b =
          std::find(collapse_state.begin() + a, collapse_state.end(), 0) -
          collapse_state.begin();

      const auto pa = p.begin() + a;
      const auto pb = p.begin() + b;

      const auto best =
          std::max_element(pa, pb, [&gradient](const auto& u, const auto& v) {
            return gradient(u.template cast<int>()) <
                   gradient(v.template cast<int>());
          });
      p_collapsed.emplace_back(*best);

      i = b;
    }

    return p_collapsed;
  }


  //! @brief Split the polyline into smaller and straighter polylines.
  //! @{
  template <typename T, int N>
  auto split(const std::vector<Point<T, N>>& ordered_points,
             const T angle_threshold = static_cast<T>(M_PI / 6))
      -> std::vector<std::vector<Point<T, N>>>
  {
    if (angle_threshold >= M_PI)
      throw std::runtime_error{"Invalid angle threshold!"};

    if (ordered_points.size() < 3)
      return {ordered_points};

    const auto& p = ordered_points;
    const auto& cos_threshold = std::cos(angle_threshold);

    // Calculate the orientation of line segments.
    auto deltas = std::vector<Point<T, N>>(p.size() - 1);
    for (auto i = 0u; i < deltas.size(); ++i)
      deltas[i] = (p[i + 1] - p[i]).normalized();

    // Find the cuts.
    auto cuts = std::vector<std::uint8_t>(p.size(), 0);
    for (auto i = 0u; i < deltas.size() - 1; ++i)
    {
      const auto cosine = deltas[i].dot(deltas[i + 1]);
      // a, b, c
      // a = p[i]
      // b = p[i + 1]
      // c = p[i + 2]
      cuts[i + 1] = cosine < cos_threshold;
    }

    auto pp = std::vector<std::vector<Point<T, N>>>{};
    pp.push_back({p[0]});
    for (auto i = 1u; i < p.size(); ++i)
    {
      pp.back().push_back(p[i]);
      if (cuts[i] == 1)
        pp.push_back({p[i]});
    }

    return pp;
  }

  template <typename T, int N>
  auto split(const std::vector<std::vector<Point<T, N>>>& edges,
             const T angle_threshold = static_cast<T>(M_PI / 6))
      -> std::vector<std::vector<Point<T, N>>>
  {
    auto edges_split = std::vector<std::vector<Point<T, N>>>{};
    edges_split.reserve(2 * edges.size());
    for (const auto& e : edges)
      append(edges_split, split(e, angle_threshold));

    return edges_split;
  }
  //! @}

}  // namespace DO::Sara
