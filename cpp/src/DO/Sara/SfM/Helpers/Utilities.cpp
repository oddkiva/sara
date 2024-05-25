// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/SfM/Helpers/Utilities.hpp>

#include <DO/Sara/Graphics/ImageDraw.hpp>
#include <DO/Sara/Visualization/Features/Draw.hpp>

#include <fmt/format.h>

#include <fstream>


namespace DO::Sara {

  auto draw_feature_tracks(
      DO::Sara::ImageView<Rgb8>& display,  //
      const CameraPoseGraph& pgraph,       //
      const FeatureGraph& fgraph,          //
      const CameraPoseGraph::Vertex pose_u,
      const std::vector<FeatureTracker::Track>& tracks_alive,
      [[maybe_unused]] const std::vector<std::size_t>& track_visibility_count,
      const float scale) -> void
  {
    for (auto t = 0u; t < tracks_alive.size(); ++t)
    {
      const auto& track = tracks_alive[t];

      // Just to double check.
      // if (track_visibility_count[t] < 3u)
      //   continue;

      auto p = std::array<Eigen::Vector2f, 2>{};
      auto r = std::array<float, 2>{};
      {
        const auto& [pose_v0, k] = fgraph[track[0]];
        const auto& f0 = features(pgraph[pose_v0].keypoints)[k];
        p[0] = f0.center();
        r[0] = std::max(f0.radius() * scale, 1.f);
        const auto color =
            pose_v0 == pose_u ? Rgb8{0, 255, 255} : Rgb8{0, 127, 127};
        draw_circle(display, p[0], r[0], color, 3);
      }

      for (auto v1 = 1u; v1 < track.size(); ++v1)
      {
        const auto& [pose_v1, k] = fgraph[track[v1]];
        const auto& f1 = features(pgraph[pose_v1].keypoints)[k];
        p[1] = f1.center();
        r[1] = std::max(f1.radius() * scale, 1.f);
        const auto color =
            pose_v1 == pose_u ? Rgb8{255, 0, 0} : Rgb8{0, 127, 127};
        draw_circle(display, p[1], r[1], color, 3);

        draw_arrow(display, p[0], p[1], color, 3);

        p[0] = p[1];
        r[0] = r[1];
      }
    }
  }

  auto write_point_correspondences(
      const CameraPoseGraph& pgraph,  //
      const FeatureGraph& fgraph,     //
      const std::vector<FeatureTracker::Track>& tracks_alive,
      const std::filesystem::path& out_csv_fp) -> void
  {
    std::ofstream out{out_csv_fp.string()};
    if (!out)
      throw std::runtime_error{"Cannot create output CSV!"};

    for (auto t = 0u; t < tracks_alive.size(); ++t)
    {
      const auto& track = tracks_alive[t];

      auto p = std::array<Eigen::Vector2f, 2>{};

      const auto& [pose_v0, k0] = fgraph[track.front()];
      const auto& f0 = features(pgraph[pose_v0].keypoints)[k0];
      p[0] = f0.center();

      const auto& [pose_v1, k1] = fgraph[track.back()];
      const auto& f1 = features(pgraph[pose_v1].keypoints)[k1];
      p[1] = f1.center();

      out << fmt::format("{},{},{},{}\n", p[0].x(), p[0].y(), p[1].x(),
                         p[1].y());
    }
  }

}  // namespace DO::Sara
