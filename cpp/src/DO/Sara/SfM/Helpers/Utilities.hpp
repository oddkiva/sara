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

#pragma once

#include <DO/Sara/SfM/Graph/FeatureTracker.hpp>

#include <filesystem>


namespace DO::Sara {

  auto
  draw_feature_tracks(ImageView<Rgb8>&,        //
                      const CameraPoseGraph&,  //
                      const FeatureGraph&,     //
                      const CameraPoseGraph::Vertex pose_vertex,
                      const std::vector<FeatureTracker::Track>& tracks,
                      const std::vector<std::size_t>& track_visibility_count,
                      const float scale = 1.f) -> void;


  auto
  write_point_correspondences(const CameraPoseGraph&, const FeatureGraph&,
                              const std::vector<FeatureTracker::Track>& tracks,
                              const std::filesystem::path& out_csv_fp) -> void;

}  // namespace DO::Sara
