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

#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/Graphics/ImageDraw.hpp>
#include <DO/Sara/SfM/Odometry/ImageDistortionCorrector.hpp>
#include <DO/Sara/SfM/Odometry/VideoStreamer.hpp>
#include <DO/Sara/Visualization/Features/Draw.hpp>

#include <DO/Sara/SfM/BuildingBlocks/RelativePoseEstimator.hpp>
#include <DO/Sara/SfM/Graph/CameraPoseGraph.hpp>
#include <DO/Sara/SfM/Graph/FeatureTracker.hpp>

namespace DO::Sara::v2 {

  class OdometryPipeline
  {
  public:
    auto set_config(const std::filesystem::path& video_path,
                    const v2::BrownConradyDistortionModel<double>& camera)
        -> void;

    auto read() -> bool;

    auto process() -> void;

    auto make_display_frame() const -> Image<Rgb8>;

  private: /* computer vision tasks */
    auto detect_keypoints(const ImageView<float>&) const
        -> KeypointList<OERegion, float>;

    auto estimate_relative_pose(const CameraPoseGraph::Vertex u,
                                const CameraPoseGraph::Vertex v) const
        -> std::pair<RelativePoseData, TwoViewGeometry>;

  private: /* graph update tasks */
    auto add_camera_pose_and_grow_point_cloud() -> bool;

  private: /* data members */
    VideoStreamer _video_streamer;
    v2::BrownConradyDistortionModel<double> _camera;

    std::unique_ptr<ImageDistortionCorrector> _distortion_corrector;
    v2::RelativePoseEstimator _relative_pose_estimator;


    //! @brief SfM data.
    //! @{
    FeatureParams _feature_params;
    FeatureTracker _feature_tracker;
    CameraPoseGraph _pose_graph;
    //! @}

    //! @brief SfM state.
    //! @{
    CameraPoseGraph::Vertex _pose_prev;
    CameraPoseGraph::Vertex _pose_curr;
    CameraPoseGraph::Edge _relative_pose_edge;
    FeatureTracker::TrackArray _tracks_alive;
    FeatureTracker::TrackVisibilityCountArray _track_visibility_count;
    Eigen::Matrix3d _current_global_rotation = Eigen::Matrix3d::Identity();
    //! @}
  };

}  // namespace DO::Sara::v2
