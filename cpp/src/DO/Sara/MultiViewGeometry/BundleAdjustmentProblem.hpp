// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>

#include <vector>


namespace DO::Sara {

struct ObservationRef
{
  FeatureGID gid;
  int camera_id;
  int point_id;
};

struct BundleAdjustmentProblem
{
  //! @brief observation = 2D points in images.
  Tensor_<double, 2> observations;
  //! @brief the corresponding 3D point index for observation 'o'
  std::vector<int> point_indices;
  //! @brief the corresponding 3D camera index for observation 'o'
  std::vector<int> camera_indices;

  //! @{
  //! @brief camera parameters + 3D point coordinates.
  int num_cameras;
  int num_points;
  int camera_dof;
  std::vector<double> parameters;
  TensorView_<double, 2> points_abs_coords_3d;
  TensorView_<double, 2> camera_parameters;
  //! @}

  auto resize(int num_observations,  //
              int num_points_, int num_cameras_, int camera_dof_)
  {
    SARA_DEBUG << "Resizing data..." << std::endl;
    observations = Tensor_<double, 2>{{num_observations, 2}};
    point_indices = std::vector<int>(num_observations);
    camera_indices = std::vector<int>(num_observations);

    num_points = num_points_;
    num_cameras = num_cameras_;
    camera_dof = camera_dof_;
    const auto num_parameters = num_cameras * camera_dof + num_points * 3;
    parameters = std::vector<double>(num_parameters);

    auto points_abs_coords_3d_new = TensorView_<double, 2>{
        parameters.data() + 9 * num_cameras, {num_points, 3}};
    auto camera_parameters_new =
        TensorView_<double, 2>{parameters.data(), {num_cameras, camera_dof}};

    points_abs_coords_3d.swap(points_abs_coords_3d_new);
    camera_parameters.swap(camera_parameters_new);
  }

  auto populate_observations(
      const std::vector<ObservationRef>& obs_refs,
      const std::vector<KeypointList<OERegion, float>>& keypoints) -> void
  {
    SARA_DEBUG << "Populating observations..." << std::endl;
    const auto num_observations = observations.size(0);
    for (int i = 0; i < num_observations; ++i)
    {
      const auto& ref = obs_refs[i];

      // Easy things first.
      point_indices[i] = ref.point_id;
      camera_indices[i] = ref.camera_id;

      // Initialize the 2D observations.
      const auto& image_id = ref.gid.image_id;
      const auto& local_id = ref.gid.local_id;
      const double x = features(keypoints[image_id])[local_id].x();
      const double y = features(keypoints[image_id])[local_id].y();
      observations(i, 0) = x;
      observations(i, 1) = y;
    }
  }

  auto populate_3d_points_from_two_view_geometry(
      const std::set<std::set<FeatureGID>>& feature_tracks,
      const std::multimap<FeatureGID, MatchGID>& match_index,
      const TwoViewGeometry& two_view_geometry) -> void
  {
    SARA_DEBUG << "Populating 3D points..." << std::endl;

    auto points_view = points_abs_coords_3d.colmajor_view().matrix();
    for (auto [t, track] = std::make_pair(0, feature_tracks.begin());
         track != feature_tracks.end(); ++t, ++track)
    {
      const auto p = match_index.find(*track->begin())->second.m;
      const auto point_p = two_view_geometry.X.col(p);
      points_view.col(t) = point_p.hnormalized();
#if DEBUG
      SARA_DEBUG << "Point[" << t << "] = "  //
                 << points_view.col(t).transpose() << std::endl;
#endif
    }
  }

  auto populate_camera_parameters() -> void
  {
    SARA_DEBUG << "Populating camera parameters..." << std::endl;

    auto cam_matrix = camera_parameters.matrix();
    for (auto c = 0; c < num_cameras; ++c)
    {
      cam_matrix.row(c) << VectorXd::Zero(9);
    }
  }

  auto populate_data_from_two_view_geometry(
      const std::set<std::set<FeatureGID>>& feature_tracks,
      const std::vector<KeypointList<OERegion, float>>& keypoints,
      const std::multimap<FeatureGID, MatchGID>& match_index,
      const TwoViewGeometry& two_view_geometry) -> void
  {
    const auto num_points = static_cast<int>(feature_tracks.size());
    SARA_CHECK(num_points);

    auto num_observations_per_points = std::vector<int>(num_points);
    std::transform(
        std::begin(feature_tracks), std::end(feature_tracks),
        std::begin(num_observations_per_points),
        [](const auto& track) { return static_cast<int>(track.size()); });

    const auto num_observations =
        std::accumulate(std::begin(num_observations_per_points),
                        std::end(num_observations_per_points), 0);
    SARA_CHECK(num_observations);

    // 3. Count the number of cameras, which should be equal to the number of
    //    images.
    auto image_ids = std::set<int>{};
    for (const auto& track : feature_tracks)
      for (const auto& f : track)
        image_ids.insert(f.image_id);
    const auto num_cameras = static_cast<int>(image_ids.size());
    SARA_CHECK(num_cameras);

    const auto camera_dof_ = 9;
    const auto num_parameters = camera_dof_ * num_cameras + 3 * num_points;
    SARA_CHECK(num_parameters);

    // 4. Transform the data for convenience.
    auto obs_refs = std::vector<ObservationRef>{};
    {
      obs_refs.reserve(num_observations);
      for (auto [point_id, track] = std::make_pair(0, feature_tracks.begin());
           track != feature_tracks.end(); ++point_id, ++track)
      {
        for (const auto& f : *track)
          obs_refs.push_back({f, f.image_id, point_id});
      }
    }

    resize(num_observations, num_points, num_cameras, camera_dof_);
    populate_observations(obs_refs, keypoints);
    populate_3d_points_from_two_view_geometry(feature_tracks, match_index,
                                              two_view_geometry);
    populate_camera_parameters();
  }

};

} /* namespace DO::Sara */
