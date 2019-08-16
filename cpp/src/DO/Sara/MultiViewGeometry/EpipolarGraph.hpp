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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/EssentialMatrixEstimators.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/FundamentalMatrixEstimators.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>


namespace DO::Sara {

struct DO_SARA_EXPORT ViewAttributes
{
  std::vector<std::string> image_paths;
  std::vector<std::string> group_names;

  std::vector<Image<Rgb8>> images;
  std::vector<KeypointList<OERegion, float>> keypoints;
  std::vector<PinholeCamera> cameras;

  auto list_images(const std::string& dirpath) -> void;

  auto read_images() -> void;
  auto read_keypoints(H5File& h5_file) -> void;
};


struct DO_SARA_EXPORT EpipolarEdgeAttributes
{
  using EEstimator = NisterFivePointAlgorithm;
  using FEstimator = EightPointAlgorithm;

  //! @brief An edge 'e' is an index the range [0, N * (N - 1)/ 2[.
  //! where N is the number of photographs.
  Tensor_<int, 1> edge_ids;

  // @brief An edge 'e' identifies a photograph pair (i,j) where
  // i, j are in [0, N[.
  std::vector<std::pair<int, int>> edges;

  //! @{
  //! @brief List of feature-based matches for edge (i,j).
  std::vector<std::vector<IndexMatch>> index_matches;
  std::vector<std::vector<Match>> matches;
  //! @}

  //! @brief Fundamental matrix F[i,j] for each edge (i,j).
  std::vector<FundamentalMatrix> F;
  //! @{
  //! @brief RANSAC metadata for each F[i,j].
  std::vector<int> F_num_samples;
  std::vector<double> F_noise;
  std::vector<Tensor_<bool, 1>> F_inliers;
  Tensor_<int, 2> F_best_samples;
  //! @}

  //! @brief Essential matrix E[i,j] for each edge (i,j).
  std::vector<EssentialMatrix> E;
  //! @{
  //! @brief RANSAC metadata for each E[i,j].
  std::vector<int> E_num_samples;
  std::vector<double> E_noise;
  std::vector<Tensor_<bool, 1>> E_inliers;
  Tensor_<int, 2> E_best_samples;
  //! @}

  // Two-view geometry G[i,j] for each edge (i,j).
  std::vector<TwoViewGeometry> two_view_geometries;

  auto initialize_edges(int num_vertices) -> void;
  auto resize_fundamental_edge_list() -> void;
  auto resize_essential_edge_list() -> void;

  auto read_matches(H5File& h5_file, const ViewAttributes& view_attributes)
      -> void;

  auto read_fundamental_matrices(const ViewAttributes& view_attributes,
                                 H5File& h5_file) -> void;
  auto read_essential_matrices(const ViewAttributes& view_attributes,
                               H5File& h5_file) -> void;
  auto read_two_view_geometries(const ViewAttributes& view_attributes,
                                H5File& h5_file) -> void;
};

} /* namespace DO::Sara */
