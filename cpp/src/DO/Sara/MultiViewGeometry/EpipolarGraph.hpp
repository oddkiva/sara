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
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>

#include <memory>


namespace DO::Sara {

struct EpipolarEdge
{
  int i;  // left
  int j;  // right
  Eigen::Matrix3d m;
};

struct DO_SARA_EXPORT PhotoAttributes
{
  std::vector<std::string> image_paths;
  std::vector<std::string> group_names;

  std::vector<Image<Rgb8>> images;
  std::vector<KeypointList<OERegion, float>> keypoints;
  std::vector<PinholeCamera> cameras;

  auto list_images(const std::string& dirpath) -> void;

  auto read_keypoints(H5File& h5_file) -> void;
};

struct DO_SARA_EXPORT EpipolarEdgeAttributes
{
  std::vector<int> edge_ids;
  std::vector<EpipolarEdge> edges;
  std::vector<std::vector<IndexMatch>> index_matches;
  std::vector<std::vector<Match>> matches;

  std::vector<double> noise;
  std::vector<int> num_inliers;
  Tensor_<int, 2> best_samples;

  auto initialize_edges(int num_vertices) -> void;

  auto read_matches(H5File& h5_file, const PhotoAttributes& photo_attributes)
      -> void;
};

} /* namespace DO::Sara */
