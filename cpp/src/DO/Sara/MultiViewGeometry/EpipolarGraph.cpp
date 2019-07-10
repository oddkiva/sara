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

#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>


namespace DO::Sara {

auto PhotoAttributes::list_images(const std::string& dirpath) -> void
{
  if (!image_paths.empty())
    image_paths.clear();
  append(image_paths, ls(dirpath, ".png"));
  append(image_paths, ls(dirpath, ".jpg"));
  std::sort(image_paths.begin(), image_paths.end());

  if (!group_names.empty())
    group_names.clear();

  group_names.reserve(image_paths.size());
  std::transform(
      std::begin(image_paths), std::end(image_paths),
      std::back_inserter(group_names),
      [&](const std::string& image_path) { return basename(image_path); });
}

auto PhotoAttributes::read_keypoints(H5File& h5_file)
{
  if (!keypoints.empty())
    keypoints.clear();

  keypoints.reserve(image_paths.size());
  std::transform(std::begin(group_names), std::end(group_names),
                 std::back_inserter(keypoints),
                 [&](const std::string& group_name) {
                   return read_keypoints(h5_file, group_name);
                 });
}

auto EpipolarEdgeAttributes::initialize_edges(int num_vertices) -> void
{
  if (!edge_ids.empty())
    edge_ids.clear();

  if (!edges.empty())
    edges.clear();

  const auto& N = num_vertices;
  const auto edge_ids = range(N * (N - 1) / 2);

  edges.reserve(N * (N - 1) / 2);
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j)
      edges.emplace_back(i, j, Eigen::Matrix3d::Zero());
}

auto EpipolarEdgeAttributes::read_matches(
    H5File& h5_file, const PhotoAttributes& photo_attributes) -> void
{
  if (!index_matches.empty() || !matches.empty())
  {
    index_matches.clear();
    matches.clear();
  }

  index_matches.reserve(edges.size());
  std::transform(std::begin(edges), std::end(edges),
                 std::back_inserter(index_matches),
                 [&](const EpipolarEdge& edge) {
                   const auto i = edge.i;
                   const auto j = edge.j;

                   const auto match_dataset = std::string{"matches"} + "/" +
                                              std::to_string(i) + "_" +
                                              std::to_string(j);

                   auto mij = std::vector<IndexMatch>{};
                   h5_file.read_dataset(match_dataset, mij);

                   return mij;
                 });

  matches.reserve(edges.size());
  std::transform(std::begin(edge_ids), std::end(edge_ids),
                 std::back_inserter(matches), [&](int ij) {
                   const auto i = edges[ij].i;
                   const auto j = edges[ij].j;
                   return to_match(index_matches[ij],
                                   photo_attributes.keypoints[i],
                                   photo_attributes.keypoints[j]);
                 });
}

  } /* namespace DO::Sara */
