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

#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Features.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/MultiViewGeometry/HDF5.hpp>


namespace DO::Sara {

auto ViewAttributes::list_images(const std::string& dirpath) -> void
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

auto ViewAttributes::read_keypoints(H5File& h5_file) -> void
{
  if (!keypoints.empty())
    keypoints.clear();

  keypoints.reserve(image_paths.size());
  std::transform(std::begin(group_names), std::end(group_names),
                 std::back_inserter(keypoints),
                 [&](const std::string& group_name) {
                   return DO::Sara::read_keypoints(h5_file, group_name);
                 });
}

auto ViewAttributes::read_images() -> void
{
  images.resize(image_paths.size());
  std::transform(std::begin(image_paths), std::end(image_paths),
                 std::begin(images), [](const auto image_path) {
                   SARA_DEBUG << "Reading image from:\n\t" << image_path
                              << std::endl;
                   return imread<Rgb8>(image_path);
                 });
}


auto EpipolarEdgeAttributes::initialize_edges(int num_vertices)
    -> void
{
  const auto& N = num_vertices;
  edge_ids = range(N * (N - 1) / 2);

  if (!edges.empty())
    edges.clear();

  edges.reserve(N * (N - 1) / 2);
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j)
      edges.push_back(std::make_pair(i, j));
}

auto EpipolarEdgeAttributes::read_matches(H5File& h5_file,
                                          const ViewAttributes& view_attributes)
    -> void
{
  if (!index_matches.empty() || !matches.empty())
  {
    index_matches.clear();
    matches.clear();
  }

  index_matches.reserve(edges.size());
  std::transform(std::begin(edges), std::end(edges),
                 std::back_inserter(index_matches),
                 [&](const std::pair<int, int>& edge) {
                   const auto i = edge.first;
                   const auto j = edge.second;

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
                   const auto i = edges[ij].first;
                   const auto j = edges[ij].second;
                   return to_match(index_matches[ij],
                                   view_attributes.keypoints[i],
                                   view_attributes.keypoints[j]);
                 });
}

auto EpipolarEdgeAttributes::resize_fundamental_edge_list()
    -> void
{
  if (edges.empty())
    return;

  F.resize(edges.size());
  F_num_samples.resize(edges.size());
  F_noise.resize(edges.size());
  F_inliers.resize(edges.size());
  F_best_samples.resize(edges.size(), FEstimator::num_points);
}

auto EpipolarEdgeAttributes::resize_essential_edge_list()
    -> void
{
  if (edges.empty())
    return;

  E.resize(edges.size());
  E_num_samples.resize(edges.size());
  E_noise.resize(edges.size());
  E_inliers.resize(edges.size());
  E_best_samples.resize(edges.size(), EEstimator::num_points);
}

auto EpipolarEdgeAttributes::read_fundamental_matrices(
    const ViewAttributes& view_attributes, H5File& h5_file) -> void
{
  h5_file.read_dataset("F", F);
  h5_file.read_dataset("F_num_samples", F_num_samples);
  h5_file.read_dataset("F_noise", F_noise);
  h5_file.read_dataset("F_best_samples", F_best_samples);

  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
        const auto& eij = edges[ij];
        const auto i = eij.first;
        const auto j = eij.second;

        SARA_DEBUG << "Reading fundamental matrices between images:\n"
                   << "- image[" << i << "] = "  //
                   << view_attributes.group_names[i] << "\n"
                   << "- image[" << j << "] = "  //
                   << view_attributes.group_names[j] << "\n";
        std::cout.flush();

        // Estimate the fundamental matrix.
        h5_file.read_dataset(format("F_inliers/%d_%d", i, j),
                             F_inliers[ij]);
      });
}

auto EpipolarEdgeAttributes::read_essential_matrices(
    const ViewAttributes& view_attributes, H5File& h5_file) -> void
{
  h5_file.read_dataset("E", E);
  h5_file.read_dataset("E_num_samples", E_num_samples);
  h5_file.read_dataset("E_noise", E_noise);
  h5_file.read_dataset("E_best_samples", E_best_samples);

  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
        const auto& eij = edges[ij];
        const auto i = eij.first;
        const auto j = eij.second;

        SARA_DEBUG << "Reading fundamental matrices between images:\n"
                   << "- image[" << i << "] = "  //
                   << view_attributes.group_names[i] << "\n"
                   << "- image[" << j << "] = "  //
                   << view_attributes.group_names[j] << "\n";
        std::cout.flush();

        // Estimate the fundamental matrix.
        h5_file.read_dataset(format("E_inliers/%d_%d", i, j),
                             E_inliers[ij]);
      });
}


auto EpipolarEdgeAttributes::read_two_view_geometries(
    const ViewAttributes& view_attributes, H5File& h5_file) -> void
{
  two_view_geometries.resize(edges.size());

  // Get the left and right cameras.
  auto cameras = Tensor_<PinholeCamera, 1>{2};

  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
        const auto& eij = edges[ij];
        const auto i = eij.first;
        const auto j = eij.second;

        SARA_DEBUG << "Reading two-view geometry between:\n"
                   << "- image[" << i << "] = "  //
                   << view_attributes.group_names[i] << "\n"
                   << "- image[" << j << "] = "  //
                   << view_attributes.group_names[j] << "\n";
        std::cout.flush();

        // Estimate the fundamental matrix.
        h5_file.read_dataset(format("two_view_geometries/cameras/%d_%d", i, j),
                             cameras);

        two_view_geometries[ij].C1 = cameras(0);
        two_view_geometries[ij].C2 = cameras(1);
      });
}

} /* namespace DO::Sara */
