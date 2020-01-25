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

#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>

#include <boost/filesystem.hpp>


namespace fs = boost::filesystem;


namespace DO::Sara {

auto match(const KeypointList<OERegion, float>& keys1,
           const KeypointList<OERegion, float>& keys2,
           float lowe_ratio)
    -> std::vector<Match>
{
  AnnMatcher matcher{keys1, keys2, lowe_ratio};
  return matcher.compute_matches();
}


auto match_keypoints(const std::string& dirpath, const std::string& h5_filepath,
                     bool overwrite) -> void
{
  // Create a backup.
  if (!fs::exists(h5_filepath + ".bak"))
    cp(h5_filepath, h5_filepath + ".bak");

  auto h5_file = H5File{h5_filepath, H5F_ACC_RDWR};

  auto image_paths = std::vector<std::string>{};
  append(image_paths, ls(dirpath, ".png"));
  append(image_paths, ls(dirpath, ".jpg"));
  std::sort(image_paths.begin(), image_paths.end());

  auto group_names = std::vector<std::string>{};
  group_names.reserve(image_paths.size());
  std::transform(std::begin(image_paths), std::end(image_paths),
                 std::back_inserter(group_names),
                 [&](const std::string& image_path) {
                   return basename(image_path);
                 });

  auto keypoints = std::vector<KeypointList<OERegion, float>>{};
  keypoints.reserve(image_paths.size());
  std::transform(std::begin(group_names), std::end(group_names),
                 std::back_inserter(keypoints),
                 [&](const std::string& group_name) {
                   return read_keypoints(h5_file, group_name);
                 });

  const auto N = int(image_paths.size());
  auto edges = std::vector<std::pair<int, int>>{};
  edges.reserve(N * (N - 1) / 2);
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j)
      edges.emplace_back(i, j);

  auto matches = std::vector<std::vector<Match>>{};
  matches.reserve(edges.size());
  std::transform(std::begin(edges), std::end(edges),
                 std::back_inserter(matches),
                 [&](const auto& edge) {
                   const auto i = edge.first;
                   const auto j = edge.second;
                   return match(keypoints[i], keypoints[j]);
                 });

  // Save matches to HDF5.
  auto edge_ids = range(edges.size());
  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& e) {
        const auto& ij = edges[e];
        const auto i = ij.first;
        const auto j = ij.second;
        const auto& matches_ij = matches[e];

        // Transform the data.
        auto Mij = std::vector<IndexMatch>{};
        std::transform(
            std::begin(matches_ij), std::end(matches_ij),
            std::back_inserter(Mij), [](const auto& m) {
              return IndexMatch{m.x_index(), m.y_index(), m.score()};
            });

        // Save the keypoints to HDF5
        const auto group_name = std::string{"matches"};
        h5_file.get_group(group_name);

        const auto match_dataset =
            group_name + "/" + std::to_string(i) + "_" + std::to_string(j);
        h5_file.write_dataset(match_dataset, tensor_view(Mij), overwrite);
      });
}

} /* namespace DO::Sara */
