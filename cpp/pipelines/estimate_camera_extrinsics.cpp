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

#include <DO/Sara/Features.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <Eigen/Sparse>

#include <boost/filesystem.hpp>


namespace fs = boost::filesystem;
namespace sara = DO::Sara;

const auto file1 = src_path("../../../data/All.tif");
const auto file2 = src_path("../../../data/GuardOnBlonde.tif");


auto match(const sara::KeypointList<sara::OERegion, float>& keys1,
           const sara::KeypointList<sara::OERegion, float>& keys2)
    -> std::vector<sara::Match>
{
  // Compute/read matches
  sara::AnnMatcher matcher{keys1, keys2, 1.0f};
  return matcher.compute_matches();
}

GRAPHICS_MAIN()
{
  const auto dirpath = fs::path{"/mnt/a1cc5981-3655-4f74-9c62-37253d79c82d/sfm/Trafalgar/images"};
  const auto image_paths = sara::ls(dirpath.string(), ".jpg");
  const auto N = int(image_paths.size());


  using SpMatrixXi = Eigen::SparseMatrix<int>;

  auto edges = std::vector<Eigen::Triplet<int>>{};
  auto adj_list = SpMatrixXi{N, N};
  adj_list.setFromTriplets(edges.begin(), edges.end());

  //using EssentialGraph =
  //    std::unordered_map<Eigen::Vector2i, sara::EssentialMatrix<double>>;
  //using FundamentalGraph =
  //    std::unordered_map<Eigen::Vector2i, sara::FundamentalMatrix<double>>;

  //// Load essential graph.
  //auto E = EssentialGraph{};

  //// Load focal length.
  //auto focal_lengths = std::vector<double>(N);

  //// Calculate the camera extrinsics.
  //auto C = std::vector<sara::Camera>(N);

  // Bundler blabla...


  return 0;
}
