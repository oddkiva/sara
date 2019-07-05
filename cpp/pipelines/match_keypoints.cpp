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
#include <DO/Sara/MultiViewGeometry/Geometry/Essential.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/Fundamental.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <Eigen/Sparse>


namespace sara = DO::Sara;

using namespace std;


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

  using EssentialGraph =
      std::unordered_map<Eigen::Vector2i, sara::EssentialMatrix<double>>;
  using FundamentalGraph =
      std::unordered_map<Eigen::Vector2i, sara::FundamentalMatrix<double>>;

  for (int i = 0; i < N; ++i)
  {
    for (int j = i + 1; j < N; ++j)
    {
      // Load images.
      auto Ki = sara::KeypointList<sara::OERegion, float>{};
      auto Kj = sara::KeypointList<sara::OERegion, float>{};

      auto Mij = match(Ki, Kj);

      auto Ii = sara::Image<sara::Rgb8>{};
      auto Ij = sara::Image<sara::Rgb8>{};

      auto scale = 1.0f;
      auto w = int((Ii.width() + Ij.width()) * scale);
      auto h = max(Ii.height(), Ij.height());
      auto off = sara::Point2f{float(Ii.width()), 0.f};

      sara::create_window(w, h);
      sara::set_antialiasing();
      sara::check_matches(Ii, Ij, Mij, true, scale);

      auto Fij = sara::estimate_fundamental_matrix(Mij);
      auto Eij = sara::estimate_essential_matrix(Mij);

      for (size_t m = 0; m < Mij.size(); ++m)
      {
        sara::draw_image_pair(Ii, Ij, off, scale);
        sara::draw_match(Mij[m], sara::Red8, off, scale);
        cout << Mij[m] << endl;
        sara::get_key();
      }
    }

  }

  return 0;
}
