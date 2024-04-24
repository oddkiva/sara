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

#include <DO/Sara/SfM/Helpers/FundamentalMatrixEstimation.hpp>

#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/RANSAC/RANSACv2.hpp>
#include <DO/Sara/Visualization.hpp>


namespace DO::Sara {

  using FSolver = SevenPointAlgorithmDoublePrecision;

  auto estimate_fundamental_matrix(const std::vector<Match>& Mij,
                                   const KeypointList<OERegion, float>& ki,
                                   const KeypointList<OERegion, float>& kj,
                                   const int num_samples,
                                   const double err_thres)
      -> std::tuple<FundamentalMatrix, Tensor_<bool, 1>, Tensor_<int, 1>>
  {
    const auto& fi = features(ki);
    const auto& fj = features(kj);
    const auto pi = homogeneous(extract_centers(fi).cast<double>());
    const auto pj = homogeneous(extract_centers(fj).cast<double>());
    const auto Mij_tensor = to_tensor(Mij);

    const auto Xij = PointCorrespondenceList{Mij_tensor, pi, pj};
    const auto data_normalizer =
        std::make_optional(Normalizer<FundamentalMatrix>{Xij});

    auto inlier_predicate = InlierPredicate<SampsonEpipolarDistance>{};
    inlier_predicate.err_threshold = err_thres;

    static constexpr auto confidence = 0.99;
    const auto [F, inliers, sample_best] = v2::ransac(  //
        Xij,                                            //
        FSolver{},                                      //
        inlier_predicate,                               //
        num_samples, confidence,                        //
        data_normalizer,                                //
        true);

    return std::make_tuple(F, inliers, sample_best);
  }

  auto check_epipolar_constraints(const Image<Rgb8>& Ii, const Image<Rgb8>& Ij,
                                  const FundamentalMatrix& F,
                                  const std::vector<Match>& Mij,
                                  const TensorView_<int, 1>& sample_best,
                                  const TensorView_<bool, 1>& inliers,
                                  int display_step, bool wait_key) -> void
  {
    const auto scale = 0.25f;
    const auto w = int((Ii.width() + Ij.width()) * scale + 0.5f);
    const auto h = int(std::max(Ii.height(), Ij.height()) * scale + 0.5f);

    if (!active_window())
    {
      create_window(w, h);
      set_antialiasing();
    }

    if (get_sizes(active_window()) != Eigen::Vector2i(w, h))
      resize_window(w, h);

    PairWiseDrawer drawer(Ii, Ij);
    drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);

    drawer.display_images();

    for (auto m = 0; m < static_cast<int>(Mij.size()); ++m)
    {
      const Eigen::Vector3d X1 = Mij[m].x_pos().cast<double>().homogeneous();
      const Eigen::Vector3d X2 = Mij[m].y_pos().cast<double>().homogeneous();

      if (!inliers(m))
        continue;

      if (m % display_step == 0)
      {
        drawer.draw_match(Mij[m], Blue8, false);

        const auto proj_X1 = F.right_epipolar_line(X1);
        const auto proj_X2 = F.left_epipolar_line(X2);

        drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Cyan8, 1);
        drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Cyan8, 1);
      }
    }

    for (auto m = 0; m < static_cast<int>(sample_best.size()); ++m)
    {
      // Draw the best elemental subset drawn by RANSAC.
      drawer.draw_match(Mij[sample_best(m)], Red8, true);

      const Eigen::Vector3d X1 =
          Mij[sample_best(m)].x_pos().cast<double>().homogeneous();
      const Eigen::Vector3d X2 =
          Mij[sample_best(m)].y_pos().cast<double>().homogeneous();

      const auto proj_X1 = F.right_epipolar_line(X1);
      const auto proj_X2 = F.left_epipolar_line(X2);

      // Draw the corresponding epipolar lines.
      drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Magenta8, 1);
      drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Magenta8, 1);
    }

    if (wait_key)
      get_key();
  }

} /* namespace DO::Sara */
