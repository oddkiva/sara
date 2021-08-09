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

#include <DO/Sara/Core/MultiArray/DataTransformations.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>
#include <DO/Sara/SfM/BuildingBlocks.hpp>


using namespace std;


namespace DO::Sara {

using EEstimator = NisterFivePointAlgorithm;

void inspect_geometry(const TwoViewGeometry& g,
                      const Matrix<double, 3, EEstimator::num_points>& un1_s,
                      const Matrix<double, 3, EEstimator::num_points>& un2_s)
{
  const Matrix34d C1 = g.C1;
  const Matrix34d C2 = g.C2;

  SARA_DEBUG << "Camera matrices" << std::endl;
  SARA_DEBUG << "C1 =\n" << C1 << std::endl;
  SARA_DEBUG << "C2 =\n" << C2 << std::endl;

  SARA_DEBUG << "Triangulated points" << std::endl;
  const MatrixXd C1X = C1 * g.X;
  const MatrixXd C2X = C2 * g.X;

  SARA_DEBUG << "Check cheirality" << std::endl;
  const auto R = C2.topLeftCorner<3, 3>();
  const auto t = C2.col(3);
  const auto X_euclidean = g.X.topRows(3);
  const Eigen::Vector3d C = -R.transpose() * t;
  const Eigen::RowVector3d e3 = R.row(2);
  const Eigen::VectorXd dots = e3 * (X_euclidean.colwise() - C);
  const Eigen::Array<bool, 1, Eigen::Dynamic> cheirality = dots.array() > 0;
  SARA_DEBUG << "cheirality dots =\n" << dots << std::endl;
  SARA_DEBUG << "cheirality =\n" << cheirality << std::endl;

  SARA_DEBUG << "C1 * X =\n" << C1X << std::endl;
  SARA_DEBUG << "C2 * X =\n" << C2X << std::endl;


  SARA_DEBUG << "Comparison with normalized coordinates" << std::endl;
  SARA_DEBUG << "(C1 * X).hnormalized() =\n"
             << C1X.colwise().hnormalized() << std::endl;
  SARA_DEBUG << "u1n_s =\n" << un1_s.colwise().hnormalized() << std::endl;

  SARA_DEBUG << "(C2 * X).hnormalized() =\n"
             << C2X.colwise().hnormalized() << std::endl;
  SARA_DEBUG << "u2n_s.hnormalized() =\n"
             << un2_s.colwise().hnormalized() << std::endl;
  std::cout << std::endl;

  const double residual1 =
      (C1X.colwise().hnormalized() - un1_s.colwise().hnormalized()).norm() /
      (un1_s.colwise().hnormalized()).norm();
  const double residual2 =
      (C2X.colwise().hnormalized() - un2_s.colwise().hnormalized()).norm() /
      (un1_s.colwise().hnormalized()).norm();
  SARA_DEBUG << "Residual 1 = " << residual1 << std::endl;
  SARA_DEBUG << "Residual 2 = " << residual2 << std::endl;

  SARA_DEBUG << "Cheirality =\n" << g.cheirality << std::endl << std::endl;
}


auto estimate_two_view_geometry(const TensorView_<int, 2>& M,
                                const TensorView_<double, 2>& un1,
                                const TensorView_<double, 2>& un2,
                                const EssentialMatrix& E,
                                const TensorView_<bool, 1>& inliers,
                                const TensorView_<int, 1>& sample_best)
  -> TwoViewGeometry
{
  // Visualize the best sample drawn by RANSAC.
  constexpr auto L = EEstimator::num_points;
  const auto sample_best_reshaped = sample_best.reshape(Vector2i{1, L});
  const auto I = to_point_indices(sample_best_reshaped, M);
  const auto un_s = to_coordinates(I, un1, un2).transpose({0, 2, 1, 3});
  const Matrix<double, 3, L> un1_s = un_s[0][0].colmajor_view().matrix();
  const Matrix<double, 3, L> un2_s = un_s[0][1].colmajor_view().matrix();

  const auto candidate_motions = extract_relative_motion_horn(E);

  // Triangulate the points from the best samples and calculate their
  // cheirality.
  auto geometries = std::vector<TwoViewGeometry>{};
  std::transform(std::begin(candidate_motions), std::end(candidate_motions),
                 std::back_inserter(geometries), [&](const Motion& m) {
                   return two_view_geometry(m, un1_s, un2_s);
                 });
#define DEBUG_CHEIRALITY
#ifdef DEBUG_CHEIRALITY
  // Check the cheirality.
  for (auto i = 0u; i != geometries.size(); ++i)
  {
    SARA_DEBUG << "INSPECTING CANDIDATE GEOMETRY " << i << std::endl;
    const auto& g = geometries[i];
    inspect_geometry(g, un1_s, un2_s);
  }
#endif

  // Find the best geometry, i.e., the one with the high cheirality degree.
  const auto best_geom =
      std::max_element(std::begin(geometries), std::end(geometries),
                       [](const auto& g1, const auto& g2) {
                         return g1.cheirality.count() < g2.cheirality.count();
                       });


  const auto cheiral_degree = best_geom->cheirality.count();
  if (cheiral_degree == 0)
    throw std::runtime_error{"The cheirality degree can't be zero!"};

  inspect_geometry(*best_geom, un1_s, un2_s);
  // Data transformations.
  const Matrix34d P1 = best_geom->C1;
  const Matrix34d P2 = best_geom->C2;

  // Extract the matched coordinates.
  const auto card_M = M.size(0);
  const auto mindices = range(card_M);
  const auto card_M_filtered = mindices.size(0);
  SARA_CHECK(card_M_filtered);

  auto coords_matched = Tensor_<double, 3>{{2, card_M_filtered, 3}};
  auto un1_matched_mat = coords_matched[0].colmajor_view().matrix();
  auto un2_matched_mat = coords_matched[1].colmajor_view().matrix();
  {
    const auto un1_mat = un1.colmajor_view().matrix();
    const auto un2_mat = un2.colmajor_view().matrix();
    std::for_each(std::begin(mindices), std::end(mindices), [&](int m) {
      un1_matched_mat.col(m) = un1_mat.col(M(m, 0));
      un2_matched_mat.col(m) = un2_mat.col(M(m, 1));
    });
  }
#ifdef DEBUG
  SARA_DEBUG << "un1_matched_mat =\n"
             << un1_matched_mat.leftCols(10) << std::endl;
  SARA_DEBUG << "un2_matched_mat =\n"
             << un2_matched_mat.leftCols(10) << std::endl;
#endif

  SARA_DEBUG << "Triangulating all matches..." << std::endl;
  best_geom->X = triangulate_linear_eigen(P1, P2, un1_matched_mat, un2_matched_mat);
  SARA_CHECK(best_geom->X.cols());
  SARA_CHECK(inliers.flat_array().count());

  SARA_DEBUG << "Calculating cheirality..." << std::endl;
  best_geom->cheirality = relative_motion_cheirality_predicate(best_geom->X, P2);
  SARA_CHECK(best_geom->cheirality.count());

  SARA_DEBUG << "Cheiral inliers count = "
             << (best_geom->cheirality && inliers.row_vector().array()).count()
             << std::endl;

  return *best_geom;
}

auto keep_cheiral_inliers_only(TwoViewGeometry& complete_geom,
                               const TensorView_<bool, 1>& inliers) -> void
{
  auto& X = complete_geom.X;
  const auto& cheirality = complete_geom.cheirality;
  SARA_DEBUG << "Keep cheiral inliers..." << std::endl;
  const auto X_cheiral =
      range(static_cast<int>(X.cols()))                               //
      | filtered([&](int i) { return cheirality(i) && inliers(i); })  //
      | transformed([&](int i) -> Vector4d { return X.col(i); });
  SARA_CHECK(X_cheiral.size());

  const auto P2 = complete_geom.C2.matrix();

  complete_geom.X.resize(4, X_cheiral.size(0));
  for (int i = 0; i < X_cheiral.size(0); ++i)
    complete_geom.X.col(i) = X_cheiral(i);

  complete_geom.cheirality =
      relative_motion_cheirality_predicate(complete_geom.X, P2);

  SARA_DEBUG << "complete_geom.X =\n"
             << complete_geom.X.leftCols(10) << std::endl;
  SARA_DEBUG << "complete_geom.cheirality = "
             << complete_geom.cheirality.count() << std::endl;
}


auto extract_colors(const Image<Rgb8>& image1,             //
                    const Image<Rgb8>& image2,             //
                    const TwoViewGeometry& geometry)  //
  -> Tensor_<double, 2>
{
  const auto num_points = static_cast<int>(geometry.X.cols());
  const auto indices = range(num_points);

  // Convert the image to a pixel format with floating-point channel type.
  const auto I1d = image1.convert<Rgb64f>();
  const auto I2d = image2.convert<Rgb64f>();
  // Image sizes.
  const Vector2d imsizes1 = image1.sizes().cast<double>();
  const Vector2d imsizes2 = image2.sizes().cast<double>();

  // Retrieve the camera matrices.
  const auto P1 = geometry.C1.matrix();
  const auto P2 = geometry.C2.matrix();

  // Calculate the image coordinates from the normalized camera coordinates.
  const MatrixXd u1 = (P1 * geometry.X).colwise().hnormalized();
  const MatrixXd u2 = (P2 * geometry.X).colwise().hnormalized();

  // Only keep image coordinates that lie within the image domain.
  const Array<bool, 1, Dynamic> in_image_1 =
      (u1.array() >= 0).colwise().all() &&
      (u1.array().row(0) < imsizes1.x()) &&
      (u1.array().row(1) < imsizes1.y());
  SARA_CHECK(in_image_1.count());

  const Array<bool, 1, Dynamic> in_image_2 =
      (u2.array() >= 0).colwise().all() &&
      (u2.array().row(0) < imsizes2.x()) &&
      (u2.array().row(1) < imsizes2.y());
  SARA_CHECK(in_image_2.count());

  // Finally retrieve the colors for each 3D points.
  auto colors = Tensor_<double, 2>{num_points, 3};
  auto colors_mat = colors.matrix();
  std::for_each(std::begin(indices), std::end(indices), [&](int i) {
    const Vector2d u1_i = u1.col(i);
    const Vector2d u2_i = u2.col(i);

    if (in_image_1(i) && in_image_2(i))
      colors_mat.row(i) =
          0.5 * (interpolate(I1d, u1_i) + interpolate(I2d, u2_i)).transpose();
    else
      colors_mat.row(i) = RowVector3d::Zero().eval();
  });

  return colors;
}

} /* namespace DO::Sara */
