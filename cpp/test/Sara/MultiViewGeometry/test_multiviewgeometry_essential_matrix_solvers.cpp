// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018-2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "MultiViewGeometry/Essential Matrix"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/Core/Numpy.hpp>
#include <DO/Sara/Core/TensorDebug.hpp>
#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/EssentialMatrixSolvers.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/Triangulation.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>

#include <boost/test/unit_test.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestMultiViewGeometry)

BOOST_AUTO_TEST_CASE(test_extract_centers)
{
  auto features = std::vector<OERegion>{{Point2f::Ones() * 0, 1.f},
                                        {Point2f::Ones() * 1, 1.f},
                                        {Point2f::Ones() * 2, 1.f}};

  const auto x = extract_centers(features);
  auto expected_x = Tensor_<float, 2>{x.sizes()};
  expected_x.matrix() <<
    0, 0,
    1, 1,
    2, 2;

  BOOST_CHECK(x.matrix() == expected_x.matrix());

  const auto X = homogeneous(x);
  auto expected_X = Tensor_<float, 2>{X.sizes()};
  expected_X.matrix() <<
    0, 0, 1,
    1, 1, 1,
    2, 2, 1;
  BOOST_CHECK(X.matrix() == expected_X.matrix());
}

BOOST_AUTO_TEST_CASE(test_to_point_indices)
{
  constexpr auto num_matches = 5;
  constexpr auto num_samples = 2;
  constexpr auto sample_size = 4;

  auto matches = Tensor_<int, 2>{num_matches, 2};
  matches.matrix() <<
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 0;

  auto samples = Tensor_<int, 2>{num_samples, sample_size};
  samples.matrix() <<
    0, 1, 2, 3,
    4, 2, 3, 1;

  auto point_indices = to_point_indices(samples, matches);

  auto expected_point_indices = Tensor_<int, 3>{num_samples, sample_size, 2};
  expected_point_indices[0].matrix() <<
    0, 0,
    1, 1,
    2, 2,
    3, 3;
  expected_point_indices[1].matrix() <<
    4, 0,
    2, 2,
    3, 3,
    1, 1;

  BOOST_CHECK(point_indices.vector() == expected_point_indices.vector());
}

BOOST_AUTO_TEST_CASE(test_to_coordinates)
{
  constexpr auto num_matches = 5;
  constexpr auto num_samples = 2;
  constexpr auto sample_size = 4;

  const auto features1 = std::vector<OERegion>{{Point2f::Ones() * 0, 1.f},
                                               {Point2f::Ones() * 1, 1.f},
                                               {Point2f::Ones() * 2, 1.f}};
  const auto features2 = std::vector<OERegion>{{Point2f::Ones() * 1, 1.f},
                                               {Point2f::Ones() * 2, 1.f},
                                               {Point2f::Ones() * 3, 1.f}};

  const auto points1 = extract_centers(features1);
  const auto points2 = extract_centers(features2);

  auto matches = Tensor_<int, 2>{num_matches, 2};
  matches.matrix() <<
    0, 0,
    1, 1,
    2, 2,
    0, 1,
    1, 2;

  auto samples = Tensor_<int, 2>{num_samples, sample_size};
  samples.matrix() <<
    0, 1, 2, 3,
    1, 2, 3, 4;

  const auto point_indices = to_point_indices(samples, matches);
  const auto coords = to_coordinates(point_indices, points1, points2);

  //                                        N            K            P  C
  auto expected_coords = Tensor_<float, 4>{{num_samples, sample_size, 2, 2}};
  expected_coords[0].flat_array() <<
    0.f, 0.f, 1.f, 1.f,
    1.f, 1.f, 2.f, 2.f,
    2.f, 2.f, 3.f, 3.f,
    0.f, 0.f, 2.f, 2.f;

  expected_coords[1].flat_array() <<
    1.f, 1.f, 2.f, 2.f,
    2.f, 2.f, 3.f, 3.f,
    0.f, 0.f, 2.f, 2.f,
    1.f, 1.f, 3.f, 3.f;

  BOOST_CHECK(expected_coords.vector() == coords.vector());
  BOOST_CHECK(expected_coords.sizes() == coords.sizes());


  const auto coords_t = coords.transpose({0, 2, 1, 3});
  const auto sample1 = coords_t[0];

  auto expected_sample1 = Tensor_<float, 3>{sample1.sizes()};
  expected_sample1.flat_array() <<
    // P1
    0.f, 0.f,
    1.f, 1.f,
    2.f, 2.f,
    0.f, 0.f,
    // P2
    1.f, 1.f,
    2.f, 2.f,
    3.f, 3.f,
    2.f, 2.f;

  // print_3d_interleaved_array(expected_sample1);
  // print_3d_interleaved_array(sample1);
  BOOST_CHECK(expected_sample1.vector() == sample1.vector());
}


BOOST_AUTO_TEST_CASE(test_compute_normalizer)
{
  auto X = Tensor_<float, 2>{3, 3};
  X.matrix() <<
    1, 1, 1,
    2, 2, 1,
    3, 3, 1;

  auto T = compute_normalizer(X);

  Matrix3f expected_T;
  expected_T <<
    0.5, 0.0, -0.5,
    0.0, 0.5, -0.5,
    0.0, 0.0,  1.0;

  BOOST_CHECK((T - expected_T).norm() < 1e-12);
}

BOOST_AUTO_TEST_CASE(test_apply_transform)
{
  auto X = Tensor_<double, 2>{3, 3};
  X.matrix() <<
    1, 1, 1,
    2, 2, 1,
    3, 3, 1;

  // From Oxford affine covariant features dataset, graf, H1to5P homography.
  auto H = Matrix3d{};
  H <<
    6.2544644e-01,  5.7759174e-02,  2.2201217e+02,
    2.2240536e-01,  1.1652147e+00, -2.5605611e+01,
    4.9212545e-04, -3.6542424e-05,  1.0000000e+00;

  auto HX = apply_transform(H, X);

  BOOST_CHECK(HX.matrix().col(2) == Vector3d::Ones());
}


BOOST_AUTO_TEST_CASE(test_skew_symmetric_matrix)
{
  Vector3f t{1, 2, 3};
  Matrix3f T;
  T <<
     0, -3,  2,
     3,  0, -1,
    -2,  1,  0;

  BOOST_CHECK(skew_symmetric_matrix(t) == T);
}


struct TestData
{
  MatrixXd X;

  Matrix3d R;
  Vector3d t;
  EssentialMatrix E;

  Matrix34d C1;
  Matrix34d C2;

  MatrixXd u1, u2;
};

auto generate_test_data() -> TestData
{
  // 3D points.
  MatrixXd X(4, 5);  // coefficients are in [-1, 1].
  X.topRows<3>() <<
    -1.49998,   -0.5827,  -1.40591,  0.369386,  0.161931, //
    -1.23692, -0.434466, -0.142271, -0.732996,  -1.43086, //
     1.51121,  0.437918,   1.35859,   1.03883,  0.106923; //
  X.bottomRows<1>().fill(1.);

  const Matrix3d R = rotation(0.3, 0.2, 0.1);
  const Vector3d t{0.1, 0.2, 0.3};

  const auto E = essential_matrix(R, t);

  const Matrix34d C1 = BasicPinholeCamera{Matrix3d::Identity(),
                                          Matrix3d::Identity(),
                                          Vector3d::Zero()};
  const Matrix34d C2 = BasicPinholeCamera{Matrix3d::Identity(), R, t};
  MatrixXd x1 = C1 * X; x1.array().rowwise() /= x1.row(2).array();
  MatrixXd x2 = C2 * X; x2.array().rowwise() /= x2.row(2).array();

  return {X, R, t, E, C1, C2, x1, x2};
}


BOOST_AUTO_TEST_CASE(test_null_space_extraction)
{
  const auto test_data = generate_test_data();
  const auto& x1 = test_data.u1;
  const auto& x2 = test_data.u2;

  auto solver = NisterFivePointAlgorithm{};

  const auto null = solver.extract_null_space(x1, x2);
  {
    const auto [A, B, C, D] = solver.reshape_null_space(null);

    for (auto j = 0; j < x1.cols(); ++j)
    {
      BOOST_CHECK_SMALL(double(x2.col(j).transpose() * A * x1.col(j)), 1e-12);
      BOOST_CHECK_SMALL(double(x2.col(j).transpose() * B * x1.col(j)), 1e-12);
      BOOST_CHECK_SMALL(double(x2.col(j).transpose() * C * x1.col(j)), 1e-12);
      BOOST_CHECK_SMALL(double(x2.col(j).transpose() * D * x1.col(j)), 1e-12);
    }
  }
}


BOOST_AUTO_TEST_CASE(test_nister_five_point_algorithm)
{
  const auto test_data = generate_test_data();
  const auto& x1 = test_data.u1;
  const auto& x2 = test_data.u2;

  auto solver = NisterFivePointAlgorithm{};

  // 1. Extract the null space.
  const auto E_bases = solver.extract_null_space(x1, x2);

  // 2. Form the epipolar constraints.
  const auto E_bases_reshaped = solver.reshape_null_space(E_bases);
  const auto E_expr = solver.essential_matrix_expression(E_bases_reshaped);
  const auto E_constraints = solver.build_essential_matrix_constraints(E_expr);

  // 3. Solve the epipolar constraints.
  const auto Es = solver.solve_essential_matrix_constraints(E_bases_reshaped,
                                                            E_constraints);

  // Check essential matrix constraints.
  for (auto i = 0u; i < Es.size(); ++i)
  {
    const auto& Ei = Es[i].matrix();

    // SARA_DEBUG << "i = " << i << endl;
    // SARA_DEBUG << "Ein =\n" << Ei.normalized() << endl;
    // SARA_DEBUG << "En =\n" << E.normalized() << endl;
    // SARA_DEBUG << "norm(Ein - En) = "
    //           << (Ei.normalized() - E.normalized()).norm() << endl;
    // SARA_DEBUG << "norm(Ein + En) = "
    //           << (Ei.normalized() + E.normalized()).norm() << endl;

    BOOST_CHECK_SMALL(Ei.determinant(), 1e-10);
    BOOST_CHECK_SMALL(
        (2. * Ei * Ei.transpose() * Ei - (Ei * Ei.transpose()).trace() * Ei)
            .norm(),
        1e-10);

    // Paranoid check.
    for (auto j = 0; j < x1.cols(); ++j)
      BOOST_CHECK_SMALL(double(x2.col(j).transpose() * Ei * x1.col(j)), 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(test_stewenius_five_point_algorithm)
{
  const auto test_data = generate_test_data();
  const auto& x1 = test_data.u1;
  const auto& x2 = test_data.u2;

  auto solver = SteweniusFivePointAlgorithm{};

  // 1. Extract the null space.
  const auto E_bases = solver.extract_null_space(x1, x2);

  // 2. Form the epipolar constraints.
  const auto E_bases_reshaped = solver.reshape_null_space(E_bases);
  const auto E_expr = solver.essential_matrix_expression(E_bases_reshaped);
  const auto E_constraints = solver.build_essential_matrix_constraints(E_expr);

  // 3. Solve the epipolar constraints.
  const auto Es =
      solver.solve_essential_matrix_constraints(E_bases, E_constraints);

  // Check essential matrix constraints.
  for (auto i = 0u; i < Es.size(); ++i)
  {
    const auto& Ei = Es[i].matrix();

    // SARA_DEBUG << "i = " << i << endl;
    // SARA_DEBUG << "Ein =\n" << Ei.normalized() << endl;
    // SARA_DEBUG << "En =\n" << E.normalized() << endl;
    // SARA_DEBUG << "norm(Ein - En) = "
    //           << (Ei.normalized() - E.normalized()).norm() << endl;
    // SARA_DEBUG << "norm(Ein + En) = "
    //           << (Ei.normalized() + E.normalized()).norm() << endl;

    BOOST_CHECK_SMALL(Ei.determinant(), 1e-12);
    BOOST_CHECK_SMALL(
        (2. * Ei * Ei.transpose() * Ei - (Ei * Ei.transpose()).trace() * Ei)
            .norm(),
        1e-12);

    // Paranoid check.
    for (auto j = 0; j < x1.cols(); ++j)
      BOOST_CHECK_SMALL(double(x2.col(j).transpose() * Ei * x1.col(j)), 1e-12);
  }
}


BOOST_AUTO_TEST_CASE(test_extract_relative_motions_functions)
{
  const auto [X, R, t, E, C1, C2, x1, x2] = generate_test_data();
  (void) C1;
  (void) C2;

  const auto true_motion = Motion{R, t};
  const double thres = 1e-12;

  auto motion_equality_predicate = [&](const auto& motion) {
    const auto rotation_equality =
        (motion.R - true_motion.R).norm() / true_motion.R.norm() < thres;
    const auto translation_equality =
        (motion.t.normalized() - true_motion.t.normalized()).norm() /
            true_motion.t.normalized().norm() <
        thres;
    return rotation_equality && translation_equality;
  };


  {
    SARA_DEBUG << "Check SVD-based motion extraction" << std::endl;
    auto motions = extract_relative_motion_svd(E);
    const auto motion_found =
        std::find_if(motions.begin(), motions.end(), motion_equality_predicate);
    BOOST_CHECK(motion_found != motions.end());
    SARA_DEBUG << "Motion found" << std::endl;
    SARA_DEBUG << "R =\n" << motion_found->R << endl;
    SARA_DEBUG << "t = " << motion_found->t.normalized().transpose() << endl;
    SARA_DEBUG << "ΔR = " << (motion_found->R - true_motion.R).norm() << endl;
    SARA_DEBUG
        << "Δt = "
        << (motion_found->t.normalized() - true_motion.t.normalized()).norm()
        << endl;
  }

  {
    SARA_DEBUG << "Check Horn's motion extraction method" << std::endl;
    auto motions = extract_relative_motion_horn(E);
    const auto motion_found =
        std::find_if(motions.begin(), motions.end(), motion_equality_predicate);
    BOOST_CHECK(motion_found != motions.end());
    SARA_DEBUG << "Motion found" << std::endl;
    SARA_DEBUG << "R =\n" << motion_found->R << endl;
    SARA_DEBUG << "t = " << motion_found->t.normalized().transpose() << endl;
    SARA_DEBUG << "ΔR = " << (motion_found->R - true_motion.R).norm() << endl;
    SARA_DEBUG
        << "Δt = "
        << (motion_found->t.normalized() - true_motion.t.normalized()).norm()
        << endl;
  }
}

BOOST_AUTO_TEST_SUITE_END()
