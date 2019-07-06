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
#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/FivePointAlgorithms.hpp>

#include <boost/test/unit_test.hpp>

#include <iomanip>
#include <sstream>


using namespace std;
using namespace DO::Sara;


template <typename T>
void print_3d_array(const TensorView_<T, 3>& x)
{
  const auto max = x.flat_array().abs().maxCoeff();
  std::stringstream ss;
  ss << max;
  const auto pad_size = ss.str().size();


  cout << "[";
  for (auto i = 0; i < x.size(0); ++i)
  {
    cout << "[";
    for (auto j = 0; j < x.size(1); ++j)
    {
      cout << "[";
      for (auto k = 0; k < x.size(2); ++k)
      {
        cout << std::setw(pad_size) << x(i,j,k);
        if (k != x.size(2) - 1)
          cout << ", ";
      }
      cout << "]";

      if (j != x.size(1) - 1)
        cout << ", ";
      else
        cout << "]";
    }

    if (i != x.size(0) - 1)
      cout << ",\n ";
  }
  cout << "]" << endl;
}

void print_3d_array(const TensorView_<float, 3>& x)
{
  cout << "[";
  for (auto i = 0; i < x.size(0); ++i)
  {
    cout << "[";
    for (auto j = 0; j < x.size(1); ++j)
    {
      cout << "[";
      for (auto k = 0; k < x.size(2); ++k)
      {
        cout << fixed << x(i,j,k);
        if (k != x.size(2) - 1)
          cout << ", ";
      }
      cout << "]";

      if (j != x.size(1) - 1)
        cout << ", ";
      else
        cout << "]";
    }

    if (i != x.size(0) - 1)
      cout << ",\n ";
  }
  cout << "]" << endl;
}


BOOST_AUTO_TEST_SUITE(TestMultiViewGeometry)

BOOST_AUTO_TEST_CASE(test_range)
{
  auto a = range(3);
  BOOST_CHECK(a.vector() == Vector3i(0, 1, 2));
}

BOOST_AUTO_TEST_CASE(test_random_shuffle)
{
  auto a = range(4);
  a = shuffle(a);
  BOOST_CHECK(a.vector() != Vector4i(0, 1, 2, 3));
}

BOOST_AUTO_TEST_CASE(test_random_samples)
{
  constexpr auto num_samples = 2;
  constexpr auto sample_size = 5;
  constexpr auto num_data_points = 10;
  auto samples = random_samples(num_samples, sample_size, num_data_points);

  BOOST_CHECK_EQUAL(samples.size(0), num_samples);
  BOOST_CHECK_EQUAL(samples.size(1), sample_size);
  BOOST_CHECK(samples.matrix().minCoeff() >=  0);
  BOOST_CHECK(samples.matrix().maxCoeff() <  10);
}


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

  //print_3d_array(expected_sample1);
  //print_3d_array(sample1);
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


using Matrix34d = Matrix<double, 3, 4>;

Matrix3d rot_x(double angle) {
  return Eigen::AngleAxisd(angle, Vector3d::UnitX()).toRotationMatrix();
}

Matrix3d rot_y(double angle) {
  return Eigen::AngleAxisd(angle, Vector3d::UnitY()).toRotationMatrix();
}

Matrix3d rot_z(double angle) {
  return Eigen::AngleAxisd(angle, Vector3d::UnitZ()).toRotationMatrix();
}

auto essential_matrix = [](auto R, auto t) -> Matrix3d {
  return skew_symmetric_matrix(t) * R;
};

auto camera_matrix = [](auto K, auto R, auto t) -> Matrix34d {
  Matrix34d Rt = Matrix34d::Zero();
  Rt.topLeftCorner(3, 3) = R;
  Rt.col(3) = t;
  return K * Rt;
};


auto generate_test_data()
{
  // 3D points.
  MatrixXd X(4, 5);  // coefficients are in [-1, 1].
  X.topRows<3>() <<
    -1.49998,   -0.5827,  -1.40591,  0.369386,  0.161931, //
    -1.23692, -0.434466, -0.142271, -0.732996,  -1.43086, //
     1.51121,  0.437918,   1.35859,   1.03883,  0.106923; //
  X.bottomRows<1>().fill(1.);

  Matrix3d R = rot_z(0.3) * rot_x(0.1) * rot_y(0.2);
  Vector3d t{0.1, 0.2, 0.3};

  const auto E = essential_matrix(R, t);

  const auto C1 = camera_matrix(Matrix3d::Identity(), Matrix3d::Identity(),
                                Vector3d::Zero());
  const auto C2 = camera_matrix(Matrix3d::Identity(), R, t);
  MatrixXd x1 = C1 * X; x1.array().rowwise() /= x1.row(2).array();
  MatrixXd x2 = C2 * X; x2.array().rowwise() /= x2.row(2).array();

  return std::make_tuple(X, R, t, E, C1, C2, x1, x2);
}


BOOST_AUTO_TEST_CASE(test_null_space_extraction)
{
  auto [X, R, t, E, C1, C2, x1, x2] = generate_test_data();
  //SARA_DEBUG << "3D points = \n" << X << endl;
  //SARA_DEBUG << "Left points = \n" << x1 << endl;
  //SARA_DEBUG << "Right points = \n" << x2 << endl;

  auto solver = NisterFivePointAlgorithm{};

  const auto Ker = solver.extract_null_space(x1, x2);
  {
    const auto [A, B, C, D] = solver.reshape_null_space(Ker);

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
  const auto [X, R, t, E, C1, C2, x1, x2] = generate_test_data();

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
    const auto& Ei = Es[i];

    //SARA_DEBUG << "i = " << i << endl;
    //SARA_DEBUG << "Ein =\n" << Ei.normalized() << endl;
    //SARA_DEBUG << "En =\n" << E.normalized() << endl;
    //SARA_DEBUG << "norm(Ein - En) = "
    //           << (Ei.normalized() - E.normalized()).norm() << endl;
    //SARA_DEBUG << "norm(Ein + En) = "
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
  const auto [X, R, t, E, C1, C2, x1, x2] = generate_test_data();

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
    const auto& Ei = Es[i];

    //SARA_DEBUG << "i = " << i << endl;
    //SARA_DEBUG << "Ein =\n" << Ei.normalized() << endl;
    //SARA_DEBUG << "En =\n" << E.normalized() << endl;
    //SARA_DEBUG << "norm(Ein - En) = "
    //           << (Ei.normalized() - E.normalized()).norm() << endl;
    //SARA_DEBUG << "norm(Ein + En) = "
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


struct Motion {
  Matrix3d R;
  Vector3d t;
};

auto extract_relative_motion_svd(const Matrix3d& E) -> std::vector<Motion>
{
  auto svd =
      Eigen::BDCSVD<Matrix3d>(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto U = svd.matrixU();
  Matrix3d Vt = svd.matrixV().transpose();

  static_assert(std::is_same<decltype(U), Matrix3d>::value);

  if (U.determinant() < 0)
    U.col(2) *= -1;
  if (Vt.determinant() < 0)
    Vt.row(2) *= -1;

  Matrix3d W;
  W << 0, 1, 0,
      -1, 0, 0,
       0, 0, 1;

  Matrix3d Ra, Rb;
  Vector3d ta, tb;

  Ra = U * W * Vt;
  Rb = U * W.transpose() * Vt;

  ta = U.col(2);
  tb = -ta;

  return  {{Ra, ta}, {Ra, tb}, {Rb, ta}, {Rb, tb}};
}

auto cofactors_transposed(const Matrix3d& E)
{
  Matrix3d cofE;
  cofE.col(0) = E.col(1).cross(E.col(2));
  cofE.col(1) = E.col(2).cross(E.col(0));
  cofE.col(2) = E.col(0).cross(E.col(1));
  return cofE;
}

auto extract_relative_motion_horn(const Matrix3d& E) -> std::vector<Motion>
{
  const Matrix3d EEt = E * E.transpose();
  const Matrix3d cofET = cofactors_transposed(E);
  const RowVector3d norm_cofE = cofET.colwise().norm();

  auto i = int{};
  norm_cofE.maxCoeff(&i);

  const Vector3d ta = cofET.col(i) / norm_cofE(i) * std::sqrt(0.5 * EEt.trace());
  const Vector3d tb = -ta;

  const double ta_sq_norm = ta.squaredNorm();
  const Matrix3d Ra =
      (cofET - skew_symmetric_matrix(ta) * E) / ta_sq_norm;
  const auto F =  2. * (ta * ta.transpose()) / ta_sq_norm - Matrix3d::Identity();

  const Matrix3d Rb = F * Ra;

  return  {{Ra, ta}, {Ra, tb}, {Rb, ta}, {Rb, tb}};
}


BOOST_AUTO_TEST_CASE(test_extract_relative_motions_functions)
{
  const auto [X, R, t, E, C1, C2, x1, x2] = generate_test_data();

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
    auto motions = extract_relative_motion_svd(E);
    const auto motion_found =
        std::find_if(motions.begin(), motions.end(), motion_equality_predicate);
    BOOST_CHECK(motion_found != motions.end());
    //SARA_DEBUG << motion_found->R - true_motion.R << endl;
    //SARA_DEBUG << motion_found->t - true_motion.t << endl;
  }

  {
    auto motions = extract_relative_motion_horn(E);
    const auto motion_found =
        std::find_if(motions.begin(), motions.end(), motion_equality_predicate);
    BOOST_CHECK(motion_found != motions.end());
    //SARA_DEBUG << motion_found->R - true_motion.R << endl;
    //SARA_DEBUG << motion_found->t - true_motion.t << endl;
  }
}

BOOST_AUTO_TEST_SUITE_END()
