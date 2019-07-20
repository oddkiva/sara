#define BOOST_TEST_MODULE "Test ceres-solver backend"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <boost/test/unit_test.hpp>


struct CostFunctor
{
  template <typename T>
  bool operator()(const T* const x, T* residual) const
  {
    residual[0] = T(10.0) - x[0];
    return true;
  }
};


struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
    : observed_x{observed_x}
    , observed_y{observed_y}
  {
  }

  template <typename T>
  bool operator()(const T* const camera, // (1) camera parameters to optimize.
                  const T* const point,  // (2) 3D points to optimize
                  T* residuals) const
  {
    T p[3];
    // camera[0, 1, 2] are the angle axis rotation.
    ceres::AngleAxisRotatePoint(camera, point, p);

    // Camera
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Center of the distortion.
    // Snavely assumes the camera coordinate system has a negative z axis.
    T xp = -p[0] / p[2];
    T yp = -p[1] / p[2];

    // Apply second and fourth order order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    const auto r2 = xp * xp + yp * yp;
    const auto distortion = T(1) + r2 * (l1 + l2 * r2);

    const T& focal = camera[6];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;


    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y)
  {
    constexpr auto NumParams = 6 /* camera paramters */ + 3 /* points */;
    return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2,
                                           NumParams, 3>{
        new SnavelyReprojectionError{observed_x, observed_y}};
  }

  double observed_x;
  double observed_y;
};


BOOST_AUTO_TEST_CASE(test_ceres_solver_hello_world_tutorial)
{
  auto initial_x = 5.;
  auto x = initial_x;

  ceres::Problem problem{};

  auto cost_fn =
      new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>{new CostFunctor{}};

  problem.AddResidualBlock(cost_fn, nullptr, &x);

  auto options = ceres::Solver::Options{};
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  auto summary = ceres::Solver::Summary{};
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";
}

//BOOST_AUTO_TEST_CASE(test_ceres_solver_bundle_adjustment_tutorial)
//{
//  ceres::Problem problem;
//  for (int i = 0; i < bal_problem.num_observations(); ++i)
//  {
//    auto cost_fn =
//        SnavelyReprojectionError::Create(bal_problem.observations()[2 * i + 0],
//                                         bal_problem.observations()[2 * i + 1]);
//
//    problem.AddResidualBlock(cost_function,
//                             nullptr /* squared loss */,
//                             bal_problem.mutable_camera_for_observation(i),
//                             bal_problem.mutable_point_for_observation(i));
//  }
//
//  auto options = ceres::Solver::Options{};
//  options.linear_solver_type = ceres::DENSE_SCHUR;
//  options.minimizer_progress_to_stdout = true;
//  auto summary = ceres::Solver::Summary{};
//
//  ceres::Solve(options, &problem, &summary);
//  std::cout << summary.FullReport() << "\n";
//}
