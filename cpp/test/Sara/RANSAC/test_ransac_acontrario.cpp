#define BOOST_TEST_MODULE "RANSAC/A Contrario Method"

#include <DO/Sara/RANSAC/AContrario.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <random>


namespace sara = DO::Sara;

struct LineSolver
{
  using model_type = Eigen::Vector3d;
  using matrix_type = Eigen::Matrix2d;
  using matrix_view_type = Eigen::Map<const matrix_type>;

  static constexpr auto num_points = 2;
  static constexpr auto num_candidate_models = 1;

  auto operator()(const matrix_type& x) const
      -> std::array<model_type, num_candidate_models>
  {
    const auto u = x.col(0);
    const auto v = x.col(1);
    Eigen::Vector3d l = u.homogeneous().cross(v.homogeneous());
    l /= l.head(2).norm();
    return {l};
  }
};

struct NormalizedPointToLineDistance
{
  NormalizedPointToLineDistance(const Eigen::Vector2d& domain)
    : domain{domain}
    , norm_factor{normalization_factor(domain)}
  {
    SARA_CHECK(domain.transpose());
    SARA_CHECK(norm_factor);
  }

  // L1-norm
  auto operator()(const Eigen::Vector3d& l, const Eigen::Vector2d& x) const
      -> double
  {
    return std::abs(l.dot(x.homogeneous()) * norm_factor);
  }

  static auto normalization_factor(const Eigen::Vector2d& sizes) -> double
  {
    const auto diagonal = sizes.norm();
    const auto area = static_cast<double>(sizes.x() * sizes.y());
    return 2 * diagonal / area;
  };

  const Eigen::Vector2d& domain;
  const double norm_factor = normalization_factor(domain);
};


BOOST_AUTO_TEST_SUITE(TestAContrario)

BOOST_AUTO_TEST_CASE(test_line_solver)
{
  auto x = Eigen::Matrix2d{};
  // clang-format off
  x << 0., 5.,
       0., 5.;
  // clang-format on

  const auto [l] = LineSolver{}(x);
  const auto l_unnormalized = Eigen::Vector3d{-1., 1, 0.};
  const Eigen::Vector3d l_true = l_unnormalized / l_unnormalized.head(2).norm();
  BOOST_CHECK_SMALL((l - l_true).norm(), 1e-12);

  // The point to line distance must be very small.
  BOOST_CHECK_SMALL((x.colwise().homogeneous().transpose() * l).norm(), 1e-12);

  // ChecK the normalized point-to-line distance.
  const auto domain = Eigen::Vector2d{10., 10.};
  const auto d = NormalizedPointToLineDistance{domain};
  for (auto i = 0; i < x.cols(); ++i)
  {
    const auto di = d(l, x.col(i));
    BOOST_CHECK_SMALL(di, 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(test_log_nfa_calculus_in_line_fitting_problems)
{
  // Initialize the pseudo-random engine.
  auto rd = std::random_device{};
  auto rng = std::mt19937{rd()};

  // The inlier ratio.
  static constexpr auto p = 0.1;

  // Scope the domain of data points.
  static constexpr auto w = 1000.;
  static constexpr auto h = 1000.;
  const auto domain = Eigen::Vector2d{w, h};

  // Define the ground-truth line.
  static constexpr auto a = 1.;
  static constexpr auto b = 0.;
  auto l_true = Eigen::Vector3d{a, -1., b};
  l_true /= l_true.head(2).norm();

  // Specify the Gaussian distribution of residuals.
  static constexpr auto μ = 0.;
  static constexpr auto σ = 1.;
  auto N = std::normal_distribution{μ, σ};

  // Define the distribution of outliers.
  auto U = std::array{
      std::uniform_real_distribution{0., w},
      std::uniform_real_distribution{0., h}  //
  };

  //
  auto π = std::bernoulli_distribution{p};

  // The point sample function.
  const auto sample = [&]() {
    // Inlier or outlier.
    const auto type = π(rng);

    auto x = Eigen::Vector2d{};

    if (type == 0)
      x << U[0](rng), U[1](rng);
    else
    {
      x.x() = U[0](rng);
      x.y() = a * x.x() + b + N(rng);
    }

    return std::make_tuple(x, type);
  };

  // Generate the data from the mixture of inliers and outliers.
  //
  // The number of data points.
  static constexpr auto n = 100;
  auto x = std::vector<Eigen::Vector2d>(n);

  // The normalized point-to-line distance.
  const auto d = NormalizedPointToLineDistance{domain};

  // Statistics.
  auto count = std::array<int, 2>{};

  // Sample the data points.
  for (auto i = 0; i < n; ++i)
  {
    // Inlier or outlier.
    const auto [xi, type] = sample();
    x[i] = xi;

    ++count[type];
    if (type == 1)
      std::cout << (type == 0 ? "O: " : "I: ") << x[i].transpose()
                << "|ε:" << d(l_true, x[i]) / d.norm_factor << std::endl;
  }

  std::cout << "inliers  = " << count[1] << std::endl;
  std::cout << "outliers = " << count[0] << std::endl;

  auto S = sara::DataPointIndices(n);
  std::iota(S.begin(), S.end(), 0u);
  const auto ε = sara::normalized_residuals(S, l_true, x, d);

#if 0
  for (auto i = LineSolver::num_points + 1; i < n; ++i)
    std::cout << "i:" << i << " ε:" << ε[i].value << std::endl;
#endif

  auto log_nfa = std::vector<double>(n);

  static_assert(LineSolver::num_candidate_models == 1);
  static_assert(LineSolver::num_points == 2);

  for (auto i = LineSolver::num_points + 1; i < n; ++i)
  {
    const auto& candidate_subset_size = i;

    log_nfa[i] =
        sara::log_nfa<LineSolver>(n, candidate_subset_size, ε[i].value);

    // Normalized distances are not "physically interpretable" and in practice,
    // setting some pixel threshold on point-to-line distance makes sense in
    // computer vision problems.
    if (ε[i].value > 10. * d.norm_factor)
      break;

    if (log_nfa[i] >= 0)
      break;

    std::cout << "i:" << i << " x:" << x[ε[i].index].transpose()
              << " ε[i]:" << ε[i].value / d.norm_factor
              << " log(nfa):" << log_nfa[i] << std::endl;
  }
}

BOOST_AUTO_TEST_SUITE_END()
