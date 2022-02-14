#include "HomographyDecomposition.hpp"

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/Math/RQFactorization.hpp>


namespace sara = DO::Sara;


auto decompose_H_RQ_factorization(const Eigen::Matrix3d& H,
                                  const Eigen::Matrix3d& K,
                                  std::vector<Eigen::Matrix3d>& Rs,
                                  std::vector<Eigen::Vector3d>& ts,
                                  std::vector<Eigen::Vector3d>& ns) -> void
{
  const Eigen::Matrix3d invK = K.inverse();
  const Eigen::Matrix3d P = (invK * H).normalized();
  std::cout << "P =\n" << P << std::endl;

  const Eigen::Vector3d n = Eigen::Vector3d::UnitZ();

  auto R = Eigen::Matrix3d{};
  R.col(0) = P.col(0);
  R.col(1) = P.col(1);
  R.col(2) = R.col(0).cross(R.col(1));

  // My trick is to use the RQ factorization described in Multiple View
  // Geometry.
  auto K1 = Eigen::Matrix3d{};
  auto R1 = Eigen::Matrix3d{};
  sara::rq_factorization_3x3(R, K1, R1);

  const auto svd = K1.jacobiSvd();
  const Eigen::Vector3d S = svd.singularValues();

  auto t = Eigen::Vector3d{};
  t = P.col(2);

  SARA_DEBUG << "R1=\n" << R1 << std::endl;
  SARA_CHECK(R1.determinant());

  // And voila!
  Rs = {R1};
  // ts = {t / std::pow(S(0) * S(1) * S(2), 1 / 3.)};
  ts = {t / S(0)};
  ns = {n};
}

auto decompose_H_faugeras(const Eigen::Matrix3d& H, const Eigen::Matrix3d& K,
                          std::vector<Eigen::Matrix3d>& Rs,
                          std::vector<Eigen::Vector3d>& ts,
                          std::vector<Eigen::Vector3d>& ns) -> void
{
  const Eigen::Matrix3d invK = K.inverse();
  const Eigen::Matrix3d P = invK * H;

  const auto svd = P.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Vector3d S = svd.singularValues();

  const Eigen::Matrix3d U = svd.matrixU();
  const Eigen::Matrix3d V = svd.matrixV();

  Rs.clear();
  ts.clear();
  ns.clear();

  const auto& d1 = S(0);
  const auto& d2 = S(1);
  const auto& d3 = S(2);
  const auto detU_times_detV = U.determinant() * V.determinant();

  const auto R_prime_when_d_prime_positive =
      [d1, d2, d3](auto eps1, auto eps3) -> Eigen::Matrix3d {
    const auto q1 = (sara::square(d1) - sara::square(d2)) *
                    (sara::square(d2) - sara::square(d3));
    const auto q2_inv = 1 / ((d1 + d3) * d2);
    const auto q3 = sara::square(d2) + d1 * d3;

    const auto s = eps1 * eps3 * std::sqrt(q1) * q2_inv;
    const auto c = q3 * q2_inv;

    auto R = Eigen::Matrix3d{};
    // clang-format off
    R << c, 0, -s,
         0, 1,  0,
         s, 0,  c;
    // clang-format on

    return R;
  };

  const auto t_prime_and_n_prime_when_d_prime_positive =
      [d1, d2, d3, detU_times_detV](
          auto eps1, auto eps3) -> std::pair<Eigen::Vector3d, Eigen::Vector3d> {
    const auto q1 = sara::square(d1) - sara::square(d2);
    const auto q2_inv = 1 / (sara::square(d1) - sara::square(d3));
    const auto q3 = sara::square(d2) - sara::square(d3);

    const auto x1 = eps1 * std::sqrt(q1 * q2_inv);
    const auto x3 = eps3 * std::sqrt(q3 * q2_inv);

    const auto factor = d1 - d3;
    const auto d_prime = -d2;
    const auto d = detU_times_detV * d_prime;

    const auto t = Eigen::Vector3d{factor * x1 / d, 0, -factor * x3 / d};
    const auto n = Eigen::Vector3d{x1, 0, x3};

    return std::make_pair(t, n);
  };

  const auto R_prime_when_d_prime_negative =
      [d1, d2, d3](auto eps1, auto eps3) -> Eigen::Matrix3d {
    // R' is a symmetry i.e. a rotation of angle π.
    const auto q1 = (sara::square(d1) - sara::square(d2)) *
                    (sara::square(d2) - sara::square(d3));
    const auto q2_inv = 1 / ((d1 - d3) * d2);
    const auto q3 = d1 * d3 - sara::square(d2);

    const auto s = eps1 * eps3 * std::sqrt(q1) * q2_inv;
    const auto c = q3 * q2_inv;

    auto R = Eigen::Matrix3d{};
    // clang-format off
    R << c, 0,  s,
         0, 1,  0,
         s, 0, -c;
    // clang-format on

    return R;
  };

  const auto t_prime_and_n_prime_when_d_prime_negative =
      [d1, d2, d3, detU_times_detV](
          auto eps1, auto eps3) -> std::pair<Eigen::Vector3d, Eigen::Vector3d> {
    const auto q1 = sara::square(d1) - sara::square(d2);
    const auto q2_inv = 1 / (sara::square(d1) - sara::square(d3));
    const auto q3 = sara::square(d2) - sara::square(d3);

    const auto x1 = eps1 * q1 * q2_inv;
    const auto x3 = eps3 * q3 * q2_inv;

    const auto factor = d1 + d3;
    const auto d_prime = -d2;
    const auto d = detU_times_detV * d_prime;

    const auto t = Eigen::Vector3d{factor * x1 / d, 0, factor * x3 / d};
    const auto n = Eigen::Vector3d{x1, 0, x3};

    return std::make_pair(t, n);
  };

  const auto s = U.determinant() * V.determinant();

  // That case should be rare in practice.
  for (const auto& eps1 : {-1, 1})
  {
    for (const auto& eps3 : {-1, 1})
    {
      const Eigen::Matrix3d R =
          s * U * R_prime_when_d_prime_positive(eps1, eps3) * V.transpose();
      Rs.push_back(R);

      const auto [t, n] = t_prime_and_n_prime_when_d_prime_positive(eps1, eps3);
      ts.push_back(U * t);
      ns.push_back(V * n);
    }
  }

  // This should be the case most of the time in practice.
  for (const auto& eps1 : {-1, 1})
  {
    for (const auto& eps3 : {-1, 1})
    {
      SARA_CHECK(eps1);
      SARA_CHECK(eps3);
      const Eigen::Matrix3d R =
          s * U * R_prime_when_d_prime_negative(eps1, eps3) * V.transpose();
      Rs.push_back(R);

      const auto [t, n] = t_prime_and_n_prime_when_d_prime_negative(eps1, eps3);
      ts.push_back(U * t);
      ns.push_back(V * n);
    }
  }
}
