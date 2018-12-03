#define BOOST_TEST_MODULE "MultiViewGeometry/Eight Point Algorithm"

#include <DO/Sara/MultiViewGeometry/Estimators/EightPointAlgorithms.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/Fundamental.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_SUITE(TestEightPoingAlgorithm)

BOOST_AUTO_TEST_CASE(test_solve)
{
  auto Xn = Matrix<double, 3, 8>{};
  auto Yn = Matrix<double, 3, 8>{};

  Xn <<
    0.494292, 0.449212, 0.513487, 0.474079, 0.468652, 0.442959, 0.276826, 0.253816,
    0.734069, 0.595362, 0.685816,  0.58693, 0.689338, 0.577366, 0.117057, 0.675353,
           1,        1,        1,        1,        1,        1,        1,        1;

  Yn <<
    0.792952, 0.734874, 0.814332, 0.763281,   0.7605, 0.727001, 0.537151, 0.530029,
    0.644436, 0.515263, 0.596448, 0.504156, 0.603078, 0.498954, 0.115756, 0.604387,
           1,        1,        1,        1,        1,        1,        1,        1;


  auto F = FundamentalMatrix<>{};
  eight_point_fundamental_matrix(Xn, Yn, F);

  std::cout << "Algebraic errors" << std::endl;
  for (int i = 0; i < 8; ++i)
  {
    std::cout << Xn.col(i).transpose() * F.matrix() * Yn.col(i) << std::endl;
    const double error = Xn.col(i).transpose() * F.matrix() * Yn.col(i);
    BOOST_CHECK_LE(error, 1e-3);
  }
}

BOOST_AUTO_TEST_SUITE_END()
