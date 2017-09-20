#define BOOST_TEST_MODULE "MultiViewGeometry/Eight Point Algorithm"

#include <DO/Sara/MultiViewGeometry/Estimators/FundamentalMatrixEstimator.hpp>

#include <boost/test/unit_test.hpp>


using namespace DO::Sara;
using namespace std;


class TestNotImplementedException : public std::exception
{
public:
  virtual const char * what() const throw()
  {
    return "Test is not implemented!";
  }
};

BOOST_AUTO_TEST_SUITE(TestEightPoingAlgorithm)

BOOST_AUTO_TEST_CASE(test_solve)
{
  throw TestNotImplementedException{};
}

BOOST_AUTO_TEST_SUITE_END()
