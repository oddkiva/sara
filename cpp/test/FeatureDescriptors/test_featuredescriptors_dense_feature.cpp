#define BOOST_TEST_MODULE "FeatureDescriptors/Dense Feature"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/FeatureDescriptors/DenseFeature.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestDenseFeature)

BOOST_AUTO_TEST_CASE(test_function)
{
  auto image = Image<float>{ 10, 10 };
  auto dense_sifts = compute_dense_sift(image);

  BOOST_CHECK_EQUAL(image.sizes(), dense_sifts.sizes());
  BOOST_CHECK_EQUAL(Vector128f::Zero(), dense_sifts(0, 0));
}

BOOST_AUTO_TEST_SUITE_END()