#include <gtest/gtest.h>

#include <DO/Sara/FeatureDescriptors/DenseFeature.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestDenseFeature, test_function)
{
  auto image = Image<float>{ 10, 10 };
  auto dense_sifts = compute_dense_sift(image);

  EXPECT_MATRIX_EQ(image.sizes(), dense_sifts.sizes());
  EXPECT_MATRIX_EQ(Vector128f::Zero(), dense_sifts(0, 0));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}