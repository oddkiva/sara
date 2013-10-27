#include <gtest/gtest.h>
#include <DO/ImageProcessing.hpp>
#include <DO/Graphics.hpp>
#include <exception>

using namespace DO;
using namespace std;

//TEST(DO_ImageProcessing_test2, localExtremumTest)
//{
//  // Simple test case.
//  Image<float> I(10,10);
//  I.matrix().fill(1.f);
//
//  // Local maximality and minimality
//  CompareWithNeighborhood3<greater_equal, float> greater_equal33;
//  CompareWithNeighborhood3<less_equal, float> less_equal33;
//  // Strict local maximality and minimality
//  CompareWithNeighborhood3<greater, float> greater33;
//  CompareWithNeighborhood3<less, float> less33;
//
//  // Check local maximality
//  EXPECT_FALSE(greater33(I(1,1), 1, 1, I, true));
//  EXPECT_FALSE(greater33(I(1,1), 1, 1, I, false));
//  EXPECT_TRUE(greater_equal33(I(1,1),1,1,I,true));
//  EXPECT_TRUE(greater_equal33(I(1,1),1,1,I,false));
//  // Check local minimality
//  EXPECT_FALSE(less33(I(1,1), 1, 1, I, true));
//  EXPECT_FALSE(less33(I(1,1), 1, 1, I, false));
//  EXPECT_TRUE(less_equal33(I(1,1),1,1,I,true));
//  EXPECT_TRUE(less_equal33(I(1,1),1,1,I,false));
//  // Check that aliases are correctly defined.
//  EXPECT_FALSE(StrictLocalMax<float>()(1, 1, I));
//  EXPECT_FALSE(StrictLocalMin<float>()(1, 1, I));
//  vector<Point2i> maxima;
//  vector<Point2i> minima;
//  
//  maxima = strictLocalMaxima(I);
//  EXPECT_TRUE(maxima.empty());
//  maxima = localMaxima(I);
//  EXPECT_TRUE(maxima.size() == 8*8);
//  
//  minima = strictLocalMinima(I);
//  EXPECT_TRUE(minima.empty());
//  minima = localMinima(I);
//  EXPECT_TRUE(minima.size() == 8*8);
//
//  I(1,1) = 10.f;
//  I(7,7) = 10.f;
//  EXPECT_TRUE(greater33(I(1,1), 1, 1, I, false));
//  EXPECT_FALSE(greater33(I(1,1), 1, 1, I, true));
//  EXPECT_TRUE(LocalMax<float>()(1, 1, I));
//  EXPECT_TRUE(StrictLocalMax<float>()(1, 1, I));
//  EXPECT_FALSE(LocalMin<float>()(1, 1, I));
//  EXPECT_FALSE(StrictLocalMin<float>()(1, 1, I));
//  
//  maxima = strictLocalMaxima(I);
//  EXPECT_EQ(maxima.size(), 2);
//  minima = strictLocalMinima(I);
//  EXPECT_TRUE(minima.empty());
//
//  I.matrix() *= -1;
//  EXPECT_TRUE(less33(I(1,1), 1, 1, I, false));
//  EXPECT_FALSE(less33(I(1,1), 1, 1, I, true));
//  EXPECT_TRUE(StrictLocalMin<float>()(1, 1, I));
//  EXPECT_TRUE(LocalMin<float>()(1, 1, I));
//  maxima = strictLocalMaxima(I);
//  minima = strictLocalMinima(I);
//
//  EXPECT_TRUE(maxima.empty());
//  EXPECT_EQ(minima.size(), 2);
//}
//
//TEST(DO_ImageProcessing_test2, imagePyramidTest)
//{
//  ImagePyramid<float> I;
//  I.reset(2,3,1.6f,pow(2.f, 1.f/3.f));
//  EXPECT_EQ(I.numOctaves(), 2);
//  for (int i = 0; i < I.numOctaves(); ++i)
//    EXPECT_EQ(I(i).size(), I.numScalesPerOctave());
//}
//
//TEST(DO_ImageProcessing_test2, localScaleSpaceExtremumTest)
//{
//  ImagePyramid<double> I;
//  I.reset(1,3,1.6f,pow(2., 1./3.));
//  for (int i = 0; i < 3; ++i)
//  {
//    I(i,0).resize(10,10);
//    I(i,0).matrix().fill(1);
//  }
//  EXPECT_FALSE(StrictLocalScaleSpaceMax<double>()(1,1,1,0,I));
//  EXPECT_FALSE(StrictLocalScaleSpaceMin<double>()(1,1,1,0,I));
//
//  cout << "Local scale-space extrema test 1" << endl;
//  I(1,1,1,0) = 10.f;
//  I(7,7,1,0) = 10.f;
//  EXPECT_TRUE(StrictLocalScaleSpaceMax<double>()(1,1,1,0,I));
//  EXPECT_FALSE(StrictLocalScaleSpaceMin<double>()(1,1,1,0,I));
//  
//  vector<Point2i> maxima, minima;
//  maxima = strictLocalScaleSpaceMaxima(I,1,0);
//  minima = strictLocalScaleSpaceMinima(I,1,0);
//  EXPECT_EQ(maxima.size(), 2);
//  EXPECT_TRUE(minima.empty());
//
//  cout << "Local scale-space extrema test 2" << endl;
//  I(1,1,1,0) *= -1.f;
//  I(7,7,1,0) *= -1.f;
//  maxima = strictLocalScaleSpaceMaxima(I,1,0);
//  minima = strictLocalScaleSpaceMinima(I,1,0);
//  EXPECT_FALSE(LocalScaleSpaceMax<double>()(1,1,1,0,I));
//  EXPECT_FALSE(StrictLocalScaleSpaceMax<double>()(1,1,1,0,I));
//  EXPECT_TRUE(LocalScaleSpaceMin<double>()(1,1,1,0,I));
//  EXPECT_TRUE(StrictLocalScaleSpaceMin<double>()(1,1,1,0,I));
//  EXPECT_TRUE(maxima.empty());
//  EXPECT_TRUE(minima.size() == 2);
//}

template <class ChannelType>
class GaussPyrTest : public testing::Test
{
protected:
  typedef testing::Test Base;
  GaussPyrTest() : Base() {}
};

// For types Rgb32f, Rgb64f, the test compiles with MSVC10 but not with gcc.
typedef testing::Types<float/*, double, Rgb32f, Rgb64f*/> ChannelTypes;

TYPED_TEST_CASE_P(GaussPyrTest);

template <typename T>
void displayImagePyramid(const ImagePyramid<T>& pyramid, bool rescale = false)
{
  for (int o = 0; o < pyramid.numOctaves(); ++o)
  {
    cout << "Octave " << o << endl;
    for (int s = 0; s != int(pyramid(o).size()); ++s)
    {
      cout << "image " << s << endl;
      cout << pyramid.octaveScalingFactor(o) << endl;
      display(rescale ? colorRescale(pyramid(s,o)) : pyramid(s,o), 
        0, 0, pyramid.octaveScalingFactor(o));
      getKey();
    }
  }
}

static HighResTimer timer;
inline void tic() { timer.restart(); }
inline void toc(string task)
{
  double elapsed = timer.elapsedMs();
  cout << task << " time = " << elapsed << " ms" << endl;
}

TYPED_TEST_P(GaussPyrTest, gaussianPyramidTest)
{
  typedef TypeParam T;
  Image<T> I;
  ASSERT_TRUE(load(I, srcPath("sunflowerField.jpg")));

  openWindow(I.width(), I.height());
  tic();
  ImagePyramid<T> G(gaussianPyramid(I, ImagePyramidParams(-1)));
  toc("Gaussian pyramid");
  displayImagePyramid(G);

  tic();
  ImagePyramid<T> D(DoGPyramid(G));
  toc("DoG pyramid");
  displayImagePyramid(D, true);

  tic();
  ImagePyramid<T> L(LoGPyramid(G));
  toc("LoG pyramid");
  displayImagePyramid(L, true);

  closeWindow();
}

REGISTER_TYPED_TEST_CASE_P(GaussPyrTest, gaussianPyramidTest);
INSTANTIATE_TYPED_TEST_CASE_P(DO_ImageProcessing_Pyramid_Test,
                              GaussPyrTest, ChannelTypes);

//#undef main

int main(/*int argc, char** argv*/)
{
  int argc = 0;
  char **argv = 0;
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS(); 

  return 0;
}
