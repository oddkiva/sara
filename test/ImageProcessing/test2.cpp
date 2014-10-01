#include <gtest/gtest.h>
#include <DO/ImageProcessing.hpp>
#include <DO/Graphics.hpp>
#include <exception>

using namespace DO;
using namespace std;

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
