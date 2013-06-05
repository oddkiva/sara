#include <DO/FeatureDetectors.hpp>
#include <DO/Graphics.hpp>
#include <algorithm>
#include <cmath>

using namespace DO;
using namespace std;

static HighResTimer timer;
double elapsed = 0.0;
void tic()
{
  timer.restart();
}

void toc()
{
  elapsed = timer.elapsedMs();
  cout << "Elapsed time = " << elapsed << " ms" << endl << endl;
}

void testHessian(const Image<float>& I)
{
  // Chek out the image.
  openWindow(I.width(), I.height());
  display(I);
  getKey();

  // Determinant of Hessian
  Image<float> DoH;
  DoH = I.convert<float>().
    compute<Gaussian>(1.6f).
    compute<Hessian>().
    compute<Determinant>();
  display(DoH.compute<ColorRescale>());
  getKey();

  // Find local maxima
  vector<Point2i> extrema( localMaxima(DoH) );
  display(I);
  setAntialiasing();
  for (size_t i = 0; i != extrema.size(); ++i)
    fillCircle(extrema[i], 5, Red8);
  getKey();
}

void testMultiScaleHessian(const Image<float>& I)
{
  openWindow(I.width(), I.height());
  setAntialiasing();

  int firstOctave = -1;
  int numScalesPerOctave = 3;
  ImagePyramidParams pyrParams(firstOctave, numScalesPerOctave+2, 
                               pow(2.f, 1.f/numScalesPerOctave), 2);
  ImagePyramid<float> G(gaussianPyramid(I, pyrParams));
  //checkImagePyramid(G);

  ImagePyramid<float> D(DoHPyramid(G));
  //checkImagePyramid(D, true);

  display(I);
  for (int o = 0; o < D.numOctaves(); ++o)
  {
    // Verbose.
    printStage("Processing octave");
    cout << "Octave " << o << endl;
    cout << "Octave scaling factor = " << D.octaveScalingFactor(o) << endl;

    // Be careful of the bounds. We go from 1 to N-1.
    for (int s = 1; s < D.numScalesPerOctave()-1; ++s)
    {
      vector<OERegion> extrema( localScaleSpaceExtrema(D,s,o,1e-6f,10.f,2) );

      // Verbose.
      printStage("Detected extrema");
      cout << "[" << s << "] sigma = " << D.scale(s,o) << endl;
      cout << "    num extrema = " << extrema.size() << endl;

      // Draw the keypoints.
      //drawExtrema(D, extrema, s, o);
      //getKey();

      //display(D(s,o).compute<ColorRescale>(), 0, 0, D.octaveScalingFactor(o));
      //display(I);
      for (size_t i = 0; i != extrema.size(); ++i)
        extrema[i].drawOnScreen(
        extrema[i].extremumType()==OERegion::Max?Red8:Blue8,
        D.octaveScalingFactor(o));
      getKey();
    }
  }

  closeWindow();
}

int main()
{
  Image<float> I;
  string name;
  name = srcPath("sunflowerField.jpg");
  if (!load(I, name))
    return -1;

  testMultiScaleHessian(I);

  return 0;
}