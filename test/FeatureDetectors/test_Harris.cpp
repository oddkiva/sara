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

//#define CHECK_STEP_BY_STEP
#ifdef CHECK_STEP_BY_STEP
void testSimplifiedHarrisLaplace(const Image<float>& I,
                                 int firstOctave = -1,
                                 int numScalesPerOctaves = 2,
                                 bool debug = false)
{
  // Open window to check each step.
  Window win = openWindow(I.width(), I.height());
  setAntialiasing();

  // Harris pyramid
  printStage("Constructing Harris Pyramid");
  tic();
  ImagePyramid<float> cornerness;
  ImagePyramidParams harrisPyrParams(
    firstOctave, numScalesPerOctaves+1,
    pow(2.f, 1.f/numScalesPerOctaves), 1);
  cornerness = harrisCornernessPyramid(I, 0.04f, harrisPyrParams);
  toc();

  // Gaussian Pyramid
  printStage("Constructing Gaussian Pyramid");
  tic();
  ImagePyramid<float> G;
  G = gaussianPyramid(I, harrisPyrParams);
  toc();

  AdaptFeatureAffinelyToLocalShape adaptShape;
  
  display(I);
  for (int o = 0; o < cornerness.numOctaves(); ++o)
  {
    cout << "Octave " << o << endl;
    for (int s = 1; s < cornerness.numScalesPerOctave(); ++s)
    {
      cout << "image " << s << endl;
      cout << cornerness.octaveScalingFactor(o) << endl;

      // Find local maxima.
      vector<OERegion> corners;
      corners = laplaceMaxima(cornerness, G, s, o);
      cout << "Cornerness extrema: " << corners.size() << endl;

      // Display local maxima.
      float fact = cornerness.octaveScalingFactor(o);
      display(colorRescale(cornerness(s,o)), 0, 0, fact);

      // Check the affine shape adaptation.
      for (size_t i = 0; i != corners.size(); ++i)
      {
        Matrix2f shapeMat;
        if (adaptShape(shapeMat, G(s,o), corners[i]))
        {
          corners[i].shapeMat() = shapeMat*corners[i].shapeMat();
          // I believe Mikolajczyk dilates the shape to get the size ratio
          // of the feature of the shape.
          const float binFactorSz = 3.f;
          const float numBins = 2.f;
          corners[i].shapeMat() *= pow(binFactorSz*numBins,-2);
          corners[i].drawOnScreen(Blue8, fact);
        }        
      }
      getKey();
    }
  }
}
#endif

// A helper function
// Be aware that detection parameters are those set by default, e.g.,
// - thresholds like on extremum responses,
// - number of iterations in the keypoint localization,...
// Keypoints are described with the SIFT descriptor.
template <typename ComputeFeature>
vector<OERegion> computeAffineAdaptedKeypoints(const Image<float>& I,
                                               bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    printStage("Localizing interest points");
    tic();
  }
  ComputeFeature computeFeatures;
  vector<OERegion> feats;
  vector<Point2i> scaleOctPairs;
  feats = computeFeatures(I, &scaleOctPairs);
  if (verbose)
    toc();

  // 2. Affine shape adaptation
  if (verbose)
  {
    printStage("Affine shape adaptation");
    tic();
  }
  const ImagePyramid<float>& gaussPyr = computeFeatures.gaussians();
  AdaptFeatureAffinelyToLocalShape adaptShape;
  vector<int> keepFeatures(feats.size(), 0);
  for (size_t i = 0; i != feats.size(); ++i)
  {
    Matrix2f shapeMat;
    int s = scaleOctPairs[i](0);
    int o = scaleOctPairs[i](1);
    if (adaptShape(shapeMat, gaussPyr(s,o), feats[i]))
    {
      feats[i].shapeMat() = shapeMat*feats[i].shapeMat();
      keepFeatures[i] = 1;
    }
  }
  if (verbose)
    toc();

  // 3. Dominant orientations and photometric description, e.g., SIFT.
  if (verbose)
  {
    printStage("Dominant orientations and feature description");
    tic();
  }
  /*
    Compute the affinity that maps the normalized patch to the local region
    around feature $f$.
    We denote a point in the normalized patch by $(u,v) \in [0,w]^2$
    The center point is $(w/2, w/2)$ corresponds to the center $(x_f, y_f)$
    of feature $f$.

    We introduce the notion of 'scale unit', i.e.,
    $1$ scale unit is equivalent $\sigma$ pixels in the image.
  */
  /* 
    Let us set some important constants needed for the computation of the 
    normalized patch computation. 
   */
  // Patch "radius"
  const int patchRadius = 20;
  // Patch side length
  const int patchSideLength = 2*patchRadius+1;
  // Gaussian smoothing is involved in the computation of gradients orientations
  // to compute dominant orientations and the SIFT descriptor.
  const float gaussTruncFactor = 3.f; 
  // A normalized patch is composed of a grid of NxN square patches, i.e. bins,
  // centered on the feature
  const float binSideLength = 3.f; // side length of a bin in scale unit.
  const float numBins = 4.f;
  const float scaleRelRadius = sqrt(2.f)*binSideLength*(numBins+1)/2.f;
  // Store the keypoints here.
  vector<OERegion> keptFeats;
  keptFeats.reserve(2*feats.size());
  for (size_t i = 0; i != feats.size(); ++i)
  {
    if (keepFeatures[i] == 1)
    {
      // The linear transform computed from the SVD
      const Matrix2f& shapeMat = feats[i].shapeMat();
      JacobiSVD<Matrix2f> svd(shapeMat, ComputeFullU);
      Vector2f S(svd.singularValues().cwiseInverse().cwiseSqrt());
      S *= scaleRelRadius/patchRadius; // Scaling
      Matrix2f U(svd.matrixU()); // Rotation
      Matrix2f L(U*S.asDiagonal()*U.transpose()); // Linear transform.
      // The translation vector
      Vector2f t(L*Point2f::Ones()*(-patchRadius) + feats[i].center());
      // The affinity that maps the patch to the local region around the feature
      Matrix3f T(Matrix3f::Zero());
      T.block<2,2>(0,0) = L;
      T.col(2) << t, 1.f;

      // Get the normalized patch.
      Image<float> normalizedPatch(patchSideLength,patchSideLength);
      int s = scaleOctPairs[i](0);
      int o = scaleOctPairs[i](1);
      if (!warp(normalizedPatch, gaussPyr(s,o), T, 0.f, true))
        continue;


      // Rescale the feature position and shape to the original image
      // dimensions.
      double fact = gaussPyr.octaveScalingFactor(o);
      feats[i].shapeMat() *= pow(fact/**scaleRelRadius*/, -2);
      feats[i].center() *= fact;
      // Store the keypoint.
      keptFeats.push_back(feats[i]);
    }
  }
  if (verbose)
    toc();
  return keptFeats;
}

void checkKeys(const Image<float>& I, const vector<OERegion>& features)
{
  display(I);
  setAntialiasing();
  drawOERegions(features, Red8);
  getKey();
}

int main()
{
  Image<float> I;
  string name;
  name = srcPath("sunflowerField.jpg");
  if (!load(I, name))
    return -1;

  //testSimplifiedHarrisLaplace(I);

  openWindow(I.width(), I.height());
  vector<OERegion> features;
  
  features = computeAffineAdaptedKeypoints<ComputeDoGExtrema>(I);
  checkKeys(I, features);

  features = computeAffineAdaptedKeypoints<ComputeHarrisLaplaceCorners>(I);
  checkKeys(I, features);

  return 0;
}