#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
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

// A helper function
// Be aware that detection parameters are those set by default, e.g.,
// - thresholds like on extremum responses,
// - number of iterations in the keypoint localization,...
// Keypoints are described with the SIFT descriptor.
vector<OERegion> computeHarrisLaplaceAffineCorners(const Image<float>& I,
                                                   bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    printStage("Localizing Harris-Laplace interest points");
    tic();
  }
  ComputeHarrisLaplaceCorners computeCorners;
  vector<OERegion> corners;
  vector<Point2i> scaleOctPairs;
  corners = computeCorners(I, &scaleOctPairs);
  if (verbose)
    toc();

  const ImagePyramid<float>& gaussPyr = computeCorners.gaussians();
  const ImagePyramid<float>& harrisPyr = computeCorners.harris();

  // 2. Affine shape adaptation
  if (verbose)
  {
    printStage("Affine shape adaptation");
    tic();
  }
  AdaptFeatureAffinelyToLocalShape adaptShape;
  vector<int> keepFeatures(corners.size(), 0);
  for (size_t i = 0; i != corners.size(); ++i)
  {
    const int s = scaleOctPairs[i](0);
    const int o = scaleOctPairs[i](1);

    Matrix2f affAdaptTransformMat;
    if (adaptShape(affAdaptTransformMat, gaussPyr(s,o), corners[i]))
    {
      corners[i].shapeMat() = affAdaptTransformMat*corners[i].shapeMat();
      keepFeatures[i] = 1;
    }
  }
  if (verbose)
    toc();

  // 3. Rescale the kept features to original image dimensions.
  size_t num_kept_features =
    std::accumulate(keepFeatures.begin(), keepFeatures.end(), 0);

  vector<OERegion> keptCorners;
  keptCorners.reserve(num_kept_features);
  for (size_t i = 0; i != keepFeatures.size(); ++i)
  {
    if (keepFeatures[i] == 1)
    {
      keptCorners.push_back(corners[i]);
      const float fact = harrisPyr.octaveScalingFactor(scaleOctPairs[i](1));
      keptCorners.back().shapeMat() *= pow(fact,-2);
      keptCorners.back().coords() *= fact;

    }
  }

  CHECK(keptCorners.size());

  return keptCorners;
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
  name = srcPath("../../datasets/sunflowerField.jpg");
  if (!load(I, name))
    return -1;

  openWindow(I.width(), I.height());
  vector<OERegion> features;
  features = computeHarrisLaplaceAffineCorners(I);
  checkKeys(I, features);

  return 0;
}