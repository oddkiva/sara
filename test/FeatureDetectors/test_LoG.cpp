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

vector<OERegion> computeLoGExtrema(const Image<float>& I,
                                   bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    printStage("Localizing LoG extrema");
    tic();
  }
  ImagePyramidParams pyrParams(0, 3+2);
  ComputeLoGExtrema computeLoGs(pyrParams);
  vector<OERegion> LoGs;
  vector<Point2i> scaleOctPairs;
  LoGs = computeLoGs(I, &scaleOctPairs);
  if (verbose)
    toc();
  CHECK(LoGs.size());

  // 2. Rescale detected features to original image dimension.
  const ImagePyramid<float>& DoGPyr = computeLoGs.laplaciansOfGaussians();
  for (int i = 0; i < LoGs.size(); ++i)
  {
    float octScaleFact = DoGPyr.octaveScalingFactor(scaleOctPairs[i](1));
    LoGs[i].center() *= octScaleFact;
    LoGs[i].shapeMat() /= pow(octScaleFact, 2);
  }

  return LoGs;
}

vector<OERegion> computeLoGAffineExtrema(const Image<float>& I,
                                         bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    printStage("Localizing LoG affine-adapted extrema");
    tic();
  }
  ImagePyramidParams pyrParams(0);
  ComputeLoGExtrema computeLoGs(pyrParams);
  vector<OERegion> LoGs;
  vector<Point2i> scaleOctPairs;
  LoGs = computeLoGs(I, &scaleOctPairs);
  if (verbose)
    toc();
  CHECK(LoGs.size());

  const ImagePyramid<float>& gaussPyr = computeLoGs.gaussians();
  const ImagePyramid<float>& dogPyr = computeLoGs.laplaciansOfGaussians();

  // 2. Affine shape adaptation
  if (verbose)
  {
    printStage("Affine shape adaptation");
    tic();
  }
  AdaptFeatureAffinelyToLocalShape adaptShape;
  vector<int> keepFeatures(LoGs.size(), 0);
  for (size_t i = 0; i != LoGs.size(); ++i)
  {
    const int s = scaleOctPairs[i](0);
    const int o = scaleOctPairs[i](1);

    Matrix2f affAdaptTransformMat;
    if (adaptShape(affAdaptTransformMat, gaussPyr(s,o), LoGs[i]))
    {
      LoGs[i].shapeMat() = affAdaptTransformMat*LoGs[i].shapeMat();
      keepFeatures[i] = 1;
    }
  }
  if (verbose)
    toc();

  // 3. Rescale the kept features to original image dimensions.
  size_t num_kept_features =
    std::accumulate(keepFeatures.begin(), keepFeatures.end(), 0);

  vector<OERegion> keptDoGs;
  keptDoGs.reserve(num_kept_features);
  for (size_t i = 0; i != keepFeatures.size(); ++i)
  {
    if (keepFeatures[i] == 1)
    {
      keptDoGs.push_back(LoGs[i]);
      const float fact = dogPyr.octaveScalingFactor(scaleOctPairs[i](1));
      keptDoGs.back().shapeMat() *= pow(fact,-2);
      keptDoGs.back().coords() *= fact;

    }
  }

  CHECK(keptDoGs.size());

  return keptDoGs;
}

void checkKeys(const Image<float>& I, const vector<OERegion>& features)
{
  display(I);
  setAntialiasing();
  for (size_t i = 0; i != features.size(); ++i)
    features[i].draw(features[i].extremumType() == OERegion::Max ?
                     Red8 : Blue8);
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

  features = computeLoGExtrema(I);
  checkKeys(I, features);

  features = computeLoGAffineExtrema(I);
  checkKeys(I, features);

  return 0;
}