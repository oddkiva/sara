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

vector<OERegion> computeDoGExtrema(const Image<float>& I,
                                   bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    printStage("Localizing DoG extrema");
    tic();
  }
  ImagePyramidParams pyrParams(0);
  ComputeDoGExtrema computeDoGs(pyrParams);
  vector<OERegion> DoGs;
  vector<Point2i> scaleOctPairs;
  DoGs = computeDoGs(I, &scaleOctPairs);
  if (verbose)
    toc();
  CHECK(DoGs.size());

  // 2. Rescale detected features to original image dimension.
  const ImagePyramid<float>& DoGPyr = computeDoGs.diffOfGaussians();
  for (int i = 0; i < DoGs.size(); ++i)
  {
    float octScaleFact = DoGPyr.octaveScalingFactor(scaleOctPairs[i](1));
    DoGs[i].center() *= octScaleFact;
    DoGs[i].shapeMat() /= pow(octScaleFact, 2);
  }

  return DoGs;
}


vector<OERegion> computeDoGAffineExtrema(const Image<float>& I,
                                         bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    printStage("Localizing DoG affine-adapted extrema");
    tic();
  }
  ImagePyramidParams pyrParams(0);
  ComputeDoGExtrema computeDoGs(pyrParams);
  vector<OERegion> DoGs;
  vector<Point2i> scaleOctPairs;
  DoGs = computeDoGs(I, &scaleOctPairs);
  if (verbose)
    toc();
  CHECK(DoGs.size());

  const ImagePyramid<float>& gaussPyr = computeDoGs.gaussians();
  const ImagePyramid<float>& dogPyr = computeDoGs.diffOfGaussians();

  // 2. Affine shape adaptation
  if (verbose)
  {
    printStage("Affine shape adaptation");
    tic();
  }
  AdaptFeatureAffinelyToLocalShape adaptShape;
  vector<int> keepFeatures(DoGs.size(), 0);
  for (size_t i = 0; i != DoGs.size(); ++i)
  {
    const int s = scaleOctPairs[i](0);
    const int o = scaleOctPairs[i](1);

    Matrix2f affAdaptTransformMat;
    if (adaptShape(affAdaptTransformMat, gaussPyr(s,o), DoGs[i]))
    {
      DoGs[i].shapeMat() = affAdaptTransformMat*DoGs[i].shapeMat();
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
      keptDoGs.push_back(DoGs[i]);
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
  name = srcPath("sunflowerField.jpg");
  if (!load(I, name))
    return -1;

  openWindow(I.width(), I.height());
  vector<OERegion> features;
  
  features = computeDoGExtrema(I);
  checkKeys(I, features);

  features = computeDoGAffineExtrema(I);
  checkKeys(I, features);

  return 0;
}