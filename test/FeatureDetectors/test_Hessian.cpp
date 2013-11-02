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

vector<OERegion> computeHessianLaplaceAffineCorners(const Image<float>& I,
                                                    bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    printStage("Localizing Hessian-Laplace interest points");
    tic();
  }
  ComputeHessianLaplaceMaxima computeDoHs;
  vector<OERegion> heslapMaxima;
  vector<Point2i> scaleOctPairs;
  heslapMaxima = computeDoHs(I, &scaleOctPairs);
  if (verbose)
    toc();

  const ImagePyramid<float>& gaussPyr = computeDoHs.gaussians();
  const ImagePyramid<float>& detHessians = computeDoHs.detOfHessians();

  // 2. Affine shape adaptation
  if (verbose)
  {
    printStage("Affine shape adaptation");
    tic();
  }
  AdaptFeatureAffinelyToLocalShape adaptShape;
  vector<int> keepFeatures(heslapMaxima.size(), 0);
  for (size_t i = 0; i != heslapMaxima.size(); ++i)
  {
    const int s = scaleOctPairs[i](0);
    const int o = scaleOctPairs[i](1);

    Matrix2f affAdaptTransformMat;
    if (adaptShape(affAdaptTransformMat, gaussPyr(s,o), heslapMaxima[i]))
    {
      heslapMaxima[i].shapeMat() = affAdaptTransformMat*heslapMaxima[i].shapeMat();
      keepFeatures[i] = 1;
    }
  }
  if (verbose)
    toc();


  // 3. Rescale the kept features to original image dimensions.
  size_t num_kept_features = 
    std::accumulate(keepFeatures.begin(), keepFeatures.end(), 0);

  vector<OERegion> keptDoHs;
  keptDoHs.reserve(num_kept_features);
  for (size_t i = 0; i != keepFeatures.size(); ++i)
  {
    if (keepFeatures[i] == 1)
    {
      keptDoHs.push_back(heslapMaxima[i]);
      const float fact = detHessians.octaveScalingFactor(scaleOctPairs[i](1));
      keptDoHs.back().shapeMat() *= pow(fact,-2);
      keptDoHs.back().coords() *= fact;
    }    
  }

  return keptDoHs;
}

vector<OERegion> computeDoHExtrema(const Image<float>& I,
  bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    printStage("Localizing DoH interest points");
    tic();
  }
  ComputeDoHExtrema computeDoHs;
  vector<OERegion> DoHs;
  vector<Point2i> scaleOctPairs;
  DoHs = computeDoHs(I, &scaleOctPairs);
  if (verbose)
    toc();
  CHECK(DoHs.size());

  const ImagePyramid<float>& gaussPyr = computeDoHs.gaussians();
  const ImagePyramid<float>& detHessians = computeDoHs.detOfHessians();

  // 2. Rescale feature points to original image dimensions.
  for (size_t i = 0; i != DoHs.size(); ++i)
  {
    const float fact = detHessians.octaveScalingFactor(scaleOctPairs[i](1));
    DoHs[i].shapeMat() *= pow(fact,-2);
    DoHs[i].coords() *= fact;  
  }

  return DoHs;
}

vector<OERegion> computeDoHAffineExtrema(const Image<float>& I,
                                   bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    printStage("Localizing DoH affine interest points");
    tic();
  }
  ComputeDoHExtrema computeDoHs;
  vector<OERegion> DoHs;
  vector<Point2i> scaleOctPairs;
  DoHs = computeDoHs(I, &scaleOctPairs);
  if (verbose)
    toc();
  CHECK(DoHs.size());

  const ImagePyramid<float>& gaussPyr = computeDoHs.gaussians();
  const ImagePyramid<float>& detHessians = computeDoHs.detOfHessians();

  // 2. Affine shape adaptation
  if (verbose)
  {
    printStage("Affine shape adaptation");
    tic();
  }
  AdaptFeatureAffinelyToLocalShape adaptShape;
  vector<int> keepFeatures(DoHs.size(), 0);
  for (size_t i = 0; i != DoHs.size(); ++i)
  {
    const int s = scaleOctPairs[i](0);
    const int o = scaleOctPairs[i](1);

    Matrix2f affAdaptTransformMat;
    if (adaptShape(affAdaptTransformMat, gaussPyr(s,o), DoHs[i]))
    {
      DoHs[i].shapeMat() = affAdaptTransformMat*DoHs[i].shapeMat();
      keepFeatures[i] = 1;
    }
  }
  if (verbose)
    toc();


  // 3. Rescale the kept features to original image dimensions.
  size_t num_kept_features = 
    std::accumulate(keepFeatures.begin(), keepFeatures.end(), 0);

  vector<OERegion> keptDoHs;
  keptDoHs.reserve(num_kept_features);
  for (size_t i = 0; i != keepFeatures.size(); ++i)
  {
    if (keepFeatures[i] == 1)
    {
      keptDoHs.push_back(DoHs[i]);
      const float fact = detHessians.octaveScalingFactor(scaleOctPairs[i](1));
      keptDoHs.back().shapeMat() *= pow(fact,-2);
      keptDoHs.back().coords() *= fact;

    }    
  }

  return keptDoHs;
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
  features = computeHessianLaplaceAffineCorners(I);
  checkKeys(I, features);

  /*features = computeDoHExtrema(I);
  checkKeys(I, features);

  features = computeDoHAffineExtrema(I);
  checkKeys(I, features);*/

  return 0;
}