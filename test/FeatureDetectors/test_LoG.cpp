#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <algorithm>
#include <cmath>

using namespace DO;
using namespace std;

static Timer timer;
double elapsed = 0.0;
void tic()
{
  timer.restart();
}

void toc()
{
  elapsed = timer.elapsed_ms();
  cout << "Elapsed time = " << elapsed << " ms" << endl << endl;
}

vector<OERegion> computeLoGExtrema(const Image<float>& I,
                                   bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing LoG extrema");
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
    float octScaleFact = DoGPyr.octave_scaling_factor(scaleOctPairs[i](1));
    LoGs[i].center() *= octScaleFact;
    LoGs[i].shape_matrix() /= pow(octScaleFact, 2);
  }

  return LoGs;
}

vector<OERegion> computeLoGAffineExtrema(const Image<float>& I,
                                         bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing LoG affine-adapted extrema");
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
    print_stage("Affine shape adaptation");
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
      LoGs[i].shape_matrix() = affAdaptTransformMat*LoGs[i].shape_matrix();
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
      const float fact = dogPyr.octave_scaling_factor(scaleOctPairs[i](1));
      keptDoGs.back().shape_matrix() *= pow(fact,-2);
      keptDoGs.back().coords() *= fact;

    }
  }

  CHECK(keptDoGs.size());

  return keptDoGs;
}

void checkKeys(const Image<float>& I, const vector<OERegion>& features)
{
  display(I);
  set_antialiasing();
  for (size_t i = 0; i != features.size(); ++i)
    features[i].draw(features[i].extremum_type() == OERegion::Max ?
                     Red8 : Blue8);
  get_key();
}

int main()
{
  Image<float> I;
  string name;
  name = src_path("../../datasets/sunflowerField.jpg");
  if (!load(I, name))
    return -1;

  create_window(I.width(), I.height());
  vector<OERegion> features;

  features = computeLoGExtrema(I);
  checkKeys(I, features);

  features = computeLoGAffineExtrema(I);
  checkKeys(I, features);

  return 0;
}