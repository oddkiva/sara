#include <algorithm>
#include <cmath>

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>


using namespace DO::Sara;
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

vector<OERegion> computeDoGExtrema(const Image<float>& I,
                                   bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing DoG extrema");
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
    float octScaleFact = DoGPyr.octave_scaling_factor(scaleOctPairs[i](1));
    DoGs[i].center() *= octScaleFact;
    DoGs[i].shape_matrix() /= pow(octScaleFact, 2);
  }

  return DoGs;
}


vector<OERegion> computeDoGAffineExtrema(const Image<float>& I,
                                         bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing DoG affine-adapted extrema");
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
    print_stage("Affine shape adaptation");
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
      DoGs[i].shape_matrix() = affAdaptTransformMat*DoGs[i].shape_matrix();
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
      const float fact = dogPyr.octave_scaling_factor(scaleOctPairs[i](1));
      keptDoGs.back().shape_matrix() *= pow(fact,-2);
      keptDoGs.back().coords() *= fact;

    }
  }

  CHECK(keptDoGs.size());

  return keptDoGs;
}

void check_keys(const Image<float>& I, const vector<OERegion>& features)
{
  display(I);
  set_antialiasing();
  for (size_t i = 0; i != features.size(); ++i)
    features[i].draw(features[i].extremum_type() == OERegion::Max ?
                     Red8 : Blue8);
  get_key();
}

GRAPHICS_MAIN()
{
  Image<float> I;
  string name;
  name = src_path("../../datasets/sunflowerField.jpg");
  if (!load(I, name))
    return -1;

  create_window(I.width(), I.height());
  vector<OERegion> features;

  features = computeDoGExtrema(I);
  check_keys(I, features);

  features = computeDoGAffineExtrema(I);
  check_keys(I, features);

  return 0;
}