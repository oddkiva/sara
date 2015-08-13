#include <algorithm>
#include <cmath>

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>


using namespace DO::Sara;
using namespace std;


static Timer timer;

void tic()
{
  timer.restart();
}

void toc()
{
  auto elapsed = timer.elapsed_ms();
  cout << "Elapsed time = " << elapsed << " ms" << endl << endl;
}

vector<OERegion> compute_hessian_laplace_affine_maxima(const Image<float>& I,
                                                   bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing Hessian-Laplace interest points");
    tic();
  }
  ComputeHessianLaplaceMaxima compute_DoHs;
  auto hessian_laplace_maxima = vector<OERegion>{};
  auto scale_octave_pairs = vector<Point2i>{};
  hessian_laplace_maxima = compute_DoHs(I, &scale_octave_pairs);
  if (verbose)
    toc();

  const auto& G = compute_DoHs.gaussians();
  const auto& DoHs = compute_DoHs.det_of_hessians();

  // 2. Affine shape adaptation
  if (verbose)
  {
    print_stage("Affine shape adaptation");
    tic();
  }
  AdaptFeatureAffinelyToLocalShape adaptShape;
  auto keep_features = vector<unsigned char>(hessian_laplace_maxima.size(), 0);
  for (size_t i = 0; i != hessian_laplace_maxima.size(); ++i)
  {
    const int s = scale_octave_pairs[i](0);
    const int o = scale_octave_pairs[i](1);

    Matrix2f affine_adapt_transform;
    if (adaptShape(affine_adapt_transform, G(s,o), hessian_laplace_maxima[i]))
    {
      hessian_laplace_maxima[i].shape_matrix() = affine_adapt_transform*hessian_laplace_maxima[i].shape_matrix();
      keep_features[i] = 1;
    }
  }
  if (verbose)
    toc();


  // 3. Rescale the kept features to original image dimensions.
  auto num_kept_features = std::accumulate(
    keep_features.begin(), keep_features.end(), 0);

  auto kept_DoHs = vector<OERegion>{};
  kept_DoHs.reserve(num_kept_features);
  for (size_t i = 0; i != keep_features.size(); ++i)
  {
    if (keep_features[i] == 1)
    {
      kept_DoHs.push_back(hessian_laplace_maxima[i]);
      const float fact = DoHs.octave_scaling_factor(scale_octave_pairs[i](1));
      kept_DoHs.back().shape_matrix() *= pow(fact,-2);
      kept_DoHs.back().coords() *= fact;
    }
  }

  return kept_DoHs;
}

vector<OERegion> compute_DoH_extrema(const Image<float>& I,
                                     bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing DoH interest points");
    tic();
  }
  ComputeDoHExtrema computeDoHs;
  vector<OERegion> DoHs;
  vector<Point2i> scaleOctPairs;
  DoHs = computeDoHs(I, &scaleOctPairs);
  if (verbose)
    toc();
  CHECK(DoHs.size());

  const ImagePyramid<float>& detHessians = computeDoHs.detOfHessians();

  // 2. Rescale feature points to original image dimensions.
  for (size_t i = 0; i != DoHs.size(); ++i)
  {
    const float fact = detHessians.octave_scaling_factor(scaleOctPairs[i](1));
    DoHs[i].shape_matrix() *= pow(fact,-2);
    DoHs[i].coords() *= fact;
  }

  return DoHs;
}

vector<OERegion> compute_DoH_affine_extrema(const Image<float>& I,
                                            bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing DoH affine interest points");
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
    print_stage("Affine shape adaptation");
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
      DoHs[i].shape_matrix() = affAdaptTransformMat*DoHs[i].shape_matrix();
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
      const float fact = detHessians.octave_scaling_factor(scaleOctPairs[i](1));
      keptDoHs.back().shape_matrix() *= pow(fact,-2);
      keptDoHs.back().coords() *= fact;

    }
  }

  return keptDoHs;
}

void check_keys(const Image<float>& image, const vector<OERegion>& features)
{
  display(image);
  set_antialiasing();
  for (size_t i = 0; i != features.size(); ++i)
    features[i].draw(features[i].extremum_type() == OERegion::Max ?
                     Red8 : Blue8);
  get_key();
}

GRAPHICS_MAIN()
{
  auto image = Image<float>{};
  auto image_filepath = src_path("../../datasets/sunflowerField.jpg");
  if (!load(image, image_filepath))
    return -1;

  create_window(image.width(), image.height());
  auto features = compute_hessian_laplace_affine_maxima(image);
  check_keys(image, features);

  features = compute_DoH_extrema(image);
  check_keys(image, features);

  features = compute_DoH_affine_extrema(image);
  check_keys(image, features);

  return 0;
}