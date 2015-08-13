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

vector<OERegion> compute_dog_extrema(const Image<float>& I,
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
  const ImagePyramid<float>& DoGPyr = computeDoGs.diff_of_gaussians();
  for (int i = 0; i < DoGs.size(); ++i)
  {
    float octScaleFact = DoGPyr.octave_scaling_factor(scaleOctPairs[i](1));
    DoGs[i].center() *= octScaleFact;
    DoGs[i].shape_matrix() /= pow(octScaleFact, 2);
  }

  return DoGs;
}

vector<OERegion> compute_dog_affine_extrema(const Image<float>& I,
                                            bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing DoG affine-adapted extrema");
    tic();
  }
  ImagePyramidParams pyr_params(0);
  ComputeDoGExtrema compute_DoGs(pyr_params);
  auto DoGs = vector<OERegion>{};
  auto scale_octave_pairs = vector<Point2i>{};
  DoGs = compute_DoGs(I, &scale_octave_pairs);
  if (verbose)
    toc();
  CHECK(DoGs.size());

  const auto& G = compute_DoGs.gaussians();
  const auto& D = compute_DoGs.diff_of_gaussians();

  // 2. Affine shape adaptation
  if (verbose)
  {
    print_stage("Affine shape adaptation");
    tic();
  }
  AdaptFeatureAffinelyToLocalShape adapt_shape;
  auto keep_features = vector<unsigned char>(DoGs.size(), 0);
  for (size_t i = 0; i != DoGs.size(); ++i)
  {
    const int s = scale_octave_pairs[i](0);
    const int o = scale_octave_pairs[i](1);

    Matrix2f affine_adaptation_transform;
    if (adapt_shape(affine_adaptation_transform, G(s,o), DoGs[i]))
    {
      DoGs[i].shape_matrix() = affine_adaptation_transform*DoGs[i].shape_matrix();
      keep_features[i] = 1;
    }
  }
  if (verbose)
    toc();

  // 3. Rescale the kept features to original image dimensions.
  auto num_kept_features = std::accumulate(
    keep_features.begin(), keep_features.end(), 0);

  auto kept_DoGs = vector<OERegion>{};
  kept_DoGs.reserve(num_kept_features);
  for (size_t i = 0; i != keep_features.size(); ++i)
  {
    if (keep_features[i] == 1)
    {
      kept_DoGs.push_back(DoGs[i]);
      const float fact = D.octave_scaling_factor(scale_octave_pairs[i](1));
      kept_DoGs.back().shape_matrix() *= pow(fact,-2);
      kept_DoGs.back().coords() *= fact;

    }
  }

  CHECK(kept_DoGs.size());

  return kept_DoGs;
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
  auto name = src_path("../../datasets/sunflowerField.jpg");
  if (!load(image, name))
    return -1;

  create_window(image.width(), image.height());
  vector<OERegion> features;

  features = compute_dog_extrema(image);
  check_keys(image, features);

  features = compute_dog_affine_extrema(image);
  check_keys(image, features);

  return 0;
}