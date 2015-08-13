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

vector<OERegion> compute_LoG_extrema(const Image<float>& I,
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
  const auto& DoGPyr = computeLoGs.laplacians_of_gaussians();
  for (int i = 0; i < LoGs.size(); ++i)
  {
    float octScaleFact = DoGPyr.octave_scaling_factor(scaleOctPairs[i](1));
    LoGs[i].center() *= octScaleFact;
    LoGs[i].shape_matrix() /= pow(octScaleFact, 2);
  }

  return LoGs;
}

vector<OERegion> compute_LoG_affine_extrema(const Image<float>& I,
                                            bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing LoG affine-adapted extrema");
    tic();
  }

  ImagePyramidParams pyr_params(0);
  ComputeLoGExtrema compute_LoGs(pyr_params);
  auto LoGs = vector<OERegion>{};
  auto scale_octave_pairs = vector<Point2i>{};
  LoGs = compute_LoGs(I, &scale_octave_pairs);
  if (verbose)
    toc();
  CHECK(LoGs.size());

  const auto& G = compute_LoGs.gaussians();
  const auto& L = compute_LoGs.laplacians_of_gaussians();

  // 2. Affine shape adaptation
  if (verbose)
  {
    print_stage("Affine shape adaptation");
    tic();
  }
  AdaptFeatureAffinelyToLocalShape adaptShape;
  auto keep_features = vector<unsigned char>(LoGs.size(), 0);
  for (size_t i = 0; i != LoGs.size(); ++i)
  {
    const int s = scale_octave_pairs[i](0);
    const int o = scale_octave_pairs[i](1);

    Matrix2f affine_adapt_transform;
    if (adaptShape(affine_adapt_transform, G(s,o), LoGs[i]))
    {
      LoGs[i].shape_matrix() = affine_adapt_transform*LoGs[i].shape_matrix();
      keep_features[i] = 1;
    }
  }
  if (verbose)
    toc();

  // 3. Rescale the kept features to original image dimensions.
  size_t num_kept_features = std::accumulate(
    keep_features.begin(), keep_features.end(), 0);

  auto kept_DoGs = vector<OERegion>{};
  kept_DoGs.reserve(num_kept_features);
  for (size_t i = 0; i != keep_features.size(); ++i)
  {
    if (keep_features[i] == 1)
    {
      kept_DoGs.push_back(LoGs[i]);
      const float fact = L.octave_scaling_factor(scale_octave_pairs[i](1));
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
  auto image_filepath = src_path("../../datasets/sunflowerField.jpg");
  if (!load(image, image_filepath))
    return -1;

  create_window(image.width(), image.height());
  auto features = compute_LoG_extrema(image);
  check_keys(image, features);

  features = compute_LoG_affine_extrema(image);
  check_keys(image, features);

  return 0;
}