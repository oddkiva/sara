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

// A helper function
// Be aware that detection parameters are those set by default, e.g.,
// - thresholds like on extremum responses,
// - number of iterations in the keypoint localization,...
// Keypoints are described with the SIFT descriptor.
vector<OERegion> compute_harris_laplace_affine_corners(const Image<float>& I,
                                                       bool verbose = true)
{
  // 1. Feature extraction.
  if (verbose)
  {
    print_stage("Localizing Harris-Laplace interest points");
    tic();
  }
  auto compute_corners = ComputeHarrisLaplaceCorners{};
  auto scale_octave_pairs = vector<Point2i>{};
  auto corners = compute_corners(I, &scale_octave_pairs);
  if (verbose)
    toc();

  const auto& G = compute_corners.gaussians();
  const auto& H = compute_corners.harris();

  // 2. Affine shape adaptation
  if (verbose)
  {
    print_stage("Affine shape adaptation");
    tic();
  }
  auto adapt_shape = AdaptFeatureAffinelyToLocalShape{};
  auto keep_features = vector<unsigned char>(corners.size(), 0);
  for (size_t i = 0; i != corners.size(); ++i)
  {
    const int s = scale_octave_pairs[i](0);
    const int o = scale_octave_pairs[i](1);

    Matrix2f affine_adapt_transform;
    if (adapt_shape(affine_adapt_transform, G(s,o), corners[i]))
    {
      corners[i].shape_matrix() = affine_adapt_transform*corners[i].shape_matrix();
      keep_features[i] = 1;
    }
  }
  if (verbose)
    toc();

  // 3. Rescale the kept features to original image dimensions.
  auto num_kept_features = std::accumulate(
    keep_features.begin(), keep_features.end(), 0);

  auto kept_corners = vector<OERegion>{};
  kept_corners.reserve(num_kept_features);
  for (size_t i = 0; i != keep_features.size(); ++i)
  {
    if (keep_features[i] == 1)
    {
      kept_corners.push_back(corners[i]);
      const auto fact = H.octave_scaling_factor(scale_octave_pairs[i](1));
      kept_corners.back().shape_matrix() *= pow(fact, -2);
      kept_corners.back().coords() *= fact;
    }
  }

  SARA_CHECK(kept_corners.size());
  return kept_corners;
}

void check_keys(const Image<float>& image, const vector<OERegion>& features)
{
  display(image);
  set_antialiasing();
  draw_oe_regions(features, Red8);
  get_key();
}

GRAPHICS_MAIN()
{
  auto image = Image<float>{};
  auto image_filepath = src_path("../../../data/sunflowerField.jpg");
  if (!load(image, image_filepath))
    return -1;

  auto features = compute_harris_laplace_affine_corners(image);

  create_window(image.width(), image.height());
  check_keys(image, features);

  return 0;
}
