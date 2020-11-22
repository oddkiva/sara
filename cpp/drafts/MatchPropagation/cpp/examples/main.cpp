// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
/*!
 *  This file implements a part of the method published in:
 *
 *  Efficient and Scalable 4th-order Match Propagation
 *  David Ok, Renaud Marlet, and Jean-Yves Audibert.
 *  ACCV 2012, Daejeon, South Korea.
 */

#include "GrowMultipleRegions.hpp"

#include <DO/Sara/Core.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/Match.hpp>

using namespace std;
using namespace DO::Sara;


// ========================================================================== //
// Helper functions.
template <typename T>
Image<T> rotate_ccw(const Image<T>& image)
{
  auto dst = Image<T>{image.height(), image.width()};

  // Transpose.
  for (int y = 0; y < image.height(); ++y)
    for (int x = 0; x < image.width(); ++x)
      dst(y, x) = image(x, y);

  // Reverse rows.
  for (int y = 0; y < dst.height(); ++y)
    for (int x = 0; x < dst.width(); ++x)
    {
      int n_x = dst.width() - 1 - x;
      if (x >= n_x)
        break;
      std::swap(dst(x, y), dst(n_x, y));
    }

  return dst;
}

Window open_window_for_image_pair(const Image<Rgb8>& image1,
                                  const Image<Rgb8>& image2, float scale)
{
  const auto w = int((image1.width() + image2.width()) * scale);
  const auto h = int(max(image1.height(), image2.height()) * scale);
  return create_window(w, h);
}


// ========================================================================== //
// SIFT Detector.
class SIFTDetector
{
public:
  SIFTDetector() = default;

  void set_num_octaves(int n)
  {
    num_octaves = n;
  }

  void set_num_scales(int n)
  {
    num_scales = n;
  }

  void set_first_octave(int n)
  {
    first_octave = n;
  }

  void set_edge_threshold(float t)
  {
    edgeThresh = t;
  }

  void set_peak_threshold(float t)
  {
    peakThresh = t;
  }

  KeypointList<OERegion, float> run(const Image<float>& image) const
  {
    KeypointList<OERegion, float> keys;
    auto& DoGs = features(keys);
    auto& sift_descriptors = descriptors(keys);

    // Time everything.
    auto timer = Timer{};
    auto elapsed = 0.;
    double dog_detection_time, orientation_assignment_time,
        sift_description_time, gradients_computation_time;

    // 1. Feature extraction.
    print_stage("Computing DoG extrema");
    timer.restart();
    ImagePyramidParams pyrParams;  //(0);
    ComputeDoGExtrema computeDoGs(pyrParams, 0.01f);
    vector<Point2i> scale_octave_pairs;
    DoGs = computeDoGs(image, &scale_octave_pairs);
    dog_detection_time = timer.elapsed_ms();
    elapsed += dog_detection_time;
    cout << "DoG detection time = " << dog_detection_time << " ms" << endl;
    cout << "DoGs.size() = " << DoGs.size() << endl;

    // 2. Feature orientation.
    // Prepare the computation of gradients on gaussians.
    print_stage("Computing gradients of Gaussians");
    timer.restart();
    auto gradients = ImagePyramid<Vector2f>{};
    gradients = gradient_polar_coordinates(computeDoGs.gaussians());
    gradients_computation_time = timer.elapsed_ms();
    elapsed += gradients_computation_time;
    cout << "gradient of Gaussian computation time = "
         << gradients_computation_time << " ms" << endl;
    cout << "DoGs.size() = " << DoGs.size() << endl;

    // Find dominant gradient orientations.
    print_stage(
        "Assigning (possibly multiple) dominant orientations to DoG extrema");
    timer.restart();
    ComputeDominantOrientations assign_orientations;
    assign_orientations(gradients, DoGs, scale_octave_pairs);
    orientation_assignment_time = timer.elapsed_ms();
    elapsed += orientation_assignment_time;
    cout << "orientation assignment time = " << orientation_assignment_time << " ms" << endl;
    cout << "DoGs.size() = " << DoGs.size() << endl;


    // 3. Feature description.
    print_stage("Describe DoG extrema with SIFT descriptors");
    timer.restart();
    auto compute_sift = ComputeSIFTDescriptor<>{};
    sift_descriptors = compute_sift(DoGs, scale_octave_pairs, gradients);
    sift_description_time = timer.elapsed_ms();
    elapsed += sift_description_time;
    cout << "description time = " << sift_description_time << " ms" << endl;
    cout << "sifts.size() = " << sift_descriptors.size() << endl;

    // Summary in terms of computation time.
    print_stage("Total Detection/Description time");
    cout << "SIFT computation time = " << elapsed << " ms" << endl;

    // 4. Rescale  the feature position and scale $(x,y,\sigma)$ with the octave
    //    scale.
    for (size_t i = 0; i != DoGs.size(); ++i)
    {
      const auto octave_scale_factor =
          gradients.octave_scaling_factor(scale_octave_pairs[i](1));
      DoGs[i].center() *= octave_scale_factor;
      DoGs[i].shape_matrix /= pow(octave_scale_factor, 2);
    }

    rescale_shape_matrices(DoGs);

    return keys;
  }

  void rescale_shape_matrices(vector<OERegion>& features) const
  {
    // Dilate SIFT circular shape before matching keypoints.
    //
    // We are not cheating because the SIFT descriptor is calculated on an image
    // patch with actually a quite large radius:
    //   r = 3 * (N + 1) / 2 with N = 4
    //   r = 3 * (4 + 1) / 2
    //   r = 7.5
    constexpr auto dilation_factor = 7.5f;
    for (size_t i = 0; i != features.size(); ++i)
      features[i].shape_matrix /= dilation_factor * dilation_factor;
  }

private:
  // First Octave Index.
  int first_octave{-1};
  // Number of octaves.
  int num_octaves{-1};
  // Number of scales per octave.
  int num_scales{3};
  // Max ratio of Hessian eigenvalues.
  float edgeThresh{10.f};
  // Min contrast.
  float peakThresh{0.04};
};


// ========================================================================== //
// Matching demo.
GRAPHICS_MAIN()
{
  auto timer = Timer{};
  auto elapsed = double{};

  // Where are the images?
  const string query_image_path =
      src_path("products/garnier-shampoing.jpg");
  const string target_image_path = src_path("shelves/shelf-1.jpg");

  // Load the query and target images.
  Image<Rgb8> query, target;
  if (!load(query, target_image_path))
  {
    cerr << "Cannot load query image: " << target_image_path << endl;
    return 1;
  }
  if (!load(target, query_image_path))
  {
    cerr << "Cannot load target image: " << query_image_path << endl;
    return 1;
  }
  query = rotate_ccw(query);

  // Open a window.
  float scale = 0.5;
  open_window_for_image_pair(target, query, scale);
  set_antialiasing();

  PairWiseDrawer drawer(target, query);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);
  drawer.display_images();

  // Detect keys.
  print_stage("Detecting SIFT keypoints");
  KeypointList<OERegion, float> query_keypoints, target_keypoints;
  SIFTDetector detector;
  detector.set_first_octave(0);
  timer.restart();
  query_keypoints = detector.run(query.convert<float>());
  target_keypoints = detector.run(target.convert<float>());
  elapsed = timer.elapsed_ms();
  cout << "Detection time = " << elapsed << " ms" << endl;

  // Compute initial matches.
  print_stage("Compute initial matches");
  const auto nearest_neighbor_ratio = 1.f;
  AnnMatcher matcher(target_keypoints, query_keypoints,
                     nearest_neighbor_ratio);
  const auto initial_matches = matcher.compute_matches();

  // Match keypoints.
  print_stage("Filter matches by region growing robustly");
  timer.restart();
  auto regions = vector<Region>{};

  const auto num_region_growing = 2000;
  const auto growth_params = GrowthParams{};
  const auto verbose_level = 0;
  GrowMultipleRegions grow_regions(initial_matches, growth_params, verbose_level);
  regions = grow_regions(num_region_growing, 0, &drawer);
  elapsed = timer.elapsed_ms();
  cout << "Matching time = " << elapsed << " ms" << endl << endl;

  // Show the regions.
  grow_regions.check_regions(regions, &drawer);
  get_key();

  return 0;
}
