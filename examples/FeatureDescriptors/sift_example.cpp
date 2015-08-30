#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>


using namespace DO::Sara;
using namespace std;


void test_dog_sift_keypoints(const Image<float>& I)
{
  auto image_window = create_window(I.width(), I.height());
  set_antialiasing();

  auto G = gaussian_pyramid(I);
  //check_image_pyramid(G);

  auto D = difference_of_gaussians_pyramid(G);
  //check_image_pyramid(D, true);

  for (int o = 0; o < D.num_octaves(); ++o)
  {
    // Verbose.
    print_stage("Processing octave");
    cout << "Octave " << o << endl;
    cout << "Octave scaling factor = " << D.octave_scaling_factor(o) << endl;

    // Be careful of the bounds. We go from 1 to N-1.
    for (int s = 1; s < D.num_scales_per_octave()-1; ++s)
    {
      auto extrema = local_scale_space_extrema(D,s,o);

      // Verbose.
      print_stage("Detected extrema");
      cout << "[" << s << "] sigma = " << D.scale(s,o) << endl;
      cout << "    num extrema = " << extrema.size() << endl;

      // Draw the keypoints.
      //display(I.convert<float>());
      draw_extrema(D, extrema, s, o);
      get_key();

      // Gradient in polar coordinates.
      auto nabla_G = gradient_polar_coordinates(G(s,o));

      // Determine orientations.
      draw_extrema(G, extrema, s, o, false);
      for (size_t i = 0; i != extrema.size(); ++i)
      {
#define DEBUG_ORI
#ifdef DEBUG_ORI
        // Draw the patch on the image.
        highlight_patch(
          D, extrema[i].x(), extrema[i].y(), extrema[i].scale(), o);

        // Close-up on the image patch
        check_patch(G(s,o), extrema[i].x(), extrema[i].y(), extrema[i].scale());

        // Orientation histogram
        print_stage("Orientation histogram");
#endif

        Array<float, 36, 1> orientation_histogram;
        compute_orientation_histogram(
          orientation_histogram, nabla_G,
          extrema[i].x(),
          extrema[i].y(),
          extrema[i].scale());
        view_histogram(orientation_histogram);

        // Note that the peaks are shifted after smoothing.
#ifdef DEBUG_ORI
        print_stage("Smoothing orientation histogram");
#endif

        lowe_smooth_histogram(orientation_histogram);
        view_histogram(orientation_histogram);

        // Orientation peaks.
#ifdef DEBUG_ORI
        print_stage("Localizing orientation peaks");
#endif

        auto ori_peaks = find_peaks(orientation_histogram);
#ifdef DEBUG_ORI
        cout << "Raw peaks" << endl;
        for (size_t k = 0; k != ori_peaks.size(); ++k)
          cout << ori_peaks[k]*10 << endl;

        // Refine peaks.
        print_stage("Refining peaks");
#endif
        auto refined_ori_peaks = refine_peaks(orientation_histogram, ori_peaks);
#ifdef DEBUG_ORI
        cout << "Refined peaks" << endl;
        for (size_t k = 0; k != refined_ori_peaks.size(); ++k)
          cout << refined_ori_peaks[k]*10 << endl;
#endif
        // Convert orientation to radians.
        Map<ArrayXf> peaks(&refined_ori_peaks[0], refined_ori_peaks.size());
        peaks *= float(M_PI)/36;

        set_active_window(image_window);
        print_stage("Compute SIFT descriptor");
        for (size_t ori = 0; ori < refined_ori_peaks.size(); ++ori)
        {
          ComputeSIFTDescriptor<> compute_sift;

          print_stage("Draw patch grid");
          compute_sift.draw_grid(
            extrema[i].x(), extrema[i].y(), extrema[i].scale(),
            peaks(ori), D.octave_scaling_factor(o), 3);
          get_key();

          print_stage("Compute SIFT descriptor with specified orientation");
          Matrix<unsigned char, 128, 1> sift;
          sift = compute_sift(
            extrema[i].x(), extrema[i].y(), extrema[i].scale(),
            peaks(ori), nabla_G).cast<unsigned char>();
        }
      }
      get_key();
    }
  }

  close_window();
}

Set<OERegion, RealDescriptor> compute_sift_keypoints(const Image<float>& image)
{
  // Time everything.
  Timer timer;
  double elapsed = 0.;
  double dog_detection_time;
  double ori_assign_time;
  double sift_description_time;
  double grad_gaussian_time;

  // We describe the work flow of the feature detection and description.
  Set<OERegion, RealDescriptor> keys;
  auto& DoGs = keys.features;
  auto& SIFTDescriptors = keys.descriptors;

  // 1. Feature extraction.
  print_stage("Computing DoG extrema");
  timer.restart();
  ImagePyramidParams pyr_params;//(0);
  ComputeDoGExtrema compute_DoGs(pyr_params, 0.01f);
  auto scale_octave_pairs = vector<Point2i>{};
  DoGs = compute_DoGs(image, &scale_octave_pairs);
  dog_detection_time = timer.elapsed_ms();
  elapsed += dog_detection_time;
  cout << "DoG detection time = " << dog_detection_time << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;

  // 2. Feature orientation.
  // Prepare the computation of gradients on gaussians.
  print_stage("Computing gradients of Gaussians");
  timer.restart();
  auto nabla_G = gradient_polar_coordinates(compute_DoGs.gaussians());
  grad_gaussian_time = timer.elapsed_ms();
  elapsed += grad_gaussian_time;
  cout << "gradient of Gaussian computation time = " << grad_gaussian_time << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;


  // Find dominant gradient orientations.
  print_stage("Assigning (possibly multiple) dominant orientations to DoG extrema");
  timer.restart();
  ComputeDominantOrientations assign_dominant_orientations;
  assign_dominant_orientations(nabla_G, DoGs, scale_octave_pairs);
  ori_assign_time = timer.elapsed_ms();
  elapsed += ori_assign_time;
  cout << "orientation assignment time = " << ori_assign_time << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;


  // 3. Feature description.
  print_stage("Describe DoG extrema with SIFT descriptors");
  timer.restart();
  ComputeSIFTDescriptor<> compute_sift;
  SIFTDescriptors = compute_sift(DoGs, scale_octave_pairs, nabla_G);
  sift_description_time = timer.elapsed_ms();
  elapsed += sift_description_time;
  cout << "description time = " << sift_description_time << " ms" << endl;
  cout << "sifts.size() = " << SIFTDescriptors.size() << endl;

  // Summary in terms of computation time.
  print_stage("Total Detection/Description time");
  cout << "SIFT computation time = " << elapsed << " ms" << endl;

  // 4. Rescale  the feature position and scale $(x,y,\sigma)$ with the octave
  //    scale.
  for (size_t i = 0; i != DoGs.size(); ++i)
  {
    auto octave_scale_factor = nabla_G.octave_scaling_factor(scale_octave_pairs[i](1));
    DoGs[i].center() *= octave_scale_factor;
    DoGs[i].shape_matrix() /= pow(octave_scale_factor, 2);
  }

  return keys;
}

bool check_descriptors(const DescriptorMatrix<float>& descriptors)
{
  for (size_t i = 0; i < descriptors.size(); ++i)
  {
    for (size_t j = 0; j < descriptors.dimension(); ++j)
    {
      if (!isfinite(descriptors[i](j)))
      {
        cerr << "Not a finite number" << endl;
        return false;
      }
    }
  }
  cout << "OK all numbers are finite" << endl;
  return true;
}

GRAPHICS_MAIN()
{
  auto image = Image<float>{};
  if (!imread(image, src_path("../../datasets/sunflowerField.jpg")))
    return -1;

  print_stage("Detecting SIFT features");
  auto keypoints = compute_sift_keypoints(image);
  const auto& features = keypoints.features;

  print_stage("Removing existing redundancies");
  remove_redundant_features(keypoints);
  CHECK(keypoints.features.size());
  CHECK(keypoints.descriptors.size());

  // Check the features visually.
  print_stage("Draw features");
  create_window(image.width(), image.height());
  set_antialiasing();
  display(image);
  for (size_t i = 0; i != features.size(); ++i)
  {
    const auto color =
      features[i].extremum_type() == OERegion::ExtremumType::Max ?
      Red8 : Blue8;
    features[i].draw(color);
  }
  get_key();

  return 0;
}
