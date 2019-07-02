#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>


using namespace DO::Sara;
using namespace std;


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
    auto octave_scale_factor =
        nabla_G.octave_scaling_factor(scale_octave_pairs[i](1));
    DoGs[i].center() *= octave_scale_factor;
    DoGs[i].shape_matrix /= pow(octave_scale_factor, 2);
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
  auto image = imread<float>(src_path("../../../data/sunflowerField.jpg"));

  print_stage("Detecting SIFT features");
  auto keypoints = compute_sift_keypoints(image);
  const auto& features = keypoints.features;

  print_stage("Removing existing redundancies");
  remove_redundant_features(keypoints);
  SARA_CHECK(keypoints.features.size());
  SARA_CHECK(keypoints.descriptors.size());

  // Check the features visually.
  print_stage("Draw features");
  create_window(image.width(), image.height());
  set_antialiasing();
  display(image);
  for (size_t i = 0; i != features.size(); ++i)
  {
    const auto color =
        features[i].extremum_type == OERegion::ExtremumType::Max ? Red8 : Blue8;
    features[i].draw(color);
  }
  get_key();

  return 0;
}
