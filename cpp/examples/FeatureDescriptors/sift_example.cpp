#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


using namespace DO::Sara;
using namespace std;


bool check_descriptors(const TensorView_<float, 2>& descriptors)
{
  for (auto i = 0; i < descriptors.rows(); ++i)
  {
    for (auto j = 0; j < descriptors.cols(); ++j)
    {
      if (!isfinite(descriptors(i, j)))
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
  const auto image_path = src_path("../../../data/sunflowerField.jpg");
  const auto image = imread<float>(image_path);

  print_stage("Detecting SIFT features");
  auto [features, descriptors] = compute_sift_keypoints(image);

  print_stage("Removing existing redundancies");
  remove_redundant_features(features, descriptors);
  SARA_CHECK(features.size());
  SARA_CHECK(descriptors.size());

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
