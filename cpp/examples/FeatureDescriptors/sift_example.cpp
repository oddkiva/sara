#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


using namespace DO::Sara;
using namespace std;


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
