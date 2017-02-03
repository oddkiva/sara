#include <DO/Sara/ImageProcessing/DataAugmentation.hpp>


using namespace std;


namespace DO { namespace Sara {

  auto expand_zoom_transforms(const Vector2i& in_image_sizes,
                              const Vector2i& out_image_sizes,
                              float zmin, float zmax, int num_samples)
      -> std::vector<ImageDataTransform>
  {
    const auto zs = logspace(zmin, zmax, num_samples);
    const auto z_image_sizes =
        (in_image_sizes.cast<float>() * zs.transpose());
    std::cout << z_image_sizes << endl;

    auto z_transforms = vector<ImageDataTransform>{};
    for (int j = 0; j < num_samples; ++j)
      if (z_image_sizes.col(j)(0) >= out_image_sizes(0) &&
          z_image_sizes.col(j)(1) >= out_image_sizes(1))
      {
        std::cout << j << std::endl;
        auto t = ImageDataTransform{};
        t.set_zoom(zs[j]);
        z_transforms.push_back(t);
      }

    return z_transforms;
  }

  auto expand_crop_transforms(const Vector2i& in_image_sizes,
                              const Vector2i& out_image_sizes,
                              int delta_x, int delta_y)
      -> std::vector<ImageDataTransform>
  {
    return std::vector<ImageDataTransform>{};
 
  }

} /* namespace Sara */
} /* namespace DO */
