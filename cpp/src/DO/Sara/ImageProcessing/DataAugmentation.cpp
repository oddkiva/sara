#include <DO/Sara/ImageProcessing/DataAugmentation.hpp>


using namespace std;


namespace DO { namespace Sara {

  VectorXf linspace(float a, float b, int num_samples)
  {
    auto range = VectorXf(num_samples);
    for (int i = 0; i < num_samples; ++i)
      range[i] = a + (b - a) * i / (num_samples - 1);
    return range;
  }

  VectorXf logspace(float a, float b, int num_samples)
  {
    return linspace(log(a), log(b), num_samples).array().exp().matrix();
  }

  auto compose_with_zooms(const Vector2i& in_image_sizes,
                          const Vector2i& out_image_sizes, float zmin, float zmax,
                          int num_samples, const ImageDataTransform& parent_t)
      -> std::vector<ImageDataTransform>
  {
    const auto zs = logspace(zmin, zmax, num_samples);
    const auto z_image_sizes =
        (in_image_sizes.cast<float>() * zs.transpose());

    auto ts = vector<ImageDataTransform>{};
    for (int j = 0; j < num_samples; ++j)
      if (z_image_sizes.col(j).x() >= out_image_sizes.x() &&
          z_image_sizes.col(j).y() >= out_image_sizes.y())
      {
        auto child_t = parent_t;
        child_t.set_zoom(zs[j]);
        ts.push_back(child_t);
      }

    return ts;
  }

  auto compose_with_shifts(const Vector2i& in_image_sizes,
                           const Vector2i& out_image_sizes, int delta_x,
                           int delta_y, const ImageDataTransform& parent_t)
      -> std::vector<ImageDataTransform>
  {
    auto ts = vector<ImageDataTransform>{};
    for (int y = 0; y + out_image_sizes.y() < in_image_sizes.y(); y += delta_y)
    {
      for (int x = 0; x + out_image_sizes.x() < in_image_sizes.y();
           x += delta_x)
      {
        auto child_t = parent_t;
        child_t.set_shift(Vector2i{x, y});
        ts.push_back(child_t);
      }
    }

    return ts;
  }

  auto compose_with_horizontal_flip(const ImageDataTransform& parent_t)
      -> std::vector<ImageDataTransform>
  {
    auto child_t = parent_t;
    child_t.set_flip(ImageDataTransform::Horizontal);
    return {child_t};
  };

  auto compose_with_random_fancy_pca(const ImageDataTransform& parent_t)
      -> std::vector<ImageDataTransform>
  {
    return std::vector<ImageDataTransform>{};
  }


  auto enumerate_image_data_transforms(const Vector2i& in_sz,
                                       const Vector2i& out_sz,
                                       float zmin, float zmax, int num_z,
                                       int delta_x, int delta_y,
                                       int num_fancy_pca_alpha)
      -> std::vector<ImageDataTransform>
  {
    //auto ts = std::vector<ImageDataTransform>{};

    //auto z_range = logspace(zmin, zmax, num_z);

    //auto t0 = ImageDataTransform{};
    //t0.out_sizes = out_sz;

    //auto tz = compose_with_zooms(in_sz, out_sz, zmin, zmax, num_z);

    //auto tt = tz;


    return std::vector<ImageDataTransform>{};
  }


} /* namespace Sara */
} /* namespace DO */
