//! @example

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <DO/Sara/Core.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>

#include <DO/Shakti/ImageProcessing.hpp>
#include <DO/Shakti/Utilities.hpp>


namespace shakti = DO::Shakti;

using namespace std;
using namespace DO::Sara;


template <int N, int O>
void draw_grid(float x, float y, float sigma, float theta, int pen_width = 1)
{
  const float lambda = 3.f;
  const float l = lambda * sigma;
  Vector2f grid[N + 1][N + 1];
  Matrix2f T;
  theta = 0;
  T << cos(theta), -sin(theta), sin(theta), cos(theta);
  T *= l;
  for (int v = 0; v < N + 1; ++v)
    for (int u = 0; u < N + 1; ++u)
      grid[u][v] = (Vector2f{x, y} + T * Vector2f{u - N / 2.f, v - N / 2.f});
  for (int i = 0; i < N + 1; ++i)
    draw_line(grid[0][i], grid[N][i], Green8, pen_width);
  for (int i = 0; i < N + 1; ++i)
    draw_line(grid[i][0], grid[i][N], Green8, pen_width);

  Vector2f a(x, y);
  Vector2f b;
  b = a + N / 2.f * T * Vector2f(1, 0);
  draw_line(a, b, Red8, pen_width + 2);
}

template <int N, int O>
void draw_dense_sifts(const Image<Vector128f>& dense_sifts,
                      const Vector2i& num_blocks, float sigma = 1.6f,
                      float bin_scale_length = 3.f)
{
  int w = dense_sifts.width();
  int h = dense_sifts.height();

  for (int j = 0; j < num_blocks.y(); ++j)
  {
    for (int i = 0; i < num_blocks.x(); ++i)
    {
      const Point2f a{float(i) / num_blocks.x() * w,
                      float(j) / num_blocks.y() * h};
      const Point2f b{float(i + 1) / num_blocks.x() * w,
                      float(j + 1) / num_blocks.y() * h};
      const Point2f c{(a + b) / 2.f};

      float r = b.y() - c.y();

      draw_rect(a.x(), a.y(), b.x() - a.x(), b.y() - a.y(), Green8);
    }
  }
}

void draw_sift(const Vector128f& sift, float x, float y, float s,
               float bin_scale_length = 3.f, int N = 4, int O = 8)
{
  auto r = s * bin_scale_length * N / 2.f;
  Point2f a{x - r, y - r};
  Point2f b{x + r, y + r};

  for (int j = 0; j < N; ++j)
  {
    for (int i = 0; i < N; ++i)
    {
      Point2f c_ij;
      c_ij << -N / 2.f + 0.5f + i, -N / 2.f + 0.5f + j;
      c_ij *= s * bin_scale_length;
      c_ij += Point2f{x, y};

      auto x_r = (-N / 2.f + i) * s * bin_scale_length + x;
      auto y_r = (-N / 2.f + j) * s * bin_scale_length + y;
      auto w_r = s * bin_scale_length;
      auto h_r = s * bin_scale_length;

      draw_rect(int(x_r), int(y_r), int(w_r), int(h_r), Green8, 2);

      Matrix<float, 8, 1> histogram{sift.block(j * N * O + i * O, 0, 8, 1)};
      if (histogram.sum() < 1e-6f)
        continue;
      histogram /= histogram.sum();
      for (int o = 0; o < O; ++o)
      {
        auto r_b = 0.9f * s * bin_scale_length / 2.f * histogram[o];
        auto ori = 2 * float(M_PI) * o / O;
        Point2f a_ijo{c_ij + r_b * unit_vector2(ori)};

        draw_line(c_ij, a_ijo, Green8);
      }
      // CHECK(histogram.transpose());
    }
  }
}

Image<Vector128f> cpu_compute_sifts(const Image<float>& image)
{
  // Compute the image gradients in polar coordinates.
  auto gradients = gradient_polar_coordinates(image);

  // scale parameter.
  const auto sigma = 1.6f;

  Timer t;

  auto features = Image<Vector128f>{image.sizes()};
  features.flat_array().fill(Vector128f::Zero());

#ifdef _OPENMP
  omp_set_num_threads(4);
#endif

// Compute the feature vector in each pixel.
#if !defined(_WIN32)
#  pragma omp for schedule(static, 1) collapse(2)
#endif
  for (auto y = 0; y < image.height(); ++y)
  {
    for (auto x = 0; x < image.width(); ++x)
    {
      auto cpu_sift_computer = ComputeSIFTDescriptor<>{};
      features(x, y) = cpu_sift_computer(float(x), float(y), sigma, gradients);
    }
  }

  auto elapsed = t.elapsed();
  cout << "[CPU Dense SIFT] " << elapsed << "s" << endl;

  return features;
}

Image<Vector128f> gpu_compute_sifts(const Image<float>& image)
{
  // Compute the image gradients in polar coordinates.
  auto gradients = gradient_polar_coordinates(image);

  auto gpu_sift_computer = shakti::DenseSiftComputer{};

  auto features = Image<Vector128f>{image.sizes()};
  features.flat_array().fill(Vector128f::Zero());

  shakti::tic();
  gpu_sift_computer(reinterpret_cast<float*>(features.data()), image.data(),
                    image.sizes().data());
  shakti::toc("GPU Dense SIFT");

  return features;
}

GRAPHICS_MAIN()
{
  try
  {
    auto image_path = src_path("examples/Segmentation/sunflower_field.jpg");
    auto image = imread<float>(image_path);

    image = reduce(image, Vector2i{640, 480});
    cout << "image sizes = " << image.sizes().transpose() << endl;

    // Display the image.
    create_window(image.sizes());
    display(image);

    auto cpu_sifts_res = cpu_compute_sifts(image);
    auto gpu_sifts_res = gpu_compute_sifts(image);

    const int margin = 40;
    for (int y = margin; y < cpu_sifts_res.height() - margin; ++y)
      for (int x = margin; x < cpu_sifts_res.width() - margin; ++x)
      {
        const auto& f1 = cpu_sifts_res(x, y);
        const auto& f2 = gpu_sifts_res(x, y);
        auto dist = (f1 - f2).squaredNorm();
        // if (dist < 1e-3f)
        //{
        SARA_CHECK(dist);
        SARA_CHECK(x);
        SARA_CHECK(y);
        SARA_CHECK(f1.transpose());
        SARA_CHECK(f2.transpose());
        get_key();
        //}
      }

    get_key();
  }
  catch (std::exception& e)
  {
    cout << e.what() << endl;
    return 1;
  }

  return 0;
}
