#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/DisjointSets/TwoPassConnectedComponents.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/Geometry/Algorithms/Polyline.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>


namespace sara = DO::Sara;
using sara::operator""_deg;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}

// ========================================================================== //
// Chess corners are saddle points so the determinant of hessian is a good
// mathematical criterion
// ========================================================================== //
struct Saddle
{
  Eigen::Vector2i p;
  float score;

  auto operator<(const Saddle& other) const
  {
    return score < other.score;
  }
};

auto extract_saddle_points(const sara::Image<float>& hessian, float thres)
{
  auto saddle_points = std::vector<Saddle>{};
  saddle_points.reserve(std::max(hessian.width(), hessian.height()));
  for (auto y = 0; y < hessian.height(); ++y)
    for (auto x = 0; x < hessian.width(); ++x)
      if (hessian(x, y) < thres)
        saddle_points.push_back({{x, y}, std::abs(hessian(x, y))});

  return saddle_points;
}

auto nms(std::vector<Saddle>& saddle_points, const Eigen::Vector2i& image_sizes,
         int nms_radius)
{
  std::sort(saddle_points.begin(), saddle_points.end());

  auto saddle_points_filtered = std::vector<Saddle>{};
  saddle_points_filtered.reserve(saddle_points.size());

  auto saddle_map = sara::Image<std::uint8_t>{image_sizes};
  saddle_map.flat_array().fill(0);

  for (const auto& p : saddle_points)
  {
    if (saddle_map(p.p) == 1)
      continue;

    const auto vmin = std::clamp(p.p.y() - nms_radius, 0, saddle_map.height());
    const auto vmax = std::clamp(p.p.y() + nms_radius, 0, saddle_map.height());
    const auto umin = std::clamp(p.p.x() - nms_radius, 0, saddle_map.width());
    const auto umax = std::clamp(p.p.x() + nms_radius, 0, saddle_map.width());
    for (auto v = vmin; v < vmax; ++v)
      for (auto u = umin; u < umax; ++u)
        saddle_map(u, v) = 1;

    saddle_points_filtered.push_back(p);
  }
  saddle_points_filtered.swap(saddle_points);
}

// Adaptive thresholding using Otsu's method.
// Implemented from Wikipedia's explanation.
auto otsu_adaptive_binarization(const sara::Image<float>& image)
{
  const auto image_quantized = image.convert<std::uint8_t>();

  // Calculate the discrete probability distribution function.
  Eigen::Array<float, 256, 1> pdf = Eigen::Array<float, 256, 1>::Zero();
  for (const auto& pixel : image_quantized)
    pdf(pixel) += 1;
  pdf /= pdf.sum();

  // Calculate the cumulated distribution function for the 0 class (i.e., the
  // black color class).
  auto cdf = Eigen::Array<float, 256, 1>{};
  std::partial_sum(pdf.data(), pdf.data() + pdf.size(), cdf.data());

  // The mean intensity value.
  const auto mean = [&pdf]() {
    auto m = 0.f;
    for (auto t = 0; t < pdf.size(); ++t)
      m += t * pdf(t);
    return m;
  }();

  auto t = 0;

  auto cdf_0 = cdf(0);
  auto cdf_1 = 1 - cdf_0;

  auto mean_0_unnormalized = 0;
  auto mean_1_unnormalized = mean;

  auto mean_0 = mean_0_unnormalized;
  auto mean_1 = mean;

  auto best_t = cdf_0 * cdf_1 * std::pow(mean_0 - mean_1, 2);
  auto best_score = cdf_0 * cdf_1 * std::pow(mean_0 - mean_1, 2);

  for (t = 1; t < 256; ++t)
  {
    cdf_0 = cdf(t);
    cdf_1 = 1 - cdf_0;

    mean_0_unnormalized += t * pdf(t);
    mean_1_unnormalized -= t * pdf(t);

    mean_0 = mean_0_unnormalized / cdf_0;
    mean_1 = mean_1_unnormalized / cdf_1;

    const auto score = cdf_0 * cdf_1 * std::pow(mean_0 - mean_1, 2);
    if (score > best_score)
    {
      best_t = t;
      best_score = score;
    }
  }

  auto binary = sara::Image<std::uint8_t>{image.sizes()};
  std::transform(image_quantized.begin(), image_quantized.end(), binary.begin(),
                 [best_t](const auto& v) { return v < best_t ? 0 : 255; });

  return binary;
}

auto erode(const sara::Image<std::uint8_t>& image, int radius)
{
  auto image_eroded = sara::Image<std::uint8_t>{image.sizes()};

  for (auto y = radius; y < image.height() - radius; ++y)
  {
    for (auto x = radius; x < image.width() - radius; ++x)
    {
      auto value = image(x, y);
      for (auto v = y - radius; v <= y + radius; ++v)
        for (auto u = x - radius; u <= x + radius; ++u)
          value = std::min(value, image(u, v));
      image_eroded(x, y) = value;
    }
  }

  return image_eroded;
}

auto dilate(const sara::Image<std::uint8_t>& image, int radius)
{
  auto image_dilated = sara::Image<std::uint8_t>{image.sizes()};

  for (auto y = radius; y < image.height() - radius; ++y)
  {
    for (auto x = radius; x < image.width() - radius; ++x)
    {
      auto value = image(x, y);
      for (auto v = y - radius; v <= y + radius; ++v)
        for (auto u = x - radius; u <= x + radius; ++u)
          value = std::max(value, image(u, v));
      image_dilated(x, y) = value;
    }
  }

  return image_dilated;
}


auto mean_colors(const std::map<int, std::vector<Eigen::Vector2i>>& regions,
                 const sara::Image<sara::Rgb8>& image)
{
  auto colors = std::map<int, sara::Rgb8>{};
  for (const auto& [label, points] : regions)
  {
    const auto num_points = points.size();
    Eigen::Vector3f color = Eigen::Vector3f::Zero();
    for (const auto& p : points)
      color += image(p).cast<float>();
    color /= num_points;

    colors[label] = color.cast<std::uint8_t>();
  }
  return colors;
}

// ========================================================================== //
auto edge_signature(const sara::Image<sara::Rgb8>& color,
                    const sara::Image<Eigen::Vector2f>& gradients,
                    const std::vector<Eigen::Vector2i>& edge,  //
                    float delta = 1, int width = 3)
{
  auto darks = std::vector<sara::Rgb64f>{};
  auto brights = std::vector<sara::Rgb64f>{};
  for (auto s = 1; s <= width; ++s)
  {
    for (const auto& e : edge)
    {
      const Eigen::Vector2d n = gradients(e).cast<double>().normalized();

      const Eigen::Vector2d b = e.cast<double>() + s * delta * n;
      const Eigen::Vector2d d = e.cast<double>() - s * delta * n;

      if (0 <= d.x() && d.x() < color.width() &&  //
          0 <= d.y() && d.y() < color.height())
        darks.push_back(sara::interpolate(color, d));

      if (0 <= b.x() && b.x() < color.width() &&  //
          0 <= b.y() && b.y() < color.height())
        brights.push_back(sara::interpolate(color, b));
    }
  }

  Eigen::Vector2f mean_gradient = Eigen::Vector2f::Zero();
  mean_gradient = std::accumulate(edge.begin(), edge.end(), mean_gradient,
                                  [&gradients](const auto& g, const auto& e) {
                                    return g + gradients(e).normalized();  //
                                  });
  mean_gradient /= edge.size();

  return std::make_tuple(darks, brights, mean_gradient);
}


int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  const auto folder = std::string{argv[1]};

  constexpr auto sigma = 1.6f;
  constexpr auto nms_radius = 5;

  constexpr float high_threshold_ratio = static_cast<float>(10._percent);
  constexpr float low_threshold_ratio =
      static_cast<float>(high_threshold_ratio / 2.);
  constexpr float angular_threshold = static_cast<float>((10._deg).value);

  const auto color_threshold = std::sqrt(std::pow(2, 2) * 3);
  const auto segment_min_size = 50;

  auto ed = sara::EdgeDetector{{high_threshold_ratio,  //
                                low_threshold_ratio,   //
                                angular_threshold,     //
                                false}};

  for (auto i = 0; i <= 1790; i += 10)
  {
    const auto image_filepath = folder + "/" + std::to_string(i) + ".png";

    auto image = sara::imread<sara::Rgb8>(image_filepath);
    const auto image_gray = image.convert<float>();
    const auto image_blurred = image_gray.compute<sara::Gaussian>(sigma);

    // Calculate the first derivative.
    const auto gradient = image_blurred.compute<sara::Gradient>();

    // Chessboard corners are saddle points of the image, which are
    // characterized by the property det(H(x, y)) < 0.
    const auto det_hessian =
        image_blurred.compute<sara::Hessian>().compute<sara::Determinant>();

    // Adaptive thresholding.
    const auto thres = det_hessian.flat_array().minCoeff() * 0.05f;
    auto saddle_points = extract_saddle_points(det_hessian, thres);

    // Non-maxima suppression.
    nms(saddle_points, image.sizes(), nms_radius);

    if (sara::active_window() == nullptr)
    {
      sara::create_window(image.sizes());
      sara::set_antialiasing();
    }

    // Watershed to find the chessboard quadrangles.
    const auto regions = sara::color_watershed(image, color_threshold);

    // Group segments.
    ed(image_blurred);
    const auto& edges = ed.pipeline.edges_as_list;

    auto darks = std::vector<std::vector<sara::Rgb64f>>{};
    auto brights = std::vector<std::vector<sara::Rgb64f>>{};
    auto mean_gradients = std::vector<Eigen::Vector2f>{};
    for (const auto& edge : edges)
    {
      auto [dark, bright, g] = edge_signature(  //
          image,                                //
          ed.pipeline.gradient_cartesian,       //
          edge);
      darks.push_back(std::move(dark));
      brights.push_back(std::move(bright));
      mean_gradients.push_back(g);
    }

    // const auto colors = mean_colors(regions, image);
    // auto partitioning = sara::Image<sara::Rgb8>{image.sizes()};
    // for (const auto& [label, points] : regions)
    // {
    //   // Show big segments only.
    //   for (const auto& p : points)
    //     partitioning(p) = points.size() < segment_min_size  //
    //                           ? sara::Black8
    //                           : colors.at(label);
    // }

    // sara::display(partitioning);

    sara::display(image);
    for (const auto& e : edges)
    {
      const auto color = sara::Rgb8(rand() % 255, rand() % 255, rand() % 255);
      for (const auto& p : e)
        sara::fill_circle(p.x(), p.y(), 2, color);
    }

    for (auto i = 0u; i < edges.size(); ++i)
    {
      const auto& e = edges[i];

      // Discard small edges.
      if (e.size() < 2)
        continue;

      const auto& g = mean_gradients[i];
      const Eigen::Vector2f a = std::accumulate(            //
                                    e.begin(), e.end(),     //
                                    Eigen::Vector2f{0, 0},  //
                                    [](const auto& a, const auto& b) {
                                      return a + b.template cast<float>();
                                    }) /
                                e.size();
      const Eigen::Vector2f b = a + 20 * g;

      sara::draw_arrow(a, b, sara::Magenta8, 2);
    }

    // for (const auto& p : saddle_points)
    //   sara::draw_circle(p.p, 5, sara::Green8, 2);

    sara::get_key();
  }

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
