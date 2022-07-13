#pragma once

#include <DO/Sara/Graphics.hpp>

#include <DO/Shakti/Halide/SIFT/V2/ExtremumDataStructures.hpp>
#include <DO/Shakti/Halide/SIFT/V3/ExtremumDataStructures.hpp>
#include <DO/Shakti/Halide/Utilities.hpp>


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


inline auto draw_quantized_extrema(const halide::v2::QuantizedExtremumArray& e,
                                   float scale, float octave_scaling_factor = 1,
                                   int width = 2)
{
#pragma omp parallel for
  for (auto i = 0; i < static_cast<int>(e.size()); ++i)
  {
    const auto& c = e.type(i) == 1 ? sara::Cyan8 : sara::Blue8;
    const float x = e.x(i) * octave_scaling_factor;
    const float y = e.y(i) * octave_scaling_factor;

    // N.B.: the blob radius is the scale multiplied by sqrt(2).
    // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
    static constexpr auto sqrt_2 = static_cast<float>(M_SQRT2);
    const float r = scale * octave_scaling_factor * sqrt_2;

    sara::draw_circle(sara::Point2f{x, y}, r, c, width);
  }
}

inline auto draw_quantized_extrema(sara::ImageView<sara::Rgb8>& display,
                                   const halide::v2::QuantizedExtremumArray& e,
                                   float scale, float octave_scaling_factor = 1,
                                   int width = 2)
{
#pragma omp parallel for
  for (auto i = 0; i < static_cast<int>(e.size()); ++i)
  {
    const auto& c = e.type(i) == 1 ? sara::Red8 : sara::Cyan8;
    const auto x = static_cast<int>(std::round(e.x(i) * octave_scaling_factor));
    const auto y = static_cast<int>(std::round(e.y(i) * octave_scaling_factor));

    // N.B.: the blob radius is the scale multiplied by sqrt(2).
    // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
    static constexpr auto sqrt_2 = static_cast<float>(M_SQRT2);
    const auto r =
        static_cast<int>(std::round(scale * octave_scaling_factor * sqrt_2));

    sara::draw_circle(display, x, y, r, c, width);
  }
}

inline auto draw_extrema(const halide::v2::ExtremumArray& e,
                         float octave_scaling_factor = 1, int width = 2)
{
#pragma omp parallel for
  for (auto i = 0; i < e.size(); ++i)
  {
    const auto& c = e.type(i) == 1 ? sara::Blue8 : sara::Red8;
    const auto& x = e.x(i) * octave_scaling_factor;
    const auto& y = e.y(i) * octave_scaling_factor;
    const auto& s = e.s(i) * octave_scaling_factor;

    // N.B.: the blob radius is the scale multiplied by sqrt(2).
    // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
    static constexpr auto sqrt_2 = static_cast<float>(M_SQRT2);
    const float r = s * octave_scaling_factor * sqrt_2;

    sara::draw_circle(sara::Point2f{x, y}, r, c, width);
  }
}


inline auto draw_oriented_extrema(const halide::v2::OrientedExtremumArray& e,
                                  float octave_scaling_factor = 1,
                                  int width = 2)
{
  if (e.empty())
    return;

#pragma omp parallel for
  for (auto i = 0; i < static_cast<int>(e.size()); ++i)
  {
    const auto& c = e.type(i) == 1 ? sara::Red8 : sara::Blue8;
    const auto& x = e.x(i) * octave_scaling_factor;
    const auto& y = e.y(i) * octave_scaling_factor;
    const auto& s = e.s(i) * octave_scaling_factor;
    const auto& theta = e.orientations(i);

    // N.B.: the blob radius is the scale multiplied by sqrt(2).
    // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
    const float r = s * static_cast<float>(M_SQRT2);
    const auto& p1 = Eigen::Vector2f{x, y};
    const Eigen::Vector2f& p2 =
        p1 + r * Eigen::Vector2f{cos(theta), sin(theta)};

    sara::draw_line(p1, p2, c, width);
    sara::draw_circle(p1, r, c, width);
  }
}

inline auto draw_oriented_extrema(sara::ImageView<sara::Rgb8>& display,
                                  const halide::v2::OrientedExtremumArray& e,
                                  float octave_scaling_factor = 1,
                                  int width = 3, bool draw_outline = false,
                                  bool draw_orientation = false)
{
  if (e.empty())
    return;

#pragma omp parallel for
  for (auto i = 0; i < static_cast<int>(e.size()); ++i)
  {
    const auto& c = e.type(i) == 1 ? sara::Red8 : sara::Cyan8;
    const auto& x = e.x(i) * octave_scaling_factor;
    const auto& y = e.y(i) * octave_scaling_factor;
    const auto& s = e.s(i) * octave_scaling_factor;

    // N.B.: the blob radius is the scale multiplied by sqrt(2).
    // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
    const float r = s * static_cast<float>(M_SQRT2);
    const auto& p1 = Eigen::Vector2f{x, y};

    if (draw_orientation)
    {
      const auto& theta = e.orientations(i);
      const Eigen::Vector2f& p2 =
          p1 + r * Eigen::Vector2f{cos(theta), sin(theta)};

      if (draw_outline)
        sara::draw_line(display, p1, p2, sara::Black8, width + 2);
      sara::draw_line(display, p1, p2, c, width);
    }

    // Contour of orientation line.
    if (draw_outline)
      sara::draw_circle(display, p1, static_cast<int>(std::round(r)),
                        sara::Black8, width + 2);
    sara::draw_circle(display, p1, static_cast<int>(std::round(r)), c, width);
  }
}


inline auto draw_quantized_extrema(sara::ImageView<sara::Rgb8>& display,
                                   const halide::v3::QuantizedExtremumArray& e,
                                   float octave_scaling_factor = 1,
                                   int width = 2)
{
  if (e.empty())
    return;

#pragma omp parallel for
  for (auto i = 0; i < static_cast<int>(e.size()); ++i)
  {
    const auto& c = e.type(i) == 1 ? sara::Red8 : sara::Blue8;
    const float x = std::round(e.x(i) * octave_scaling_factor);
    const float y = std::round(e.y(i) * octave_scaling_factor);

    // N.B.: the blob radius is the scale multiplied by sqrt(2).
    // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
    const auto r = static_cast<int>(std::round(
        e.scale(i) * octave_scaling_factor * static_cast<float>(M_SQRT2)));

    sara::draw_circle(display, {x, y}, r, c, width);
  }
}

inline auto draw_extrema(sara::ImageView<sara::Rgb8>& display,
                         const halide::v3::ExtremumArray& e,
                         float octave_scaling_factor = 1, int width = 2)
{
  if (e.empty())
    return;

#pragma omp parallel for
  for (auto i = 0; i < static_cast<int>(e.size()); ++i)
  {
    const auto& c = e.type(i) == 1 ? sara::Magenta8 : sara::Cyan8;
    const float x = std::round(e.x(i) * octave_scaling_factor);
    const float y = std::round(e.y(i) * octave_scaling_factor);

    // N.B.: the blob radius is the scale multiplied by sqrt(2).
    // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
    const auto r = static_cast<int>(std::round(e.s(i) * octave_scaling_factor *
                                               static_cast<float>(M_SQRT2)));

    sara::draw_circle(display, {x, y}, r, c, width);
  }
}
