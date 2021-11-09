#pragma once

#include <DO/Sara/Graphics.hpp>

#include <DO/Shakti/Halide/ExtremumDataStructures.hpp>
#include <DO/Shakti/Halide/ExtremumDataStructuresV2.hpp>
#include <DO/Shakti/Halide/ExtremumDataStructuresV3.hpp>
#include <DO/Shakti/Halide/Utilities.hpp>


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


inline auto draw_quantized_extrema(const halide::v2::QuantizedExtremumArray& e,
                                   float scale, float octave_scaling_factor = 1,
                                   int width = 2)
{
#pragma omp parallel for
  for (auto i = 0; i < e.size(); ++i)
  {
    const auto& c = e.type(i) == 1 ? sara::Cyan8 : sara::Blue8;
    const float x = e.x(i) * octave_scaling_factor;
    const float y = e.y(i) * octave_scaling_factor;

    // N.B.: the blob radius is the scale multiplied by sqrt(2).
    // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
    const float r = scale * octave_scaling_factor * M_SQRT2;

    sara::draw_circle(sara::Point2f{x, y}, r, c, width);
  }
}

inline auto draw_quantized_extrema(sara::ImageView<sara::Rgb8>& display,
                                   const halide::v2::QuantizedExtremumArray& e,
                                   float scale, float octave_scaling_factor = 1,
                                   int width = 2)
{
#pragma omp parallel for
  for (auto i = 0; i < e.size(); ++i)
  {
    const auto& c = e.type(i) == 1 ? sara::Red8 : sara::Cyan8;
    const float x = std::round(e.x(i) * octave_scaling_factor);
    const float y = std::round(e.y(i) * octave_scaling_factor);

    // N.B.: the blob radius is the scale multiplied by sqrt(2).
    // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
    const float r = std::round(scale * octave_scaling_factor * float(M_SQRT2));

    sara::draw_circle(display, x, y, r, c, width);
  }
}

inline auto draw_quantized_extrema(
    const halide::Pyramid<halide::QuantizedExtremumArray>& extrema)
{
  for (const auto& so : extrema.scale_octave_pairs)
  {
    const auto& s = so.first.first;
    const auto& o = so.first.second;

    const auto& scale = so.second.first;
    const auto& octave_scaling_factor = so.second.second;

    auto eit = extrema.dict.find({s, o});
    if (eit == extrema.dict.end())
      continue;

    const auto& extrema_so = eit->second;

    for (auto i = 0u; i < extrema_so.x.size(); ++i)
    {
      const auto c0 = extrema_so.type[i] == 1 ? sara::Blue8 : sara::Red8;
      const auto& x0 = extrema_so.x[i] * octave_scaling_factor;
      const auto& y0 = extrema_so.y[i] * octave_scaling_factor;

      // N.B.: the blob radius is the scale multiplied sqrt(2).
      // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
      const auto r0 = scale * octave_scaling_factor * std::sqrt(2.f);

      const auto p1 = Eigen::Vector2f{x0, y0};
      const Eigen::Vector2f p2 = p1 + r0 * Eigen::Vector2f{cos(o), sin(o)};

      sara::draw_line(p1, p2, c0, 2);
      sara::draw_circle(x0, y0, r0, c0, 2 + 2);
    }
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
    const float r = s * octave_scaling_factor * M_SQRT2;

    sara::draw_circle(sara::Point2f{x, y}, r, c, width);
  }
}


inline auto
draw_extrema(const halide::Pyramid<halide::OrientedExtremumArray>& extrema)
{
  for (const auto& so : extrema.scale_octave_pairs)
  {
    const auto& s = so.first.first;
    const auto& o = so.first.second;

    const auto& octave_scaling_factor = so.second.second;

    auto eit = extrema.dict.find({s, o});
    if (eit == extrema.dict.end())
      continue;

    const auto& extrema_so = eit->second;
    if (extrema_so.empty())
      continue;

    for (auto i = 0u; i < extrema_so.size(); ++i)
    {
      const auto color = extrema_so.type[i] == 1 ? sara::Red8 : sara::Blue8;

      const auto x = extrema_so.x[i] * octave_scaling_factor;
      const auto y = extrema_so.y[i] * octave_scaling_factor;
      const auto& theta = extrema_so.orientations[i];
      const auto& center = Eigen::Vector2f{x, y};

      // N.B.: the blob radius is the scale multiplied sqrt(2).
      // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
      const auto r1 = extrema_so.s[i] * octave_scaling_factor * std::sqrt(2.f);
      const auto& p1 = center;
      const Eigen::Vector2f& p2 =
          center + r1 * Eigen::Vector2f{cos(theta), sin(theta)};

      sara::draw_line(p1, p2, color, 2);
      sara::draw_circle(center, r1, color, 2 + 2);
    }
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
                                  int width = 3)
{
  if (e.empty())
    return;

#pragma omp parallel for
  for (auto i = 0; i < e.size(); ++i)
  {
    const auto& c = e.type(i) == 1 ? sara::Red8 : sara::Cyan8;
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

    // Contour of orientation line.
    sara::draw_line(display, p1.x(), p1.y(), p2.x(), p2.y(), sara::Black8,
                    width + 2);
    sara::draw_circle(display, p1.x(), p1.y(), r, sara::Black8, width + 2);
    sara::draw_line(display, p1.x(), p1.y(), p2.x(), p2.y(), c, width);
    sara::draw_circle(display, p1.x(), p1.y(), r, c, width);
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
    const float r =
        std::round(e.scale(i) * octave_scaling_factor * float(M_SQRT2));

    sara::draw_circle(display, x, y, r, c, width);
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
    const float r = std::round(e.s(i) * octave_scaling_factor * float(M_SQRT2));

    sara::draw_circle(display, x, y, r, c, width);
  }
}
