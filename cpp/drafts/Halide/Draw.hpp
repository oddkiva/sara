#pragma once

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>

#include <drafts/Halide/ExtremaDataStructures.hpp>
#include <drafts/Halide/Utilities.hpp>

#include "shakti_halide_gray32f_to_rgb.h"
#include "shakti_halide_rgb_to_gray.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


auto show_dog_pyramid(sara::ImagePyramid<float>& dog_pyramid)
{
  for (auto o = 0; o < dog_pyramid.num_octaves(); ++o)
  {
    for (auto s = 0; s < dog_pyramid.num_scales_per_octave(); ++s)
    {
      auto& dog = dog_pyramid(s, o);

      auto image_rgb = sara::Image<sara::Rgb8>{dog.sizes()};
      dog.flat_array() = (dog.flat_array() + 1.f) / 2.f;
      auto buffer_gray = halide::as_runtime_buffer<float>(dog);
      auto buffer_rgb = halide::as_interleaved_runtime_buffer(image_rgb);
      shakti_halide_gray32f_to_rgb(buffer_gray, buffer_rgb);

      sara::display(image_rgb);
    }
  }
}

auto show_pyramid(const sara::ImagePyramid<float>& pyramid)
{
  for (auto o = 0; o < pyramid.num_octaves(); ++o)
    for (auto s = 0; s < pyramid.num_scales_per_octave(); ++s)
      sara::display(sara::color_rescale(pyramid(s, o)));
}

auto draw_quantized_extrema(const halide::Pyramid<halide::QuantizedExtremumArray>& extrema)
{
  for  (const auto& so: extrema.scale_octave_pairs)
  {
    const auto& s = so.first.first;
    const auto& o = so.first.second;

    const auto& scale  = so.second.first;
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

auto draw_extrema(const halide::Pyramid<halide::OrientedExtremumArray>& extrema)
{
  for  (const auto& so: extrema.scale_octave_pairs)
  {
    const auto& s = so.first.first;
    const auto& o = so.first.second;

    const auto& octave_scaling_factor = so.second.second;

    auto eit = extrema.dict.find({s, o});
    if (eit == extrema.dict.end())
      continue;

    const auto& extrema_so = eit->second;

    for (auto i = 0u; i < extrema_so.x.size(); ++i)
    {
      const auto c1 = extrema_so.type[i] == 1 ? sara::Red8 : sara::Blue8;

      const auto x1 = extrema_so.x[i] * octave_scaling_factor;
      const auto y1 = extrema_so.y[i] * octave_scaling_factor;
      const auto& o1 = extrema_so.orientations[i];
      const auto p1 = Eigen::Vector2f{x1, y1};

      // N.B.: the blob radius is the scale multiplied sqrt(2).
      // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
      const auto r1 = extrema_so.s[i] * octave_scaling_factor * std::sqrt(2.f);

      const Eigen::Vector2f p2 = p1 + r1 * Eigen::Vector2f{cos(o), sin(o)};

      sara::draw_line(p1, p2, c1, 2);
      sara::draw_circle(x1, y1, r1, c1, 2 + 2);
    }
  }
}
