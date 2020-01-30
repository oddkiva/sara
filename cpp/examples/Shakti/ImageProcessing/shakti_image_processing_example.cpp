// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Geometry/Tools/Utilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO/VideoStream.hpp>

#include <DO/Shakti/ImageProcessing.hpp>
#include <DO/Shakti/MultiArray.hpp>
#include <DO/Shakti/Utilities/DeviceInfo.hpp>
#include <DO/Shakti/Utilities/Timer.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;


using namespace std;
using namespace DO;
using namespace sara;


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
void draw_dense_sifts(const Image<Vector128f>& dense_sifts, int num_blocks_x,
                      int num_blocks_y, float sigma = 1.6f,
                      float bin_scale_length = 3.f)
{
  int w = dense_sifts.width();
  int h = dense_sifts.height();

  for (int j = 0; j < num_blocks_y; ++j)
  {
    for (int i = 0; i < num_blocks_x; ++i)
    {
      const Point2f a{float(i) / num_blocks_x * w, float(j) / num_blocks_y * h};
      const Point2f b{float(i + 1) / num_blocks_x * w,
                      float(j + 1) / num_blocks_y * h};
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


GRAPHICS_MAIN()
{
  try
  {
    auto devices = shakti::get_devices();
    devices.front().make_current_device();
    cout << devices.front() << endl;

    // const auto video_filepath = src_path("Segmentation/orion_1.mpg");
    const string video_filepath = "/home/david/Desktop/test.mp4";
    cout << video_filepath << endl;
    VideoStream video_stream{video_filepath};
    auto video_frame_index = int{0};
    auto video_frame = video_stream.frame();

    auto in_frame = Image<float>{video_stream.sizes()};
    auto out_frame = Image<float>{video_stream.sizes()};
    auto apply_gaussian_filter = shakti::GaussianFilter{3.f};

    // auto sifts = Image<Vector128f>{};
    // auto sift_computer = shakti::DenseSiftComputer{};

    auto cpu_timer = sara::Timer{};
    auto cpu_time = 0.;

    create_window(video_frame.sizes());

    while (true)
    {
      cpu_timer.restart();
      if (!video_stream.read())
        break;
      cpu_time = cpu_timer.elapsed_ms();
      std::cout << "[CPU video decoding time] " << cpu_time << "ms" << std::endl;

      cpu_timer.restart();
      std::transform(video_frame.begin(), video_frame.end(), in_frame.begin(),
                     [](const Rgb8& c) -> float {
                       auto gray = float{};
                       sara::smart_convert_color(c, gray);
                       return gray;
                     });
      cpu_time = cpu_timer.elapsed_ms();
      std::cout << "[CPU color conversion time] " << cpu_time << "ms" << std::endl;

      shakti::tic();
      apply_gaussian_filter(out_frame.data(), in_frame.data(),
                            in_frame.sizes().data());
      shakti::toc("GPU gaussian filter");

      // shakti::tic();
      // sifts.resize(in_frame.sizes());
      // sift_computer(reinterpret_cast<float*>(sifts.data()), out_frame.data(),
      //               out_frame.sizes().data());
      // shakti::toc("Dense SIFT");

      display(out_frame);
      // draw_sift(sifts(160, 120), 160, 120, 10.f);
      // draw_sift(sifts(160, 120), 160, 120, 1.6f, 3.f);

      ++video_frame_index;
      cout << endl;
    }
  }
  catch (std::exception& e)
  {
    cout << e.what() << endl;
  }

  return 0;
}
