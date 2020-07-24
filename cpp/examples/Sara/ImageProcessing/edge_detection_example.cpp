// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <set>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/DisjointSets/DisjointSets.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace std;
using namespace DO::Sara;


auto non_max_suppression(const ImageView<float>& grad_mag,
                         const ImageView<float>& grad_ori, float hiThres,
                         float loThres)
{
  auto edges = Image<uint8_t>{grad_mag.sizes()};
  edges.flat_array().fill(0);
  for (auto y = 1; y < grad_mag.height() - 1; ++y)
  {
    for (auto x = 1; x < grad_mag.width() - 1; ++x)
    {
      const auto& grad_curr = grad_mag(x, y);
      if (grad_curr < loThres)
        continue;

      const auto& theta = grad_ori(x, y);
      const Vector2d p = Vector2i(x, y).cast<double>();
      const Vector2d d = Vector2f{cos(theta), sin(theta)}.cast<double>();
      const Vector2d p0 = p - d;
      const Vector2d p2 = p + d;
      const auto grad_prev = interpolate(grad_mag, p0);
      const auto grad_next = interpolate(grad_mag, p2);

      const auto is_max = grad_curr > grad_prev &&  //
                          grad_curr > grad_next;
      if (!is_max)
        continue;

      edges(x, y) = grad_curr > hiThres ? 255 : 128;
    }
  }
  return edges;
}

auto hysteresis(ImageView<std::uint8_t>& edges)
{
  auto visited = Image<std::uint8_t>{edges.sizes()};
  visited.flat_array().fill(0);

  std::queue<Eigen::Vector2i> queue;
  for (auto y = 0; y < edges.height(); ++y)
  {
    for (auto x = 0; x < edges.width(); ++x)
    {
      if (edges(x, y) == 255)
      {
        queue.emplace(x, y);
        visited(x, y) = 1;
      }
    }
  }

  const auto dir = std::array<Eigen::Vector2i, 8>{
      Eigen::Vector2i{1, 0},  Eigen::Vector2i{1, 1},  Eigen::Vector2i{0, 1},
      Eigen::Vector2i{-1, 1}, Eigen::Vector2i{-1, 0}, Eigen::Vector2i{-1, -1},
      Eigen::Vector2i{0, -1}, Eigen::Vector2i{1, -1}};
  while (!queue.empty())
  {
    const auto& p = queue.front();

    // Promote a weak edge to a strong edge.
    if (edges(p) != 255)
      edges(p) = 255;

    // Add nonvisited weak edges.
    for (const auto& d : dir)
    {
      const Eigen::Vector2i n = p + d;
      // Boundary conditions.
      if (n.x() < 0 || n.x() >= edges.width() ||  //
          n.y() < 0 || n.y() >= edges.height())
        continue;

      if (edges(n) == 128 and not visited(n))
      {
        visited(n) = 1;
        queue.emplace(n);
      }
    }

    queue.pop();
  }
}


auto connected_components(const ImageView<std::uint8_t>& edges)
{
  const auto index = [&edges](const Eigen::Vector2i& p) {
    return p.y() * edges.width() + p.x();
  };

  const auto is_edgel = [&edges](const Eigen::Vector2i& p) {
    return edges(p) == 255;
  };

  auto ds = DisjointSets(edges.size());
  auto visited = Image<std::uint8_t>{edges.sizes()};
  visited.flat_array().fill(0);

  // Collect the edgels and make as many sets as pixels.
  std::queue<Eigen::Vector2i> q;
  for (auto y = 0; y < edges.height(); ++y)
  {
    for (auto x = 0; x < edges.width(); ++x)
    {
      ds.make_set(index({x, y}));
      if (is_edgel({x, y}))
        q.emplace(x, y);
    }
  }

  const auto dir = std::array<Eigen::Vector2i, 8>{
      Eigen::Vector2i{1, 0},    //
      Eigen::Vector2i{1, 1},    //
      Eigen::Vector2i{0, 1},    //
      Eigen::Vector2i{-1, 1},   //
      Eigen::Vector2i{-1, 0},   //
      Eigen::Vector2i{-1, -1},  //
      Eigen::Vector2i{0, -1},   //
      Eigen::Vector2i{1, -1}    //
  };

  while (!q.empty())
  {
    const auto& p = q.front();
    visited(p) = 2; // 2 = visited

    if (!is_edgel(p))
      throw std::runtime_error{"NOT AN EDGEL!"};

    // Find its corresponding node in the disjoint set.
    const auto node_p = ds.node(index(p));
    // draw_point(p.x(), p.y(), Red8);

    // Add nonvisited weak edges.
    for (const auto& d : dir)
    {
      const Eigen::Vector2i n = p + d;
      // Boundary conditions.
      if (n.x() < 0 || n.x() >= edges.width() ||  //
          n.y() < 0 || n.y() >= edges.height())
        continue;

      // Make sure that the neighbor is an edgel.
      if (!is_edgel(n))
        continue;

      // draw_point(n.x(), n.y(), Yellow8);

      // Merge component of p and component of n.
      const auto node_n = ds.node(index(n));
      ds.join(node_p, node_n);

      // Enqueue the neighbor n if it is not already enqueued
      if (visited(n) == 0)
      {
        // Enqueue the neighbor.
        q.emplace(n);
        visited(n) = 1;  // 1 = enqueued
      }
    }

    q.pop();
  }

  auto contours = std::map<int, std::vector<Point2i>>{};
  for (auto y = 0; y < edges.height();++y)
  {
    for (auto x = 0; x < edges.width();++x)
    {
      const auto p = Eigen::Vector2i{x, y};
      const auto index_p = index(p);
      if (is_edgel(p))
        contours[ds.component(index_p)].push_back(p);
    }
  }

  return contours;
}

auto test_on_image()
{
  // Read an image.
  const auto image =
      imread<float>(src_path("../../../../data/sunflowerField.jpg"));

  auto sigma = sqrt(pow(1.6f, 2) - pow(0.5f, 2));
  auto image_curr = deriche_blur(image, sigma);

  create_window(image.sizes());
  display(image_curr);

  for (auto s = 0; s < 500; ++s)
  {
    // Blur.
    const auto grad = gradient(image_curr);
    const auto grad_mag = grad.cwise_transform(  //
        [](const auto& v) { return v.norm(); });
    const auto grad_ori = grad.cwise_transform(
        [](const auto& v) { return std::atan2(v.y(), v.x()); });

    const auto hiThres = grad_mag.flat_array().maxCoeff() * 0.2;
    const auto loThres = hiThres * 0.05;

    auto edges = non_max_suppression(grad_mag, grad_ori, hiThres, loThres);

    hysteresis(edges);

    const auto contours = connected_components(edges);

    auto grouped_edgels = Image<Rgb8>{edges.sizes()};
    grouped_edgels.flat_array().fill(Black8);
    for (const auto& c: contours)
    {
      const auto& points = c.second;
      const auto color = Rgb8(rand() % 255, rand() % 255, rand() % 255);
      for (const auto& p : points)
        grouped_edgels(p) = color;
    }
    display(grouped_edgels);

    millisleep(1);

    const auto delta = std::pow(2., 1. / 100.);
    const auto sigma = 1.6 * sqrt(pow(delta, 2 * s + 2) - pow(delta, 2 * s));
    image_curr = deriche_blur(image_curr, sigma);
  }

  get_key();
}

auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  // const auto video_filepath = "/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
  const auto video_filepath = "/Users/david/Desktop/Datasets/humanising-autonomy/field.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray32f = Image<float>{frame.sizes()};

  // Show the local extrema.
  create_window(frame.sizes());
  set_antialiasing();

  while (true)
  {
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }

    convert(frame, frame_gray32f);

    inplace_deriche_blur(frame_gray32f, std::sqrt(std::pow(1.6f, 2) - 1));

    const auto grad = gradient(frame_gray32f);
    const auto grad_mag = grad.cwise_transform(  //
        [](const auto& v) { return v.norm(); });
    const auto grad_ori = grad.cwise_transform(
        [](const auto& v) { return std::atan2(v.y(), v.x()); });

    const auto hiThres = grad_mag.flat_array().maxCoeff() * 0.2;
    const auto loThres = hiThres * 0.05;

    auto edges = non_max_suppression(grad_mag, grad_ori, hiThres, loThres);

    hysteresis(edges);

    const auto contours = connected_components(edges);

    auto grouped_edgels = Image<Rgb8>{frame.sizes()};
    //convert(frame_gray32f, grouped_edgels);
    grouped_edgels.flat_array().fill(Black8);
    for (const auto& c: contours)
    {
      const auto& points = c.second;
      const auto color = Rgb8(rand() % 255, rand() % 255, rand() % 255);
      for (const auto& p : points)
        grouped_edgels(p) = color;
    }
    display(grouped_edgels);
  }
}

GRAPHICS_MAIN()
{
  test_on_video();
  return 0;
}
