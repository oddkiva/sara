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

#define BOOST_TEST_MODULE "Shakti/Segmentation/SuperPixel"

#include <memory>

#include <boost/test/unit_test.hpp>

#include <DO/Shakti/Cuda/Segmentation/SuperPixel.hpp>


using namespace std;
using namespace DO::Shakti;


BOOST_AUTO_TEST_CASE(test_sizeof_cluster)
{
  static_assert(sizeof(Cluster) == 32,
                "Check the memory alignment of the cluster data structure.");
}

class TestSegmentationSLIC
{
protected:
  TestSegmentationSLIC()
  {
    N = 3;
    B = 16;
    M = N * B;
    host_image_sizes = Vector2i{M, M};

    in_host_image.reset(new Vector4f[M * M]);

    auto at = [&](int x, int y) { return x + M * y; };
    for (int y = 0; y < M; ++y)
    {
      for (int x = 0; x < M; ++x)
      {
        auto val = float(x / B + N * (y / B)) / (N * N);
        in_host_image[at(x, y)] = Vector4f{val, val, val, 0.f};
      }
    }
  }

protected:
  // Test host image data.
  int N;
  int B;
  int M;

  Vector2i host_image_sizes;
  unique_ptr<Vector4f[]> in_host_image;
};

BOOST_FIXTURE_TEST_CASE(test_constructor, TestSegmentationSLIC)
{
  SegmentationSLIC slic;
  BOOST_CHECK_EQUAL(slic.get_distance_weight(), 0.f);
}

BOOST_FIXTURE_TEST_CASE(test_getters_and_setters, TestSegmentationSLIC)
{
  SegmentationSLIC slic;

  slic.set_image_sizes({2, 2}, 128);
  BOOST_CHECK_EQUAL(Vector2i(2, 2), slic.get_image_sizes());
  BOOST_CHECK_EQUAL(128, slic.get_image_padded_width());
}

BOOST_FIXTURE_TEST_CASE(test_set_image_sizes_helper, TestSegmentationSLIC)
{
  // Transfer data to device memory.
  MultiArray<Vector4f, 2> in_device_image{in_host_image.get(),
                                          host_image_sizes};

  // Test helper function for image dimensions.
  SegmentationSLIC slic;
  slic.set_image_sizes(in_device_image);
  BOOST_CHECK_EQUAL(in_device_image.sizes(), slic.get_image_sizes());
  BOOST_CHECK_EQUAL(in_device_image.padded_width(),
                    slic.get_image_padded_width());
}

BOOST_FIXTURE_TEST_CASE(test_init_clusters, TestSegmentationSLIC)
{
  // Transfer data to device memory.
  MultiArray<Vector4f, 2> in_device_image{in_host_image.get(),
                                          host_image_sizes};

  // Initialize the image segmentation.
  SegmentationSLIC slic;
  slic.set_image_sizes(in_device_image);

  // Initialize the clusters.
  unique_ptr<Cluster[]> out_host_clusters{new Cluster[N * N]};
  MultiArray<Cluster, 2> out_device_clusters{
      slic.init_clusters(in_device_image)};

  // Check the clusters.
  out_device_clusters.copy_to_host(out_host_clusters.get());
  {
    auto at = [&](int x, int y) { return x + N * y; };
    for (int y = 0; y < N; ++y)
    {
      for (int x = 0; x < N; ++x)
      {
        const auto& center = out_host_clusters[at(x, y)].center;
        const auto& color = out_host_clusters[at(x, y)].color;

        const auto expected_center =
            Vector2f{float(x * B + B / 2), float(y * B + B / 2)};
        const auto v = float(x + N * y) / (N * N);
        const auto expected_color = Vector4f{v, v, v, 0.f};
        BOOST_REQUIRE_EQUAL(expected_center, center);
        BOOST_REQUIRE_EQUAL(expected_color, color);
        BOOST_REQUIRE_EQUAL(0, out_host_clusters[at(x, y)].num_points);
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_init_labels, TestSegmentationSLIC)
{
  // Transfer data to device memory.
  MultiArray<Vector4f, 2> in_device_image{in_host_image.get(),
                                          host_image_sizes};

  // Initialize the image segmentation.
  SegmentationSLIC slic;
  slic.set_image_sizes(in_device_image);

  // Initialize the clusters.
  MultiArray<int, 2> out_device_labels{slic.init_labels(in_device_image)};

  // Check.
  BOOST_CHECK_EQUAL(in_device_image.sizes(), out_device_labels.sizes());
}

BOOST_FIXTURE_TEST_CASE(test_assign_clusters, TestSegmentationSLIC)
{
  // Transfer data to device memory.
  MultiArray<Vector4f, 2> in_device_image{in_host_image.get(),
                                          host_image_sizes};

  // Init the image segmentation.
  SegmentationSLIC slic;
  slic.set_image_sizes(in_device_image);

  // Init the clusters.
  MultiArray<Cluster, 2> out_device_clusters{
      slic.init_clusters(in_device_image)};
  unique_ptr<Cluster[]> out_host_clusters{
      new Cluster[out_device_clusters.size()]};

  // Initialize the labels.
  MultiArray<int, 2> out_device_labels{slic.init_labels(in_device_image)};
  unique_ptr<int[]> out_host_labels{new int[in_device_image.size()]};

  // Assign the means.
  slic.assign_means(out_device_labels, out_device_clusters, in_device_image);
  out_device_labels.copy_to_host(out_host_labels.get());

  // Check the labels.
  auto at = [&](int x, int y) { return x + in_device_image.width() * y; };
  for (int y = 0; y < out_device_labels.height(); ++y)
    for (int x = 0; x < out_device_labels.width(); ++x)
      BOOST_REQUIRE_EQUAL(out_host_labels[at(x, y)], x / B + y / B * N);
}

BOOST_FIXTURE_TEST_CASE(test_update_means, TestSegmentationSLIC)
{
  // Transfer data to device memory.
  MultiArray<Vector4f, 2> in_device_image{in_host_image.get(),
                                          host_image_sizes};

  // Initialize the image segmentation.
  SegmentationSLIC slic;
  slic.set_image_sizes(in_device_image);

  // Initialize the clusters.
  unique_ptr<Cluster[]> out_host_clusters{new Cluster[in_device_image.size()]};
  MultiArray<Cluster, 2> out_device_clusters{
      slic.init_clusters(in_device_image)};

  // Initialize the labels.
  unique_ptr<int[]> out_host_labels{new int[in_device_image.size()]};
  MultiArray<int, 2> out_device_labels{slic.init_labels(in_device_image)};

  // Assign the means.
  slic.assign_means(out_device_labels, out_device_clusters, in_device_image);

  // Update the means.
  slic.update_means(out_device_clusters, out_device_labels, in_device_image);
  out_device_clusters.copy_to_host(out_host_clusters.get());

  // Check the updated means.
  auto at = [&](int x, int y) { return x + out_device_clusters.width() * y; };
  for (int y = 0; y < out_device_clusters.height(); ++y)
  {
    for (int x = 0; x < out_device_clusters.width(); ++x)
    {
      // Check the color.
      const auto v = float(x + N * y) / (N * N);
      BOOST_REQUIRE_SMALL((v - out_host_clusters[at(x, y)].color.x()), 1e-4f);
      BOOST_REQUIRE_SMALL((v - out_host_clusters[at(x, y)].color.y()), 1e-4f);
      BOOST_REQUIRE_SMALL((v - out_host_clusters[at(x, y)].color.z()), 1e-4f);
      BOOST_REQUIRE_SMALL((0 - out_host_clusters[at(x, y)].color.w()), 1e-4f);

      // Check the number of points.
      BOOST_REQUIRE_EQUAL(B * B, out_host_clusters[at(x, y)].num_points);

      // Check the mean center.
      // Square patch [0, 16[ x [0, 16[ = Square patch [0, 15]x [0, 15]
      // Thus square patch is (15/2, 15/2) = 7.5
      BOOST_REQUIRE_EQUAL(Vector2f(x * B + B / 2 - 0.5f, y * B + B / 2 - 0.5f),
                out_host_clusters[at(x, y)].center);
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_segmentation_cuda, TestSegmentationSLIC)
{
  // Transfer data to device memory.
  MultiArray<Vector4f, 2> in_device_image{in_host_image.get(),
                                          host_image_sizes};

  // Initialize the image segmentation.
  SegmentationSLIC slic;
  MultiArray<int, 2> labels{slic(in_device_image)};

  // Initialize the labels.
  unique_ptr<int[]> out_host_labels{new int[in_device_image.size()]};

  // Check the labels.
  labels.copy_to_host(out_host_labels.get());

  auto at = [&](int x, int y) { return x + in_device_image.width() * y; };
  for (int y = 0; y < labels.height(); ++y)
    for (int x = 0; x < labels.width(); ++x)
      BOOST_REQUIRE_EQUAL(out_host_labels[at(x, y)], x / B + y / B * N);
}

BOOST_FIXTURE_TEST_CASE(test_segmentation, TestSegmentationSLIC)
{
  unique_ptr<int[]> out_host_labels{new int[M * M]};

  SegmentationSLIC slic;
  slic(out_host_labels.get(), in_host_image.get(), host_image_sizes.data());

  auto at = [&](int x, int y) { return x + M * y; };
  for (int y = 0; y < M; ++y)
    for (int x = 0; x < M; ++x)
      BOOST_REQUIRE_EQUAL(out_host_labels[at(x, y)], x / B + y / B * N);
}
