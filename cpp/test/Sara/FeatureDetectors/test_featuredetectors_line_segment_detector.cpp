// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "FeatureDetectors/Line Segment Detector"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/FeatureDetectors/LineSegmentDetector.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/Geometry/Algorithms/Polyline.hpp>



using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestLineSegmentDetectors)

BOOST_AUTO_TEST_CASE(test_line_segment_detector_on_synthetic_square_image)
{
  // Create a square image.
  constexpr auto N = 20;
  constexpr auto pad = 3;
  auto image = Image<float>{N, N};
  image.flat_array().fill(0);
  for (auto y = pad; y < N - pad; ++y)
    for (auto x = pad; x < N - pad; ++x)
      image(x, y) = 1;

  // Preprocess the image first.
  // Internally, we apply a centered 3x3 derivative filter instead of
  // Roberts-Cross filter to localize edges. This may not be optimal, so we
  // have to apply a strong blur variance.
  apply_gaussian_filter(image, image, 1.8f);

  // The line segment detector.
  auto lsd = LineSegmentDetector{};
  lsd.parameters.high_threshold_ratio = 5e-2f;
  lsd.parameters.low_threshold_ratio = 2e-2f;
  lsd.parameters.angular_threshold = static_cast<float>(10._deg);

  // Detect line segments.
  lsd(image);

  // std::cout << lsd.pipeline.edge_map.matrix().cast<int>() << std::endl;

  // We must find the 4 edges of the square.
  BOOST_CHECK_EQUAL(lsd.pipeline.line_segments.size(), 4u);
  for (const auto& [is_line, line] : lsd.pipeline.line_segments)
    BOOST_CHECK_GE(line.length(), 9);
}

BOOST_AUTO_TEST_CASE(test_edge_detector_on_synthetic_square_image)
{
  // Create a square image.
  constexpr auto N = 20;
  constexpr auto pad = 3;
  auto image = Image<float>{N, N};
  image.flat_array().fill(0);
  for (auto y = pad; y < N - pad; ++y)
    for (auto x = pad; x < N - pad; ++x)
      image(x, y) = 1;

  // Preprocess the image first.
  // Internally, we apply a centered 3x3 derivative filter instead of
  // Roberts-Cross filter to localize edges. This may not be optimal, so we
  // have to apply a strong blur variance.
  apply_gaussian_filter(image, image, 1.8f);

  // The line segment detector.
  auto ed = EdgeDetector{};
  ed.parameters.high_threshold_ratio = 5e-2f;
  ed.parameters.low_threshold_ratio = 2e-2f;
  ed.parameters.angular_threshold = static_cast<float>(10._deg);

  // Detect line segments.
  ed(image);

  // std::cout << ed.pipeline.edge_map.matrix().cast<int>() << std::endl;

  // We must find the 4 edges of the square.
  const auto& edges_detected = ed.pipeline.edges_simplified;
  auto edges_filtered = std::vector<std::vector<Eigen::Vector2d>>{};
  std::copy_if(edges_detected.begin(), edges_detected.end(),
               std::back_inserter(edges_filtered),
               [](const auto& e) {
                 return e.size() >= 2;
               });

  BOOST_CHECK_EQUAL(edges_filtered.size(), 4u);
  for (const auto& e : edges_filtered)
    BOOST_CHECK_GE(length(e), 9);
}

BOOST_AUTO_TEST_SUITE_END()
