// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageIO/Read write HEIF files"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/ImageIO.hpp>

#include <filesystem>


namespace sara = DO::Sara;
namespace fs = std::filesystem;


BOOST_AUTO_TEST_SUITE(TestImageIO)

BOOST_AUTO_TEST_CASE(test_webp_image_format_read_and_write)
{
  const auto w = 64;
  const auto h = 64;

  for (auto i = 0; i < 10; ++i)
  {
    // Write dummy image.
    {
      auto image = sara::Image<sara::Rgb8>{w, h};
      image.flat_array().fill(sara::Red8);
      imwrite(image, "output.webp", 100);
    }

    // Read the image we just wrote.
    {
      const auto filepath = "output.webp";

      // Check the image dimensions.
      const auto image = sara::imread<sara::Rgb8>(filepath);
      BOOST_CHECK_EQUAL(image.width(), w);
      BOOST_CHECK_EQUAL(image.height(), h);

      // Check the pixel values.
      static constexpr auto max_relative_error = 0.05f;  // 5%
      BOOST_CHECK(std::all_of(image.begin(), image.end(), [](const auto& v) {
        // Very liberal check... The webp compression is a bit peculiar.
        const auto rel_err = std::abs(static_cast<float>(v[0]) / 255.f - 1.f);
        return rel_err < max_relative_error;
      }));

      fs::remove(filepath);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
