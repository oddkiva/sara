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

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/ImageIO.hpp>

#include <libheif/heif_cxx.h>


namespace sara = DO::Sara;
namespace fs = boost::filesystem;


BOOST_AUTO_TEST_SUITE(TestImageIO)

BOOST_AUTO_TEST_CASE(test_rgb_image_read_write)
{
  const auto w = 64;
  const auto h = 64;

  for (auto i = 0; i < 10; ++i)
  {
    // Write dummy image.
    //
    // CAVEAT: libheif/libx265 is leaking when writing (!!!)
    // Even if I try to mitigate the amount of leaks by using a singleton
    // object, it will still leak as much.
    //
    // But also is it a true memory leak?
    {
      auto image = sara::Image<sara::Rgb8>{w, h};
      image.flat_array().fill(sara::Red8);
      imwrite(image, "output.heic", 50);
    }

    // Read the image we just wrote.
    {
      const auto filepath = "output.heic";

      const auto image = sara::imread<sara::Rgb8>(filepath);
      BOOST_CHECK_EQUAL(image.width(), w);
      BOOST_CHECK_EQUAL(image.height(), h);
      BOOST_CHECK(std::all_of(image.begin(), image.end(), [](const auto& v) {
        return (v.template cast<float>() - sara::Red8.cast<float>()).norm() < 2;
      }));

      // fs::remove(filepath);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
