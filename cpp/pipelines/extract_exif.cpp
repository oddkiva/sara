// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageIO/Details/Exif.hpp>
#include <DO/Sara//Graphics.hpp>
#include <DO/Sara/FileSystem.hpp>

#include <iterator>


namespace sara = DO::Sara;


void extract_exif()
{
  const auto dirpath = fs::path{"/mnt/a1cc5981-3655-4f74-9c62-37253d79c82d/sfm/Trafalgar/images"};
  const auto image_paths = sara::ls(dirpath.string(), ".jpg");

  auto exif_data = std::vector<EXIFInfo>{};
  exif_data.reserve(image_paths.size());

  std::transform(std::begin(image_paths), std::end(image_paths),
                 std::back_inserter(exif_data), [&](const auto& path) {
                   SARA_DEBUG << "Reading exif data from image " << path
                              << "..." << std::endl;
                   auto exif_info = EXIFInfo{};
                   sara::read_exif_info(exif_info, path);

                   SARA_DEBUG << "EXIF DATA:\n" << exif_info << std::endl;
                   SARA_DEBUG << "shutter speed value: " << exif_info.ShutterSpeedValue << std::endl;

                   return exif_info;
                 });

  SARA_CHECK(exif_data.size());
}


GRAPHICS_MAIN()
{
  extract_exif();
  return 0;
}
