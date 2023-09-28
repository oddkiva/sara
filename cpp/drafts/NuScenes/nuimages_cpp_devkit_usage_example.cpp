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

//! @file

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>

#include <drafts/NuScenes/NuImages.hpp>


namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
  using namespace std::string_literals;

  const auto nuimages_version = "v1.0-mini"s;
  const auto nuimages_root_path = "/Users/david/Downloads/nuimages-v1.0-mini"s;
  const auto nuimages = NuImages{nuimages_version, nuimages_root_path, true};



  return 0;
}
