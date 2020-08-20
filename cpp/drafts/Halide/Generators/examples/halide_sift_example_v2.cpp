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

//! @example

#include <algorithm>
#include <cmath>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Differential.hpp>
#include <drafts/Halide/LocalExtrema.hpp>
#include <drafts/Halide/Pyramids.hpp>
#include <drafts/Halide/RefineExtrema.hpp>
#include <drafts/Halide/Utilities.hpp>

#include <drafts/Halide/DominantGradientOrientations.hpp>
#include <drafts/Halide/Draw.hpp>
#include <drafts/Halide/Resize.hpp>
#include <drafts/Halide/SIFT.hpp>


namespace shakti = DO::Shakti;
namespace halide = DO::Shakti::HalideBackend;


namespace DO::Shakti::HalideBackend {

}  // namespace DO::Shakti::HalideBackend


auto test_on_image()
{
  const auto image_filepath =
#ifdef __APPLE__
      "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#else
      "/home/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#endif
  // "/Users/david/GitLab/DO-CV/sara/cpp/drafts/MatchPropagation/cpp/examples/shelves/shelf-1.jpg";
  auto image = sara::imread<float>(image_filepath);

  auto image_buffer = halide::as_buffer(image);
  auto out_buffer = Halide::Runtime::Buffer(;

  timer.restart();
  sift_extractor(image);
  SARA_DEBUG << "Halide SIFT computation time: "  //
             << timer.elapsed_ms() << " ms" << std::endl;
  SARA_CHECK(sift_extractor.pipeline.num_keypoints());

  // Show the local extrema.
  sara::create_window(image.sizes());
  sara::set_antialiasing();
  sara::display(image);
  draw_extrema(sift_extractor.pipeline.oriented_extrema);
  sara::get_key();
}


GRAPHICS_MAIN()
{
  test_on_image();
  return 0;
}
