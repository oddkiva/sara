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

#include <drafts/NeuralNetworks/Darknet/Network.hpp>
#include <drafts/NeuralNetworks/Darknet/Parser.hpp>
#include <drafts/NeuralNetworks/Darknet/YoloUtilities.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <boost/filesystem.hpp>

#include <iomanip>

#include <omp.h>

#define COMPARE_WITH_DARKNET_OUTPUT
#if defined(COMPARE_WITH_DARKNET_OUTPUT)
#  include <drafts/NeuralNetworks/Darknet/Debug.hpp>
#endif


namespace sara = DO::Sara;
namespace d = DO::Sara::Darknet;
namespace fs = boost::filesystem;


inline auto check_yolo_implementation(d::Network& model,
                                      const std::string& output_dir)
{
  namespace fs = boost::filesystem;

  if (!fs::exists(output_dir))
    throw std::runtime_error{"Ouput directory " + output_dir +
                             "does not exist!"};

  // Check the weights.
  d::check_convolutional_weights(model, output_dir);

  const auto x = d::read_tensor(                     //
      (fs::path{output_dir} / "input.bin").string()  //
  );
  const auto xt = x.transpose({0, 2, 3, 1});

  const auto image = sara::ImageView<sara::Rgb32f>{
      reinterpret_cast<sara::Rgb32f*>(const_cast<float*>(xt.data())),
      {xt.size(2), xt.size(1)}};
  const auto& image_resized = image;

  sara::create_window(3 * image.width(), image.height());
  sara::display(image);

  model.debug = true;

  model.forward(x);

  // Compare my layer outputs with Darknet's.
  const auto gt = d::read_all_intermediate_outputs(output_dir);

  const auto& net = model.net;
  for (auto layer = 1u; layer < net.size(); ++layer)
  {
    std::cout << "CHECKING LAYER " << layer << ": " << net[layer]->type
              << std::endl
              << *net[layer] << std::endl;
    d::check_against_ground_truth(gt[layer - 1], net[layer]->output,
                                  image_resized.sizes(),
                                  /* max_diff_thres */ 2e-4f,
                                  /* show_errors */ true);
  }

  SARA_DEBUG << "EVERYTHING OK" << std::endl;
  SARA_DEBUG << "EVERYTHING OK" << std::endl;
  SARA_DEBUG << "EVERYTHING OK" << std::endl;
  SARA_DEBUG << "EVERYTHING OK" << std::endl;
  SARA_DEBUG << "EVERYTHING OK" << std::endl;
  SARA_DEBUG << "EVERYTHING OK" << std::endl;
}


auto graphics_main(int, char**) -> int
{
  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});

  static constexpr auto yolo_version = 4;
  static constexpr auto is_tiny = false;
  const auto yolo_dirpath = data_dir_path / "trained_models" /
                            ("yolov" + std::to_string(yolo_version));
  auto model =
      sara::Darknet::load_yolo_model(yolo_dirpath, yolo_version, is_tiny);

  const auto yolo_intermediate_output_dir = "/home/david/GitHub/darknet/yolov4";
  check_yolo_implementation(model, yolo_intermediate_output_dir);

  return 0;
}


auto main(int argc, char** argv) -> int
{
#ifndef __APPLE__
  Eigen::initParallel();
#endif

  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(graphics_main);
  return app.exec();
}
