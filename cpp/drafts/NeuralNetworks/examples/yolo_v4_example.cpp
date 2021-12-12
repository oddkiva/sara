// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/MultiArray/Slice.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include <drafts/NeuralNetworks/Darknet/Network.hpp>
#include <drafts/NeuralNetworks/Darknet/Parser.hpp>


namespace sara = DO::Sara;


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int argc, char** argv)
{
  namespace fs = boost::filesystem;

  const auto data_dir_path =
      fs::canonical(fs::path{src_path("../../../../data")});
  const auto cfg_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.cfg";
  const auto weights_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.weights";

  auto model = sara::Darknet::Network{};
  auto& net = model.net;
  net = sara::Darknet::NetworkParser{}.parse_config_file(cfg_filepath.string());
  sara::Darknet::NetworkWeightLoader{weights_filepath.string()}.load(net);

  const auto image = argc < 2
                         ? sara::imread<sara::Rgb32f>(
                               (data_dir_path / "GuardOnBlonde.tif").string())
                         : sara::imread<sara::Rgb32f>(argv[1]);
  sara::create_window(image.sizes());
  sara::display(image);

  // Resize the image to the network input sizes.
  // TODO: optimize later.
  const auto& input_layer =
      dynamic_cast<const sara::Darknet::Input&>(*net.front());
  const auto image_resized =
      sara::resize(image, {input_layer.width, input_layer.height});
  sara::display(image_resized);


  // Feed the input to the network.
  model.forward(sara::tensor_view(image_resized)
                      .reshape(Eigen::Vector4i{1, image_resized.height(),
                                               image_resized.width(), 3})
                      .transpose({0, 3, 1, 2}));

#ifdef CHECK_AGAIN
  auto& x = net[0]->output;

  // Intermediate inputs
  auto& y1 = net[1]->output;
  auto& y2 = net[2]->output;
  auto& y3 = net[3]->output;
  auto& y4 = net[4]->output;
  auto& y5 = net[5]->output;
  auto& y6 = net[6]->output;
  auto& y7 = net[7]->output;
  auto& y8 = net[8]->output;
#endif
  std::cout << "Layer 18:\n" << *net[18] << std::endl;
  auto& y18 = net[18]->output;

  std::cout << "Layer 30:\n" << *net[30] << std::endl;
  auto& y30 = net[30]->output;

  // Visualize.
  auto visualize = [](const auto& y, const auto& sizes) {
    for (auto i = 0; i < y.size(1); ++i)
    {
      const auto y_i = y[0][i];
      const auto im_i = sara::image_view(y_i);
      const auto im_i_rescaled = sara::resize(sara::color_rescale(im_i), sizes);
      sara::display(im_i_rescaled);
      sara::get_key();
    }
  };

  const auto x_sizes = Eigen::Vector2i{
      image_resized.width(), image_resized.height()  //
  };
  // visualize(y1, x_sizes);
  // visualize(y2, x_sizes);
  // visualize(y3, x_sizes);
  // visualize(y4, x_sizes);
  // visualize(y5, x_sizes);
  // visualize(y6, x_sizes);
  // visualize(y7, x_sizes);
  // visualize(y8, x_sizes);
  visualize(y18, x_sizes);
  // visualize(y30, x_sizes);

  return 0;
}
