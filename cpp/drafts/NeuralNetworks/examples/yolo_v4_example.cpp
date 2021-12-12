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
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include <drafts/NeuralNetworks/Darknet/Layer.hpp>
#include <drafts/NeuralNetworks/Darknet/Parser.hpp>


namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
  namespace fs = boost::filesystem;

  const auto data_dir_path =
      fs::canonical(fs::path{src_path("../../../../data")});
  const auto cfg_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.cfg";
  const auto weights_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.weights";

  auto net = sara::Darknet::NetworkParser{}.parse_config_file(cfg_filepath.string());
  sara::Darknet::NetworkWeightLoader{weights_filepath.string()}.load(net);

  const auto image = sara::imread<sara::Rgb32f>((data_dir_path / "GuardOnBlonde.tif").string());
  sara::create_window(image.sizes());
  sara::display(image);
  // sara::get_key();

  // Resize the image to the network input sizes.
  // TODO: optimize later.
  const auto& input_layer = dynamic_cast<const sara::Darknet::Input &>(*net.front());
  const auto image_resized = sara::resize(image, {input_layer.width, input_layer.height});
  sara::display(image_resized);
  // sara::get_key();

  // Create the gaussian smoothing kernel for RGB color values.
  const auto& conv_1 = dynamic_cast<const sara::Darknet::Convolution&>(*net[1]);
  const auto& conv_2 = dynamic_cast<const sara::Darknet::Convolution&>(*net[2]);
  const auto& conv_3 = dynamic_cast<const sara::Darknet::Convolution&>(*net[3]);

  // Implement the convolutional forward function.
  auto forward = [](const auto& conv_layer, const auto& x, auto& y) {
    const auto& w = conv_layer.weights.w;
    const auto& b = conv_layer.weights.b;
    const auto& stride = conv_layer.stride;
    std::cout << "Forwarding to\n" << conv_layer << std::endl;

    // Convolve.
    im2col_gemm_convolve(
        y,
        x,                                       // the signal
        w,                                       // the transposed kernel.
        sara::make_constant_padding(0.f),        // the padding type
        {x.size(0), x.size(1), stride, stride},  // strides in the convolution
        {0, 0, -1, -1});  // Be careful about the C dimension.

    // Bias.
    for (auto n = 0; n < y.size(0); ++n)
      for (auto c = 0; c < y.size(1); ++c)
        y[n][c].flat_array() += b(c);

    // TODO: activation.
    // y.flat_array() = ...
  };

  auto visualize = [](const auto& y) {
    for (auto i = 0; i < y.size(1); ++i)
    {
      const auto y_i = y[0][i];
      const auto im_i = sara::image_view(y_i);
      const auto im_i_rescaled = sara::color_rescale(im_i);
      sara::display(im_i_rescaled);
      sara::get_key();
    }
  };


  // Transpose the image from NHWC to NCHW storage order.
  //                          0123    0312
  const auto x = sara::tensor_view(image_resized)
                     .reshape(Eigen::Vector4i{1, image_resized.height(),
                                              image_resized.width(), 3})
                     .transpose({0, 3, 1, 2});

  // Intermediate inputs
  auto y1 = sara::Tensor_<float, 4>{conv_1.output_sizes};
  auto y2 = sara::Tensor_<float, 4>{conv_2.output_sizes};
  auto y3 = sara::Tensor_<float, 4>{conv_3.output_sizes};

  forward(conv_1, x, y1);
  // visualize(y1);

  forward(conv_2, y1, y2);
  visualize(y2);

  // forward(conv_3, y2, y3);
  // visualize(y3);

  return 0;
}
