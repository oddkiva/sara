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

#include <drafts/NeuralNetworks/Darknet/Layer.hpp>
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

  auto net = sara::Darknet::NetworkParser{}.parse_config_file(cfg_filepath.string());
  using Net = decltype(net);
  sara::Darknet::NetworkWeightLoader{weights_filepath.string()}.load(net);

  const auto image = argc < 2
                         ? sara::imread<sara::Rgb32f>(
                               (data_dir_path / "GuardOnBlonde.tif").string())
                         : sara::imread<sara::Rgb32f>(argv[1]);
  sara::create_window(image.sizes());
  sara::display(image);

  // Resize the image to the network input sizes.
  // TODO: optimize later.
  const auto& input_layer = dynamic_cast<const sara::Darknet::Input &>(*net.front());
  const auto image_resized = sara::resize(image, {input_layer.width, input_layer.height});
  sara::display(image_resized);

  // Create the gaussian smoothing kernel for RGB color values.
  const auto& conv_1 = dynamic_cast<const sara::Darknet::Convolution&>(*net[1]);
  const auto& conv_2 = dynamic_cast<const sara::Darknet::Convolution&>(*net[2]);
  const auto& conv_3 = dynamic_cast<const sara::Darknet::Convolution&>(*net[3]);
  const auto& route = dynamic_cast<const sara::Darknet::Route&>(*net[4]);
  const auto& conv_4 = dynamic_cast<const sara::Darknet::Convolution&>(*net[5]);
  const auto& conv_5 = dynamic_cast<const sara::Darknet::Convolution&>(*net[6]);


  // Implement the convolutional forward function.
  auto forward_to_conv = [](const auto& conv, const auto& x, auto& y) {
    const auto& w = conv.weights.w;
    const auto& b = conv.weights.b;
    const auto& stride = conv.stride;
    std::cout << "Forwarding to\n" << conv << std::endl;

    sara::tic();

    // Convolve.
    sara::im2col_gemm_convolve(
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

    if (conv.activation != "leaky")
      throw std::runtime_error{"Unsupported activation!"};

    // Leaky activation.
    y.cwise_transform_inplace([](float& x) {
      if (x < 0)
        x *= 0.1f;
    });

    sara::toc("Convoluation Forward Pass");
  };

  auto forward_to_route = [](const sara::Darknet::Route& route,
                             const sara::Tensor_<float, 4>& x,
                             sara::Tensor_<float, 4>& y) {
    sara::tic();
    if (route.layers.size() != 1)
      throw std::runtime_error{"Route layer implementation incomplete!"};

    const auto start = (Eigen::Vector4i{} << 0,
                        route.group_id * (x.size(1) / route.groups), 0, 0)
                           .finished();
    const auto end = x.sizes();
#ifdef DEBUG_ROUTE
    std::cout << "start = " << start.transpose() << std::endl;
    std::cout << "end   = " << end.transpose() << std::endl;
#endif

    y = sara::slice(x,
                    {
                        {start(0), end(0), 1},
                        {start(1), end(1), 1},
                        {start(2), end(2), 1},
                        {start(3), end(3), 1},  //
                    })
            .make_copy();
#ifdef DEBUG_ROUTE
    std::cout << "y sizes = " << y.sizes().transpose() << std::endl;
#endif

    sara::toc("Route Forward Pass");
  };

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


  // Transpose the image from NHWC to NCHW storage order.
  //                          0123    0312
  const auto x = sara::tensor_view(image_resized)
                     .reshape(Eigen::Vector4i{1, image_resized.height(),
                                              image_resized.width(), 3})
                     .transpose({0, 3, 1, 2});
  const auto x_sizes =
      Eigen::Vector2i{image_resized.width(), image_resized.height()};

  // Intermediate inputs
  auto y1 = sara::Tensor_<float, 4>{conv_1.output_sizes};
  auto y2 = sara::Tensor_<float, 4>{conv_2.output_sizes};
  auto y3 = sara::Tensor_<float, 4>{conv_3.output_sizes};
  auto y4 = sara::Tensor_<float, 4>{route.output_sizes};
  auto y5 = sara::Tensor_<float, 4>{conv_4.output_sizes};
  auto y6 = sara::Tensor_<float, 4>{conv_5.output_sizes};

  forward_to_conv(conv_1, x, y1);
  forward_to_conv(conv_2, y1, y2);
  forward_to_conv(conv_3, y2, y3);
  forward_to_route(route, y3, y4);
  forward_to_conv(conv_4, y4, y5);
  forward_to_conv(conv_5, y5, y6);

  // visualize(y1, x_sizes);
  // visualize(y2, x_sizes);
  // visualize(y3, x_sizes);
  // visualize(y4, x_sizes);
  // visualize(y5, x_sizes);
  visualize(y6, x_sizes);

  return 0;
}
