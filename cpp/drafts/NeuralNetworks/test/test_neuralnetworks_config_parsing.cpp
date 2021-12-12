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

#define BOOST_TEST_MODULE "NeuralNetworks/Yolo Configuration Parsing"

#include <DO/Sara/Core.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include <drafts/NeuralNetworks/Darknet/Layer.hpp>
#include <drafts/NeuralNetworks/Darknet/Parser.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_SUITE(TestLayers)

BOOST_AUTO_TEST_CASE(test_yolov4_tiny_config_parsing)
{
  namespace fs = boost::filesystem;

  const auto data_dir_path =
      fs::canonical(fs::path{src_path("../../../../data")});
  const auto cfg_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.cfg";
  const auto weights_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.weights";
  BOOST_CHECK(fs::exists(cfg_filepath));

  auto net = sara::Darknet::NetworkParser{}.parse_config_file(cfg_filepath.string());
  sara::Darknet::NetworkWeightLoader{weights_filepath.string()}.load(net);

  const auto image = sara::imread<sara::Rgb32f>((data_dir_path / "dog.jpg").string());

  // Resize the image to the network input sizes.
  // TODO: optimize later.
  const auto& input_layer = dynamic_cast<const sara::Darknet::Input &>(*net.front());
  const auto image_resized = sara::resize(image, {input_layer.width, input_layer.height});

  // Transpose the image from NHWC to NCHW storage order.
  //                          0123    0312
  const auto x = sara::tensor_view(image_resized)
                     .reshape(Eigen::Vector4i{1, image_resized.height(),
                                              image_resized.width(), 3})
                     .transpose({0, 3, 1, 2});

  // Create the gaussian smoothing kernel for RGB color values.
  const auto& conv_layer = dynamic_cast<const sara::Darknet::Convolution&>(*net[1]);
  auto kt = conv_layer.weights.w;

  // Convolve the image using the GEMM BLAS routine.
  auto y = gemm_convolve(
      x,                  // the signal
      kt,                 // the transposed kernel.
      sara::make_constant_padding(0.f),  // the padding type
      // make_constant_padding(0.f),      // the padding type
      {1, kt.size(0), 2, 2},  // strides in the convolution
      {0, 1, 0, 0});  // pay attention to the offset here for the C dimension.

  std::cout << y.sizes().transpose() << std::endl;

  // Transpose the tensor data back to NHWC storage order to view the image.
  y = y.transpose({0, 2, 3, 1});
}

BOOST_AUTO_TEST_SUITE_END()
