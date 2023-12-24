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

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/NeuralNetworks/Darknet/Debug.hpp>
#include <DO/Sara/NeuralNetworks/Darknet/Network.hpp>
#include <DO/Sara/NeuralNetworks/Darknet/Parser.hpp>
#include <DO/Sara/NeuralNetworks/Darknet/YoloUtilities.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <fmt/core.h>


#include <filesystem>
#include <iomanip>

#ifdef _OPENMP
#  include <omp.h>
#endif


namespace d = DO::Sara::Darknet;
namespace fs = std::filesystem;
namespace sara = DO::Sara;


auto check_yolo_implementation(d::Network& model, const fs::path& output_dir)
    -> void
{
  if (!fs::exists(output_dir))
    throw std::runtime_error{fmt::format("Ouput directory '{}' does not exist!",
                                         output_dir.string())};

  // Check the weights.
  d::check_convolutional_weights(model, output_dir);

  const auto x = d::read_tensor((output_dir / "input.bin").string());
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


auto save_network_fused_conv_weights(d::Network& model,
                                     const fs::path& output_dir) -> void
{
  if (!fs::exists(output_dir))
    throw std::runtime_error{fmt::format("Ouput directory '{}' does not exist!",
                                         output_dir.string())};

  model.debug = true;

  const auto& net = model.net;
  auto conv_idx = 0;

  for (auto layer = 1u; layer < net.size(); ++layer)
  {
    auto conv = dynamic_cast<d::Convolution*>(net[layer].get());
    if (!conv)
      continue;

    d::write_tensor(conv->weights.w,
                    output_dir /
                        fmt::format("conv_weight_{conv_idx}.bin", layer));

    const auto b_tensor = sara::TensorView_<float, 1>{
        conv->weights.b.data(), static_cast<int>(conv->weights.b.size())};
    d::write_tensor(b_tensor,
                    output_dir /
                        fmt::format("conv_weight_{conv_idx}.bin", layer));
    ++conv_idx;
  }
}

auto save_network_intermediate_outputs(d::Network& model,
                                       const fs::path& image_path,
                                       const fs::path& output_dir) -> void
{
  if (!fs::exists(output_dir))
    throw std::runtime_error{fmt::format("Ouput directory '{}' does not exist!",
                                         output_dir.string())};

  sara::tic();
  auto input_hwc = sara::imread<sara::Rgb32f>(image_path.string());
  sara::toc("Image read");

  sara::tic();
  const auto input_chw = sara::tensor_view(input_hwc).transpose({2, 0, 1});
  static_assert(std::is_same_v<decltype(input_chw),  //
                               const sara::Tensor_<float, 3>>);
  sara::toc("Image transpose (HWC -> CHW)");

  sara::tic();
  const auto& input_layer = dynamic_cast<const d::Input&>(*model.net.front());
  auto input_nchw = sara::Tensor_<float, 4>{
      {1, 3, input_layer.height(), input_layer.width()}};
  for (auto i = 0; i < 3; ++i)
  {
    const auto src = sara::image_view(input_chw[i]);
    auto dst = sara::image_view(input_nchw[0][i]);
    sara::resize_v2(src, dst);
  }
  sara::toc("Image resize");

  model.debug = true;
  model.forward(input_nchw);

  // Save the input tensor.
  d::write_tensor(input_nchw, output_dir / "yolo_inter_0.bin");

  // Save the intermediate output tensors.
  const auto& net = model.net;
  for (auto layer = 1u; layer < net.size(); ++layer)
  {
    std::cout << "CHECKING LAYER " << layer << ": " << net[layer]->type
              << std::endl
              << *net[layer] << std::endl;

    d::write_tensor(net[layer]->output,
                    output_dir / fmt::format("yolo_inter_{}.bin", layer));
  }
}

auto graphics_main(int, char**) -> int
{
  const auto model_dir_path = fs::canonical(  //
      fs::path{src_path("trained_models")}    //
  );
  fmt::print("YOLO: {}\n", model_dir_path.string());
  if (!fs::exists(model_dir_path))
    throw std::runtime_error{"trained_models directory does not exist"};
  static constexpr auto yolo_version = 4;
  static constexpr auto is_tiny = true;
  const auto yolo_model_name =
      "yolov" + std::to_string(yolo_version) + (is_tiny ? "-tiny" : "");
  const auto yolo_dir_path = model_dir_path / yolo_model_name;
  if (!fs::exists(yolo_dir_path))
    throw std::runtime_error{"YOLO directory does not exist"};
  auto model = d::load_yolo_model(yolo_dir_path, yolo_version, is_tiny);
  fmt::print("Load model OK!");

  const auto yolo_out_dir_path =
      fs::path{"/Users/oddkiva/Desktop/"} / yolo_model_name;
  if (!fs::exists(yolo_out_dir_path))
    fs::create_directory(yolo_out_dir_path);

  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  const auto image_path = data_dir_path / "dog.jpg";
  if (!fs::exists(image_path))
    throw std::runtime_error{"image does not exist!"};

  save_network_intermediate_outputs(model, image_path, yolo_out_dir_path);

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
