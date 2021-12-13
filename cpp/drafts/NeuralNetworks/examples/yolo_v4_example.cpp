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

#include <iomanip>


namespace sara = DO::Sara;
namespace fs = boost::filesystem;


auto read_all_intermediate_outputs(const std::string& dir_path)
{
  auto stringify = [](int n) {
    std::ostringstream ss;
    ss << std::setw(3) << std::setfill('0') << n;
    return ss.str();
  };

  auto outputs = std::vector<sara::Tensor_<float, 4>>(38);
  for (auto i = 0u; i < outputs.size(); ++i)
  {
    const auto filepath = fs::path{dir_path} / (stringify(i) + ".bin");
    std::cout << "Parsing " << filepath << std::endl;
    outputs[i] = sara::Darknet::read_tensor(filepath.string());
  }

  return outputs;
}


auto visualize(const sara::TensorView_<float, 4>& y,  //
               const Eigen::Vector2i& sizes)
{
  for (auto i = 0; i < y.size(1); ++i)
  {
    const auto y_i = y[0][i];
    const auto im_i = sara::image_view(y_i);
    const auto im_i_rescaled = sara::resize(sara::color_rescale(im_i), sizes);
    sara::display(im_i_rescaled);
    sara::get_key();
  }
}

auto visualize2(const sara::TensorView_<float, 4>& y1,
                const sara::TensorView_<float, 4>& y2,  //
                const Eigen::Vector2i& sizes)
{
  auto reformat = [&sizes](const auto& y) {
    const auto y_i = y;
    const auto im_i = sara::image_view(y_i);
    const auto im_i_rescaled = sara::resize(sara::color_rescale(im_i), sizes);
    return im_i_rescaled;
  };

  for (auto i = 0; i < y1.size(1); ++i)
  {
    // Calculate on the actual tensor.
    auto diff = sara::Tensor_<float, 2>{y1[0][i].sizes()};
    diff.matrix() = y1[0][i].matrix() - y2[0][i].matrix();

    const auto residual = diff.matrix().norm();
    const auto minDiff = diff.matrix().cwiseAbs().minCoeff();
    const auto maxDiff = diff.matrix().cwiseAbs().maxCoeff();

    if (false)
    {
      std::cout << "residual " << i << " = " << residual << std::endl;
      std::cout << "min residual value " << i << " = " << minDiff << std::endl;
      std::cout << "max residual value " << i << " = " << maxDiff << std::endl;

      std::cout << "GT\n" << y1[0][i].matrix().block(0, 0, 5, 5) << std::endl;
      std::cout << "ME\n" << y2[0][i].matrix().block(0, 0, 5, 5) << std::endl;
    }

    if (maxDiff > 6e-5f)
    {
      // Resize and color rescale the data to show it nicely.
      const auto im1 = reformat(y1[0][i]);
      const auto im2 = reformat(y2[0][i]);
      const auto imdiff = reformat(diff);

      sara::display(im1);
      sara::display(im2, {im1.width(), 0});
      sara::display(imdiff, {2 * im1.width(), 0});

      sara::get_key();
      throw std::runtime_error{"FISHY COMPUTATION ERROR!"};
    }
  }
}


int __main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
  const auto data_dir_path =
      fs::canonical(fs::path{src_path("../../../../data")});
  const auto cfg_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.cfg";
  const auto weights_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.weights";
  const auto yolov4_tiny_out_dir = "/home/david/GitHub/darknet/yolov4-tiny";

  auto model = sara::Darknet::Network{};
  auto& net = model.net;
  net = sara::Darknet::NetworkParser{}.parse_config_file(cfg_filepath.string());
  sara::Darknet::NetworkWeightLoader{weights_filepath.string()}.load(net);
  // Check the weights.
  model.check_convolutional_weights(yolov4_tiny_out_dir);


#ifdef USE_SARA_IO
  const auto image =
      argc < 2
          ? sara::imread<sara::Rgb32f>((data_dir_path / "dog.jpg").string())
          : sara::imread<sara::Rgb32f>(argv[1]);
  sara::create_window(416 * 3, 416);
  sara::display(image);

  // Resize the image to the network input sizes.
  // TODO: optimize later.
  const auto& input_layer =
      dynamic_cast<const sara::Darknet::Input&>(*net.front());
  const auto image_resized =
      sara::resize(image, {input_layer.width, input_layer.height});
  const auto x_sizes = Eigen::Vector2i{
      image_resized.width(), image_resized.height()  //
  };

  sara::display(image_resized);


  // Feed the input to the network.
  model.debug = false;
  model.forward(sara::tensor_view(image_resized)
                    .reshape(Eigen::Vector4i{1, image_resized.height(),
                                             image_resized.width(), 3})
                    .transpose({0, 3, 1, 2}));

#else
  const auto x = sara::Darknet::read_tensor(                  //
      (fs::path{yolov4_tiny_out_dir} / "input.bin").string()  //
  );
  const auto xt = x.transpose({0, 2, 3, 1});
  const auto x_image = sara::ImageView<sara::Rgb32f>{
      reinterpret_cast<sara::Rgb32f*>(const_cast<float*>(xt.data())),
      {xt.size(2), xt.size(1)}};
  const auto x_sizes = x_image.sizes();

  sara::create_window(416 * 3, 416);
  sara::display(x_image);
  sara::get_key();

  model.debug = true;
  model.forward(x);
#endif

  // Load all the intermediate outputs calculated from Darknet.
  const auto gt = read_all_intermediate_outputs(yolov4_tiny_out_dir);

  // TODO: implement the YOLO layer.
  for (auto layer = 1u; layer < net.size(); ++layer)
  {
    std::cout << "CHECKING LAYER " << layer << ": "
              << net[layer]->type << std::endl
              << *net[layer] << std::endl;
    visualize2(gt[layer - 1], net[layer]->output, x_sizes);
  }

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
