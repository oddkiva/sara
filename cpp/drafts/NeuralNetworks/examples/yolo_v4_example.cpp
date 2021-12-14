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

#include <omp.h>


namespace sara = DO::Sara;
namespace fs = boost::filesystem;


struct Detection
{
  Eigen::Vector4f box;
  float objectness_prob;
  Eigen::VectorXf class_probs;
};

auto get_yolo_boxes(const sara::TensorView_<float, 3>& output,
                    const std::vector<int>& box_sizes_prior,
                    const std::vector<int>& masks,
                    const Eigen::Vector2i& network_input_sizes,
                    const Eigen::Vector2i& original_sizes,
                    float objectness_threshold)
{
  const Eigen::Vector2f scale = original_sizes.cast<float>().array() /
                                network_input_sizes.cast<float>().array();

  auto boxes = std::vector<Detection>{};
  for (auto box = 0; box < 3; ++box)
  {
    // Box center
    const auto rel_x = output[box * 85 + 0];
    const auto rel_y = output[box * 85 + 1];
    // Box log sizes
    const auto log_w = output[box * 85 + 2];
    const auto log_h = output[box * 85 + 3];
    // Objectness probability.
    const auto objectness = output[box * 85 + 4];

    // Fetch the box size prior.
    const auto& w_prior = box_sizes_prior[2 * masks[box] + 0];
    const auto& h_prior = box_sizes_prior[2 * masks[box] + 1];

    for (auto i = 0; i < output.size(1); ++i)
      for (auto j = 0; j < output.size(2); ++j)
      {
        if (objectness(i, j) < objectness_threshold)
          continue;

        // This is the center of the box and not the top-left corner of the box.
        auto xy = Eigen::Vector2f{
            (j + rel_x(i, j)) / output.size(2),  //
            (i + rel_y(i, j)) / output.size(1)   //
        };
        xy.array() *= original_sizes.cast<float>().array();

        // Exponentiate and rescale to get the box sizes.
        auto wh = Eigen::Vector2f{log_w(i, j), log_h(i, j)};
        wh.array() = wh.array().exp();
        wh(0) *= w_prior * scale.x();
        wh(1) *= h_prior * scale.y();

        // The final box.
        const auto xywh =
            (Eigen::Vector4f{} << (xy - wh * 0.5f), wh).finished();

        // The probabilities.
        const auto obj_prob = objectness(i, j);
        auto class_probs = Eigen::VectorXf{80};
        for (auto c = 0; c < 80; ++c)
          class_probs(c) = output[box * 85 + 5 + c](i, j);

        boxes.push_back({xywh, obj_prob, class_probs});
      }
  }

  return boxes;
}


int __main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
  const auto num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
  Eigen::setNbThreads(num_threads);

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

  const auto image =
      argc < 2
          ? sara::imread<sara::Rgb32f>((data_dir_path / "dog.jpg").string())
          : sara::imread<sara::Rgb32f>(argv[1]);
  sara::create_window(image.sizes());
  sara::display(image);

  // Resize the image to the network input sizes.
  // TODO: optimize later.
  const auto& input_layer =
      dynamic_cast<const sara::Darknet::Input&>(*net.front());
  const auto image_resized =
      sara::resize(image, {input_layer.width, input_layer.height});
  const auto image_tensor =
      sara::tensor_view(image_resized)
          .reshape(Eigen::Vector4i{1, image_resized.height(),
                                   image_resized.width(), 3})
          .transpose({0, 3, 1, 2});

  // Feed the input to the network.
  // TODO: optimize this method to avoid recopying again or better, eliminate
  // the input layer.
  auto timer = sara::Timer{};
  timer.restart();
  model.forward(image_tensor);
  const auto elapsed = timer.elapsed_ms();
  std::cout << "Total = " << elapsed << " ms" << std::endl;


  for (const auto& layer : net)
  {
    if (const auto yolo = dynamic_cast<const sara::Darknet::Yolo*>(layer.get()))
    {
      const auto dets =
          get_yolo_boxes(yolo->output[0], yolo->anchors, yolo->mask,
                         image_resized.sizes(), image.sizes(), 0.25f);
      for (const auto& det : dets)
        sara::draw_rect(det.box(0), det.box(1), det.box(2), det.box(3),
                        sara::Red8, 2);
    }
  }
  sara::get_key();

  return 0;
}


int main(int argc, char** argv)
{
  Eigen::initParallel();

  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
