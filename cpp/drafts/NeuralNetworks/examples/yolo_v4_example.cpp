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
#include <DO/Sara/VideoIO.hpp>

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

auto load_yolov4_tiny_model(const fs::path& model_dir_path)
{
  const auto cfg_filepath = model_dir_path / "yolov4-tiny.cfg";
  const auto weights_filepath = model_dir_path / "yolov4-tiny.weights";

  auto model = sara::Darknet::Network{};
  auto& net = model.net;
  net = sara::Darknet::NetworkParser{}.parse_config_file(cfg_filepath.string());
  sara::Darknet::NetworkWeightLoader{weights_filepath.string()}.load(net);

  return model;
}

//! @brief Post-processing functions.
//! @{
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

// Simple greedy NMS based on area IoU criterion.
auto nms(const std::vector<Detection>& detections, float iou_threshold = 0.4f)
    -> std::vector<Detection>
{
  auto detections_sorted = detections;
  std::sort(detections_sorted.begin(), detections_sorted.end(),
            [](const auto& a, const auto& b) {
              return a.objectness_prob > b.objectness_prob;
            });

  auto detections_filtered = std::vector<Detection>{};
  detections_filtered.reserve(detections.size());

  for (const auto& d : detections_sorted)
  {
    if (detections_filtered.empty())
    {
      detections_filtered.push_back(d);
      continue;
    }

    auto boxes_kept = Eigen::MatrixXf(detections_filtered.size(), 4);
    for (auto i = 0u; i < detections_filtered.size(); ++i)
      boxes_kept.row(i) = detections_filtered[i].box.transpose();

    const auto x1 = boxes_kept.col(0);
    const auto y1 = boxes_kept.col(1);
    const auto w = boxes_kept.col(2);
    const auto h = boxes_kept.col(3);

    const auto x2 = x1 + w;
    const auto y2 = y1 + h;

    // Intersection.
    const auto inter_x1 = x1.array().max(d.box(0));
    const auto inter_y1 = y1.array().max(d.box(1));
    const auto inter_x2 = x2.array().min(d.box(0) + d.box(2));
    const auto inter_y2 = y2.array().min(d.box(1) + d.box(3));
    const auto intersect = (inter_x1 <= inter_x2) && (inter_y1 <= inter_y2);

    // Intersection areas
    const Eigen::ArrayXf inter_area =
        intersect.cast<float>() *
        ((inter_x2 - inter_x1) * (inter_y2 - inter_y1));

    // Union areas.
    const Eigen::ArrayXf union_area =
        w.array() * h.array() + d.box(2) * d.box(3) - inter_area;

    // IoU
    const Eigen::ArrayXf iou = inter_area / union_area;

    const auto valid = (iou < iou_threshold).all();
    if (valid)
      detections_filtered.push_back(d);
  }

  return detections_filtered;
}
//! @}

// The API.
auto detect_objects(const sara::ImageView<sara::Rgb32f>& image,
                    sara::Darknet::Network& model)
{
  auto& net = model.net;
  const auto& input_layer =
      dynamic_cast<const sara::Darknet::Input&>(*net.front());

  // Resize the image to the network input sizes.
  // TODO: optimize later.
  const auto image_resized =
      sara::resize(image, {input_layer.width(), input_layer.height()});
  const auto image_tensor =
      sara::tensor_view(image_resized)
          .reshape(Eigen::Vector4i{1, image_resized.height(),
                                   image_resized.width(), 3})
          .transpose({0, 3, 1, 2});

  // Feed the input to the network.
  // TODO: optimize this method to avoid recopying again or better, eliminate
  // the input layer.
  model.forward(image_tensor);

  // Accumulate all the detection from each YOLO layer.
  auto detections = std::vector<Detection>{};
  for (const auto& layer : net)
  {
    if (const auto yolo = dynamic_cast<const sara::Darknet::Yolo*>(layer.get()))
    {
      const auto dets =
          get_yolo_boxes(yolo->output[0], yolo->anchors, yolo->mask,
                         image_resized.sizes(), image.sizes(), 0.25f);
      detections.insert(detections.end(), dets.begin(), dets.end());
    }
  }

  return nms(detections);
}


auto test_on_image(int argc, char** argv) -> void
{
#ifndef __APPLE__
  const auto num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
  Eigen::setNbThreads(num_threads);
#endif

  const auto data_dir_path =
      fs::canonical(fs::path{src_path("../../../../data")});
  const auto yolov4_tiny_dirpath = data_dir_path / "trained_models";
  const auto image =
      argc < 2
          ? sara::imread<sara::Rgb32f>((data_dir_path / "dog.jpg").string())
          : sara::imread<sara::Rgb32f>(argv[1]);
  sara::create_window(image.sizes());
  sara::display(image);

  auto model = load_yolov4_tiny_model(yolov4_tiny_dirpath);

  sara::display(image);
  const auto dets = detect_objects(image, model);
  for (const auto& det : dets)
  {
    static constexpr auto int_round = [](const float v) {
      return static_cast<int>(std::round(v));
    };
    sara::draw_rect(int_round(det.box(0)), int_round(det.box(1)),
                    int_round(det.box(2)), int_round(det.box(3)),  //
                    sara::Green8, 2);
  }
  sara::get_key();
}

auto test_on_video(int argc, char** argv) -> void
{
#ifndef __APPLE__
  const auto num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
  Eigen::setNbThreads(num_threads);
#endif

  if (argc < 2)
  {
    std::cerr << "Missing video path" << std::endl;
    return;
  }

  const auto video_filepath = argv[1];
  auto video_stream = sara::VideoStream{video_filepath};
  auto frame = video_stream.frame();

  const auto data_dir_path =
      fs::canonical(fs::path{src_path("../../../../data")});
  const auto yolov4_tiny_dirpath = data_dir_path / "trained_models";
  auto model = load_yolov4_tiny_model(yolov4_tiny_dirpath);
  model.profile = false;

  sara::create_window(frame.sizes());

  const auto skip = argc < 3 ? 0 : std::stoi(argv[2]);
  auto frames_read = 0;
  while (true)
  {
    sara::tic();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    sara::toc("Video Decoding");

    ++frames_read;
    if (frames_read % (skip + 1) != 0)
      continue;

    sara::tic();
    const auto frame32f = video_stream.frame().convert<sara::Rgb32f>();
    sara::toc("Color conversion");

    sara::tic();
    auto dets = detect_objects(frame32f, model);
    sara::toc("Yolo");

    sara::display(frame);
    for (const auto& det : dets)
      sara::draw_rect(det.box(0), det.box(1), det.box(2), det.box(3),
                      sara::Green8, 4);
  }
}

int __main(int argc, char** argv)
{
  // test_on_image(argc, argv);
  test_on_video(argc, argv);
  return 0;
}


int main(int argc, char** argv)
{
#ifndef __APPLE__
  Eigen::initParallel();
#endif

  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
