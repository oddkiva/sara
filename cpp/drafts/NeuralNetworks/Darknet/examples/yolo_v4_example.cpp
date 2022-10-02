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


namespace sara = DO::Sara;
namespace d = DO::Sara::Darknet;
namespace fs = boost::filesystem;


// The API.
auto detect_objects(const sara::ImageView<sara::Rgb8>& image,
                    sara::Darknet::Network& model)
{
  auto& net = model.net;
  const auto& input_layer =
      dynamic_cast<const sara::Darknet::Input&>(*net.front());

  sara::tic();
  const auto image_transposed = sara::tensor_view(image).transpose({2, 0, 1});
  static_assert(std::is_same_v<decltype(image_transposed),
                               const sara::Tensor_<std::uint8_t, 3>>);
  sara::toc("Image transpose");

  sara::tic();
  auto rgb_tensor = image_transposed.cwise_transform(
      [](const std::uint8_t& v) { return v / 255.f; });
  sara::toc("Image channel conversion");

  sara::tic();
  auto rgb_tensor_resized = sara::Tensor_<float, 4>{
      {1, 3, input_layer.height(), input_layer.width()}};
  for (auto i = 0; i < 3; ++i)
  {
    const auto src = sara::image_view(rgb_tensor[i]);
    auto dst = sara::image_view(rgb_tensor_resized[0][i]);
    sara::resize_v2(src, dst);
  }
  sara::toc("Image resize");

  // Feed the input to the network.
  model.forward(rgb_tensor_resized);

  // Accumulate all the detection from each YOLO layer.
  auto detections = std::vector<d::YoloBox>{};
  for (const auto& layer : net)
  {
    if (const auto yolo = dynamic_cast<const sara::Darknet::Yolo*>(layer.get()))
    {
      std::cout << *yolo << std::endl;
      const auto dets = d::get_yolo_boxes(                           //
          yolo->output[0],                                           //
          yolo->anchors, yolo->mask,                                 //
          {rgb_tensor_resized.size(3), rgb_tensor_resized.size(2)},  //
          image.sizes(),                                             //
          0.25f);
      detections.insert(detections.end(), dets.begin(), dets.end());
    }
  }

  return d::nms(detections);
}


auto test_on_image(int argc, char** argv) -> void
{
#ifndef __APPLE__
  const auto num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
  Eigen::setNbThreads(num_threads);
#endif

  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  const auto yolov4_tiny_dirpath =
      data_dir_path / "trained_models" / "yolov7-tiny";
  const auto image =
      argc < 2 ? sara::imread<sara::Rgb8>((data_dir_path / "dog.jpg").string())
               : sara::imread<sara::Rgb8>(argv[1]);
  sara::create_window(image.sizes());
  sara::display(image);

  auto model = sara::Darknet::load_yolo_model(yolov4_tiny_dirpath, 7, true);

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

#ifdef _WIN32
  const auto video_filepath = sara::select_video_file_from_dialog_box();
  if (video_filepath.empty())
    return;
#else
  if (argc < 2)
  {
    std::cerr << "Missing video path" << std::endl;
    return;
  }
  const auto video_filepath = argv[1];
#endif
  auto video_stream = sara::VideoStream{video_filepath};
  auto frame = video_stream.frame();

  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  const auto yolo_version = 4;
  const auto is_tiny = false;
  auto yolo_name = "yolov" + std::to_string(yolo_version);
  if (is_tiny)
    yolo_name += "-tiny";
  const auto yolo_dirpath = data_dir_path / "trained_models" / yolo_name;
  auto model = d::load_yolo_model(yolo_dirpath, yolo_version, is_tiny);

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
    auto dets = detect_objects(video_stream.frame(), model);
    sara::toc("Yolo");

    sara::display(frame);
    for (const auto& det : dets)
    {
      static constexpr auto int_round = [](const float v) {
        return static_cast<int>(std::round(v));
      };
      sara::draw_rect(int_round(det.box(0)), int_round(det.box(1)),
                      int_round(det.box(2)), int_round(det.box(3)),  //
                      sara::Green8, 2);
    }
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
