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

#include <drafts/NeuralNetworks/Darknet/YoloUtilities.hpp>
#include <drafts/NeuralNetworks/TensorRT/DarknetParser.hpp>
#include <drafts/NeuralNetworks/TensorRT/IO.hpp>
#include <drafts/NeuralNetworks/TensorRT/InferenceEngine.hpp>
#include <drafts/NeuralNetworks/TensorRT/Yolo.hpp>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <boost/filesystem.hpp>

#include <omp.h>


namespace sara = DO::Sara;
namespace s = sara;
namespace fs = boost::filesystem;
namespace trt = sara::TensorRT;
namespace d = sara::Darknet;


// The API.
auto detect_objects(
    const sara::ImageView<sara::Rgb8>& image,
    const trt::InferenceEngine& inference_engine,
    trt::InferenceEngine::PinnedTensor<float, 3>& cuda_in_tensor,
    std::vector<trt::InferenceEngine::PinnedTensor<float, 3>>& cuda_out_tensors,
    const float iou_thres,  //
    const std::vector<std::vector<int>>& anchor_masks,
    const std::vector<int>& anchors) -> std::vector<d::YoloBox>
{
  // N.B.: this would still be unacceptably slow and a GPU implementation is
  // still preferrable.
  // The CPU implementation takes between 5 and 7 ms on a powerful CPU...
  sara::tic();
  auto rgb_tensor = sara::Tensor_<float, 3>{3, image.height(), image.width()};
  const auto rgb = image.data();
  auto r = rgb_tensor[0].data();
  auto g = rgb_tensor[1].data();
  auto b = rgb_tensor[2].data();
  const auto size = static_cast<int>(image.size());
  if (image.size() != rgb_tensor[0].size())
    throw 0;
#pragma omp parallel for
  for (auto i = 0; i < size; ++i)
  {
    r[i] = rgb[i].channel<s::R>() / 255.f;
    g[i] = rgb[i].channel<s::G>() / 255.f;
    b[i] = rgb[i].channel<s::B>() / 255.f;
  }
  sara::toc("Uint8 Interleaved to Float Planar");

  sara::tic();
  auto rgb_tensor_resized = sara::TensorView_<float, 3>{cuda_in_tensor.data(),
                                                        cuda_in_tensor.sizes()};
  for (auto i = 0; i < 3; ++i)
  {
    const auto src = sara::image_view(rgb_tensor[i]);
    auto dst = sara::image_view(rgb_tensor_resized[i]);
    sara::resize_v2(src, dst);
  }
  sara::toc("Image resize");

  // Feed the input and outputs to the YOLO v4 tiny network.
  sara::tic();
  inference_engine(cuda_in_tensor, cuda_out_tensors, true);
  sara::toc("Inference time");

  // Accumulate all the detection from each YOLO layer.
  sara::tic();
  auto detections = std::vector<d::YoloBox>{};
  for (auto i = 0; i < 2; ++i)
  {
    const auto& yolo_out = cuda_out_tensors[i];
    const auto& anchor_mask = anchor_masks[i];
    const auto dets = d::get_yolo_boxes(                   //
        yolo_out,                                          //
        anchors, anchor_mask,                              //
        {cuda_in_tensor.size(2), cuda_in_tensor.size(1)},  //
        image.sizes(),                                     //
        0.25f);
    detections.insert(detections.end(), dets.begin(), dets.end());
  }
  sara::toc("Postprocess boxes");

  sara::tic();
  detections = d::nms(detections, iou_thres);
  sara::toc("NMS");

  return detections;
}


auto test_on_video(int argc, char** argv) -> void
{
  omp_set_num_threads(omp_get_max_threads());
  SARA_CHECK(omp_get_max_threads());

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

  const auto skip = argc < 3 ? 0 : std::stoi(argv[2]);
  const auto iou_thres = argc < 4 ? 0.4f : std::stof(argv[3]);
  SARA_CHECK(skip);
  SARA_CHECK(iou_thres);

  auto video_stream = sara::VideoStream{video_filepath};
  auto frame = video_stream.frame();

  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  static constexpr auto yolo_version = 4;
  static constexpr auto is_tiny = false;
  auto yolo_model = "yolov" + std::to_string(yolo_version);
  if (is_tiny)
    yolo_model += "-tiny";
  const auto yolo_dirpath = data_dir_path / "trained_models" / yolo_model;

  const auto yolo_plan_filepath = yolo_dirpath / (yolo_model + ".plan");

  // Load the network and get the CUDA inference engine ready.
  auto inference_engine = trt::InferenceEngine{};
  // if (fs::exists(yolo_plan_filepath))
  //   inference_engine.load_from_plan_file(yolo_plan_filepath.string());
  // else
  {
    const auto serialized_net = trt::convert_yolo_v4_network_from_darknet(
        yolo_dirpath.string(), is_tiny);
    inference_engine = trt::InferenceEngine{serialized_net};
    trt::write_plan(serialized_net, yolo_plan_filepath.string());
  }

  auto cuda_in_tensor = trt::InferenceEngine::PinnedTensor<float, 3>{};
  auto cuda_out_tensors =
      std::vector<trt::InferenceEngine::PinnedTensor<float, 3>>{};
  auto yolo_masks = std::vector<std::vector<int>>{};
  auto yolo_anchors = std::vector<int>{};

  if constexpr (is_tiny)
  {
    // The CUDA tensors.
    cuda_in_tensor = trt::InferenceEngine::PinnedTensor<float, 3>{3, 416, 416};
    cuda_out_tensors = std::vector{
        trt::InferenceEngine::PinnedTensor<float, 3>{255, 13, 13},
        trt::InferenceEngine::PinnedTensor<float, 3>{255, 26, 26}  //
    };

    yolo_masks = std::vector{
        std::vector{3, 4, 5},  //
        std::vector{1, 2, 3}   //
    };
    yolo_anchors = std::vector{
        10,  14,   //
        23,  27,   //
        37,  58,   //
        81,  82,   //
        135, 169,  //
        344, 319   //
    };
  }
  else
  {
    // The CUDA tensors.
    cuda_in_tensor = trt::InferenceEngine::PinnedTensor<float, 3>{3, 608, 608};
    cuda_out_tensors = std::vector{
        trt::InferenceEngine::PinnedTensor<float, 3>{255, 76, 76},
        trt::InferenceEngine::PinnedTensor<float, 3>{255, 38, 38},  //
        trt::InferenceEngine::PinnedTensor<float, 3>{255, 19, 19}   //
    };

    const auto yolo_masks = std::vector{
        std::vector{0, 1, 2},  //
        std::vector{3, 4, 5},  //
        std::vector{6, 7, 8},  //
    };
    const auto yolo_anchors = std::vector{
        12,  16,   //
        19,  36,   //
        40,  28,   //
        36,  75,   //
        76,  55,   //
        72,  146,  //
        142, 110,  //
        192, 243,  //
        459, 401   //
    };
  }

  sara::create_window(frame.sizes());
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
    auto dets = detect_objects(            //
        video_stream.frame(),              //
        inference_engine,                  //
        cuda_in_tensor, cuda_out_tensors,  //
        iou_thres, yolo_masks, yolo_anchors);
    sara::toc("Object detection");

    sara::tic();
    for (const auto& det : dets)
    {
      static constexpr auto int_round = [](const float v) {
        return static_cast<int>(std::round(v));
      };
      sara::draw_rect(frame,  //
                      int_round(det.box(0)), int_round(det.box(1)),
                      int_round(det.box(2)), int_round(det.box(3)),  //
                      sara::Green8, 2);
    }
    sara::toc("Draw detections");

    sara::display(frame);
  }
}


int graphics_main(int argc, char** argv)
{
  test_on_video(argc, argv);
  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(graphics_main);
  return app.exec();
}
