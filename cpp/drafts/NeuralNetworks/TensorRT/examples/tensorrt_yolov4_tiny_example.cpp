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
#include <drafts/NeuralNetworks/TensorRT/InferenceExecutor.hpp>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <boost/filesystem.hpp>


namespace sara = DO::Sara;
namespace fs = boost::filesystem;
namespace trt = sara::TensorRT;
namespace d = sara::Darknet;


// The API.
auto detect_objects(
    const sara::ImageView<sara::Rgb32f>& image,
    const trt::InferenceExecutor& inference_engine,
    trt::InferenceExecutor::PinnedTensor<float, 3>& cuda_in_tensor,
    std::array<trt::InferenceExecutor::PinnedTensor<float, 3>, 2>&
        cuda_out_tensors,
    const float iou_thres,  //
    const std::array<std::vector<int>, 2>& anchor_masks,
    const std::vector<int>& anchors) -> std::vector<d::YoloBox>
{
  // This is the bottleneck.
  sara::tic();
  const auto image_resized = sara::resize(image, {416, 416});
  sara::toc("Image resize");

  sara::tic();
  const auto image_tensor =
      sara::tensor_view(image_resized)
          .reshape(Eigen::Vector4i{1, image_resized.height(),
                                   image_resized.width(), 3})
          .transpose({0, 3, 1, 2});
  sara::toc("Tensor transpose");

  // Copy to the CUDA tensor.
  sara::tic();
  std::copy(image_tensor.begin(), image_tensor.end(), cuda_in_tensor.begin());
  sara::toc("Copy to CUDA tensor");

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
    const auto dets =
        d::get_yolo_boxes(yolo_out,              //
                          anchors, anchor_mask,  //
                          image_resized.sizes(), image.sizes(), 0.25f);
    detections.insert(detections.end(), dets.begin(), dets.end());
  }
  sara::toc("Postprocess boxes");

  sara::tic();
  detections = d::nms(detections, iou_thres);
  sara::toc("NMS");

  SARA_CHECK(iou_thres);

  return detections;
}


auto test_on_video(int argc, char** argv) -> void
{
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
  const auto yolov4_tiny_dirpath = data_dir_path / "trained_models";
  auto serialized_net = trt::convert_yolo_v4_tiny_network_from_darknet(
      yolov4_tiny_dirpath.string());

  // Load the network and get the CUDA inference engine ready.
  auto inference_executor = trt::InferenceExecutor{serialized_net};

  // The CUDA tensors.
  auto cuda_in_tensor =
      trt::InferenceExecutor::PinnedTensor<float, 3>{3, 416, 416};
  auto cuda_out_tensors = std::array{
      trt::InferenceExecutor::PinnedTensor<float, 3>{255, 13, 13},
      trt::InferenceExecutor::PinnedTensor<float, 3>{255, 26, 26}  //
  };

  const auto yolo_masks = std::array{
      std::vector{3, 4, 5},  //
      std::vector{1, 2, 3}   //
  };
  const auto yolo_anchors = std::vector{
      10,  14,   //
      23,  27,   //
      37,  58,   //
      81,  82,   //
      135, 169,  //
      344, 319   //
  };

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
    const auto frame32f = video_stream.frame().convert<sara::Rgb32f>();
    sara::toc("Color conversion");

    sara::tic();
    auto dets = detect_objects(            //
        frame32f,                          //
        inference_executor,                //
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
