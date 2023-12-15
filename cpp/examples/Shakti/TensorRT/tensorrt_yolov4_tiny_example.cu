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

#include "DO/Sara/Core/Image/Image.hpp"
#include <DO/Shakti/Cuda/MultiArray/ManagedMemoryAllocator.hpp>
#include <DO/Shakti/Cuda/TensorRT/DarknetParser.hpp>
#include <DO/Shakti/Cuda/TensorRT/InferenceExecutor.hpp>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>
#include <DO/Sara/NeuralNetworks/Darknet/YoloUtilities.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <algorithm>
#include <filesystem>


namespace sara = DO::Sara;
namespace fs = std::filesystem;
namespace trt = DO::Shakti::TensorRT;
namespace d = sara::Darknet;

using CudaManagedTensor3ub =
    trt::InferenceExecutor::ManagedTensor<std::uint8_t, 3>;
using CudaManagedTensor3f = trt::InferenceExecutor::ManagedTensor<float, 3>;


__global__ auto naive_downsample_and_transpose(float* out_chw,
                                               const std::uint8_t* in_hwc,
                                               const int wout, const int hout,
                                               const int win, const int hin)
    -> void
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int yout = blockIdx.y * blockDim.y + threadIdx.y;
  const int xout = blockIdx.z * blockDim.z + threadIdx.z;

  if (xout >= wout || yout >= hout || c >= 3)
    return;

  const float sx = float(win) / float(wout);
  const float sy = float(hin) / float(hout);

  int xin = int(xout * sx + 0.5f);
  int yin = int(yout * sy + 0.5f);

  if (xin >= win)
    xin = win - 1;
  if (yin >= hin)
    yin = hin - 1;

  const int gi_out = c * hout * wout + yout * wout + xout;
  const int gi_in = yin * win * 3 + xin * 3 + c;

  static constexpr auto normalize_factor = 1 / 255.f;
  out_chw[gi_out] = static_cast<float>(in_hwc[gi_in]) * normalize_factor;
}


// The API.
auto detect_objects(
    const trt::InferenceExecutor& inference_engine,
    const trt::InferenceExecutor::PinnedTensor<float, 3>& cuda_in_tensor,
    std::array<trt::InferenceExecutor::PinnedTensor<float, 3>, 2>&
        cuda_out_tensors,
    const float iou_thres,  //
    const std::array<std::vector<int>, 2>& anchor_masks,
    const std::vector<int>& anchors, const Eigen::Vector2i& image_sizes)
    -> std::vector<d::YoloBox>
{
  // Feed the input and outputs to the YOLO v4 tiny network.
  sara::tic();
  inference_engine(cuda_in_tensor, cuda_out_tensors, true);
  sara::toc("Inference time");

  // Accumulate all the detection from each YOLO layer.
  sara::tic();
  auto detections = std::vector<d::YoloBox>{};
  const auto wr = cuda_in_tensor.sizes()(2);
  const auto hr = cuda_in_tensor.sizes()(1);
  for (auto i = 0; i < 2; ++i)
  {
    const auto& yolo_out = cuda_out_tensors[i];
    const auto& anchor_mask = anchor_masks[i];
    const auto dets = d::get_yolo_boxes(yolo_out,              //
                                        anchors, anchor_mask,  //
                                        {wr, hr}, image_sizes, 0.25f);
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

  auto tensor_hwc_8u = CudaManagedTensor3ub{frame.height(), frame.width(), 3};
  auto tensor_hwc_32f = CudaManagedTensor3f{frame.height(), frame.width(), 3};
  auto tensor_chw_resized_32f = CudaManagedTensor3f{{3, 416, 416}};

  auto& cuda_in_tensor = tensor_chw_resized_32f;
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
    std::copy_n(reinterpret_cast<const std::uint8_t*>(frame.data()),
                sizeof(sara::Rgb8) * frame.size(),  //
                tensor_hwc_8u.begin());
    sara::toc("Copy frame data from host to CUDA");

    sara::tic();
    {
      // Data order: H W C
      //             0 1 2
      const auto in_hwc = tensor_hwc_8u.data();
      const auto win = tensor_hwc_8u.sizes()(1);
      const auto hin = tensor_hwc_8u.sizes()(0);

      // Data order: C H W
      //             0 1 2
      auto out_chw = tensor_chw_resized_32f.data();
      const auto hout = tensor_chw_resized_32f.sizes()(1);
      const auto wout = tensor_chw_resized_32f.sizes()(2);

      const auto threads_per_block = dim3(4, 16, 16);
      const auto num_blocks = dim3(  //
          1,                         //
          (hout + threads_per_block.y - 1) / threads_per_block.y,
          (wout + threads_per_block.z - 1) / threads_per_block.z  //
      );

      naive_downsample_and_transpose<<<num_blocks, threads_per_block>>>(
          out_chw, in_hwc,  //
          wout, hout,       //
          win, hin          //
      );
    }
    sara::toc("CUDA downsample+transpose");

    sara::tic();
    auto dets = detect_objects(               //
        inference_executor,                   //
        cuda_in_tensor, cuda_out_tensors,     //
        iou_thres, yolo_masks, yolo_anchors,  //
        frame.sizes());
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


auto graphics_main(int argc, char** argv) -> int
{
  test_on_video(argc, argv);
  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(graphics_main);
  return app.exec();
}
