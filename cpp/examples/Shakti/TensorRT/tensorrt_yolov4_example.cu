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

#include <DO/Shakti/Cuda/MultiArray/ManagedMemoryAllocator.hpp>
#include <DO/Shakti/Cuda/TensorRT/DarknetParser.hpp>
#include <DO/Shakti/Cuda/TensorRT/IO.hpp>
#include <DO/Shakti/Cuda/TensorRT/InferenceEngine.hpp>
#include <DO/Shakti/Cuda/TensorRT/Yolo.hpp>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>
#include <DO/Sara/NeuralNetworks/Darknet/Parser.hpp>
#include <DO/Sara/NeuralNetworks/Darknet/YoloUtilities.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <tinycolormap.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <filesystem>


namespace sara = DO::Sara;
namespace darknet = sara::Darknet;
namespace fs = std::filesystem;
namespace trt = DO::Shakti::TensorRT;
namespace d = sara::Darknet;

using CudaManagedTensor3ub =
    trt::InferenceEngine::ManagedTensor<std::uint8_t, 3>;
using CudaManagedTensor3f = trt::InferenceEngine::ManagedTensor<float, 3>;


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

auto naive_downsample_and_transpose(CudaManagedTensor3f& tensor_chw_resized_32f,
                                    const CudaManagedTensor3ub& tensor_hwc_8u)
    -> void
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


class Yolo
{
public:
  Yolo() = default;

  Yolo(const fs::path& yolo_dir_path, const int yolo_version,
       const bool is_tiny)
  {
    auto yolo_model = "yolov" + std::to_string(yolo_version);
    if (is_tiny)
      yolo_model += "-tiny";

    const auto yolo_cfg_path = yolo_dir_path / (yolo_model + ".cfg");
    const auto yolo_plan_path = yolo_dir_path / (yolo_model + ".plan");

    read_plan(yolo_dir_path, yolo_plan_path, is_tiny);
    parse_yolo_params(yolo_cfg_path);
    read_class_names(yolo_dir_path);
  }

  auto detect(const CudaManagedTensor3ub& input_image_hwc_rgb_8u,
              const float iou_thres,  //
              const Eigen::Vector2i& image_sizes) -> std::vector<d::YoloBox>
  {
    sara::tic();
    naive_downsample_and_transpose(_cuda_in_tensor, input_image_hwc_rgb_8u);
    sara::toc("CUDA downsample+transpose");

    // Feed the input and outputs to the YOLO v4 tiny network.
    sara::tic();
    static constexpr auto synchronize = true;
    _trt_engine(_cuda_in_tensor, _cuda_out_tensors, synchronize);
    sara::toc("Inference time");

    // Accumulate all the detection from each YOLO layer.
    sara::tic();
    auto detections = std::vector<d::YoloBox>{};
    const auto wr = _cuda_in_tensor.sizes()(2);
    const auto hr = _cuda_in_tensor.sizes()(1);
    for (auto i = 0u; i < _yolo_mask_sets.size(); ++i)
    {
      const auto& yolo_out = _cuda_out_tensors[i];
      const auto& yolo_masks = _yolo_mask_sets[i];
      const auto dets = d::get_yolo_boxes(yolo_out,                   //
                                          _yolo_anchors, yolo_masks,  //
                                          {wr, hr}, image_sizes, 0.25f);
      detections.insert(detections.end(), dets.begin(), dets.end());
    }
    sara::toc("Postprocess boxes");

    sara::tic();
    detections = d::nms(detections, iou_thres);
    sara::toc("NMS");

    return detections;
  }

  auto classes() const noexcept -> const std::vector<std::string>&
  {
    return _yolo_classes;
  }

private:
  auto read_plan(const fs::path& yolo_dir_path, const fs::path& yolo_plan_path,
                 const bool is_tiny) -> void
  {
    // Load the network and get the CUDA inference engine ready.
    if (!fs::exists(yolo_plan_path))
    {
      // Create the optimized network and serialize it.
      const auto serialized_net = trt::convert_yolo_v4_network_from_darknet(
          yolo_dir_path.string(), is_tiny);
      _trt_engine = trt::InferenceEngine{serialized_net};
      // Save it for later to avoid recalculating the optimized network.
      trt::write_plan(serialized_net, yolo_plan_path.string());
      return;
    }

    _trt_engine.load_from_plan_file(yolo_plan_path.string());
  }

  auto read_class_names(const fs::path& yolo_dir_path) -> void
  {
    auto yolo_classes_file =
        std::ifstream{(yolo_dir_path / "classes.txt").string()};
    if (!yolo_classes_file)
      throw std::runtime_error{"Cannot read YOLO class file!"};

    auto line = std::string{};
    while (std::getline(yolo_classes_file, line))
      _yolo_classes.emplace_back(std::move(line));
  }

  auto parse_yolo_params(const fs::path& yolo_cfg_path) -> void
  {
    _yolo_mask_sets.clear();
    _yolo_anchors.clear();

    auto model = darknet::Network{};
    auto& net = model.net;
    net = darknet::NetworkParser{}.parse_config_file(yolo_cfg_path.string());

    const auto& h_input = net[0]->output_sizes(2);
    const auto& w_input = net[0]->output_sizes(3);
    _cuda_in_tensor = CudaManagedTensor3f{{3, h_input, w_input}};

    for (const auto& layer : net)
    {
      if (const auto yolo =
              dynamic_cast<const sara::Darknet::Yolo*>(layer.get()))
      {
        if (_yolo_anchors.empty())
          _yolo_anchors = yolo->anchors;

        _yolo_mask_sets.emplace_back(yolo->mask);

        const auto& h = yolo->output_sizes(2);
        const auto& w = yolo->output_sizes(3);
        _cuda_out_tensors.emplace_back(Eigen::Vector3i{255, h, w});
      }
    }
  }

private:
  trt::InferenceEngine _trt_engine;

  CudaManagedTensor3f _cuda_in_tensor;
  std::vector<trt::InferenceEngine::PinnedTensor<float, 3>> _cuda_out_tensors;

  std::vector<std::vector<int>> _yolo_mask_sets;
  std::vector<int> _yolo_anchors;
  std::vector<std::string> _yolo_classes;
};


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
  const auto is_tiny = argc < 5 ? false : static_cast<bool>(std::stoi(argv[4]));
  SARA_CHECK(skip);
  SARA_CHECK(iou_thres);

  auto video_stream = sara::VideoStream{video_filepath};
  auto frame = video_stream.frame();

  // Instantiate the YOLO v4 object detector.
  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  static constexpr auto yolo_version = 4;
  auto yolo_model = "yolov" + std::to_string(yolo_version);
  if (is_tiny)
    yolo_model += "-tiny";
  const auto yolo_dir_path = data_dir_path / "trained_models" / yolo_model;
  auto yolo = Yolo{yolo_dir_path, yolo_version, is_tiny};

  auto tensor_hwc_8u = CudaManagedTensor3ub{frame.height(), frame.width(), 3};
  auto tensor_hwc_32f = CudaManagedTensor3f{frame.height(), frame.width(), 3};

  // Assign a list of colors for each class.
  static constexpr auto color_map = tinycolormap::ColormapType::Turbo;
  auto class_colors = std::vector<sara::Rgb8>(yolo.classes().size());
  for (auto x = 0u; x < class_colors.size(); ++x)
  {
    const auto color = tinycolormap::GetColor(
        static_cast<double>(x) / class_colors.size(), color_map);
    class_colors[x] << color.ri(), color.gi(), color.bi();
    if (x % 3 == 0)
      class_colors[x].array() = 255 - class_colors[x].array();
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
    std::copy_n(reinterpret_cast<const std::uint8_t*>(frame.data()),
                sizeof(sara::Rgb8) * frame.size(),  //
                tensor_hwc_8u.begin());
    sara::toc("Copy frame data from host to CUDA");

    sara::tic();
    const auto dets = yolo.detect(tensor_hwc_8u, iou_thres, frame.sizes());
    sara::toc("Object detection");

    sara::tic();
    for (const auto& det : dets)
    {
      static constexpr auto int_round = [](const float v) {
        return static_cast<int>(std::round(v));
      };

      auto label_index = int{};
      det.class_probs.maxCoeff(&label_index);

      const auto x = int_round(det.box(0));
      const auto y = int_round(det.box(1));
      const auto w = int_round(det.box(2));
      const auto h = int_round(det.box(3));

      sara::draw_rect(frame, x, y, w, h, class_colors[label_index], 5);

      const auto& class_name = yolo.classes()[label_index];
      const auto class_score = int_round(det.class_probs[label_index] * 100);

      const auto& label = fmt::format("{} {}%", class_name, class_score);
      sara::draw_text(frame, x, y - 3, label, sara::White8, 16, 0.f, false,
                      true, false);
    }
    sara::toc("Draw detections");

    sara::display(frame);
    static auto pause = false;
    auto ev = sara::Event{};
    sara::get_event(5, ev);
    if (ev.key == ' ')
      pause = true;
    if (ev.key == sara::KEY_ESCAPE)
      break;
    if (pause)
    {
      const auto k = sara::get_key();
      if (k == ' ')
        pause = false;
      else if (k == sara::KEY_ESCAPE)
        break;
    }
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
