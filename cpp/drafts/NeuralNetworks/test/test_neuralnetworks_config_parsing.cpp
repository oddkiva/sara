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

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <iostream>
#include <string>


namespace sara = DO::Sara;


struct Layer
{
  virtual ~Layer() = default;
  virtual auto parse_line(const std::string& line) -> void = 0;
  virtual auto to_output_stream(std::ostream& os) const -> void = 0;

  friend inline auto operator<<(std::ostream& os, const Layer& l)
      -> std::ostream&
  {
    l.to_output_stream(os);
    return os;
  }

  std::string type;
  Eigen::Vector4i input_sizes = Eigen::Vector4i::Constant(-1);
  Eigen::Vector4i output_sizes = Eigen::Vector4i::Constant(-1);
};

struct Input : Layer
{
  int width;
  int height;
  int batch;

  auto update_output_sizes() -> void
  {
    output_sizes << batch, 3, height, width;
  }

  auto parse_line(const std::string& line) -> void override
  {
    auto line_split = std::vector<std::string>{};
    boost::split(line_split, line, boost::is_any_of("="),
                 boost::token_compress_on);
    for (auto& str : line_split)
      boost::trim(str);

    const auto& key = line_split[0];
    if (key == "width")
      width = std::stoi(line_split[1]);
    else if (key == "height")
      height = std::stoi(line_split[1]);
    else if (key == "batch")
      batch = std::stoi(line_split[1]);
  }

  auto to_output_stream(std::ostream& os) const -> void override
  {
    os << "- input width  = " << width << "\n";
    os << "- input height = " << height << "\n";
    os << "- input batch  = " << batch << "\n";
  }
};

struct BatchNormalization : Layer
{
  std::vector<float> bias;
  std::vector<float> scale;
  std::vector<float> rolling_mean;
  std::vector<float> rolling_variance;
};

struct Convolution : Layer
{
  bool batch_normalize = true;

  int filters;
  int size;
  int stride;
  int pad;
  std::string activation;

  auto update_output_sizes() -> void
  {
    output_sizes = input_sizes;
    output_sizes[1] = filters;
    output_sizes.tail(2) /= stride;
  }

  auto parse_line(const std::string& line) -> void override
  {
    auto line_split = std::vector<std::string>{};
    boost::split(line_split, line, boost::is_any_of("="),
                 boost::token_compress_on);
    for (auto& str : line_split)
      boost::trim(str);

    const auto& key = line_split[0];
    if (key == "batch_normalize")
      batch_normalize = static_cast<bool>(std::stoi(line_split[1]));
    else if (key == "filters")
      filters = std::stoi(line_split[1]);
    else if (key == "size")
      size = std::stoi(line_split[1]);
    else if (key == "stride")
      stride = std::stoi(line_split[1]);
    else if (key == "pad")
      pad = std::stoi(line_split[1]);
    else if (key == "activation")
      activation = line_split[1];
  }

  auto to_output_stream(std::ostream& os) const -> void override
  {
    os << "- normalize      = " << batch_normalize << "\n";
    os << "- filters        = " << filters << "\n";
    os << "- size           = " << size << "\n";
    os << "- stride         = " << stride << "\n";
    os << "- pad            = " << pad << "\n";
    os << "- activation     = " << activation << "\n";
    os << "- input          = " << input_sizes.transpose() << "\n";
    os << "- output         = " << output_sizes.transpose() << "\n";
  }

  //  auto read(FILE* fp) -> void
  //  {
  //    // 1. Read bias weights.
  //    // 2. Read batch normalization weights.
  //    // 3. Read convolution weights.
  //
  //    // 1.
  //    const auto bias_weight_count =
  //        fread(bias.data(), sizeof(float), bias.size(), fp);
  //    if (bias_weight_count != bias.size())
  //      throw std::runtime_error{"Could not read bias weights!"};
  //
  //    // 2.
  //
  //    // 3.
  //    const auto kernel_weight_count =
  //        fread(kernel.data(), sizeof(float), kernel.size(), fp);
  //    if (kernel_weight_count != kernel.size())
  //      throw std::runtime_error{"Could not read kernel weights!"};
  //    // TODO: transpose the kernel.
  //  }
};

struct Route : Layer
{
  std::vector<std::int32_t> layers;
  int groups = 1;
  int group_id = -1;

  auto update_output_sizes(const std::vector<std::unique_ptr<Layer>>& nodes)
      -> void
  {
    // All layers must have the same width, height, and batch size.
    // Only the input channels vary.
    const auto id =
        layers.front() < 0 ? nodes.size() - 1 + layers.front() : layers.front();
    input_sizes = nodes[id]->output_sizes;
    output_sizes = nodes[id]->output_sizes;

    auto channels = 0;
    for (const auto& rel_id : layers)
    {
      const auto id = rel_id < 0 ? nodes.size() - 1 + rel_id : rel_id;
      channels += nodes[id]->output_sizes[1];
    }
    output_sizes[1] = channels;

    output_sizes[1] /= groups;
  }

  auto parse_line(const std::string& line) -> void override
  {
    auto line_split = std::vector<std::string>{};
    boost::split(line_split, line, boost::is_any_of("="),
                 boost::token_compress_on);
    for (auto& str : line_split)
      boost::trim(str);

    const auto& key = line_split[0];
    if (key == "groups")
      groups = std::stoi(line_split[1]);
    else if (key == "group_id")
      group_id = std::stoi(line_split[1]);
    else if (key == "layers")
    {
      auto layer_strings = std::vector<std::string>{};
      boost::split(layer_strings, line_split[1], boost::is_any_of(", "),
                   boost::token_compress_on);
      for (const auto& layer_str : layer_strings)
        layers.push_back(std::stoi(layer_str));
    }
  }

  auto to_output_stream(std::ostream& os) const -> void override
  {
    os << "- layers         = ";
    for (const auto& layer : layers)
      os << layer << ", ";
    os << "\n";

    os << "- groups         = " << groups << "\n";
    os << "- group_id       = " << group_id << "\n";
    os << "- input          = " << input_sizes.transpose() << "\n";
    os << "- output         = " << output_sizes.transpose() << "\n";
  }
};

struct MaxPool : Layer
{
  int size = 2;
  int stride = 2;

  auto update_output_sizes() -> void
  {
    output_sizes = input_sizes;
    output_sizes.tail(2) /= stride;
  }

  auto parse_line(const std::string& line) -> void override
  {
    auto line_split = std::vector<std::string>{};
    boost::split(line_split, line, boost::is_any_of("="),
                 boost::token_compress_on);
    for (auto& str : line_split)
      boost::trim(str);

    const auto& key = line_split[0];
    if (key == "size")
      size = std::stoi(line_split[1]);
    else if (key == "stride")
      stride = std::stoi(line_split[1]);
  }

  auto to_output_stream(std::ostream& os) const -> void override
  {
    os << "- size           = " << size << "\n";
    os << "- stride         = " << stride << "\n";
    os << "- input          = " << input_sizes.transpose() << "\n";
    os << "- output         = " << output_sizes.transpose() << "\n";
  }
};

struct Upsample : Layer
{
  int stride = 2;

  auto update_output_sizes() -> void
  {
    output_sizes = input_sizes;
    output_sizes.tail(2) *= stride;
  }

  auto parse_line(const std::string& line) -> void override
  {
    auto line_split = std::vector<std::string>{};
    boost::split(line_split, line, boost::is_any_of("="),
                 boost::token_compress_on);
    for (auto& str : line_split)
      boost::trim(str);

    const auto& key = line_split[0];
    if (key == "stride")
      stride = std::stoi(line_split[1]);
  }

  auto to_output_stream(std::ostream& os) const -> void override
  {
    os << "- stride         = " << stride << "\n";
    os << "- input          = " << input_sizes.transpose() << "\n";
    os << "- output         = " << output_sizes.transpose() << "\n";
  }
};

struct Yolo : Layer
{
  std::vector<std::int32_t> mask;
  std::vector<std::int32_t> anchors;
  std::int32_t classes;
  std::int32_t num;
  float jitter;
  float scale_x_y;
  float cls_normalizer;
  float iou_normalizer;
  std::string iou_loss;
  float ignore_thresh;
  float truth_thresh;
  int random;
  float resize;
  std::string nms_kind;
  float beta_nms;

  auto parse_line(const std::string& line) -> void override
  {
    auto line_split = std::vector<std::string>{};
    boost::split(line_split, line, boost::is_any_of("="),
                 boost::token_compress_on);
    for (auto& str : line_split)
      boost::trim(str);

    const auto& key = line_split[0];
    if (key == "mask")
    {
      auto mask_strings = std::vector<std::string>{};
      boost::split(mask_strings, line_split[1], boost::is_any_of(", "),
                   boost::token_compress_on);
      for (const auto& mask_str : mask_strings)
        mask.push_back(std::stoi(mask_str));
    }
    else if (key == "anchors")
    {
      auto anchors_strings = std::vector<std::string>{};
      boost::split(anchors_strings, line_split[1], boost::is_any_of(", "),
                   boost::token_compress_on);
      for (const auto& anchors_str : anchors_strings)
        anchors.push_back(std::stoi(anchors_str));
    }
    else if (key == "classes")
      classes = std::stoi(line_split[1]);
    else if (key == "num")
      num = std::stoi(line_split[1]);
    else if (key == "jitter")
      jitter = std::stof(line_split[1]);
    else if (key == "scale_x_y")
      scale_x_y = std::stof(line_split[1]);
    else if (key == "cls_normalizer")
      cls_normalizer = std::stof(line_split[1]);
    else if (key == "iou_normalizer")
      iou_normalizer = std::stof(line_split[1]);
    else if (key == "iou_loss")
      iou_loss = line_split[1];
    else if (key == "ignore_thresh")
      ignore_thresh = std::stof(line_split[1]);
    else if (key == "truth_thresh")
      truth_thresh = std::stof(line_split[1]);
    else if (key == "random")
      random = std::stoi(line_split[1]);
    else if (key == "resize")
      resize = std::stof(line_split[1]);
    else if (key == "nms_kind")
      nms_kind = line_split[1];
    else if (key == "beta_nms")
      beta_nms = std::stof(line_split[1]);
  }

  auto to_output_stream(std::ostream& os) const -> void override
  {
    os << "- mask           = ";
    std::copy(mask.begin(), mask.end(),
              std::ostream_iterator<int>(std::cout, ", "));
    os << "\n";

    os << "- anchors        = ";
    std::copy(anchors.begin(), anchors.end(),
              std::ostream_iterator<int>(std::cout, ", "));
    os << "\n";

    os << "- classes        = " << classes << "\n";
    os << "- num            = " << num << "\n";
    os << "- jitter         = " << jitter << "\n";
    os << "- scale_x_y      = " << scale_x_y << "\n";
    os << "- cls_normalizer = " << cls_normalizer << "\n";
    os << "- iou_normalizer = " << iou_normalizer << "\n";
    os << "- iou_loss       = " << iou_loss << "\n";
    os << "- ignore_thresh  = " << ignore_thresh << "\n";
    os << "- truth_thresh   = " << truth_thresh << "\n";
    os << "- random         = " << random << "\n";
    os << "- resize         = " << resize << "\n";
    os << "- nms_kind       = " << nms_kind << "\n";
    os << "- beta_nms       = " << beta_nms << "\n";

    os << "- input          = " << input_sizes.transpose() << "\n";
    os << "- output         = " << output_sizes.transpose() << "\n";
  }
};


struct DarknetNetworkParser
{
  inline auto read_line(std::ifstream& file, std::string& line) const
  {
    if (!std::getline(file, line))
      return false;
    line = boost::algorithm::trim_copy(line);
    return true;
  }

  inline auto is_section(const std::string& line) const
  {
    return line.front() == '[';
  }

  inline auto is_comment(const std::string& line) const
  {
    return line.front() == '#';
  }

  inline auto section_name(const std::string& line) const
  {
    auto line_trimmed = line;
    boost::algorithm::trim_if(
        line_trimmed, [](const auto ch) { return ch == '[' or ch == ']'; });
    return line_trimmed;
  }

  auto make_new_layer(const std::string& layer_type,
                      std::vector<std::unique_ptr<Layer>>& nodes) const
  {
    std::cout << "MAKING NEW LAYER: " << layer_type << std::endl;

    if (layer_type == "net")
      nodes.emplace_back(new Input);
    else if (layer_type == "convolutional")
      nodes.emplace_back(new Convolution);
    else if (layer_type == "route")
      nodes.emplace_back(new Route);
    else if (layer_type == "maxpool")
      nodes.emplace_back(new MaxPool);
    else if (layer_type == "upsample")
      nodes.emplace_back(new Upsample);
    else if (layer_type == "yolo")
      nodes.emplace_back(new Yolo);

    nodes.back()->type = layer_type;
  }

  auto finish_layer_init(std::vector<std::unique_ptr<Layer>>& nodes) const
  {
    const auto& layer_type = nodes.back()->type;
    if (layer_type != "net")
    {
      if (nodes.size() < 2)
        throw std::runtime_error{"Invalid network!"};
      const auto& previous_node = *(nodes.rbegin() + 1);
      nodes.back()->input_sizes = previous_node->output_sizes;
    }

    if (layer_type == "net")
      dynamic_cast<Input&>(*nodes.back()).update_output_sizes();
    else if (layer_type == "convolutional")
      dynamic_cast<Convolution&>(*nodes.back()).update_output_sizes();
    else if (layer_type == "route")
      dynamic_cast<Route&>(*nodes.back()).update_output_sizes(nodes);
    else if (layer_type == "maxpool")
      dynamic_cast<MaxPool&>(*nodes.back()).update_output_sizes();
    else if (layer_type == "upsample")
      dynamic_cast<Upsample&>(*nodes.back()).update_output_sizes();

    std::cout << "CHECKING CURRENT LAYER: " << std::endl;
    std::cout << *nodes.back() << std::endl;
  }

  auto parse_config_file(const std::string& cfg_filepath) const
  {
    namespace fs = boost::filesystem;

    const auto data_dir_path =
        fs::canonical(fs::path{src_path("../../../../data")});

    auto file = std::ifstream{cfg_filepath};

    auto line = std::string{};

    auto section = std::string{};
    auto in_current_section = false;
    auto enter_new_section = false;

    auto nodes = std::vector<std::unique_ptr<Layer>>{};

    while (read_line(file, line))
    {
      if (line.empty())
        continue;

      if (is_comment(line))
        continue;

      // Enter a new section.
      if (is_section(line))
      {
        // Finish initialization of the previous layer if there was one.
        if (!section.empty())
          finish_layer_init(nodes);

        // Create a new layer.
        section = section_name(line);
        make_new_layer(section, nodes);

        enter_new_section = true;
        in_current_section = false;
        continue;
      }

      if (enter_new_section)
      {
        in_current_section = true;
        enter_new_section = false;
      }

      if (in_current_section)
        nodes.back()->parse_line(line);
    }

    finish_layer_init(nodes);

    return nodes;
  }
};


BOOST_AUTO_TEST_SUITE(TestLayers)

BOOST_AUTO_TEST_CASE(test_yolov4_tiny_config_parsing)
{
  namespace fs = boost::filesystem;

  const auto data_dir_path =
      fs::canonical(fs::path{src_path("../../../../data")});
  const auto cfg_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.cfg";
  BOOST_CHECK(fs::exists(cfg_filepath));

  const auto net = DarknetNetworkParser{}  //
                       .parse_config_file(cfg_filepath.string());
}

BOOST_AUTO_TEST_SUITE_END()
