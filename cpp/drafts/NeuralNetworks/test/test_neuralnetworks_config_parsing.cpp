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

#include <DO/Sara/Defines.hpp>

#include <Eigen/Core>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <iostream>
#include <string>


auto read_line(std::ifstream& file, std::string& line)
{
  if (!std::getline(file, line))
    return false;
  line = boost::algorithm::trim_copy(line);
  return true;
}

auto is_section(const std::string& line)
{
  return line.front() == '[';
}

auto is_comment(const std::string& line)
{
  return line.front() == '#';
}

auto section_name(const std::string& line)
{
  auto line_trimmed = line;
  boost::algorithm::trim_if(
      line_trimmed, [](const auto ch) { return ch == '[' or ch == ']'; });
  return line_trimmed;
}


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
    for (auto& str: line_split)
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
    for (auto& str: line_split)
      boost::trim(str);

    std::cout << "[parsing] ";
    std::copy(line_split.begin(), line_split.end() - 1,
              std::ostream_iterator<std::string>{std::cout, " = "});
    std::cout << *(line_split.end() - 1) << std::endl;

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
};

struct Route : Layer
{
  std::vector<std::int32_t> layers;
  int groups = 1;
  int group_id = -1;

  auto update_output_sizes(const std::vector<std::unique_ptr<Layer>>& nodes) -> void
  {
    // All layers must have the same width, height, and batch size.
    // Only the input channels vary.
    const auto id = layers.front() < 0 ? nodes.size() - 1 + layers.front() : layers.front();
    input_sizes = nodes[id]->output_sizes;
    output_sizes = nodes[id]->output_sizes;

    auto channels = 0;
    for (const auto& rel_id: layers)
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
    for (auto& str: line_split)
      boost::trim(str);

    const auto& key = line_split[0];
    if (key == "groups")
      groups = std::stoi(line_split[1]);
    if (key == "group_id")
      group_id = std::stoi(line_split[1]);
    if (key == "layers")
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
    for (const auto& layer: layers)
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
    for (auto& str: line_split)
      boost::trim(str);

    const auto& key = line_split[0];
    if (key == "size")
      size = std::stoi(line_split[1]);
    if (key == "stride")
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

struct Yolo : Layer
{
  auto parse_line(const std::string& line) -> void override
  {
    auto line_split = std::vector<std::string>{};
    boost::split(line_split, line, boost::is_any_of("="),
                 boost::token_compress_on);
    for (auto& str: line_split)
      boost::trim(str);

    std::cout << "YOLO: TODO" << std::endl;
  }

  auto to_output_stream(std::ostream& os) const -> void override
  {
    os << "- input          = " << input_sizes.transpose() << "\n";
    os << "- output         = " << output_sizes.transpose() << "\n";
  }
};


BOOST_AUTO_TEST_SUITE(TestLayers)

BOOST_AUTO_TEST_CASE(test_yolov4_tiny_config_parsing)
{
  namespace fs = boost::filesystem;

  const auto data_dir_path = fs::canonical(fs::path{ src_path("../../../../data") });
  const auto cfg_filepath = data_dir_path / "trained_models" / "yolov4-tiny.cfg";
  BOOST_CHECK(fs::exists(cfg_filepath));

  auto file = std::ifstream{cfg_filepath.string()};
  BOOST_CHECK(file.is_open());

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
      if (!section.empty())
      {
        std::cout << "FINISHING PARSED SECTION: " << section << std::endl;
        if (section != "net")
        {
          if (nodes.size() < 2)
            throw std::runtime_error{"Invalid network!"};
          const auto &previous_node = *(nodes.rbegin() + 1);
          nodes.back()->input_sizes = previous_node->output_sizes;
        }
        if (section == "net")
          dynamic_cast<Input&>(*nodes.back()).update_output_sizes();
        if (section == "convolutional")
          dynamic_cast<Convolution&>(*nodes.back()).update_output_sizes();
        if (section == "route")
          dynamic_cast<Route&>(*nodes.back()).update_output_sizes(nodes);
        if (section == "maxpool")
          dynamic_cast<MaxPool&>(*nodes.back()).update_output_sizes();

        std::cout << "CHECKING PARSED SECTION: " << std::endl;
        std::cout << *nodes.back() << std::endl;
      }

      section = section_name(line);
      std::cout << "ENTERING NEW SECTION: ";
      std::cout << section << std::endl;

      if (section == "net")
        nodes.emplace_back(new Input);
      else if (section == "convolutional")
        nodes.emplace_back(new Convolution);
      else if (section == "route")
        nodes.emplace_back(new Route);
      else if (section == "maxpool")
        nodes.emplace_back(new MaxPool);
      else if (section == "yolo")
        nodes.emplace_back(new Yolo);

      nodes.back()->type = section;

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

  // Parse the model weights in the second pass.
}

BOOST_AUTO_TEST_SUITE_END()
