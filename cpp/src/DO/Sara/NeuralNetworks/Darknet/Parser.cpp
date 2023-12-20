// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/NeuralNetworks/Darknet/Parser.hpp>

#include <boost/algorithm/string.hpp>

#include <fstream>


namespace DO::Sara::Darknet {

  auto NetworkParser::read_line(std::ifstream& file, std::string& line) const
      -> bool
  {
    if (!std::getline(file, line))
      return false;
    line = boost::algorithm::trim_copy(line);
    return true;
  }

  auto NetworkParser::is_section(const std::string& line) const -> bool
  {
    return line.front() == '[';
  }

  auto NetworkParser::is_comment(const std::string& line) const -> bool
  {
    return line.front() == '#';
  }

  auto NetworkParser::section_name(const std::string& line) const -> std::string
  {
    auto line_trimmed = line;
    boost::algorithm::trim_if(
        line_trimmed, [](const auto ch) { return ch == '[' || ch == ']'; });
    return line_trimmed;
  }

  auto NetworkParser::make_new_layer(
      const std::string& layer_type,
      std::vector<std::unique_ptr<Layer>>& nodes) const -> void
  {
    std::cout << "MAKING NEW LAYER: " << layer_type << std::endl;

    if (layer_type == "net")
      nodes.emplace_back(new Input);
    else if (layer_type == "convolutional")
      nodes.emplace_back(new Convolution);
    else if (layer_type == "route")
      nodes.emplace_back(new Route);
    else if (layer_type == "shortcut")
      nodes.emplace_back(new Shortcut);
    else if (layer_type == "maxpool")
      nodes.emplace_back(new MaxPool);
    else if (layer_type == "upsample")
      nodes.emplace_back(new Upsample);
    else if (layer_type == "yolo")
      nodes.emplace_back(new Yolo);
    else
      throw std::runtime_error{"The \"" + layer_type +
                               "\" layer is not implemented!"};

    nodes.back()->type = layer_type;
  }

  auto NetworkParser::finish_layer_init(
      std::vector<std::unique_ptr<Layer>>& nodes) const -> void
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
    else if (layer_type == "yolo")
      dynamic_cast<Yolo&>(*nodes.back()).update_output_sizes(nodes);
    else if (layer_type == "shortcut")
      dynamic_cast<Shortcut&>(*nodes.back()).update_output_sizes(nodes);

    std::cout << "CHECKING CURRENT LAYER: " << std::endl;
    std::cout << *nodes.back() << std::endl;
  }

  auto NetworkParser::parse_config_file(const std::string& cfg_filepath) const
      -> std::vector<std::unique_ptr<Layer>>
  {
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

  NetworkWeightLoader::NetworkWeightLoader(const std::string& filepath,
                                           const bool debug)
    : debug{debug}
  {
    fp = fopen(filepath.c_str(), "rb");
    if (fp == nullptr)
      throw std::runtime_error{"Failed to open file: " + filepath};

    [[maybe_unused]] auto num_bytes_read = size_t{};

    num_bytes_read += fread(&major, sizeof(int), 1, fp);
    num_bytes_read += fread(&minor, sizeof(int), 1, fp);
    num_bytes_read += fread(&revision, sizeof(int), 1, fp);
    SARA_DEBUG << "version = " << major << "." << minor << "." << revision
               << std::endl;

    if ((major * 10 + minor) >= 2)
    {
      if (debug)
        SARA_DEBUG << "seen = 64\n";
      uint64_t iseen = 0;
      num_bytes_read += fread(&iseen, sizeof(uint64_t), 1, fp);
      seen = iseen;
    }
    else
    {
      if (debug)
        printf("\n seen 32");
      uint32_t iseen = 0;
      num_bytes_read += fread(&iseen, sizeof(uint32_t), 1, fp);
      seen = iseen;
    }
    if (debug)
      SARA_DEBUG << format(
          "Trained: %.0f K-images (%.0f K-batch of 64 images)\n",
          static_cast<float>(seen) / 1000, static_cast<float>(seen) / 64000);
    transpose = (major > 1000) || (minor > 1000);

    // std::cout << "Num bytes read = " << num_bytes_read << std::endl;
  }

  NetworkWeightLoader::~NetworkWeightLoader()
  {
    if (fp)
    {
      fclose(fp);
      fp = nullptr;
    }
  }

  auto NetworkWeightLoader::load(std::vector<std::unique_ptr<Layer>>& net)
      -> void
  {
    auto i = 0;
    for (auto& layer : net)
    {
      if (auto d = dynamic_cast<Convolution*>(layer.get()))
      {
        if (debug)
          std::cout << "LOADING WEIGHTS FOR CONVOLUTIONAL LAYER:\n"
                    << "[" << i << "]\n"
                    << *layer << std::endl;
        d->load_weights(fp);
        ++i;
      }
    }
  }


  auto load_yolo_model(const std::filesystem::path& model_dir_path,
                       const int version, const bool is_tiny) -> Network
  {
    auto yolo_name = "yolov" + std::to_string(version);
    if (is_tiny)
      yolo_name += "-tiny";
    const auto cfg_filepath = model_dir_path / (yolo_name + ".cfg");
    const auto weights_filepath = model_dir_path / (yolo_name + ".weights");

    auto model = Network{};
    auto& net = model.net;
    net = NetworkParser{}.parse_config_file(cfg_filepath.string());

    auto network_weight_loader =
        NetworkWeightLoader{weights_filepath.string(), true};
    network_weight_loader.load(net);

    return model;
  }

}  // namespace DO::Sara::Darknet
