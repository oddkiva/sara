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

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/Defines.hpp>

#include <fstream>
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
  std::string type;
};

struct Input : Layer
{
  int width;
  int height;
  int batch;

  auto parse_line(const std::string& line) -> void override
  {
    auto line_split = std::vector<std::string>{};
    boost::split(line_split, line, boost::is_any_of("= "),
                 boost::token_compress_on);

    const auto& key = line_split[0];
    if (key == "width")
      width = std::stoi(line_split[1]);
    else if (key == "height")
      height = std::stoi(line_split[1]);
    else if (key == "batch")
      batch = std::stoi(line_split[1]);
  }

  friend inline auto operator<<(std::ostream& os, const Input& c)
      -> std::ostream&
  {
    os << "- width  = " << c.width << "\n";
    os << "- height = " << c.height << "\n";
    os << "- batch  = " << c.batch << "\n";
    return os;
  }
};

struct Convolution : Layer
{
  int batch_normalize = 1;
  int filters;
  int size;
  int stride;
  int pad;
  std::string activation;

  auto parse_line(const std::string& line) -> void override
  {
    auto line_split = std::vector<std::string>{};
    boost::split(line_split, line, boost::is_any_of("= "),
                 boost::token_compress_on);

    const auto& key = line_split[0];
    if (key == "batch_normalize")
      batch_normalize = std::stoi(line_split[1]);
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

  friend inline auto operator<<(std::ostream& os, const Convolution& c)
      -> std::ostream&
  {
    os << "- normalize  = " << c.batch_normalize << "\n";
    os << "- filters    = " << c.filters << "\n";
    os << "- size       = " << c.size << "\n";
    os << "- stride     = " << c.stride << "\n";
    os << "- pad        = " << c.pad << "\n";
    os << "- activation = " << c.activation << "\n";
    return os;
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
  auto links = std::vector<std::pair<int, int>>{};

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
        std::cout << "FINISHED PARSING SECTION: " << section << std::endl;
        std::cout << "CHECKING PARSED SECTION: " << std::endl;

        if (section == "net")
          std::cout << dynamic_cast<const Input&>(*nodes.back())
                    << std::endl;
        if (section == "convolutional")
          std::cout << dynamic_cast<const Convolution&>(*nodes.back())
                    << std::endl;
      }

      section = section_name(line);
      std::cout << "ENTERING NEW SECTION: ";
      std::cout << section << std::endl;

      if (section == "net")
        nodes.emplace_back(new Input);
      else if (section == "convolutional")
        nodes.emplace_back(new Convolution);

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
