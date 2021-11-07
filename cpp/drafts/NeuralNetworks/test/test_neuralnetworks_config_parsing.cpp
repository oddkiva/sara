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
#include <boost/test/unit_test.hpp>

#include <filesystem>
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


template <typename Type>
class Layer
{
  Type type;
};

struct Convolution
{
  int batch_normalize = 1;
  int filters;
  int size;
  int stride;
  int pad;
  int activation;
};

struct Route
{
  std::vector<int> layer_indices;
};


auto parse_option(const std::string& line)
{
  auto line_split = std::vector<std::string>{};
  boost::split(line_split, line, boost::is_any_of("= "),
               boost::token_compress_on);
  return line_split;
}


BOOST_AUTO_TEST_SUITE(TestLayers)

BOOST_AUTO_TEST_CASE(test_yolov4_tiny_config_parsing)
{
  namespace fs = std::filesystem;

  const auto filepath =                            //
      fs::path{"/home/david/GitLab/DO-CV/sara"} /  //
      "data/trained_models/" /                     //
      "yolov4-tiny.cfg";
  BOOST_CHECK(fs::exists(filepath));

  auto file = std::ifstream{filepath.string()};
  BOOST_CHECK(file.is_open());

  auto line = std::string{};

  auto in_current_section = false;
  auto enter_new_section = false;

  auto nodes = std::vector<std::string>{};
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
      const auto section = section_name(line);
      std::cout << "ENTERING NEW SECTION: ";
      std::cout << section << std::endl;

      if (section != "net")
        nodes.emplace_back(section);

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
    {
      const auto line_split = parse_option(line);
      for (const auto& str : line_split)
        std::cout << str << ";";
      std::cout << std::endl;
    }
  }

  // Parse all the nodes as a first pass.
  for (auto i = 0u; i < nodes.size(); ++i)
    std::cout << i << " " << nodes[i] << std::endl;

  // Parse the links as a second pass.
}

BOOST_AUTO_TEST_SUITE_END()
