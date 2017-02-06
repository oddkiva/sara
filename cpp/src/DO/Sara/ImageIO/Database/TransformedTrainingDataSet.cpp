// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
// Copyright (C) 2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <tuple>
#include <fstream>
#include <sstream>

#include <DO/Sara/ImageIO/Database/TransformedTrainingDataSet.hpp>


using namespace std;


namespace DO { namespace Sara {

  template <typename Out>
  void split(const std::string& s, char delim, Out result)
  {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
      *(result++) = item;
    }
  }

  void TransformedImageClassificationTrainingDataSet::read_from_csv(
      const std::string& csv_filepath)
  {
    ifstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    auto csv_row = string{};
    auto csv_cells = vector<string>{};

    while (getline(csv_file, csv_row))
    {
      split(csv_row, ";", back_insert(csv_cells));

      _x.push_back(csv_cells[0]);
      _y.push_back(stoi(csv_cells[1]));

      auto z = std::stof(csv_cells[2]);
      auto theta = std::stof(csv_cells[3]);
      auto tx = std::stoi(csv_cells[4]);
      auto ty = std::stoi(csv_cells[5]);
      auto flip_type = csv_cells[6];
      auto alpha =
          Vector3f{stof(csv_cells[7]), stof(csv_cells[8]), stof(csv_cells[9])};
    }
  }

  void TransformedImageClassificationTrainingDataSet::write_to_csv(
      const std::string& csv_filepath) const
  {
    ofstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    auto s = begin();
    auto s_end = end();

    for (; s != s_end; ++s)
      csv_file << s.x().path() << ";" << s.y_ref() << ";" << s.t_ref().z << ";"
               << s.t_ref().theta << ";" << s.t_ref().t.x() << ";"
               << s.t_ref().t.y() << ";" << s.t_ref().flip_type << ";"
               << s.t_ref().alpha.x() << ";" << s.t_ref().alpha.y() << ";"
               << s.t_ref().alpha.z() << "\n";
  }


} /* namespace Sara */
} /* namespace DO */
