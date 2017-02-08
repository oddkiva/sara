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

  void read_from_csv(TransformedImageClassificationTrainingDataSet& data_set,
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
      split(csv_row, ';', back_inserter(csv_cells));

      data_set._x.push_back(csv_cells[0]);
      data_set._y.push_back(stoi(csv_cells[1]));

      auto t = ImageDataTransform{};
      t.set_zoom(std::stof(csv_cells[2]));
      t.theta = std::stof(csv_cells[3]);
      t.set_shift(Vector2i{std::stoi(csv_cells[4]), std::stoi(csv_cells[5])});
      t.set_flip(csv_cells[6] == "H" ? ImageDataTransform::Horizontal
                                     : ImageDataTransform::None);
      t.set_fancy_pca(Vector3f{stof(csv_cells[7]), stof(csv_cells[8]), stof(csv_cells[9])});

      data_set._t.push_back(t);
    }
  }

  void write_to_csv(
      const TransformedImageClassificationTrainingDataSet& data_set,
      const std::string& csv_filepath)
  {
    ofstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    auto s = data_set.begin();
    auto s_end = data_set.end();

    for (; s != s_end; ++s)
      csv_file << s.x().path() << ";" << s.y_ref() << ";" << s.t_ref().z << ";"
               << s.t_ref().theta << ";" << s.t_ref().t.x() << ";"
               << s.t_ref().t.y() << ";" << s.t_ref().flip_type << ";"
               << s.t_ref().alpha.x() << ";" << s.t_ref().alpha.y() << ";"
               << s.t_ref().alpha.z() << "\n";
  }


} /* namespace Sara */
} /* namespace DO */
