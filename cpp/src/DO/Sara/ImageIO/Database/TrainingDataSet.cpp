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

#include <DO/Sara/ImageIO/Database/TrainingDataSet.hpp>


using namespace std;


namespace DO { namespace Sara {

  void ImageClassificationTrainingDataSet::read_from_csv(
      const std::string& csv_filepath)
  {
    ifstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    auto csv_row = string{};
    auto x = string{};
    auto y = int{};
    auto sep = char{};

    while (getline(csv_file, csv_row))
    {
      istringstream csv_row_stream{csv_row};

      csv_row_stream >> x >> sep >> y;
      _x.push_back(x);
      _y.push_back(y);
    }
  }

  void ImageClassificationTrainingDataSet::write_to_csv(
      const std::string& csv_filepath) const
  {
    ofstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    auto s = begin();
    auto s_end = end();

    for (; s != s_end; ++s)
      csv_file << s.x().path() << ";" << s.y_ref() << "\n";
  }


} /* namespace Sara */
} /* namespace DO */
