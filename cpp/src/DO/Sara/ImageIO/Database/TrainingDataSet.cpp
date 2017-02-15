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

#include <fstream>

#include <DO/Sara/ImageIO/Database/TrainingDataSet.hpp>


using namespace std;


namespace DO { namespace Sara {

  void read_from_csv(ImageClassificationTrainingDataSet& data_set,
                     const std::string& csv_filepath)
  {
    ifstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    data_set.clear();

    auto csv_row = string{};
    auto csv_cells = vector<string>(2);

    while (getline(csv_file, csv_row))
    {
      details::split(csv_row, ',', csv_cells.begin());
      data_set._x.push_back(csv_cells[0]);
      data_set._y.push_back(stoi(csv_cells[1]));
    }
  }

  void write_to_csv(const ImageClassificationTrainingDataSet& data_set,
                    const std::string& csv_filepath)
  {
    ofstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    auto s = data_set.begin();
    auto s_end = data_set.end();

    for (; s != s_end; ++s)
      csv_file << s.x().path() << "," << s.y_ref() << "\n";
  }

  void read_from_csv(ImageSegmentationTrainingDataSet& data_set,
                     const std::string& csv_filepath)
  {
    ifstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    data_set.clear();

    auto csv_row = string{};
    auto csv_cells = vector<string>(2);

    while (getline(csv_file, csv_row))
    {
      details::split(csv_row, ',', csv_cells.begin());
      data_set._x.push_back(csv_cells[0]);
      data_set._y.push_back(csv_cells[1]);
    }
  }

  void write_to_csv(const ImageSegmentationTrainingDataSet& data_set,
                    const std::string& csv_filepath)
  {
    ofstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    auto s = data_set.begin();
    auto s_end = data_set.end();

    for (; s != s_end; ++s)
      csv_file << s.x().path() << "," << s.y().path() << "\n";
  }

} /* namespace Sara */
} /* namespace DO */
