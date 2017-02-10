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
#include <DO/Sara/ImageIO/Database/TransformedTrainingDataSet.hpp>


using namespace std;


namespace DO { namespace Sara {

  void parse_image_data_transform(ImageDataTransform& t, const vector<string>& csv_cells)
  {
    t.out_sizes = Vector2i{stoi(csv_cells[2]), stoi(csv_cells[3])};

    const auto apply_zoom = csv_cells[4] == "Y";
    if (apply_zoom)
    {
      const auto z = stof(csv_cells[5]);
      t.set_zoom(z);
    }

    const auto apply_shift = csv_cells[6] == "Y";
    if (apply_shift)
    {
      const auto shift = Vector2i{stoi(csv_cells[7]), stoi(csv_cells[8])};
      t.set_shift(shift);
    }

    const auto apply_flip = csv_cells[9] == "Y";
    if (apply_flip)
    {
      const auto flip_type = csv_cells[10] == "H"
                                 ? ImageDataTransform::Horizontal
                                 : ImageDataTransform::None;
      t.set_flip(flip_type);
    }

    const auto apply_fancy_pca = csv_cells[11] == "Y";
    if (apply_fancy_pca)
    {
      const auto alpha =
        Vector3f{stof(csv_cells[12]), stof(csv_cells[13]), stof(csv_cells[14])};
      t.set_fancy_pca(alpha);
    }
  }

  auto stringify_flip(ImageDataTransform::FlipType f) -> char
  {
    switch (f)
    {
    case ImageDataTransform::Horizontal:
      return 'H';
    case ImageDataTransform::Vertical:
      return 'V';
    case ImageDataTransform::None:
    default:
      return 'N';
    }
  }

  auto operator<<(ostream& os, const ImageDataTransform& t) -> ostream&
  {
    const auto apply_zoom =
      t.apply_transform[ImageDataTransform::Zoom] ? 'Y' : 'N';
    const auto apply_shift =
      t.apply_transform[ImageDataTransform::Shift] ? 'Y' : 'N';
    const auto apply_flip =
      t.apply_transform[ImageDataTransform::Flip] ? 'Y' : 'N';
    const auto apply_fancy_pca =
      t.apply_transform[ImageDataTransform::FancyPCA] ? 'Y' : 'N';

    os << t.out_sizes.x() << ',' << t.out_sizes.y() << ','
       << apply_zoom << ',' << t.z << ','
       << apply_shift << ',' << t.t.x() << ',' << t.t.y() << ','
       << apply_flip << ',' << stringify_flip(t.flip_type) << ','
       << apply_fancy_pca << ',' << t.alpha.x() << ',' << t.alpha.y() << ','
       << t.alpha.z();

    return os;
  }

  void read_from_csv(TransformedImageClassificationTrainingDataSet& data_set,
                     const std::string& csv_filepath)
  {
    ifstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    auto csv_row = string{};
    auto csv_cells = vector<string>(20);

    while (getline(csv_file, csv_row))
    {
      details::split(csv_row, ',', csv_cells.begin());

      data_set._x.push_back(csv_cells[0]);
      data_set._y.push_back(stoi(csv_cells[1]));

      auto t = ImageDataTransform{};
      parse_image_data_transform(t, csv_cells);
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
      csv_file << s.x().path() << ',' << s.y_ref() << ',' << s.t_ref() << "\n";
  }

  void read_from_csv(TransformedImageSegmentationTrainingDataSet & data_set,
                     const std::string& csv_filepath)
  {
    ifstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    auto csv_row = string{};
    auto csv_cells = vector<string>(20);

    while (getline(csv_file, csv_row))
    {
      details::split(csv_row, ',', csv_cells.begin());

      data_set._x.push_back(csv_cells[0]);
      data_set._y.push_back(csv_cells[1]);

      auto t = ImageDataTransform{};
      parse_image_data_transform(t, csv_cells);
      data_set._t.push_back(t);
    }
  }

  void write_to_csv(
      const TransformedImageSegmentationTrainingDataSet& data_set,
      const std::string& csv_filepath)
  {
    ofstream csv_file{csv_filepath};
    if (!csv_file)
      throw std::runtime_error{
          string{"Cannot open CSV file: " + csv_filepath}.c_str()};

    auto s = data_set.begin();
    auto s_end = data_set.end();

    for (; s != s_end; ++s)
      csv_file << s.x().path() << ',' << s.y().path() << ',' << s.t_ref() << '\n';
  }


} /* namespace Sara */
} /* namespace DO */
