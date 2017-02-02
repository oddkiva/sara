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


namespace DO { namespace Sara {

  template <typename X, typename Y>
  class TransformedTrainingSample : public TrainingSampleIterator<X, Y>
  {
  public:
    using x_type = typename TrainingSampleTraits<X, Y>::input_type;
    using y_type = typename TrainingSampleTraits<X, Y>::label_type;

    auto x() -> x_type
    {
      return x_type{};
    }

    auto y() -> y_type
    {
      return y_type{};
    }

    ImageDataTransform _transform;
  };


  void write_to_csv(
      const std::string& out_path,
      std::vector<TransformedTrainingSample>& transformed_training_samples)
  {
    ofstream out{out_path};

    if (!out)
      throw std::runtime_error{"Could not create CSV file"};

    for (const auto& s : transformed_training_samples)
    {
      out << s.x.path() << ";" << s.y.path() << ";"
          << int(s._use_original) << ";"
          << "[" << s._transform.out_sizes << "]" << ";"
          << s._transform.z << ";"
          << "[" << s._transform.t[0] << "," << s._transform.t[1] << "]" << ";"
          << s._transform.flip_type << ";"
          << s._transform.alpha << "\n";
    }
  }

  std::vector<TransformedTrainingSample>
  read_from_csv(const std::string& csv_path)
  {
    ifstream in{csv_path};
  }


} /* namespace Sara */
} /* namespace DO */
