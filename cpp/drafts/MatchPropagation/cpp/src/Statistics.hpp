// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
/*!
 *  This file implements a part of the method published in:
 *
 *  Efficient and Scalable 4th-order Match Propagation
 *  David Ok, Renaud Marlet, and Jean-Yves Audibert.
 *  ACCV 2012, Daejeon, South Korea.
 */

#pragma once

#include <DO/Sara/Defines.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

namespace DO::Sara {

  //! @addtogroup MatchPropagation
  //! @{

  class Statistics
  {
  public:
    double min, max, median, mean, sigma;
    size_t size;

    template <typename T>
    void compute_statistics(const std::vector<T>& numbers)
    {
      if (!numbers.empty())
      {
        std::vector<T> sorted_numbers(numbers);
        sort(sorted_numbers.begin(), sorted_numbers.end());

        size = numbers.size();
        median = sorted_numbers[sorted_numbers.size() / 2];
        this->min = sorted_numbers.front();
        this->max = sorted_numbers.back();
        double sum = 0.;
        for (size_t i = 0; i != sorted_numbers.size(); ++i)
          sum += sorted_numbers[i];
        mean = sum / double(sorted_numbers.size());
        sigma = 0.;
        for (size_t i = 0; i != sorted_numbers.size(); ++i)
          sigma += sorted_numbers[i] * sorted_numbers[i] - mean * mean;
        sigma /= double(sorted_numbers.size());
        sigma = sqrt(sigma);
      }
      else
      {
        min = max = median = mean = sigma = 0.;
        size = 0;
      }
    }

    DO_SARA_EXPORT
    friend std::ostream& operator<<(std::ostream& os, const Statistics& s);
  };

  DO_SARA_EXPORT
  void write_statistics(std::ofstream& out, const std::vector<Statistics>& stats);

  //! @}

} /* namespace DO::Sara */
