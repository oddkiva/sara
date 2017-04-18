// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#ifndef DO_STAT_STAT_HPP
#define DO_STAT_STAT_HPP

#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

namespace DO {

  class Stat {
  public:
    double min, max, median, mean, sigma;
    size_t size;
    template <typename T>
    void computeStats(const std::vector<T>& numbers)
    {
      if (!numbers.empty())
      {
        std::vector<T> sorted_numbers(numbers);
        sort(sorted_numbers.begin(), sorted_numbers.end());

        size = numbers.size();
        median = sorted_numbers[sorted_numbers.size()/2];
        this->min = sorted_numbers.front();
        this->max = sorted_numbers.back();
        double sum = 0.;
        for (size_t i = 0; i != sorted_numbers.size(); ++i)
          sum += sorted_numbers[i];
        mean = sum/double(sorted_numbers.size());
        sigma = 0.;
        for (size_t i = 0; i != sorted_numbers.size(); ++i)
          sigma += sorted_numbers[i]*sorted_numbers[i] - mean*mean;
        sigma /= double(sorted_numbers.size());
        sigma = sqrt(sigma);
      }
      else
      {
        min = max = median = mean = sigma = 0.;
        size = 0;
      }
    }
    friend std::ostream& operator<<(std::ostream& os, const Stat& s);
  };
  void writeStats(std::ofstream& out, const std::vector<Stat>& stats);

} /* namespace DO */

#endif /* DO_STAT_STAT_HPP */