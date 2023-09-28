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

//! @file

#pragma once

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>


namespace DO::Sara {

  /*!
   *  @ingroup ImageProcessing
   *  @defgroup Extrema Extremum Localization
   *  @{
   */

  //! @brief Generic neighborhood comparison functor.
  template <template <typename> class Compare, typename T>
  struct CompareWithNeighborhood3
  {
    bool operator()(T val, int x, int y, const ImageView<T>& I,
                    bool compareWithCenter) const
    {
      for (int v = -1; v <= 1; ++v)
      {
        for (int u = -1; u <= 1; ++u)
        {
          if (u == 0 && v == 0 && !compareWithCenter)
            continue;
          if (!_compare(val, I(x + u, y + v)))
            return false;
        }
      }
      return true;
    }

    Compare<T> _compare;
  };

  //! @brief Local spatial extremum test.
  template <template <typename> class Compare, typename T>
  struct LocalExtremum
  {
    inline bool operator()(int x, int y, const ImageView<T>& I) const
    {
      return _compare(I(x, y), x, y, I, false);
    }

    CompareWithNeighborhood3<Compare, T> _compare;
  };

  //! @brief Local scale-space extremum test.
  template <template <typename> class Compare, typename T>
  struct LocalScaleSpaceExtremum
  {
    inline bool operator()(int x, int y, int s, int o,
                           const ImagePyramid<T, 2>& I) const
    {
      return _compare(I(x, y, s, o), x, y, I(s - 1, o), true) &&
             _compare(I(x, y, s, o), x, y, I(s, o), false) &&
             _compare(I(x, y, s, o), x, y, I(s + 1, o), true);
    }

    CompareWithNeighborhood3<Compare, T> _compare;
  };

  //! @brief Get local spatial extrema.
  template <template <typename> class Compare, typename T>
  std::vector<Point2i> local_extrema(const ImageView<T>& I)
  {
    LocalExtremum<Compare, T> local_extremum;
    auto extrema = std::vector<Point2i>{};
    for (int y = 1; y < I.height() - 1; ++y)
      for (int x = 1; x < I.width() - 1; ++x)
        if (local_extremum(x, y, I))
          extrema.push_back(Point2i(x, y));
    return extrema;
  }

  //! @brief Get local scale-space extrema at scale \f$\sigma(s,o)\f$
  template <template <typename> class Compare, typename T>
  std::vector<Point2i> local_scale_space_extrema(const ImagePyramid<T>& I,
                                                 int s, int o)
  {
    LocalScaleSpaceExtremum<Compare, T> local_extremum;
    auto extrema = std::vector<Point2i>{};
    for (int y = 1; y < I(s, o).height() - 1; ++y)
      for (int x = 1; x < I(s, o).width() - 1; ++x)
        if (local_extremum(x, y, s, o, I))
          extrema.push_back(Point2i(x, y));
    return extrema;
  }

  //! @brief Local spatial maximum test.
  template <typename T>
  struct LocalMax : LocalExtremum<std::greater_equal, T>
  {
  };

  //! @brief Local spatial minimum test.
  template <typename T>
  struct LocalMin : LocalExtremum<std::less_equal, T>
  {
  };

  //! @brief Local scale-space maximum test.
  template <typename T>
  struct LocalScaleSpaceMax : LocalScaleSpaceExtremum<std::greater_equal, T>
  {
  };

  //! @brief Local scale-space minimum test.
  template <typename T>
  struct LocalScaleSpaceMin : LocalScaleSpaceExtremum<std::less_equal, T>
  {
  };

  //! @brief Strict local spatial maximum test.
  template <typename T>
  struct StrictLocalMax : LocalExtremum<std::greater, T>
  {
  };

  //! @brief Strict local spatial minimum test.
  template <typename T>
  struct StrictLocalMin : LocalExtremum<std::less, T>
  {
  };

  //! @brief Strict local scale-space maximum test.
  template <typename T>
  struct StrictLocalScaleSpaceMax : LocalScaleSpaceExtremum<std::greater, T>
  {
  };

  //! @brief Strict local scale-space minimum test.
  template <typename T>
  struct StrictLocalScaleSpaceMin : LocalScaleSpaceExtremum<std::less, T>
  {
  };

  //! @brief Get local spatial maxima.
  template <typename T>
  inline std::vector<Point2i> local_maxima(const ImageView<T>& I)
  {
    return local_extrema<std::greater_equal, T>(I);
  }

  //! @brief Get local spatial minima.
  template <typename T>
  inline std::vector<Point2i> local_minima(const ImageView<T>& I)
  {
    return local_extrema<std::less_equal, T>(I);
  }

  //! @brief Get strict local spatial maxima.
  template <typename T>
  inline std::vector<Point2i> strict_local_maxima(const ImageView<T>& I)
  {
    return local_extrema<std::greater, T>(I);
  }

  //! @brief Get strict local spatial minima.
  template <typename T>
  inline std::vector<Point2i> strict_local_minima(const ImageView<T>& I)
  {
    return local_extrema<std::less, T>(I);
  }

  //! @brief Get local scale space maxima.
  template <typename T>
  inline std::vector<Point2i> local_scale_space_maxima(const ImagePyramid<T>& I,
                                                       int s, int o)
  {
    return local_scale_space_extrema<std::greater_equal, T>(I, s, o);
  }

  //! @brief Get local scale space minima.
  template <typename T>
  inline std::vector<Point2i> local_scale_space_minima(const ImagePyramid<T>& I,
                                                       int s, int o)
  {
    return local_scale_space_extrema<std::less_equal, T>(I, s, o);
  }

  //! @brief Get strict local scale space maxima.
  template <typename T>
  inline std::vector<Point2i>
  strict_local_scale_space_maxima(const ImagePyramid<T>& I, int s, int o)
  {
    return local_scale_space_extrema<std::greater, T>(I, s, o);
  }

  //! @brief Get strict local scale space minima.
  template <typename T>
  inline std::vector<Point2i>
  strict_local_scale_space_minima(const ImagePyramid<T>& I, int s, int o)
  {
    return local_scale_space_extrema<std::less, T>(I, s, o);
  }

  //! @}

}  // namespace DO::Sara
