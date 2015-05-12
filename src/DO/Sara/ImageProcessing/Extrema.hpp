// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_IMAGEPROCESSING_EXTREMA_HPP
#define DO_IMAGEPROCESSING_EXTREMA_HPP


#include <DO/ImageProcessing/ImagePyramid.hpp>


namespace DO {

   /*!
    \ingroup ImageProcessing
    \defgroup Extrema Extremum Localization
    @{
   */

  template <template <typename> class Compare, typename T>
  struct CompareWithNeighborhood3
  {
    bool operator()(T val, int x, int y, const Image<T>& I,
                    bool compareWithCenter) const
    {
      for (int v=-1; v <= 1; ++v)
      {
        for (int u=-1; u <= 1; ++u)
        {
          if ( u==0 && v==0 && !compareWithCenter )
            continue;
          if ( !compare_(val,I(x+u,y+v)) )
            return false;
        }
      }
      return true;
    }
    Compare<T> compare_;
  };

  //! Local spatial extremum test.
  template <template <typename> class Compare, typename T>
  struct LocalExtremum
  {
    inline bool operator()(int x, int y, const Image<T>& I) const
    {
      return compare_(I(x,y), x, y, I, false);
    }
    CompareWithNeighborhood3<Compare, T> compare_;
  };

  //! Local scale-space extremum test.
  template <template <typename> class Compare, typename T>
  struct LocalScaleSpaceExtremum
  {
    inline bool operator()(int x, int y, int s, int o,
                           const ImagePyramid<T, 2>& I) const
    {
      return compare_(I(x,y,s,o), x, y, I(s-1,o), true ) &&
             compare_(I(x,y,s,o), x, y, I(s  ,o), false) && 
             compare_(I(x,y,s,o), x, y, I(s+1,o), true );
    }
    CompareWithNeighborhood3<Compare, T> compare_;
  };

  //! Get local spatial extrema.
  template <template <typename> class Compare, typename T>
  std::vector<Point2i> local_extrema(const Image<T>& I)
  {
    LocalExtremum<Compare, T> local_extremum;
    std::vector<Point2i> extrema;
    for (int y = 1; y < I.height()-1; ++y)
      for (int x = 1; x < I.width()-1; ++x)
        if (local_extremum(x,y,I))
          extrema.push_back(Point2i(x,y));
    return extrema;
  }

  //! Get local scale-space extrema at scale \f$\sigma(s,o)\f$
  template <template <typename> class Compare, typename T>
  std::vector<Point2i> local_scale_space_extrema(const ImagePyramid<T>& I,
                                                 int s, int o)
  {
    LocalScaleSpaceExtremum<Compare, T> local_extremum;
    std::vector<Point2i> extrema;
    for (int y = 1; y < I(s,o).height()-1; ++y)
      for (int x = 1; x < I(s,o).width()-1; ++x)
        if (local_extremum(x,y,s,o,I))
          extrema.push_back(Point2i(x,y));
    return extrema;
  }

  //! Local spatial maximum test.
  template <typename T>
  struct LocalMax : LocalExtremum<std::greater_equal, T> {};

  //! Local spatial minimum test.
  template <typename T>
  struct LocalMin : LocalExtremum<std::less_equal, T> {};

  //! Local scale-space maximum test.
  template <typename T>
  struct LocalScaleSpaceMax : LocalScaleSpaceExtremum<std::greater_equal, T> {};

  //! Local scale-space minimum test.
  template <typename T>
  struct LocalScaleSpaceMin : LocalScaleSpaceExtremum<std::less_equal, T> {};

  //! Strict local spatial maximum test.
  template <typename T>
  struct StrictLocalMax : LocalExtremum<std::greater, T> {};

  //! Strict local spatial minimum test.
  template <typename T>
  struct StrictLocalMin : LocalExtremum<std::less, T> {};

  //! Strict local scale-space maximum test.
  template <typename T>
  struct StrictLocalScaleSpaceMax : LocalScaleSpaceExtremum<std::greater, T> {};

  //! Strict local scale-space minimum test.
  template <typename T>
  struct StrictLocalScaleSpaceMin : LocalScaleSpaceExtremum<std::less, T> {};

  //! Get local spatial maxima.
  template <typename T>
  inline std::vector<Point2i> local_maxima(const Image<T>& I)
  {
    return local_extrema<std::greater_equal, T>(I);
  }

  //! Get local spatial minima.
  template <typename T>
  inline std::vector<Point2i> local_minima(const Image<T>& I)
  {
    return local_extrema<std::less_equal, T>(I);
  }

  //! Get strict local spatial maxima.
  template <typename T>
  inline std::vector<Point2i> strict_local_maxima(const Image<T>& I)
  {
    return local_extrema<std::greater, T>(I);
  }

  //! Get strict local spatial minima.
  template <typename T>
  inline std::vector<Point2i> strict_local_minima(const Image<T>& I)
  {
    return local_extrema<std::less, T>(I);
  }

  //! Get local scale space maxima.
  template <typename T>
  inline std::vector<Point2i>
  local_scale_space_maxima(const ImagePyramid<T>& I, int s, int o)
  {
    return local_scale_space_extrema<std::greater_equal, T>(I,s,o);
  }

  //! Get local scale space minima.
  template <typename T>
  inline std::vector<Point2i>
  local_scale_space_minima(const ImagePyramid<T>& I, int s, int o)
  {
    return local_scale_space_extrema<std::less_equal, T>(I,s,o);
  }

  //! Get strict local scale space maxima.
  template <typename T>
  inline std::vector<Point2i>
  strict_local_scale_space_maxima(const ImagePyramid<T>& I, int s, int o)
  {
    return local_scale_space_extrema<std::greater, T>(I,s,o);
  }

  //! Get strict local scale space minima.
  template <typename T>
  inline std::vector<Point2i>
  strict_local_scale_space_minima(const ImagePyramid<T>& I, int s, int o)
  {
    return local_scale_space_extrema<std::less, T>(I,s,o);
  }

  //! @}

} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_EXTREMA_HPP */