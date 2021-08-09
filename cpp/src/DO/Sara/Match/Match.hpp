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

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image.hpp>

#include <DO/Sara/Features/Feature.hpp>


namespace DO { namespace Sara {

  //! @addtogroup Match
  //! @{

  class Match
  {
  public:
    enum class Direction : std::uint8_t
    {
      SourceToTarget,
      TargetToSource
    };

  public:
    //! @{
    //! @brief Constructors.
    inline Match() = default;

    inline Match(const OERegion* x, const OERegion* y,
                 float score = std::numeric_limits<float>::max(),
                 Direction matching_dir = Direction::SourceToTarget,
                 int x_index = -1, int y_index = -1)
      : _x(x)
      , _y(y)
      , _x_index(x_index)
      , _y_index(y_index)
      , _rank(-1)
      , _score(score)
      , _matching_dir(matching_dir)
    {
    }
    //! @}

    //! @{
    //! Constant accessors.
    const OERegion* x_pointer() const
    {
      return _x;
    }

    const OERegion* y_pointer() const
    {
      return _y;
    }

    const OERegion& x() const
    {
      if (_x == nullptr)
        throw std::runtime_error{"x is null"};
      return *_x;
    }

    const OERegion& y() const
    {
      if (_y == nullptr)
        throw std::runtime_error{"y is null"};
      return *_y;
    }

    const Point2f& x_pos() const
    {
      return x().center();
    }

    const Point2f& y_pos() const
    {
      return y().center();
    }

    int rank() const
    {
      return _rank;
    }

    float score() const
    {
      return _score;
    }

    Direction matching_direction() const
    {
      return _matching_dir;
    }

    int x_index() const
    {
      return _x_index;
    }

    int y_index() const
    {
      return _y_index;
    }

    Vector2i index_pair() const
    {
      return Vector2i(_x_index, _y_index);
    }
    //! @}

    //! @{
    //! Non-constant accessors.
    const OERegion *& x_pointer()
    {
      return _x;
    }

    const OERegion *& y_pointer()
    {
      return _y;
    }

    int& rank()
    {
      return _rank;
    }

    float& score()
    {
      return _score;
    }

    Direction& matching_direction()
    {
      return _matching_dir;
    }

    int& x_index()
    {
      return _x_index;
    }

    int& y_index()
    {
      return _y_index;
    }
    //! @}

    //! Key match equality.
    bool operator==(const Match& m) const
    {
      return x() == m.x() && y() == m.y();
    }

  private: /* data members */
    const OERegion* _x{nullptr};
    const OERegion* _y{nullptr};
    int _x_index{-1};
    int _y_index{-1};
    int _rank{-1};
    float _score{std::numeric_limits<float>::max()};
    Direction _matching_dir{Direction::SourceToTarget};
  };

  inline Match make_index_match(int i1, int i2,
                                float score = std::numeric_limits<float>::max())
  {
    return Match{nullptr, nullptr, score, Match::Direction::SourceToTarget,
                 i1,      i2};
  }

  //! @{
  //! I/O
  DO_SARA_EXPORT
  std::ostream & operator<<(std::ostream & os, const Match& m);

  DO_SARA_EXPORT
  bool write_matches(const std::vector<Match>& matches, const std::string& fileName);

  DO_SARA_EXPORT
  bool read_matches(std::vector<Match>& matches, const std::string& filepath, float score_thres = 10.f);

  DO_SARA_EXPORT
  bool read_matches(std::vector<Match>& matches,
                    const std::vector<OERegion>& source_keys,
                    const std::vector<OERegion>& target_keys,
                    const std::string& filepath,
                    float score_thres = 10.f);
  //! @}

  //! @}

} /* namespace Sara */
} /* namespace DO */
