// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_MATCH_MATCH_HPP
#define DO_SARA_MATCH_MATCH_HPP

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  class Match
  {
  public:
    enum MatchingDirection { SourceToTarget, TargetToSource };

  public:
    //! @{
    //! \brief Constructors.
    inline Match()
      : _x(0), _y(0)
      , _target_rank(-1), _score(std::numeric_limits<float>::max())
      , _matching_dir(SourceToTarget)
      , _x_idx(-1), _y_idx(-1)
    {
    }

    inline Match(const OERegion *x,
                 const OERegion *y,
                 float score = std::numeric_limits<float>::max(),
                 MatchingDirection matchingDir = SourceToTarget,
                 int indX = -1, int indY = -1)
      : _x(x), _y(y)
      , _target_rank(-1), _score(score)
      , _matching_dir(matchingDir), _x_idx(indX), _y_idx(indY)
    {
    }
    //! @}

    //! Constant accessors.
    bool is_x_null() const { return _x == 0; }

    bool is_y_null() const { return _y == 0; }

    const OERegion& x() const { if (is_x_null()) throw 0; return *_x; }

    const OERegion& y() const { if (is_y_null()) throw 0; return *_y; }

    const Point2f& x_pos() const { return x().center(); }

    const Point2f& y_pos() const { return y().center(); }

    int rank() const { return _target_rank; }

    float score() const { return _score; }

    MatchingDirection matchingDir() const { return _matching_dir; }

    int x_idx() const { return _x_idx; }

    int y_idx() const { return _y_idx; }

    Vector2i index_pair() const { return Vector2i(_x_idx, _y_idx); }

    //! Non-constant accessors.
    const OERegion *& x_ptr() { return _x; }

    const OERegion *& y_ptr() { return _y; }

    int& rank() { return _target_rank; }

    float& score() { return _score; }

    MatchingDirection& matchingDir() { return _matching_dir; }

    int& x_idx() { return _x_idx; }

    int& y_idx() { return _y_idx; }

    //! Key match equality.
    bool operator==(const Match& m) const
    {
      return (x() == m.x() && y() == m.y());
    }

  private: /* data members */
    const OERegion *_x;
    const OERegion *_y;
    int _target_rank;
    float _score;
    MatchingDirection _matching_dir;
    int _x_idx, _y_idx;
  };

  inline Match index_match(int i1, int i2)
  {
    return Match(0, 0, std::numeric_limits<float>::max(), Match::SourceToTarget, i1, i2);
  }

  //! @{
  //! I/O
  DO_EXPORT
  std::ostream & operator<<(std::ostream & os, const Match& m);

  DO_EXPORT
  bool write_matches(const std::vector<Match>& matches, const std::string& fileName);

  DO_EXPORT
  bool read_matches(std::vector<Match>& matches, const std::string& filepath, float score_thres = 10.f);

  DO_EXPORT
  bool read_matches(std::vector<Match>& matches,
                    const std::vector<OERegion>& source_keys,
                    const std::vector<OERegion>& target_keys,
                    const std::string& filepath,
                    float score_thres = 10.f);
  //! @}


  //! @{
  //! View matches.
  DO_EXPORT
  void draw_image_pair(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                       const Point2f& off2, float scale = 1.0f);

  inline void draw_image_pair(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                              float scale = 1.0f)
  {
    draw_image_pair(I1, I2, Point2f(I1.width()*scale, 0.f), scale);
  }

  inline void draw_image_pair_v(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                                float scale = 1.0f)
  {
    draw_image_pair(I1, I2, Point2f(0.f, I1.height()*scale), scale);
  }

  DO_EXPORT
  void draw_match(const Match& m, const Color3ub& c, const Point2f& off2,
                  float z = 1.f);

  DO_EXPORT
  void draw_matches(const std::vector<Match>& matches,
                    const Point2f& off2, float z = 1.f);

  DO_EXPORT
  void check_matches(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                     const std::vector<Match>& matches,
                     bool redraw_everytime = false, float z = 1.f);
  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_MATCH_MATCH_HPP */