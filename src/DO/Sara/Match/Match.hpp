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

namespace DO {

  class Match
  {
  public:

    enum MatchingDirection { SourceToTarget, TargetToSource };
    //! Default constructor
    inline Match()
      : x_(0), y_(0)
      , target_rank_(-1), score_(std::numeric_limits<float>::max())
      , matching_dir_(SourceToTarget)
      , x_ind_(-1), y_ind_(-1) {}

    inline Match(const OERegion *x,
                 const OERegion *y,
                 float score = std::numeric_limits<float>::max(),
                 MatchingDirection matchingDir = SourceToTarget,
                 int indX = -1, int indY = -1)
      : x_(x), y_(y)
      , target_rank_(-1), score_(score)
      , matching_dir_(matchingDir), x_ind_(indX), y_ind_(indY)
    {}

    //! Constant accessors.
    bool isXNull() const { return x_ == 0; }
    bool isYNull() const { return y_ == 0; }
    const OERegion& x() const { if (isXNull()) throw 0; return *x_; }
    const OERegion& y() const { if (isYNull()) throw 0; return *y_; }
    const Point2f& posX() const { return x().center(); }
    const Point2f& posY() const { return y().center(); }
    int rank() const { return target_rank_; }
    float score() const { return score_; }
    MatchingDirection matchingDir() const { return matching_dir_; }
    int indX() const { return x_ind_; }
    int indY() const { return y_ind_; }
    Vector2i indexPair() const { return Vector2i(x_ind_, y_ind_); }

    //! Non-constant accessors.
    const OERegion *& ptrX() { return x_; }
    const OERegion *& ptrY() { return y_; }
    int& rank() { return target_rank_; }
    float& score() { return score_; }
    MatchingDirection& matchingDir() { return matching_dir_; }
    int& indX() { return x_ind_; }
    int& indY() { return y_ind_; }

    //! Key match equality.
    bool operator==(const Match& m) const
    { return (x() == m.x() && y() == m.y()); }

  private: /* data members */
    const OERegion *x_;
    const OERegion *y_;
    int target_rank_;
    float score_;
    MatchingDirection matching_dir_;
    int x_ind_, y_ind_;
  };

  inline Match indexMatch(int i1, int i2)
  { return Match(0, 0, std::numeric_limits<float>::max(), Match::SourceToTarget, i1, i2); }

  //! I/O
  std::ostream & operator<<(std::ostream & os, const Match& m);

  bool writeMatches(const std::vector<Match>& matches, const std::string& fileName);

  bool readMatches(std::vector<Match>& matches, const std::string& fileName, float scoreT = 10.f);

  bool readMatches(std::vector<Match>& matches,
    const std::vector<OERegion>& sKeys, const std::vector<OERegion>& tKeys,
    const std::string& fileName, float scoreT = 10.f);

  //! View matches.
  void drawImPair(const Image<Rgb8>& I1, const Image<Rgb8>& I2, const Point2f& off2, float scale = 1.0f);

  inline void drawImPairH(const Image<Rgb8>& I1, const Image<Rgb8>& I2, float scale = 1.0f)
  { drawImPair(I1, I2, Point2f(I1.width()*scale, 0.f), scale); }
  inline void drawImPairV(const Image<Rgb8>& I1, const Image<Rgb8>& I2, float scale = 1.0f)
  { drawImPair(I1, I2, Point2f(0.f, I1.height()*scale), scale); }

  void drawMatch(const Match& m, const Color3ub& c, const Point2f& off2, float z = 1.f);

  void drawMatches(const std::vector<Match>& matches, const Point2f& off2, float z = 1.f);

  void checkMatches(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                    const std::vector<Match>& matches, bool redrawEverytime = false, float z = 1.f);

} /* namespace DO */

#endif /* DO_SARA_MATCH_MATCH_HPP */