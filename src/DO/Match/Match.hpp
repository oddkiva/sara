/*
 * =============================================================================
 *
 *       Filename:  Match.hpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/05/2010 12:31:00
 *       Revision:  none
 *       Compiler:  msvc
 *
 *         Author:  David OK (DO), david.ok@imagine.enpc.fr 
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

#ifndef DO_MATCH_MATCH_HPP
#define DO_MATCH_MATCH_HPP

namespace DO {

	class Match
	{
  public:

		enum MatchingDirection { SourceToTarget, TargetToSource };
		//! Default constructor
		inline Match()
		  : source_(0), target_(0)
      , targetRank_(-1), score_(std::numeric_limits<float>::max())
      , matchingDir_(SourceToTarget)
      , sInd_(-1), tInd_(-1) {}

		inline Match(const Keypoint *source,
						     const Keypoint *target,
						     float score = std::numeric_limits<float>::max(),
						     MatchingDirection matchingDir = SourceToTarget,
                 int i1 = -1, int i2 = -1)
		  : source_(source), target_(target)
      , targetRank_(-1), score_(score)
      , matchingDir_(matchingDir), sInd_(i1), tInd_(i2)
    {}

		//! Constant accessors.
    bool isSKeyNull() const { return source_ == 0; }
    bool isTKeyNull() const { return target_ == 0; }
    const Keypoint& source() const { if (isSKeyNull()) exit(-1); return *source_; }
		const Keypoint& target() const { if (isTKeyNull()) exit(-1); return *target_; }
		const OERegion& sFeat() const { return source().feat(); }
		const OERegion& tFeat() const { return target().feat(); }
    const Point2f& sPos() const { return sFeat().center(); }
    const Point2f& tPos() const { return tFeat().center(); }
		int rank() const { return targetRank_; }
		float score() const { return score_; }
		MatchingDirection matchingDir() const { return matchingDir_; }
    int sInd() const { return sInd_; }
    int tInd() const { return tInd_; }
    Vector2i indexPair() const { return Vector2i(sInd_, tInd_); }

		//! Non-constant accessors.
    const Keypoint *& sPtr() { return source_; }
    const Keypoint *& tPtr() { return target_; }
    int& rank() { return targetRank_; }
		float& score() { return score_; }
		MatchingDirection& matchingDir() { return matchingDir_; }
    int& sInd() { return sInd_; }
    int& tInd() { return tInd_; }

    //! Key match equality.
    bool operator==(const Match& m) const
    { return (source() == m.source() && target() == m.target()); }

	private: /* data members */
		const Keypoint *source_;
		const Keypoint *target_;
		int targetRank_;
    float score_;
		MatchingDirection matchingDir_;
    int sInd_, tInd_;
	};

  inline Match indexMatch(int i1, int i2)
  { return Match(0, 0, std::numeric_limits<float>::max(), Match::SourceToTarget, i1, i2); }

	//! I/O
	std::ostream & operator<<(std::ostream & os, const Match& m);

  bool writeMatches(const std::vector<Match>& matches, const std::string& fileName);

  bool readMatches(std::vector<Match>& matches, const std::string& fileName, float scoreT = 10.f);

  bool readMatches(std::vector<Match>& matches,
    const std::vector<Keypoint>& sKeys, const std::vector<Keypoint>& tKeys,
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

#endif /* DO_MATCH_MATCH_HPP */