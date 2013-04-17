#ifndef DO_MATCH_PAIRWISEDRAWER_HPP
#define DO_MATCH_PAIRWISEDRAWER_HPP

namespace DO {

  class Match;

  class PairWiseDrawer
  {
  public:
    enum CatType { CatH, CatV };

    PairWiseDrawer(
      const Image<Rgb8>& I1,
      const Image<Rgb8>& I2)
      : image1(I1), image2(I2) {}

    //! Set visualization parameters.
    void setVizParams(float s1, float s2, CatType concatType)
    {
      z1 = s1; z2 = s2;
      catType = concatType;
      off2 = catType == CatH ? Point2i(image1.width(), 0) : Point2i(0, image1.height());
    }

    void displayImages() const;

    void drawPoint(int i, const Point2f& p, const Color3ub& c, int r = 2) const;

    void drawLine(int i, const Point2f& pa, const Point2f& pb, const Color3ub& c, int penWidth = 1) const;

    void drawArrow(int i, const Point2f& pa, const Point2f& pb, const Color3ub& c, int penWidth = 1) const;

    void drawTriangle(int i, const Point2f& pa, const Point2f& pb, const Point2f& pc,
                      const Color3ub& c = Cyan8, int r = 2) const;

    void drawRect(int i, const Point2f& p1, const Point2f& p2, int r, const Color3ub& c = Yellow8) const;

    template<typename LineIterator>
    void drawLines(int i, LineIterator first, LineIterator last,
                   const Color3ub& c = Black8, int r = 2) const
    {
      assert(i == 0 || i ==1);
      for(LineIterator line = first; line != last; ++line)
        drawLine(i, line->first, line->second, c, r);
    }

    void drawLineFromEqn(int i, const Vector3f& eqn, const Color3ub& c = Cyan8, int r = 2) const;

    template<typename EqnIterator>
    inline void drawLinesFromEqns(int i, EqnIterator first, EqnIterator last,
                                  const Color3ub& c = Cyan8, int r = 2) const
    {
      assert(i == 0 || i ==1);
      for(EqnIterator eqn = first; eqn != last; ++eqn)
        drawLineFromEqn(i, *eqn, c, r);
    }

    template<typename VHIterator>
    inline void drawVertices(int i, VHIterator first, VHIterator last, int r = 2, 
                             const Color3ub& c = Yellow8) const
    {
      assert(i == 0 || i ==1);
      for(VHIterator vh = first; vh != last; ++vh)
        drawPoint(i, Point2f( (*vh)->point().x(), (*vh)->point().y() ), c, r);
    }

    void drawKeypoint(int i, const Keypoint& k, const Color3ub& c = Red8) const;

    void drawMatch(const Match& m, const Color3ub& c = Magenta8, bool drawLine = false) const;

    const Image<Rgb8>& image(int i) const
    {
      assert(i == 0 || i == 1);
      return (i == 0) ? image1 : image2;
    }

    Point2i off(int i) const
    {
      assert(i == 0 || i == 1);
      return (i == 0) ? Point2i::Zero() : off2;
    }

    Point2f offF(int i) const
    { return off(i).cast<float>(); }

    float scale(int i) const
    {
      assert(i == 0 || i == 1);
      return (i == 0) ? z1 : z2;
    }

  private:
    //! Images and features
    const Image<Rgb8>& image1, image2;

    CatType catType;
    Point2i off2;
    float z1, z2;
  };

} /* namespace DO */

#endif /* DO_MATCH_PAIRWISEDRAWER_HPP */