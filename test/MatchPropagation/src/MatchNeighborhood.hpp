// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ===========================================================================

#ifndef DO_GROWREGION_MATCHNEIGHBORHOOD_HPP
#define DO_GROWREGION_MATCHNEIGHBORHOOD_HPP

#include <DO/Core.hpp>
#include <DO/KDTree.hpp>
#include <DO/Match.hpp>

namespace DO {

  class Rho_m
  {
  public:
    Rho_m(const Match& m)
      : m_(m)
      , Sigma_x_(m.x().shapeMat())
      , Sigma_y_(m.y().shapeMat())
    {
    }

    float dx(const Match& m) const
    { return (m.posX() - m_.posX()).dot(Sigma_x_*(m.posX() - m_.posX())); }

    float dy(const Match& m) const
    { return (m.posY() - m_.posY()).dot(Sigma_x_*(m.posY() - m_.posY())); }

    float operator()(const Match& m) const
    {
      float dxx = dx(m);
      float dyy = dy(m);
      return std::min(dxx, dyy)/std::max(dxx, dyy);
    }

  private:
    const Match& m_;
    const Matrix2f& Sigma_x_;
    const Matrix2f& Sigma_y_;
  };

  inline float rho(const Match& m1, const Match& m2)
  {
    Rho_m rho_m1(m1), rho_m2(m2);
    return std::min(rho_m1(m2), rho_m2(m1));
  }

  // SVD-based computation with a > b.
  void ellRadii(float& a, float& b, const Matrix2f& M);

  float sqIsoRadius(const Matrix2f& M);

  class ComputeN_K
  {
  public: /* interface. */
    ComputeN_K(const std::vector<Match>& matches,
               size_t neighborhoodMaxSize = 1e3,
               const PairWiseDrawer *pDrawer = 0,
               bool verbose = false);
    ~ComputeN_K();
    std::vector<size_t> operator()(size_t i, size_t K, double squaredRhoMin);
    std::vector<std::vector<size_t> >
    operator()(const std::vector<size_t>& indices,
               size_t K, double squaredRhoMin);
    std::vector<std::vector<size_t> > operator()(size_t K, double squaredRhoMin)
    { return computeNeighborhoods(K, squaredRhoMin); }
    void operator()(std::vector<std::vector<size_t> >& components,
                    std::vector<size_t>& representers,
                    const std::vector<Match>& matches,
                    double thres)
    {
      getRedundancyComponentsAndRepresenters(components, representers, matches,
                                             thres);
    }

  private: /* member functions. */
    // Comparison by lexicographical order.
    typedef std::pair<Vector2f, size_t> PosIndex;
    typedef std::pair<Vector4f, size_t> MatchIndex;
    typedef std::pair<size_t, float> IndexScore;
    struct CompareByPos
    {
      inline bool operator()(const PosIndex& v1, const PosIndex& v2) const
      { return lexCompare(v1.first, v2.first); }
    };
    struct CompareByXY
    {
      inline bool operator()(const MatchIndex& m1, const MatchIndex& m2) const
      { return lexCompare(m1.first, m2.first); }
    };
    struct EqualIndexScore1
    {
      inline bool operator()(const IndexScore& i1, const IndexScore& i2) const
      { return i1.first == i2.first; }
    };
    struct CompareIndexScore1
    {
      inline bool operator()(const IndexScore& i1, const IndexScore& i2) const
      { return i1.first > i2.first; }
    };
    struct CompareIndexScore2
    {
      inline bool operator()(const IndexScore& i1, const IndexScore& i2) const
      { return i1.second > i2.second; }
    };
    // 1. Sort matches by positions and match positions.
    void createAndSortXY();
    // 2. Create data matrix for nearest neighbor search.
    size_t countUniquePositions(const std::vector<PosIndex>& X);
    size_t countUniquePosMatches();
    MatrixXd createPosMat(const std::vector<PosIndex>& X);
    void createXMat();
    void createYMat();
    void createXYMat();
    std::vector<std::vector<size_t> >
    createPosToMatchTable(const std::vector<PosIndex>& X, const MatrixXd& xMat);
    // 3. Create lookup tables to find matches with corresponding positions.
    void createXToM();
    void createYToM();
    void createXYToM();
    // 4. Build kD-trees.
    void buildKDTrees();
    // 5. Compute neighborhoods with kD-tree data structures for efficient
    //    neighbor search.
    std::vector<std::vector<size_t> > computeNeighborhoods(size_t K,
                                                           double squaredRhoMin);
    void getMatchesFromX(std::vector<IndexScore>& indexScores,
                         size_t index,
                         const std::vector<int>& xIndices,
                         double squaredRhoMin);
    void getMatchesFromY(std::vector<IndexScore>& indexScores,
                         size_t index,
                         const std::vector<int>& yIndices,
                         double squaredRhoMin);
    void keepBestScaleConsistentMatches(std::vector<size_t>& N_K_i,
                                         std::vector<IndexScore>& indexScores,
                                         size_t K);
    // 6. (Optional) Compute redundancies before computing the neighborhoods.
    std::vector<std::vector<size_t> > computeRedundancies(double thres);
    void getRedundancyComponentsAndRepresenters(std::vector<std::vector<size_t> >& components,
                                                std::vector<size_t>& representers,
                                                const std::vector<Match>& matches,
                                                double thres);

  private: /* data members. */
    const std::vector<Match>& M_;
    // For profiling.
    Timer timer_;
    double elapsed_;
    bool verbose_;
    const PairWiseDrawer *drawer_ptr_;
    // Internal allocation.
    size_t neighborhood_max_size_;
    // For internal computation.
    std::vector<PosIndex> X_, Y_;
    std::vector<MatchIndex> XY_;
    MatrixXd X_mat_, Y_mat_, XY_mat_;
    std::vector<std::vector<size_t> > X_to_M_, Y_to_M_, XY_to_M_;
    // KDTree
    KDTree *x_index_ptr_;
    KDTree *y_index_ptr_;
    std::vector<int> x_indices_, y_indices_;
    std::vector<double> sq_dists_;
    std::vector<IndexScore> index_scores_;
    EqualIndexScore1 equal_index_score_1_;
    CompareIndexScore2 compare_index_score_1_;
    CompareIndexScore2 compare_index_score_2_;
  };

  //! Symmetrized version of ComputeN_K.
  std::vector<std::vector<size_t> >
  computeHatN_K(const std::vector<std::vector<size_t> >& N_K);

}

#endif /* DO_GROWREGION_MATCHNEIGHBORHOOD_HPP */