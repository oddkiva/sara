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

// ========================================================================== //
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ===========================================================================

#pragma once

#include <DO/Sara/Core.hpp>
#include <DO/Sara/KDTree.hpp>
#include <DO/Sara/Match.hpp>


namespace DO { namespace Sara { namespace extra {

  class Rho_m
  {
  public:
    Rho_m(const Match& m)
      : _m(m)
      , _Sigma_x(m.x().shapeMat())
      , _Sigma_y(m.y().shapeMat())
    {
    }

    auto dx(const Match& m) const -> float
    {
      return (m.x_pos() - _m.x_pos()).dot(_Sigma_x * (m.x_pos() - _m.x_pos()));
    }

    auto dy(const Match& m) const -> float
    {
      return (m.y_pos() - _m.y_pos()).dot(_Sigma_x * (m.y_pos() - _m.y_pos()));
    }

    auto operator()(const Match& m) const -> float
    {
      float dxx = dx(m);
      float dyy = dy(m);
      return std::min(dxx, dyy) / std::max(dxx, dyy);
    }

  private:
    const Match& _m;
    const Matrix2f& _Sigma_x;
    const Matrix2f& _Sigma_y;
  };

  inline auto rho(const Match& m1, const Match& m2) -> float
  {
    Rho_m rho_m1(m1), rho_m2(m2);
    return std::min(rho_m1(m2), rho_m2(m1));
  }

  // SVD-based computation with a > b.
  void ellipse_radii(float& a, float& b, const Matrix2f& M);

  float square_isometric_radius(const Matrix2f& M);


  class ComputeN_K
  {
  public: /* interface. */
    ComputeN_K(const std::vector<Match>& matches,
               size_t neighborhoodMaxSize = 1e3,
               const PairWiseDrawer* pDrawer = 0, bool verbose = false);

    ~ComputeN_K();

    auto operator()(size_t i, size_t K, double squaredRhoMin)
        -> std::vector<size_t>;

    auto operator()(const std::vector<size_t>& indices, size_t K,
                    double squaredRhoMin) -> std::vector<std::vector<size_t>>;

    auto operator()(size_t K, double squared_rho_min)
        -> std::vector<std::vector<size_t>>
    {
      return computeNeighborhoods(K, squared_rho_min);
    }

    auto operator()(std::vector<std::vector<size_t>>& components,
                    std::vector<size_t>& representers,
                    const std::vector<Match>& matches, double thres) -> void
    {
      get_redundancy_components_and_representers(components, representers,
                                                 matches, thres);
    }

  private: /* member functions. */
    // Comparison by lexicographical order.
    typedef std::pair<Vector2f, size_t> PosIndex;
    typedef std::pair<Vector4f, size_t> MatchIndex;
    typedef std::pair<size_t, float> IndexScore;

    struct CompareByPos
    {
      inline bool operator()(const PosIndex& v1, const PosIndex& v2) const
      {
        return lexicographical_compare(v1.first, v2.first);
      }
    };

    struct CompareByXY
    {
      inline bool operator()(const MatchIndex& m1, const MatchIndex& m2) const
      {
        return lexicographical_compare(m1.first, m2.first);
      }
    };

    struct EqualIndexScore1
    {
      inline bool operator()(const IndexScore& i1, const IndexScore& i2) const
      {
        return i1.first == i2.first;
      }
    };

    struct CompareIndexScore1
    {
      inline bool operator()(const IndexScore& i1, const IndexScore& i2) const
      {
        return i1.first > i2.first;
      }
    };

    struct CompareIndexScore2
    {
      inline bool operator()(const IndexScore& i1, const IndexScore& i2) const
      {
        return i1.second > i2.second;
      }
    };

    // 1. Sort matches by positions and match positions.
    void create_and_sort_XY();

    // 2. Create data matrix for nearest neighbor search.
    size_t countUniquePositions(const std::vector<PosIndex>& X);

    size_t countUniquePosMatches();

    MatrixXd createPosMat(const std::vector<PosIndex>& X);

    void createXMat();
    void createYMat();
    void createXYMat();

    std::vector<std::vector<size_t>>
    createPosToMatchTable(const std::vector<PosIndex>& X, const MatrixXd& xMat);

    // 3. Create lookup tables to find matches with corresponding positions.
    void createXToM();
    void createYToM();
    void createXYToM();

    // 4. Build kD-trees.
    void buildKDTrees();

    // 5. Compute neighborhoods with kD-tree data structures for efficient
    //    neighbor search.
    std::vector<std::vector<size_t>> computeNeighborhoods(size_t K,
                                                          double squaredRhoMin);

    void getMatchesFromX(std::vector<IndexScore>& indexScores, size_t index,
                         const std::vector<int>& xIndices,
                         double squaredRhoMin);

    void getMatchesFromY(std::vector<IndexScore>& indexScores, size_t index,
                         const std::vector<int>& yIndices,
                         double squaredRhoMin);

    void keepBestScaleConsistentMatches(std::vector<size_t>& N_K_i,
                                        std::vector<IndexScore>& indexScores,
                                        size_t K);

    // 6. (Optional) Compute redundancies before computing the neighborhoods.
    std::vector<std::vector<size_t>> computeRedundancies(double thres);

    void getRedundancyComponentsAndRepresenters(
        std::vector<std::vector<size_t>>& components,
        std::vector<size_t>& representers, const std::vector<Match>& matches,
        double thres);

  private: /* data members. */
    const std::vector<Match>& _M;

    // For profiling.
    Timer _timer;
    double _elapsed;
    bool _verbose;
    const PairWiseDrawer *_drawer;

    // Internal allocation.
    size_t neighborhood_max_size_;

    // For internal computation.
    std::vector<PosIndex> _X, _Y;
    std::vector<MatchIndex> _XY;
    MatrixXd _X_mat, _Y_mat, _XY_mat;
    std::vector<std::vector<size_t>> _X_to_M, _Y_to_M, _XY_to_M;

    // KDTree
    KDTree *_x_index_ptr;
    KDTree *_y_index_ptr;
    std::vector<int> _x_indices, _y_indices;
    std::vector<double> _sq_dists;
    std::vector<IndexScore> _index_scores;
    EqualIndexScore1 _equal_index_score_1;
    CompareIndexScore2 _compare_index_score_1;
    CompareIndexScore2 _compare_index_score_2;
  };

  //! Symmetrized version of ComputeN_K.
  std::vector<std::vector<size_t>>
  compute_hat_N_K(const std::vector<std::vector<size_t>>& N_K);


} /* namespace extra */
} /* namespace Sara */
} /* namespace DO */
