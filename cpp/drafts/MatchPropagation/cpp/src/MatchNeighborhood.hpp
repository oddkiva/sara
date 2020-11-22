// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
/*!
 *  This file implements a part of the method published in:
 *
 *  Efficient and Scalable 4th-order Match Propagation
 *  David Ok, Renaud Marlet, and Jean-Yves Audibert.
 *  ACCV 2012, Daejeon, South Korea.
 */

#pragma once

#include <DO/Sara/Core.hpp>
#include <DO/Sara/KDTree.hpp>
#include <DO/Sara/Match.hpp>


namespace DO::Sara {

  //! @addtogroup MatchPropagation
  //! @{

  //! Affine Covariant Match Distance denoted as @f$\rho_m @f$
  class AffineCovariantMatchDistance
  {
  public:
    AffineCovariantMatchDistance(const Match& m)
      : _m(m)
      , _Sigma_x(m.x().shape_matrix)
      , _Sigma_y(m.y().shape_matrix)
    {
    }

    inline auto dx(const Match& m) const -> float
    {
      return (m.x_pos() - _m.x_pos()).transpose() * _Sigma_x *
             (m.x_pos() - _m.x_pos());
    }

    inline auto dy(const Match& m) const -> float
    {
      return (m.y_pos() - _m.y_pos()).transpose() * _Sigma_x * (m.y_pos() - _m.y_pos());
    }

    inline auto operator()(const Match& m) const -> float
    {
      const auto dxx = dx(m);
      const auto dyy = dy(m);
      return std::min(dxx, dyy) / std::max(dxx, dyy);
    }

  private:
    const Match& _m;
    const Matrix2f& _Sigma_x;
    const Matrix2f& _Sigma_y;
  };


  //! Compute the symmetrized affine covariant match distance.
  inline auto rho(const Match& m1, const Match& m2) -> float
  {
    const AffineCovariantMatchDistance rho_m1(m1);
    const AffineCovariantMatchDistance rho_m2(m2);
    return std::min(rho_m1(m2), rho_m2(m1));
  }

  //! SVD-based computation with a > b.
  auto ellipse_radii(float& a, float& b, const Matrix2f& M) -> void;

  auto square_isometric_radius(const Matrix2f& M) -> float;


  //! @brief Functor class that computes the K nearest matches denoted as
  //! @f$ \mathcal{N}_K(.) @f$.
  class DO_SARA_EXPORT NearestMatchNeighborhoodComputer
  {
  public: /* interface. */
    NearestMatchNeighborhoodComputer(const std::vector<Match>& matches,
                                      size_t neighborhood_max_size = 1e3,
                                      const PairWiseDrawer *drawer = 0,
                                      bool verbose = false);

    auto operator()(size_t i, size_t K, double squared_rho_min)
        -> std::vector<size_t>;

    auto operator()(const std::vector<size_t>& indices, size_t K,
                    double squared_rho_min) -> std::vector<std::vector<size_t>>;

    auto operator()(size_t K, double squared_rho_min)
        -> std::vector<std::vector<size_t>>
    {
      return compute_neighborhoods(K, squared_rho_min);
    }

    auto operator()(std::vector<std::vector<size_t>>& components,
                    std::vector<size_t>& representers,
                    const std::vector<Match>& matches, double thres) -> void
    {
      get_redundancy_components_and_representers(components, representers,
                                                 matches, thres);
    }

  private: /* member functions. */
    //! @brief Comparison by lexicographical order.
    //! @{
    using PosIndex = std::pair<Vector2f, size_t>;
    using MatchIndex = std::pair<Vector4f, size_t>;
    using IndexScore = std::pair<size_t, float>;

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
    //! @}

    // @brief  1. Sort matches by positions and match positions.
    auto create_and_sort_xy() -> void;

    //! @brief  2. Create data matrix for nearest neighbor search.
    //! @{
    auto count_unique_positions(const std::vector<PosIndex>& x) -> size_t;

    auto count_unique_pos_matches() -> size_t;

    auto create_position_matrix(const std::vector<PosIndex>& X) -> MatrixXd;

    auto create_x_matrix() -> void;
    auto create_y_matrix() -> void;
    auto create_xy_matrix() -> void;

    auto create_position_to_match_table(const std::vector<PosIndex>& X,
                                        const MatrixXd& X_mat)
        -> std::vector<std::vector<size_t>>;
    //! @}

    //! @brief  3. Create lookup tables to find matches with corresponding
    //! positions.
    void create_x_to_m();
    void create_y_to_m();
    void create_xy_to_m();

    //! @brief  4. Build kD-trees.
    void build_kdtrees();

    //! @brief 5. Compute neighborhoods with kD-tree data structures for
    //! efficient neighbor search.
    //! @{
    auto compute_neighborhoods(size_t K, double squared_rho_min)
        -> std::vector<std::vector<size_t>>;

    auto get_matches_from_x(std::vector<IndexScore>& index_scores, size_t index,
                            const std::vector<int>& x_indices,
                            double squared_rho_min) -> void;

    auto get_matches_from_y(std::vector<IndexScore>& index_scores, size_t index,
                            const std::vector<int>& y_indices,
                            double squared_rho_min) -> void;

    auto
    keep_best_scale_consistent_matches(std::vector<size_t>& N_K_i,
                                       std::vector<IndexScore>& index_scores,
                                       size_t K) -> void;
    //! @}

    //! @brief  6. (Optional) Compute redundancies before computing the
    //! neighborhoods.
    //! @{
    auto compute_redundancies(double thres) -> std::vector<std::vector<size_t>>;

    auto get_redundancy_components_and_representers(
        std::vector<std::vector<size_t>>& components,
        std::vector<size_t>& representers, const std::vector<Match>& matches,
        double thres) -> void;
    //! @}

  private: /* data members. */
    const std::vector<Match>& _M;

    // @{
    // @brief For profiling.
    Timer _timer;
    double _elapsed;
    bool _verbose;
    const PairWiseDrawer *_drawer;
    // @}

    // Internal allocation.
    size_t _neighborhood_max_size;

    //! @brief For internal computation.
    //! @{
    std::vector<PosIndex> _X, _Y;
    std::vector<MatchIndex> _XY;
    MatrixXd _X_mat, _Y_mat, _XY_mat;
    std::vector<std::vector<size_t>> _X_to_M, _Y_to_M, _XY_to_M;
    //! @}

    //! @brief KDTree
    //! @{
    std::unique_ptr<KDTree> _x_index_ptr;
    std::unique_ptr<KDTree> _y_index_ptr;
    std::vector<int> _x_indices, _y_indices;
    std::vector<double> _sq_dists;
    std::vector<IndexScore> _index_scores;
    EqualIndexScore1 _equal_index_score_1;
    CompareIndexScore2 _compare_index_score_1;
    CompareIndexScore2 _compare_index_score_2;
    //! @}
  };

  //! Symmetrized version of ComputeN_K.
  DO_SARA_EXPORT
  auto compute_hat_N_K(const std::vector<std::vector<size_t>>& N_K)
      -> std::vector<std::vector<size_t>>;

  //! @}

}  // namespace DO::Sara
