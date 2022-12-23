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

#include <DO/Sara/Visualization.hpp>

#include "MatchNeighborhood.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif


using namespace std;


namespace DO::Sara {

  NearestMatchNeighborhoodComputer::NearestMatchNeighborhoodComputer(
      const vector<Match>& matches, size_t neighborhood_max_size,
      const PairWiseDrawer* drawer, bool verbose)
    : _M(matches)
    , _verbose(verbose)
    , _drawer(drawer)
    , _neighborhood_max_size(neighborhood_max_size)
  {
    create_and_sort_xy();
    create_x_matrix();
    create_y_matrix();
    create_xy_matrix();
    create_x_to_m();
    create_y_to_m();
    create_xy_to_m();
    build_kdtrees();
  }

  vector<size_t>
  NearestMatchNeighborhoodComputer::operator()(size_t i, size_t K,
                                               double squared_rho_min)
  {
    vector<size_t> N_K_i;
    N_K_i.reserve(_neighborhood_max_size);

    // Match $m_i = (x_i, y_i)$
    const Vector2d xi = _M[i].x_pos().cast<double>();
    const Vector2d yi = _M[i].y_pos().cast<double>();
    const auto Ki = static_cast<int>(K);
    // Collect the K nearest matches $\mathb{x}_j$ to $\mathb{x}_i$.
    _x_index_ptr->knn_search(xi, Ki, _x_indices, _sq_dists);
    // Collect the K nearest matches $\mathb{y}_j$ to $\mathb{y}_i$
    _y_index_ptr->knn_search(yi, Ki, _y_indices, _sq_dists);


    // Reset list of neighboring matches.
    _index_scores.clear();
    //#define DEBUG
#ifdef DEBUG
    if (_verbose && _drawer)
      _drawer->display_images();
#endif
    get_matches_from_x(_index_scores, i, _x_indices, squared_rho_min);
    get_matches_from_y(_index_scores, i, _y_indices, squared_rho_min);
    keep_best_scale_consistent_matches(N_K_i, _index_scores, 10 * K);
#ifdef DEBUG
    if (_verbose && _drawer)
    {
      cout << "_M[" << i << "] =\n" << _M[i] << endl;
      cout << "N_K[" << i << "].size() = " << N_K_i.size() << endl;
      cout << "index_scores.size() = " << _index_scores.size() << endl;
      for (size_t j = 0; j != _index_scores.size(); ++j)
        cout << j << "\t" << _index_scores[j].first << " "
             << _index_scores[j].second << endl;
      for (size_t j = 0; j != K; ++j)
      {
        _drawer->draw_match(_M[N_K_i[j]]);
        _drawer->draw_point(0, _M[N_K_i[j]].x_pos(), Cyan8, 3);
        _drawer->draw_point(1, _M[N_K_i[j]].y_pos(), Cyan8, 3);
      }
      _drawer->draw_match(_M[i], Cyan8);
      get_key();
    }
#endif
    return N_K_i;
  }

  vector<vector<size_t>>
  NearestMatchNeighborhoodComputer::operator()(const vector<size_t>& indices,
                                               size_t K, double squared_rho_min)
  {
    vector<vector<size_t>> N_K_indices;
    N_K_indices.resize(indices.size());

    vector<vector<IndexScore>> index_scores;
    vector<vector<int>> x_indices, y_indices;
    vector<vector<double>> squared_distances;
    index_scores.resize(indices.size());
    x_indices.resize(indices.size());
    y_indices.resize(indices.size());
    squared_distances.resize(indices.size());

//#define PARALLEL_QUERY
#ifdef PARALLEL_QUERY
    cout << "Parallel neighborhood query" << endl;
    // Match $m_i = (x_i, y_i)$.
    MatrixXd xis(2, indices.size());
    MatrixXd yis(2, indices.size());
    for (size_t i = 0; i != indices.size(); ++i)
    {
      xis.col(i) = _M[indices[i]].x_pos().cast<double>();
      yis.col(i) = _M[indices[i]].y_pos().cast<double>();
    }
    // Collect the K nearest matches $\mathb{x}_j$ to $\mathb{x}_i$.
    _x_index_ptr->knn_search(xis, K, x_indices, squared_distances, true);
    // Collect the K nearest matches $\mathb{y}_j$ to $\mathb{y}_i$
    _y_index_ptr->knn_search(yis, K, y_indices, squared_distances, true);
#else
    for (size_t i = 0; i != indices.size(); ++i)
    {
      const Vector2d xi = _M[indices[i]].x_pos().cast<double>();
      const Vector2d yi = _M[indices[i]].y_pos().cast<double>();
      const auto Ki = static_cast<int>(K);
      _x_index_ptr->knn_search(xi, Ki, x_indices[i], squared_distances[0]);
      // Collect the K nearest matches $\mathb{y}_j$ to $\mathb{y}_i$
      _y_index_ptr->knn_search(yi, Ki, y_indices[i], squared_distances[0]);
    }
#endif

    int num_indices = static_cast<int>(indices.size());
#ifdef _OPENMP
#  pragma omp parallel for
#endif
    for (int i = 0; i < num_indices; ++i)
    {
      get_matches_from_x(index_scores[i], indices[i], x_indices[i],
                         squared_rho_min);
      get_matches_from_y(index_scores[i], indices[i], y_indices[i],
                         squared_rho_min);
      keep_best_scale_consistent_matches(N_K_indices[i], index_scores[i],
                                         10 * K);
    }

    return N_K_indices;
  }

  // ======================================================================== //
  // 1. Sort matches by positions and match positions.
  void NearestMatchNeighborhoodComputer::create_and_sort_xy()
  {
    if (_verbose)
      _timer.restart();
    // Construct the set of matches ordered by position x and y,
    // and by position matches (x,y).
    _X.resize(_M.size());
    _Y.resize(_M.size());
    _XY.resize(_M.size());
    for (size_t i = 0; i != _M.size(); ++i)
    {
      _X[i] = make_pair(_M[i].x_pos(), i);
      _Y[i] = make_pair(_M[i].y_pos(), i);
      _XY[i].first << _M[i].x_pos(), _M[i].y_pos();
      _XY[i].second = i;
    }
    // Sort the set of matches.
    CompareByPos compareByPos;
    CompareByXY compareByXY;
    sort(_X.begin(), _X.end(), compareByPos);
    sort(_Y.begin(), _Y.end(), compareByPos);
    sort(_XY.begin(), _XY.end(), compareByXY);
    if (_verbose)
    {
      _elapsed = _timer.elapsed();
      print_stage("Storing matrices of unique positions and match positions");
      cout << "Time elapsed = " << _elapsed << " seconds" << endl;
    }
  }

  // ======================================================================== //
  // 2. Create data matrix for nearest neighbor search.
  size_t NearestMatchNeighborhoodComputer::count_unique_positions(
      const vector<PosIndex>& X)
  {
    // Count the number of unique positions $\mathbf{x}_i$.
    size_t numPosX = 1;
    Vector2f pos(X[0].first);
    for (size_t i = 1; i != X.size(); ++i)
    {
      if (X[i].first != pos)
      {
        ++numPosX;
        pos = X[i].first;
      }
    }
    if (_verbose)
    {
      cout << "Check number of unique positions x" << endl;
      cout << "numPosX = " << numPosX << endl;
      cout << "X.size() = " << X.size() << endl;
      cout << "_M.size() = " << _M.size() << endl;
    }
    return numPosX;
  }

  size_t NearestMatchNeighborhoodComputer::count_unique_pos_matches()
  {
    if (_verbose)
      _timer.restart();
    // Count the number of unique positions $(\mathbf{x}_i, \mathbf{y}_i)$.
    size_t num_pos_xy = 1;
    Vector4f match = _XY[0].first;
    for (size_t i = 1; i != _XY.size(); ++i)
    {
      if (_XY[i].first != match)
      {
        ++num_pos_xy;
        match = _XY[i].first;
      }
    }
    if (_verbose)
    {
      cout << "Check number of unique position match (x,y)" << endl;
      ;
      cout << "num_pos_xy = " << num_pos_xy << endl;
      cout << "matchesByXY.size() = " << _XY.size() << endl;
      cout << "matches.size() = " << _M.size() << endl;
      _elapsed = _timer.elapsed();
      cout << "Counting number of unique positions (x,y)." << endl;
      cout << "Time elapsed = " << _elapsed << " seconds" << endl;
    }
    return num_pos_xy;
  }

  MatrixXd NearestMatchNeighborhoodComputer::create_position_matrix(
      const vector<PosIndex>& X)
  {
    size_t numPosX = count_unique_positions(X);
    // Store the matrix of positions $\mathbf{x}_i$ without duplicate.
    MatrixXd xMat(2, numPosX);
    size_t xInd = 0;
    Vector2f pos(X[0].first);
    xMat.col(0) = pos.cast<double>();
    for (size_t i = 1; i != X.size(); ++i)
    {
      if (X[i].first != pos)
      {
        ++xInd;
        pos = X[i].first;
        xMat.col(xInd) = pos.cast<double>();
      }
    }
    if (_verbose)
    {
      cout << "Stacking position vectors." << endl;
      cout << "xInd = " << xInd << endl;
    }
    return xMat;
  }

  void NearestMatchNeighborhoodComputer::create_x_matrix()
  {
    if (_verbose)
      print_stage("X matrix");
    _X_mat = create_position_matrix(_X);
    if (_verbose && _drawer)
    {
      for (auto i = 0; i != _X_mat.cols(); ++i)
      {
        Vector2f xi(_X_mat.col(i).cast<float>());
        // _drawer->draw_point(0, xi, Red8, 5);
      }
    }
  }

  void NearestMatchNeighborhoodComputer::create_y_matrix()
  {
    if (_verbose)
      print_stage("Y matrix");

    _Y_mat = create_position_matrix(_Y);

    if (_verbose && _drawer)
    {
      for (auto i = 0; i != _Y_mat.cols(); ++i)
      {
        Vector2f yi(_Y_mat.col(i).cast<float>());
        // _drawer->draw_point(1, yi, Blue8, 5);
      }
    }
  }

  void NearestMatchNeighborhoodComputer::create_xy_matrix()
  {
    if (_verbose)
      print_stage("XY matrix");

    // Store the matrix of position matches $(\mathbf{x}_i, \mathbf{y}_i)$
    // without duplicate.
    size_t num_pos_xy = count_unique_pos_matches();
    _XY_mat = MatrixXd(4, num_pos_xy);

    size_t xy_index = 0;
    Vector4f match(_XY[0].first);
    _XY_mat.col(0) = match.cast<double>();
    for (size_t i = 1; i != _XY.size(); ++i)
    {
      if (_XY[i].first != match)
      {
        ++xy_index;
        match = _XY[i].first;
        _XY_mat.col(xy_index) = match.cast<double>();
      }
    }
    if (_verbose && _drawer)
    {
      cout << "xy_index = " << xy_index << endl;
      /*for (size_t i = 0; i != _XY_mat.cols(); ++i)
      {
        Vector2f xi(_XY_mat.block(0,i,2,1).cast<float>());
        _drawer->drawPoint(0, xi, Magenta8, 5);
        Vector2f yi(_XY_mat.block(2,i,2,1).cast<float>());
        _drawer->drawPoint(1, yi, Magenta8, 5);
      }*/
    }
  }

  auto NearestMatchNeighborhoodComputer::create_position_to_match_table(
      const vector<PosIndex>& X, const MatrixXd& xMat) -> vector<vector<size_t>>
  {
    // Store the indices of matches $(\mathbf{x}_i, .)$.
    size_t numPosX = xMat.cols();
    vector<vector<size_t>> xToM(numPosX);
    // Allocate memory first.
    for (size_t i = 0; i != numPosX; ++i)
      xToM[i].reserve(_neighborhood_max_size);
    // Loop
    size_t xInd = 0;
    Vector2f pos(X[0].first);
    xToM[0].push_back(X[0].second);
    for (size_t i = 1; i != X.size(); ++i)
    {
      if (X[i].first != pos)
      {
        ++xInd;
        pos = X[i].first;
      }
      xToM[xInd].push_back(X[i].second);
    }
    if (_verbose)
    {
      print_stage("Check number of positions for x");
      size_t num_positions_x = xToM.size();
      size_t num_matches_x = 0;
      for (size_t i = 0; i != xToM.size(); ++i)
        num_matches_x += xToM[i].size();
      cout << "num_positions_x = " << num_positions_x << endl;
      cout << "num_matches_x = " << num_matches_x << endl;
    }
    return xToM;
  }

  // ======================================================================== //
  // 3. Create lookup tables to find matches with corresponding positions.
  void NearestMatchNeighborhoodComputer::create_x_to_m()
  {
    if (_verbose)
    {
      print_stage("Create X to M");
      _timer.restart();
    }
    _X_to_M = create_position_to_match_table(_X, _X_mat);
    if (_verbose)
    {
      size_t num_positions_x = _X_to_M.size();
      size_t num_matches_x = 0;
      for (size_t i = 0; i != _X_to_M.size(); ++i)
        num_matches_x += _X_to_M[i].size();
      cout << "num_positions_x = " << num_positions_x << endl;
      cout << "num_matches_x = " << num_matches_x << endl;
    }
  }

  void NearestMatchNeighborhoodComputer::create_y_to_m()
  {
    if (_verbose)
    {
      print_stage("Create Y to M");
      _timer.restart();
    }
    _Y_to_M = create_position_to_match_table(_Y, _Y_mat);
    if (_verbose)
    {
      print_stage("Check number of positions for y");
      size_t num_positions_y = _Y_to_M.size();
      size_t num_matches_y = 0;
      for (size_t i = 0; i != _Y_to_M.size(); ++i)
        num_matches_y += _Y_to_M[i].size();
      cout << "num_positions_y = " << num_positions_y << endl;
      cout << "num_matches_y = " << num_matches_y << endl;
    }
  }

  void NearestMatchNeighborhoodComputer::create_xy_to_m()
  {
    if (_verbose)
    {
      print_stage("Create XY to M");
      _timer.restart();
    }
    // Store the indices of matches with corresponding positions
    // $(\mathbf{x}_i,\mathbf{y}_i)$.
    size_t numXY = _XY_mat.cols();
    _XY_to_M.resize(numXY);
    // Allocate memory first.
    for (size_t i = 0; i != numXY; ++i)
      _XY_to_M[i].reserve(_neighborhood_max_size);
    // Loop
    size_t xy_index = 0;
    Vector4f xy(_XY[0].first);
    _XY_to_M[0].push_back(_XY[0].second);
    for (size_t i = 1; i != _XY.size(); ++i)
    {
      if (_XY[i].first != xy)
      {
        ++xy_index;
        xy = _XY[i].first;
      }
      _XY_to_M[xy_index].push_back(_XY[i].second);
    }

    if (_verbose)
    {
      print_stage("Check number of positions for xy");
      size_t num_positions_xy = _XY_to_M.size();
      size_t num_matches_xy = 0;
      for (size_t i = 0; i != _XY_to_M.size(); ++i)
        num_matches_xy += _XY_to_M[i].size();
      cout << "num_positions_xy = " << num_positions_xy << endl;
      cout << "num_matches_xy = " << num_matches_xy << endl;

      _elapsed = _timer.elapsed();
      cout << "Storing list of indices of matches with positions (x_i,y_i)."
           << endl;
      cout << "Time elapsed = " << _elapsed << " seconds" << endl;
    }

    // _drawer->display_images();
    // for (size_t i = 1798; i != _XY_to_M.size(); ++i)
    //{
    //  /*if (_XY_to_M[i].size() == 1)
    //    continue;*/

    //  cout << "\ni = " << i << " ";
    //  cout << "_XY_to_M["<<i<<"].size() = " << _XY_to_M[i].size() << endl;
    //  cout << "_XY_mat.col("<<i<<") = " << _XY_mat.col(i).transpose() << endl;

    //  Rgb8 col(rand()%256, rand()%256, rand()%256);
    //  for (size_t j = 0; j != _XY_to_M[i].size(); ++j)
    //  {
    //    cout << " " << _XY_to_M[i][j] << endl;
    //    cout << _M[_XY_to_M[i][j]] << endl;
    //    _drawer->draw_match(_M[_XY_to_M[i][j]], col);
    //  }
    //  get_key();
    //}
  }

  // ======================================================================== //
  // 4. Build kD-trees.
  void NearestMatchNeighborhoodComputer::build_kdtrees()
  {
    // Construct the kD-trees.
    if (_verbose)
    {
      print_stage("kD-Tree construction");
      _timer.restart();
    }
    _x_index_ptr.reset(new KDTree{_X_mat});
    _y_index_ptr.reset(new KDTree{_Y_mat});
    if (_verbose)
    {
      _elapsed = _timer.elapsed();
      cout << "KD-Tree building time" << endl;
      cout << "Time elapsed = " << _elapsed << " seconds" << endl;
    }

    // Prepare work arrays.
    _x_indices.reserve(2 * _neighborhood_max_size);
    _y_indices.reserve(2 * _neighborhood_max_size);
    _sq_dists.reserve(2 * _neighborhood_max_size);
    _index_scores.reserve(2 * _neighborhood_max_size * 100);
  }

  // ======================================================================== //
  // 5. Compute neighborhoods with kD-tree data structures for efficient
  //    neighbor search.
  vector<vector<size_t>>
  NearestMatchNeighborhoodComputer::compute_neighborhoods(size_t K,
                                                          double squared_rho_min)
  {
    // Preallocate array of match neighborhoods.
    vector<vector<size_t>> N_K(_M.size());
    for (size_t i = 0; i != N_K.size(); ++i)
      N_K[i].reserve(_neighborhood_max_size);

    // Now query the kD-trees.
    if (_verbose)
    {
      print_stage("Querying neighborhoods");
      _timer.restart();
    }

    // Let's go.
    for (size_t i = 0; i != _M.size(); ++i)
    {
      // Match $m_i = (x_i, y_i)$
      const Vector2d xi = _M[i].x_pos().cast<double>();
      const Vector2d yi = _M[i].y_pos().cast<double>();
      const auto Ki = static_cast<int>(K);
      // Collect the indices of $\mathb{x}_j$ such that
      // $|| \mathbf{x}_j - \mathbf{x}_i ||_2 < thres$.
      _x_index_ptr->knn_search(xi, Ki, _x_indices, _sq_dists);
      // Collect the indices of $\mathb{y}_j$ such that
      // $|| \mathbf{y}_j - \mathbf{x}_i ||_2 < thres$.
      _y_index_ptr->knn_search(yi, Ki, _y_indices, _sq_dists);
      // Reset list of neighboring matches.
      _index_scores.clear();
//#define DEBUG
#ifdef DEBUG
      if (_verbose && _drawer)
        _drawer->display_images();
#endif
      get_matches_from_x(_index_scores, i, _x_indices, squared_rho_min);
      get_matches_from_y(_index_scores, i, _y_indices, squared_rho_min);
      keep_best_scale_consistent_matches(N_K[i], _index_scores, 10 * K);
#ifdef DEBUG
      if (_verbose && _drawer)
      {
        cout << "_M[" << i << "] =\n" << _M[i] << endl;
        cout << "N_K[" << i << "].size() = " << N_K[i].size() << endl;
        cout << "index_scores.size() = " << _index_scores.size() << endl;
        for (size_t j = 0; j != _index_scores.size(); ++j)
          cout << j << "\t" << _index_scores[j].first << " "
               << _index_scores[j].second << endl;
        for (size_t j = 0; j != K; ++j)
        {
          _drawer->draw_match(_M[N_K[i][j]]);
          _drawer->draw_point(0, _M[N_K[i][j]].x_pos(), Cyan8, 3);
          _drawer->draw_point(1, _M[N_K[i][j]].y_pos(), Cyan8, 3);
        }
        _drawer->draw_match(_M[i], Cyan8);
        get_key();
      }
#endif
    }

    if (_verbose)
    {
      _elapsed = _timer.elapsed();
      cout << "Match nearest neighbor search." << endl;
      cout << "Time elapsed = " << _elapsed << " seconds" << endl;
    }

    return N_K;
  }

  void NearestMatchNeighborhoodComputer::get_matches_from_x(
      vector<IndexScore>& index_scores, size_t i, const vector<int>& x_indices,
      double squared_rho_min)
  {
    for (size_t j = 0; j != x_indices.size(); ++j)
    {
      size_t ind = x_indices[j];
      for (size_t k = 0; k != _X_to_M[ind].size(); ++k)
      {
        size_t ix = _X_to_M[ind][k];
        double score = rho(_M[i], _M[ix]);
#ifdef DEBUG
        if (_verbose && _drawer)
        {
          _drawer->draw_point(0, Vector2f(_X_mat.col(ind).cast<float>()), Blue8,
                              10);
          // _drawer->draw_match(_M[ix], Blue8);
        }
#endif
        if (score >= squared_rho_min)
          index_scores.push_back(make_pair(ix, score));
      }
    }
  }

  void NearestMatchNeighborhoodComputer::get_matches_from_y(
      vector<IndexScore>& index_scores, size_t i, const vector<int>& y_indices,
      double squared_rho_min)
  {
    for (size_t j = 0; j != y_indices.size(); ++j)
    {
      size_t ind = y_indices[j];
      for (size_t k = 0; k != _Y_to_M[ind].size(); ++k)
      {
        size_t ix = _Y_to_M[ind][k];
        double score = rho(_M[i], _M[ix]);
#ifdef DEBUG
        if (_verbose && _drawer)
        {
          _drawer->draw_point(1, Vector2f(_Y_mat.col(ind).cast<float>()), Blue8,
                              10);
          // _drawer->draw_match(_M[ix], Blue8);
        }
#endif
        if (score >= squared_rho_min)
          index_scores.push_back(make_pair(ix, score));
      }
    }
  }

  // ======================================================================== //
  // 6. (Optional) Compute redundancies before computing the neighborhoods.
  void NearestMatchNeighborhoodComputer::keep_best_scale_consistent_matches(
      vector<size_t>& N_K_i, vector<IndexScore>& index_scores, size_t N)
  {
    // Sort matches by scaled consistency ratio/distrust score.
    sort(index_scores.begin(), index_scores.end(), _compare_index_score_1);
    vector<IndexScore>::iterator it =
        unique(index_scores.begin(), index_scores.end(), _equal_index_score_1);
    index_scores.resize(it - index_scores.begin());
    for (size_t j = 0; j != index_scores.size(); ++j)
      index_scores[j].second = -_M[index_scores[j].first].score();
    sort(index_scores.begin(), index_scores.end(), _compare_index_score_2);
    size_t sz = min(index_scores.size(), N);

    N_K_i.resize(sz);
    for (size_t j = 0; j != sz; ++j)
      N_K_i[j] = index_scores[j].first;
  }

  vector<vector<size_t>>
  NearestMatchNeighborhoodComputer::compute_redundancies(double thres)
  {
    vector<vector<size_t>> redundancies(_M.size());
    for (size_t i = 0; i != redundancies.size(); ++i)
      redundancies[i].reserve(1000);

    // Construct the kD-trees.
    if (_verbose)
    {
      print_stage("kD-Tree construction");
      _timer.restart();
    }
    KDTree xIndex(_X_mat);
    KDTree yIndex(_Y_mat);
    if (_verbose)
    {
      _elapsed = _timer.elapsed();
      cout << "KD-Tree building time" << endl;
      cout << "Time elapsed = " << _elapsed << " seconds" << endl;
    }

    // Now query the kD-trees.
    if (_verbose)
    {
      print_stage("Querying neighborhoods");
      _timer.restart();
    }
    vector<int> x_indices, y_indices;
    vector<double> squared_distances;
    vector<size_t> match_indices;
    x_indices.reserve(_neighborhood_max_size);
    y_indices.reserve(_neighborhood_max_size);
    squared_distances.reserve(_neighborhood_max_size);
    match_indices.reserve(_neighborhood_max_size * 2);

    // Do the dumb way first.
    for (size_t i = 0; i != _M.size(); ++i)
    {
//#define DEBUG
#ifdef DEBUG
      if (_verbose && _drawer)
      {
        cout << i << endl;
        _drawer->display_images();
      }
#endif
      // Match $m_i = (x_i, y_i)$
      Vector2d xi(_M[i].x_pos().cast<double>());
      Vector2d yi(_M[i].y_pos().cast<double>());
#ifdef DEBUG
      if (_verbose && _drawer)
      {
        _drawer->draw_match(_M[i]);
        cout << _M[i] << endl;
        _drawer->draw_point(0, xi.cast<float>(), Red8, 5);
        _drawer->draw_point(1, yi.cast<float>(), Blue8, 5);
      }
#endif

      // Collect the indices of $\mathb{x}_j$ such that
      // $|| \mathbf{x}_j - \mathbf{x}_i ||_2 < thres$.
      xIndex.radius_search(xi, thres * thres, x_indices, squared_distances);
      // Collect the indices of $\mathb{y}_j$ such that
      // $|| \mathbf{y}_j - \mathbf{x}_i ||_2 < thres$.
      yIndex.radius_search(yi, thres * thres, y_indices, squared_distances);

#ifdef DEBUG
      if (_drawer)
      {
        cout << "x_indices.size() = " << x_indices.size() << endl;
        cout << "y_indices.size() = " << y_indices.size() << endl;
        for (size_t j = 0; j < x_indices.size(); ++j)
          _drawer->draw_point(
              0, Vector2f(_X_mat.col(x_indices[j]).cast<float>()), Red8);
        for (size_t j = 0; j < y_indices.size(); ++j)
          _drawer->draw_point(
              1, Vector2f(_Y_mat.col(y_indices[j]).cast<float>()), Blue8);
        get_key();
      }
#endif

      match_indices.clear();
      match_indices.clear();
      // Collect all the matches that have position $\mathbf{x}_j$.
      for (size_t j = 0; j != x_indices.size(); ++j)
      {
        size_t indXj = x_indices[j];  // index of $\mathbf{x}_j$
        for (size_t k = 0; k != _X_to_M[indXj].size(); ++k)
        {
          size_t match_index = _X_to_M[indXj][k];
          match_indices.push_back(match_index);
        }
      }
      // Collect all the matches that have position $\mathbf{y}_j$.
      for (size_t j = 0; j != y_indices.size(); ++j)
      {
        size_t indYj = y_indices[j];  // index of $\mathbf{y}_j$
        for (size_t k = 0; k != _Y_to_M[indYj].size(); ++k)
        {
          size_t match_index = _Y_to_M[indYj][k];
          match_indices.push_back(match_index);
        }
      }
#ifdef DEBUG
      if (_verbose)
        cout << "match_indices = " << match_indices.size() << endl;
#endif

      // Keep only matches $m_j$ such that
      // $|| \mathbf{x}_j - \mathbf{x}_i || < thres$ and
      // $|| \mathbf{y}_j - \mathbf{y}_i || < thres$.
      const Match& mi = _M[i];
      for (size_t j = 0; j != match_indices.size(); ++j)
      {
        size_t indj = match_indices[j];
        const Match& mj = _M[indj];
        if ((mi.x_pos() - mj.x_pos()).squaredNorm() < thres * thres &&
            (mi.y_pos() - mj.y_pos()).squaredNorm() < thres * thres)
        {
          redundancies[i].push_back(indj);
#ifdef DEBUG
          if (_drawer)
            _drawer->draw_match(mj, Cyan8);
#endif
        }
#ifdef DEBUG
        else if (_drawer)
          _drawer->draw_match(mj, Black8);
#endif
      }
#ifdef DEBUG
      if (_verbose)
      {
        cout << "redundancies[i].size() = " << redundancies[i].size() << endl;
        get_key();
      }
#endif
    }
    if (_verbose)
    {
      _elapsed = _timer.elapsed();
      cout << "Nearest neighbor search." << endl;
      cout << "Time elapsed = " << _elapsed << " seconds" << endl;
    }
    return redundancies;
  }

  auto
  NearestMatchNeighborhoodComputer::get_redundancy_components_and_representers(
      vector<vector<size_t>>& /*components*/,  //
      vector<size_t>& /* representers */,      //
      const vector<Match>& /* matches */,      //
      double /* thres */) -> void
  {
    // if (_verbose)
    //  print_stage("Compute match redundancy component");

    // vector<vector<size_t>> redundancies = compute_redundancies(thres);

    // using namespace boost;
    // typedef adjacency_list <vecS, vecS, undirectedS> Graph;

    //// Construct the graph.
    // Graph g(matches.size());
    // for (size_t i = 0; i < redundancies.size(); ++i)
    //  for (vector<int>::size_type j = 0; j < redundancies[i].size(); ++j)
    //    add_edge(i, redundancies[i][j], g);

    //// Compute the connected components.
    // vector<size_t> component(num_vertices(g));
    // size_t num_components = connected_components(g, &component[0]);

    //// Store the components.
    // components = vector<vector<size_t> >(num_components);
    // for (size_t i = 0; i != component.size(); ++i)
    //  components.reserve(100);
    // for (size_t i = 0; i != component.size(); ++i)
    //  components[component[i]].push_back(i);

    //// Store the best representer for each component.
    // representers.resize(num_components);
    // for (size_t i = 0; i != components.size(); ++i)
    //{
    //  size_t index_best_match = components[i][0];
    //  for (size_t j = 0; j < components[i].size(); ++j)
    //  {
    //    size_t index = components[i][j];
    //    if (matches[index_best_match].score() > matches[index].score())
    //      index_best_match = index;
    //  }
    //  representers[i] = index_best_match;
    //}

    // if (_verbose)
    //{
    //  cout << "Checksum components" << endl;
    //  size_t num_matches_from_components = 0;
    //  for (size_t i = 0; i != components.size(); ++i)
    //    num_matches_from_components += components[i].size();
    //  cout << "num_matches_from_components = " << num_matches_from_components
    //       << endl;
    //  cout << "num_components = " << num_components << endl;

    //  if (_drawer)
    //    _drawer->display_images();
    //  for (size_t i = 0; i != components.size(); ++i)
    //  {
    //    for (size_t j = 0; j < components[i].size(); ++j)
    //    {
    //      size_t index = components[i][j];
    //      if (_drawer)
    //        _drawer->draw_match(_M[index]);
    //    }
    //    if (_drawer)
    //    {
    //      _drawer->draw_match(_M[representers[i]], Red8);
    //      cout << "components[" << i << "].size() = " << components[i].size()
    //           << endl;
    //    }
    //    if (_drawer)
    //      get_key();
    //  }
    //  get_key();
    //}
  }

  // ======================================================================== //
  // Symmetrized version of NearestMatchNeighborhoodComputer.
  vector<vector<size_t>> compute_hat_N_K(const vector<vector<size_t>>& N_K)
  {
    print_stage("\\hat{N}_K(m)");
    Timer t;
    double elapsed;

    vector<vector<size_t>> hatNK(N_K.size());
    for (size_t i = 0; i != hatNK.size(); ++i)
      hatNK[i].reserve(N_K[0].size() * 2);
    for (size_t i = 0; i != hatNK.size(); ++i)
    {
      hatNK[i] = N_K[i];
      for (size_t j = 0; j != N_K[i].size(); ++j)
        hatNK[N_K[i][j]].push_back(i);
    }

    for (size_t i = 0; i != hatNK.size(); ++i)
    {
      sort(hatNK[i].begin(), hatNK[i].end());
      vector<size_t>::iterator it = unique(hatNK[i].begin(), hatNK[i].end());
      hatNK[i].resize(it - hatNK[i].begin());
    }

    elapsed = t.elapsed();
    cout << "Computation time = " << elapsed << " seconds" << endl;

    return hatNK;
  }

}  // namespace DO::Sara
