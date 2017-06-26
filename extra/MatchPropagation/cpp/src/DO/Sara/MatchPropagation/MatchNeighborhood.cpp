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

#include <DO/Sara/Graphics.hpp>

#include "MatchNeighborhood.hpp"

#ifdef _OPENMP
# include <omp.h>
#endif


using namespace std;

namespace DO { namespace Sara {

  inline void ellRadii(float& a, float& b, const Matrix2f& M)
  {
    JacobiSVD<Matrix2f> svd(M, ComputeFullU);
    const Vector2f& D = svd.singularValues();
    const Vector2f radii(D.cwiseSqrt().cwiseInverse());
    b = radii(0);
    a = radii(1);
  }

  float sqIsoRadius(const Matrix2f& M)
  {
    return 1.f/sqrt(M.determinant());
  }

  ComputeN_K::
  ComputeN_K(const vector<Match>& matches,
             size_t neighborhoodMaxSize,
             const PairWiseDrawer *pDrawer,
             bool verbose)
    : M_(matches)
    , verbose_(verbose), drawer_ptr_(pDrawer)
    , neighborhood_max_size_(neighborhoodMaxSize)
    , x_index_ptr_(0)
    , y_index_ptr_(0)
  {
    createAndSortXY();
    createXMat();
    createYMat();
    createXYMat();
    createXToM();
    createYToM();
    createXYToM();
    buildKDTrees();
  }

  ComputeN_K::~ComputeN_K()
  {
    if (x_index_ptr_)
      delete x_index_ptr_;
    if (y_index_ptr_)
      delete y_index_ptr_;
  }

  vector<size_t>
  ComputeN_K::operator()(size_t i, size_t K, double squaredRhoMin)
  {
    vector<size_t> N_K_i;
    N_K_i.reserve(neighborhood_max_size_);

    // Match $m_i = (x_i, y_i)$
    Vector2d xi(M_[i].posX().cast<double>());
    Vector2d yi(M_[i].posY().cast<double>());
    // Collect the K nearest matches $\mathb{x}_j$ to $\mathb{x}_i$.
    x_index_ptr_->knnSearch(xi, K, x_indices_, sq_dists_, true);
    // Collect the K nearest matches $\mathb{y}_j$ to $\mathb{y}_i$
    y_index_ptr_->knnSearch(yi, K, y_indices_, sq_dists_, true);


    // Reset list of neighboring matches.
    index_scores_.clear();
    //#define DEBUG
#ifdef DEBUG
    if (verbose_ && drawer_ptr_)
      drawer_ptr_->displayImages();
#endif
    getMatchesFromX(index_scores_, i, x_indices_, squaredRhoMin);
    getMatchesFromY(index_scores_, i, y_indices_, squaredRhoMin);
    keepBestScaleConsistentMatches(N_K_i, index_scores_, 10*K);
#ifdef DEBUG
    if (verbose_ && drawer_ptr_)
    {
      cout << "M_["<<i<<"] =\n" << M_[i] << endl;
      cout << "N_K["<<i<<"].size() = " << N_K_i.size() << endl;
      cout << "indexScores.size() = " << index_scores_.size() << endl;
      for (size_t j = 0; j != index_scores_.size(); ++j)
        cout << j << "\t" << index_scores_[j].first << " " << index_scores_[j].second << endl;
      for (size_t j = 0; j != K; ++j)
      {
        drawer_ptr_->drawMatch(M_[N_K_i[j]]);
        drawer_ptr_->drawPoint(0, M_[N_K_i[j]].posX(), Cyan8, 3);
        drawer_ptr_->drawPoint(1, M_[N_K_i[j]].posY(), Cyan8, 3);
      }
      drawer_ptr_->drawMatch(M_[i], Cyan8);
      getKey();
    }
#endif
    return N_K_i;
  }

  vector<vector<size_t> >
  ComputeN_K::operator()(const vector<size_t>& indices,
                         size_t K, double squaredRhoMin)
  {
    vector<vector<size_t> > N_K_indices;
    N_K_indices.resize(indices.size());

    vector<vector<IndexScore> > indexScores;
    vector<vector<int> > xIndices, yIndices;
    vector<vector<double> > sqDists;
    indexScores.resize(indices.size());
    xIndices.resize(indices.size());
    yIndices.resize(indices.size());
    sqDists.resize(indices.size());

//#define PARALLEL_QUERY
#ifdef PARALLEL_QUERY
    cout << "Parallel neighborhood query" << endl;
    // Match $m_i = (x_i, y_i)$.
    MatrixXd xis(2, indices.size());
    MatrixXd yis(2, indices.size());
    for (size_t i = 0; i != indices.size(); ++i)
    {
      xis.col(i) = M_[indices[i]].posX().cast<double>();
      yis.col(i) = M_[indices[i]].posY().cast<double>();
    }
    // Collect the K nearest matches $\mathb{x}_j$ to $\mathb{x}_i$.
    x_index_ptr_->knnSearch(xis, K, xIndices, sqDists, true);
    // Collect the K nearest matches $\mathb{y}_j$ to $\mathb{y}_i$
    y_index_ptr_->knnSearch(yis, K, yIndices, sqDists, true);
#else
    for (size_t i = 0; i != indices.size(); ++i)
    {
      Vector2d xi(M_[indices[i]].posX().cast<double>());
      Vector2d yi(M_[indices[i]].posY().cast<double>());
      x_index_ptr_->knnSearch(xi, K, xIndices[i], sqDists[0], true);
      // Collect the K nearest matches $\mathb{y}_j$ to $\mathb{y}_i$
      y_index_ptr_->knnSearch(yi, K, yIndices[i], sqDists[0], true);
    }
#endif

    int num_indices = static_cast<int>(indices.size());
#ifdef _OPENMP
# pragma omp parallel for
#endif
    for (int i = 0; i < num_indices; ++i)
    {
      getMatchesFromX(indexScores[i], indices[i], xIndices[i], squaredRhoMin);
      getMatchesFromY(indexScores[i], indices[i], yIndices[i], squaredRhoMin);
      keepBestScaleConsistentMatches(N_K_indices[i], indexScores[i], 10*K);
    }

    return N_K_indices;
  }

  // ======================================================================== //
  // 1. Sort matches by positions and match positions.
  void ComputeN_K::createAndSortXY()
  {
    if (verbose_)
      timer_.restart();
    // Construct the set of matches ordered by position x and y,
    // and by position matches (x,y).
    X_.resize(M_.size());
    Y_.resize(M_.size());
    XY_.resize(M_.size());
    for (size_t i = 0; i != M_.size(); ++i)
    {
      X_[i] = make_pair(M_[i].posX(), i);
      Y_[i] = make_pair(M_[i].posY(), i);
      XY_[i].first << M_[i].posX(), M_[i].posY();
      XY_[i].second = i;
    }
    // Sort the set of matches.
    CompareByPos compareByPos;
    CompareByXY compareByXY;
    sort(X_.begin(), X_.end(), compareByPos);
    sort(Y_.begin(), Y_.end(), compareByPos);
    sort(XY_.begin(), XY_.end(), compareByXY);
    if (verbose_)
    {
      elapsed_ = timer_.elapsed();
      printStage("Storing matrices of unique positions and match positions");
      cout << "Time elapsed = " << elapsed_ << " seconds" << endl;
    }
  }

  // ======================================================================== //
  // 2. Create data matrix for nearest neighbor search.
  size_t
  ComputeN_K::
  countUniquePositions(const vector<PosIndex>& X)
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
    if(verbose_)
    {
      cout << "Check number of unique positions x" << endl;
      cout << "numPosX = " << numPosX << endl;
      cout << "X.size() = " << X.size() << endl;
      cout << "M_.size() = " << M_.size() << endl;
    }
    return numPosX;
  }

  size_t ComputeN_K::countUniquePosMatches()
  {
    if (verbose_)
      timer_.restart();
    // Count the number of unique positions $(\mathbf{x}_i, \mathbf{y}_i)$.
    size_t numPosXY = 1;
    Vector4f match = XY_[0].first;
    for (size_t i = 1; i != XY_.size(); ++i)
    {
      if (XY_[i].first != match)
      {
        ++numPosXY;
        match = XY_[i].first;
      }
    }
    if (verbose_)
    {
      cout << "Check number of unique position match (x,y)" << endl;;
      cout << "numPosXY = " << numPosXY << endl;
      cout << "matchesByXY.size() = " << XY_.size() << endl;
      cout << "matches.size() = " << M_.size() << endl;
      elapsed_ = timer_.elapsed();
      cout << "Counting number of unique positions (x,y)." << endl;
      cout << "Time elapsed = " << elapsed_ << " seconds" << endl;
    }
    return numPosXY;
  }

  MatrixXd
  ComputeN_K::
  createPosMat(const vector<PosIndex>& X)
  {
    size_t numPosX = countUniquePositions(X);
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
    if (verbose_)
    {
      cout << "Stacking position vectors." << endl;
      cout << "xInd = " << xInd << endl;
    }
    return xMat;
  }

  void ComputeN_K::createXMat()
  {
    if (verbose_)
      printStage("X matrix");
    X_mat_ = createPosMat(X_);
    if (verbose_ && drawer_ptr_)
    {
      for (size_t i = 0; i != X_mat_.cols(); ++i)
      {
        Vector2f xi(X_mat_.col(i).cast<float>());
        //drawer_ptr_->drawPoint(0, xi, Red8, 5);
      }
    }
  }

  void ComputeN_K::createYMat()
  {
    if (verbose_)
      printStage("Y matrix");

    Y_mat_ = createPosMat(Y_);

    if (verbose_ && drawer_ptr_)
    {
      for (size_t i = 0; i != Y_mat_.cols(); ++i)
      {
        Vector2f yi(Y_mat_.col(i).cast<float>());
        //drawer_ptr_->drawPoint(1, yi, Blue8, 5);
      }
    }
  }

  void ComputeN_K::createXYMat()
  {
    if (verbose_)
      printStage("XY matrix");
    // Store the matrix of position matches $(\mathbf{x}_i, \mathbf{y}_i)$
    // without duplicate.
    size_t numPosXY = countUniquePosMatches();
    XY_mat_ = MatrixXd(4, numPosXY);
    size_t xyInd = 0;
    Vector4f match(XY_[0].first);
    XY_mat_.col(0) = match.cast<double>();
    for (size_t i = 1; i != XY_.size(); ++i)
    {
      if (XY_[i].first != match)
      {
        ++xyInd;
        match = XY_[i].first;
        XY_mat_.col(xyInd) = match.cast<double>();
      }
    }
    if (verbose_ && drawer_ptr_)
    {
      cout << "xyInd = " << xyInd << endl;
      /*for (size_t i = 0; i != XY_mat_.cols(); ++i)
      {
        Vector2f xi(XY_mat_.block(0,i,2,1).cast<float>());
        drawer_ptr_->drawPoint(0, xi, Magenta8, 5);
        Vector2f yi(XY_mat_.block(2,i,2,1).cast<float>());
        drawer_ptr_->drawPoint(1, yi, Magenta8, 5);
      }*/
    }
  }

  vector<vector<size_t> >
  ComputeN_K::
  createPosToMatchTable(const vector<PosIndex>& X, const MatrixXd& xMat)
  {
    // Store the indices of matches $(\mathbf{x}_i, .)$.
    size_t numPosX = xMat.cols();
    vector<vector<size_t> > xToM(numPosX);
    // Allocate memory first.
    for (size_t i = 0; i != numPosX; ++i)
      xToM[i].reserve(neighborhood_max_size_);
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
      xToM[xInd].push_back( X[i].second );
    }
    if (verbose_)
    {
      printStage("Check number of positions for x");
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
  void ComputeN_K::createXToM()
  {
    if (verbose_)
    {
      printStage("Create X to M");
      timer_.restart();
    }
    X_to_M_ = createPosToMatchTable(X_, X_mat_);
    if (verbose_)
    {
      size_t num_positions_x = X_to_M_.size();
      size_t num_matches_x = 0;
      for (size_t i = 0; i != X_to_M_.size(); ++i)
        num_matches_x += X_to_M_[i].size();
      cout << "num_positions_x = " << num_positions_x << endl;
      cout << "num_matches_x = " << num_matches_x << endl;
    }
  }

  void ComputeN_K::createYToM()
  {
    if (verbose_)
    {
      printStage("Create Y to M");
      timer_.restart();
    }
    Y_to_M_ = createPosToMatchTable(Y_, Y_mat_);
    if (verbose_)
    {
      printStage("Check number of positions for y");
      size_t num_positions_y = Y_to_M_.size();
      size_t num_matches_y = 0;
      for (size_t i = 0; i != Y_to_M_.size(); ++i)
        num_matches_y += Y_to_M_[i].size();
      cout << "num_positions_y = " << num_positions_y << endl;
      cout << "num_matches_y = " << num_matches_y << endl;
    }
  }

  void ComputeN_K::createXYToM()
  {
    if (verbose_)
    {
      printStage("Create XY to M");
      timer_.restart();
    }
    // Store the indices of matches with corresponding positions
    // $(\mathbf{x}_i,\mathbf{y}_i)$.
    size_t numXY = XY_mat_.cols();
    XY_to_M_.resize(numXY);
    // Allocate memory first.
    for (size_t i = 0; i != numXY; ++i)
      XY_to_M_[i].reserve(neighborhood_max_size_);
    // Loop
    size_t xyInd = 0;
    Vector4f xy(XY_[0].first);
    XY_to_M_[0].push_back(XY_[0].second);
    for (size_t i = 1; i != XY_.size(); ++i)
    {
      if (XY_[i].first != xy)
      {
        ++xyInd;
        xy = XY_[i].first;
      }
      XY_to_M_[xyInd].push_back( XY_[i].second );
    }
    if (verbose_)
    {
      printStage("Check number of positions for xy");
      size_t num_positions_xy = XY_to_M_.size();
      size_t num_matches_xy = 0;
      for (size_t i = 0; i != XY_to_M_.size(); ++i)
        num_matches_xy += XY_to_M_[i].size();
      cout << "num_positions_xy = " << num_positions_xy << endl;
      cout << "num_matches_xy = " << num_matches_xy << endl;

      elapsed_ = timer_.elapsed();
      cout << "Storing list of indices of matches with positions (x_i,y_i)." << endl;
      cout << "Time elapsed = " << elapsed_ << " seconds" << endl;
    }

    //drawer_ptr_->displayImages();
    //for (size_t i = 1798; i != XY_to_M_.size(); ++i)
    //{
    //  /*if (XY_to_M_[i].size() == 1)
    //    continue;*/

    //  cout << "\ni = " << i << " ";
    //  cout << "XY_to_M_["<<i<<"].size() = " << XY_to_M_[i].size() << endl;
    //  cout << "XY_mat_.col("<<i<<") = " << XY_mat_.col(i).transpose() << endl;

    //  Rgb8 col(rand()%256, rand()%256, rand()%256);
    //  for (size_t j = 0; j != XY_to_M_[i].size(); ++j)
    //  {
    //    cout << " " << XY_to_M_[i][j] << endl;
    //    cout << M_[XY_to_M_[i][j]] << endl;
    //    drawer_ptr_->drawMatch(M_[XY_to_M_[i][j]], col);
    //  }
    //  getKey();
    //}
  }

  // ======================================================================== //
  // 4. Build kD-trees.
  void ComputeN_K::buildKDTrees()
  {
    // Construct the kD-trees.
    if (verbose_)
    {
      printStage("kD-Tree construction");
      timer_.restart();
    }
    x_index_ptr_ = new KDTree(X_mat_);
    y_index_ptr_ = new KDTree(Y_mat_);
    if (verbose_)
    {
      elapsed_ = timer_.elapsed();
      cout << "KD-Tree building time" << endl;
      cout << "Time elapsed = " << elapsed_ << " seconds" << endl;
    }

    // Prepare work arrays.
    x_indices_.reserve(2*neighborhood_max_size_);
    y_indices_.reserve(2*neighborhood_max_size_);
    sq_dists_.reserve(2*neighborhood_max_size_);
    index_scores_.reserve(2*neighborhood_max_size_*100);
  }

  // ======================================================================== //
  // 5. Compute neighborhoods with kD-tree data structures for efficient
  //    neighbor search.
  vector<vector<size_t> >
  ComputeN_K::computeNeighborhoods(size_t K, double squaredRhoMin)
  {
    // Preallocate array of match neighborhoods.
    vector<vector<size_t> > N_K(M_.size());
    for (size_t i = 0; i != N_K.size(); ++i)
      N_K[i].reserve(neighborhood_max_size_);

    // Now query the kD-trees.
    if (verbose_)
    {
      printStage("Querying neighborhoods");
      timer_.restart();
    }

    // Let's go.
    for (size_t i = 0; i != M_.size(); ++i)
    {
      // Match $m_i = (x_i, y_i)$
      Vector2d xi(M_[i].posX().cast<double>());
      Vector2d yi(M_[i].posY().cast<double>());
      // Collect the indices of $\mathb{x}_j$ such that
      // $|| \mathbf{x}_j - \mathbf{x}_i ||_2 < thres$.
      x_index_ptr_->knnSearch(xi, K, x_indices_, sq_dists_);
      // Collect the indices of $\mathb{y}_j$ such that
      // $|| \mathbf{y}_j - \mathbf{x}_i ||_2 < thres$.
      y_index_ptr_->knnSearch(yi, K, y_indices_, sq_dists_);
      // Reset list of neighboring matches.
      index_scores_.clear();
//#define DEBUG
#ifdef DEBUG
      if (verbose_ && drawer_ptr_)
        drawer_ptr_->displayImages();
#endif
      getMatchesFromX(index_scores_, i, x_indices_, squaredRhoMin);
      getMatchesFromY(index_scores_, i, y_indices_, squaredRhoMin);
      keepBestScaleConsistentMatches(N_K[i], index_scores_, 10*K);
#ifdef DEBUG
      if (verbose_ && drawer_ptr_)
      {
        cout << "M_["<<i<<"] =\n" << M_[i] << endl;
        cout << "N_K["<<i<<"].size() = " << N_K[i].size() << endl;
        cout << "indexScores.size() = " << index_scores_.size() << endl;
        for (size_t j = 0; j != index_scores_.size(); ++j)
          cout << j << "\t" << index_scores_[j].first << " " << index_scores_[j].second << endl;
        for (size_t j = 0; j != K; ++j)
        {
          drawer_ptr_->drawMatch(M_[N_K[i][j]]);
          drawer_ptr_->drawPoint(0, M_[N_K[i][j]].posX(), Cyan8, 3);
          drawer_ptr_->drawPoint(1, M_[N_K[i][j]].posY(), Cyan8, 3);
        }
        drawer_ptr_->drawMatch(M_[i], Cyan8);
        getKey();
      }
#endif
    }

    if (verbose_)
    {
      elapsed_ = timer_.elapsed();
      cout << "Match nearest neighbor search." << endl;
      cout << "Time elapsed = " << elapsed_ << " seconds" << endl;
    }

    return N_K;
  }

  void ComputeN_K::getMatchesFromX(vector<IndexScore>& indexScores,
                                   size_t i,
                                   const vector<int>& xIndices,
                                   double squaredRhoMin)
  {
    for (size_t j = 0; j != xIndices.size(); ++j)
    {
      size_t ind = xIndices[j];
      for (size_t k = 0; k != X_to_M_[ind].size(); ++k)
      {
        size_t ix = X_to_M_[ind][k];
        double score = rho(M_[i], M_[ix]);
#ifdef DEBUG
        if (verbose_ && drawer_ptr_)
        {
          drawer_ptr_->drawPoint(0, Vector2f(X_mat_.col(ind).cast<float>()), Blue8, 10);
          //drawer_ptr_->drawMatch(M_[ix], Blue8);
        }
#endif
        if (score >= squaredRhoMin)
          indexScores.push_back(make_pair(ix, score));
      }
    }
  }

  void ComputeN_K::getMatchesFromY(vector<IndexScore>& indexScores,
                                   size_t i,
                                   const vector<int>& yIndices,
                                   double squaredRhoMin)
  {
    for (size_t j = 0; j != yIndices.size(); ++j)
    {
      size_t ind = yIndices[j];
      for (size_t k = 0; k != Y_to_M_[ind].size(); ++k)
      {
        size_t ix = Y_to_M_[ind][k];
        double score = rho(M_[i], M_[ix]);
#ifdef DEBUG
        if (verbose_ && drawer_ptr_)
        {
          drawer_ptr_->drawPoint(1, Vector2f(Y_mat_.col(ind).cast<float>()), Blue8, 10);
          //drawer_ptr_->drawMatch(M_[ix], Blue8);
        }
#endif
        if (score >= squaredRhoMin)
          indexScores.push_back(make_pair(ix, score));
      }
    }
  }

  // ======================================================================== //
  // 6. (Optional) Compute redundancies before computing the neighborhoods.
  void
  ComputeN_K::
  keepBestScaleConsistentMatches(vector<size_t>& N_K_i,
                                  vector<IndexScore>& indexScores,
                                  size_t N)
  {
    // Sort matches by scaled consistency ratio/distrust score.
    sort(indexScores.begin(), indexScores.end(), compare_index_score_1_);
    vector<IndexScore>::iterator it =
      unique(indexScores.begin(), indexScores.end(), equal_index_score_1_);
    indexScores.resize(it - indexScores.begin());
    for (size_t j = 0; j != indexScores.size(); ++j)
      indexScores[j].second = -M_[indexScores[j].first].score();
    sort(indexScores.begin(), indexScores.end(), compare_index_score_2_);
    size_t sz = min(indexScores.size(), N);

    N_K_i.resize(sz);
    for (size_t j = 0; j != sz; ++j)
      N_K_i[j] = indexScores[j].first;
  }

  vector<vector<size_t> >
  ComputeN_K::computeRedundancies(double thres)
  {
    vector<vector<size_t> > redundancies(M_.size());
    for (size_t i = 0; i != redundancies.size(); ++i)
      redundancies[i].reserve(1000);

    // Construct the kD-trees.
    if (verbose_)
    {
      printStage("kD-Tree construction");
      timer_.restart();
    }
    KDTree xIndex(X_mat_);
    KDTree yIndex(Y_mat_);
    if (verbose_)
    {
      elapsed_ = timer_.elapsed();
      cout << "KD-Tree building time" << endl;
      cout << "Time elapsed = " << elapsed_ << " seconds" << endl;
    }

    // Now query the kD-trees.
    if (verbose_)
    {
      printStage("Querying neighborhoods");
      timer_.restart();
    }
    vector<int> xIndices, yIndices;
    vector<double> sqDists;
    vector<size_t> matchIndices;
    xIndices.reserve(neighborhood_max_size_);
    yIndices.reserve(neighborhood_max_size_);
    sqDists.reserve(neighborhood_max_size_);
    matchIndices.reserve(neighborhood_max_size_*2);

    // Do the dumb way first.
    for (size_t i = 0; i != M_.size(); ++i)
    {
//#define DEBUG
#ifdef DEBUG
      if (verbose_ && drawer_ptr_)
      {
        cout << i << endl;
        drawer_ptr_->displayImages();
      }
#endif
      // Match $m_i = (x_i, y_i)$
      Vector2d xi(M_[i].posX().cast<double>());
      Vector2d yi(M_[i].posY().cast<double>());
#ifdef DEBUG
      if (verbose_ && drawer_ptr_)
      {
        drawer_ptr_->drawMatch(M_[i]);
        cout << M_[i] << endl;
        drawer_ptr_->drawPoint(0, xi.cast<float>(), Red8, 5);
        drawer_ptr_->drawPoint(1, yi.cast<float>(), Blue8, 5);
      }
#endif

      // Collect the indices of $\mathb{x}_j$ such that
      // $|| \mathbf{x}_j - \mathbf{x}_i ||_2 < thres$.
      xIndex.radiusSearch(xi, thres*thres, xIndices, sqDists);
      // Collect the indices of $\mathb{y}_j$ such that
      // $|| \mathbf{y}_j - \mathbf{x}_i ||_2 < thres$.
      yIndex.radiusSearch(yi, thres*thres, yIndices, sqDists);

#ifdef DEBUG
      if (drawer_ptr_)
      {
        cout << "xIndices.size() = " << xIndices.size() << endl;
        cout << "yIndices.size() = " << yIndices.size() << endl;
        for (size_t j = 0; j < xIndices.size(); ++j)
          drawer_ptr_->drawPoint(0, Vector2f(X_mat_.col(xIndices[j]).cast<float>()), Red8);
        for (size_t j = 0; j < yIndices.size(); ++j)
          drawer_ptr_->drawPoint(1, Vector2f(Y_mat_.col(yIndices[j]).cast<float>()), Blue8);
        getKey();
      }
#endif

      matchIndices.clear();
      matchIndices.clear();
      // Collect all the matches that have position $\mathbf{x}_j$.
      for (size_t j = 0; j != xIndices.size(); ++j)
      {
        size_t indXj = xIndices[j]; // index of $\mathbf{x}_j$
        for (size_t k = 0; k!= X_to_M_[indXj].size(); ++k)
        {
          size_t matchInd = X_to_M_[indXj][k];
          matchIndices.push_back(matchInd);
        }
      }
      // Collect all the matches that have position $\mathbf{y}_j$.
      for (size_t j = 0; j != yIndices.size(); ++j)
      {
        size_t indYj = yIndices[j]; // index of $\mathbf{y}_j$
        for (size_t k = 0; k!= Y_to_M_[indYj].size(); ++k)
        {
          size_t matchInd = Y_to_M_[indYj][k];
          matchIndices.push_back(matchInd);
        }
      }
#ifdef DEBUG
      if (verbose_)
        cout << "matchIndices = " << matchIndices.size() << endl;
#endif

      // Keep only matches $m_j$ such that
      // $|| \mathbf{x}_j - \mathbf{x}_i || < thres$ and
      // $|| \mathbf{y}_j - \mathbf{y}_i || < thres$.
      const Match& mi = M_[i];
      for (size_t j = 0; j != matchIndices.size(); ++j)
      {
        size_t indj = matchIndices[j];
        const Match& mj = M_[indj];
        if ( (mi.posX() - mj.posX()).squaredNorm() < thres*thres &&
             (mi.posY() - mj.posY()).squaredNorm() < thres*thres )
        {
          redundancies[i].push_back(indj);
#ifdef DEBUG
          if (drawer_ptr_)
            drawer_ptr_->drawMatch(mj, Cyan8);
#endif
        }
#ifdef DEBUG
        else if (drawer_ptr_)
          drawer_ptr_->drawMatch(mj, Black8);
#endif
      }
#ifdef DEBUG
      if (verbose_)
      {
        cout << "redundancies[i].size() = " << redundancies[i].size() << endl;
        getKey();
      }
#endif
    }
    if (verbose_)
    {
      elapsed_ = timer_.elapsed();
      cout << "Nearest neighbor search." << endl;
      cout << "Time elapsed = " << elapsed_ << " seconds" << endl;
    }
    return redundancies;
  }

  /*
  void
  ComputeN_K::
  getRedundancyComponentsAndRepresenters(vector<vector<size_t> >& components,
                                         vector<size_t>& representers,
                                         const vector<Match>& matches,
                                         double thres)
  {
    if (verbose_)
      printStage("Compute match redundancy component");

    vector<vector<size_t> > redundancies = computeRedundancies(thres);

    using namespace boost;
    typedef adjacency_list <vecS, vecS, undirectedS> Graph;

    // Construct the graph.
    Graph g(matches.size());
    for (size_t i = 0; i < redundancies.size(); ++i)
      for (vector<int>::size_type j = 0; j < redundancies[i].size(); ++j)
        add_edge(i, redundancies[i][j], g);

    // Compute the connected components.
    vector<size_t> component(num_vertices(g));
    size_t num_components = connected_components(g, &component[0]);

    // Store the components.
    components = vector<vector<size_t> >(num_components);
    for (size_t i = 0; i != component.size(); ++i)
      components.reserve(100);
    for (size_t i = 0; i != component.size(); ++i)
      components[component[i]].push_back(i);

    // Store the best representer for each component.
    representers.resize(num_components);
    for (size_t i = 0; i != components.size(); ++i)
    {
      size_t index_best_match = components[i][0];
      for (size_t j = 0; j < components[i].size(); ++j)
      {
        size_t index = components[i][j];
        if (matches[index_best_match].score() > matches[index].score())
          index_best_match = index;
      }
      representers[i] = index_best_match;
    }

    if (verbose_)
    {
      cout << "Checksum components" << endl;
      size_t num_matches_from_components = 0;
      for (size_t i = 0; i != components.size(); ++i)
        num_matches_from_components += components[i].size();
      cout << "num_matches_from_components = " << num_matches_from_components << endl;
      cout << "num_components = " << num_components << endl;

      if (drawer_ptr_)
        drawer_ptr_->displayImages();
      for (size_t i = 0; i != components.size(); ++i)
      {
        for (size_t j = 0; j < components[i].size(); ++j)
        {
          size_t index = components[i][j];
          if (drawer_ptr_)
            drawer_ptr_->drawMatch(M_[index]);
        }
        if (drawer_ptr_)
        {
          drawer_ptr_->drawMatch(M_[representers[i]], Red8);
          cout << "components["<<i<<"].size() = " << components[i].size() << endl;
        }
        if (drawer_ptr_)
          getKey();
      }
      getKey();
    }
  }
  */

  // ======================================================================== //
  // Symmetrized version of ComputeN_K.
  vector<vector<size_t> > computeHatN_K(const vector<vector<size_t> >& N_K)
  {
    printStage("\\hat{N}_K(m)");
    Timer t;
    double elapsed;

    vector<vector<size_t> > hatNK(N_K.size());
    for (size_t i = 0; i != hatNK.size(); ++i)
      hatNK[i].reserve(N_K[0].size()*2);
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
      hatNK[i].resize(it-hatNK[i].begin());
    }

    elapsed = t.elapsed();
    cout << "Computation time = " << elapsed << " seconds" << endl;

    return hatNK;
  }

} /* namespace Sara */
} /* namespace DO */
