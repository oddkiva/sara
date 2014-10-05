// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifdef _MSC_VER
# pragma warning( disable : 4244 4267 4291 )
#endif

#include <DO/KDTree.hpp>

namespace DO {

  KDTree::KDTree(const MatrixXd& data,
                 const flann::KDTreeIndexParams& indexParams,
                 const flann::SearchParams& searchParams)
    : wrapped_data_(true)
    , data_(const_cast<double *>(data.data()), data.cols(), data.rows())
    , index_(data_, indexParams)
    , index_params_(indexParams)
    , search_params_(searchParams)
  {
    index_.buildIndex();
  }

  KDTree::~KDTree()
  {
    if (!wrapped_data_)
      delete[] data_.ptr();
  }

  // ======================================================================== //
  // k-NN search methods.
  void KDTree::knn_search(const double *query, size_t k,
                          std::vector<int>& indices,
                          std::vector<double>& sqDists)
  {
    flann::Matrix<double> q(const_cast<double *>(query), 1, data_.cols);
    std::vector<std::vector<int> > _indices;
    std::vector<std::vector<double> > _sqDists;
    index_.knnSearch(q, _indices, _sqDists, k, search_params_);
    indices = _indices[0];
    sqDists = _sqDists[0];
  }

  void KDTree::knn_search(const MatrixXd& queries, size_t k,
                          std::vector<std::vector<int> >& indices,
                          std::vector<std::vector<double> >& sqDists,
                          bool remove1NN)
  {
    if (queries.rows() != data_.cols)
    {
      std::string errorMsg("queries.rows() != data_.cols");
      std::cerr << errorMsg << std::endl;
      throw std::runtime_error(errorMsg);
    }
    flann::Matrix<double> q(const_cast<double *>(queries.data()),
                            queries.cols(), queries.rows());
    if (remove1NN)
    {
      index_.knnSearch(q, indices, sqDists, k+1, search_params_);
      indices.erase(indices.begin());
      sqDists.erase(sqDists.begin());
    }
    else
      index_.knnSearch(q, indices, sqDists, k, search_params_);
  }

  void KDTree::knn_search(const MatrixXd& queries, size_t k,
                          std::vector<std::vector<int> >& indices,
                          bool remove1NN)
  {
    std::vector<std::vector<double> > sqDists;
    knn_search(queries, k, indices, sqDists, remove1NN);
  }

  void KDTree::knn_search(size_t i, size_t k,
                          std::vector<int>& indices, std::vector<double>& sqDists)
  {
    knn_search(data_[i], k+1, indices, sqDists);
    indices.erase(indices.begin());
    sqDists.erase(sqDists.begin());
  }

  void KDTree::knn_search(size_t i, size_t k, std::vector<int>& indices)
  {
    std::vector<double> sqDists;
    knn_search(i, k, indices, sqDists);
  }

  void KDTree::knn_search(const std::vector<size_t>& queries, size_t k,
                          std::vector<std::vector<int> >& indices,
                          std::vector<std::vector<double> >& sqDists)
  {
    MatrixXd Q(data_.cols, queries.size());
    for (size_t i = 0; i != queries.size(); ++i)
      for (size_t j = 0; j != data_.cols; ++j)
        Q(j,i) = data_[i][j];
    knn_search(Q, k+1, indices, sqDists);
    for (size_t i = 0; i != indices.size(); ++i)
    {
      indices[i].erase(indices[i].begin());
      sqDists[i].erase(sqDists[i].begin());
    }
  }

  void KDTree::knn_search(const std::vector<size_t>& queries, size_t k,
                          std::vector<std::vector<int> >& indices)
  {
    std::vector<std::vector<double> > sqDists;
    knn_search(queries, k+1, indices, sqDists);
  }

  // ======================================================================== //
  // Radius search methods.
  int KDTree::radius_search(const double *query, double sqSearchRadius,
                            std::vector<int>& indices,
                            std::vector<double>& sqDists)
  {   
    flann::Matrix<double> q(const_cast<double *>(query), 1, data_.cols);
    flann::Matrix<int> i(&indices[0], 1, indices.size());
    flann::Matrix<double> s(&sqDists[0], 1, sqDists.size());
    index_.radiusSearch(q, i, s, static_cast<float>(sqSearchRadius),
                        search_params_);
    return indices.size();
  }

  //! Basic radius search wrapped function.
  void KDTree::radius_search(const MatrixXd& queries, double sqSearchRadius,
                             std::vector<std::vector<int> >& indices,
                             std::vector<std::vector<double> >& sqDists,
                             bool remove1NN)
  {
    if (queries.rows() != data_.cols)
    {
      std::string errorMsg("queries.rows() != data_.cols");
      std::cerr << errorMsg << std::endl;
      throw std::runtime_error(errorMsg);
    }
    flann::Matrix<double> q(const_cast<double *>(queries.data()),
                            queries.cols(), queries.rows());
    if (remove1NN)
    {
      index_.radiusSearch(q, indices, sqDists,
                          static_cast<float>(sqSearchRadius), search_params_);
      indices.erase(indices.begin());
      sqDists.erase(sqDists.begin());
    }
    else
      index_.radiusSearch(q, indices, sqDists,
                           static_cast<float>(sqSearchRadius), search_params_);
  }

  int KDTree::radius_search(size_t i, double sqSearchRadius,
                           std::vector<int>& indices, std::vector<double>& sqDists)
  {
    radius_search(data_[i], sqSearchRadius, indices, sqDists);
    indices.erase(indices.begin());
    sqDists.erase(sqDists.begin());
    return indices.size();
  }

  int KDTree::radius_search(size_t i, double sqSearchRadius, std::vector<int>& indices)
  {
    std::vector<double> sqDists;
    return radius_search(i, sqSearchRadius, indices, sqDists);
  }

  void KDTree::radius_search(const std::vector<size_t>& queries,
                             double sqSearchRadius,
                             std::vector<std::vector<int> >& indices,
                             std::vector<std::vector<double> >& sqDists)
  {
    MatrixXd Q(data_.cols, queries.size());
    for (size_t i = 0; i != queries.size(); ++i)
      for (size_t j = 0; j != data_.cols; ++j)
        Q(j,i) = data_[i][j];
    radius_search(Q, sqSearchRadius, indices, sqDists);
    for (size_t i = 0; i != indices.size(); ++i)
    {
      indices[i].erase(indices[i].begin());
      sqDists[i].erase(sqDists[i].begin());
    }
  }

  void KDTree::radius_search(const std::vector<size_t>& queries,
                             double sqSearchRadius,
                             std::vector<std::vector<int> >& indices)
  {
    std::vector<std::vector<double> > sqDists;
    radius_search(queries, sqSearchRadius, indices, sqDists);
  }

}