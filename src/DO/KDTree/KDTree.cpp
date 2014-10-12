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


using namespace std;


namespace DO {

  KDTree::KDTree(const MatrixXd& data_matrix,
                 const flann::KDTreeIndexParams& index_params,
                 const flann::SearchParams& search_params)
    : _has_data_ownership(true)
    , _row_major_data_matrix(const_cast<double *>(data_matrix.data()),
                             data_matrix.cols(),
                             data_matrix.rows())
    , _index(_row_major_data_matrix, index_params)
    , _index_params(index_params)
    , _search_params(search_params)
  {
    _index.buildIndex();
  }

  KDTree::~KDTree()
  {
    if (!_has_data_ownership)
      delete[] _row_major_data_matrix.ptr();
  }


  // ======================================================================== //
  // k-NN search methods.
  void KDTree::knn_search(const double *query_vector,
                          size_t num_nearest_neighbors,
                          vector<int>& nn_indices,
                          vector<double>& nn_squared_distances)
  {
    flann::Matrix<double> query_row_vectors(
      const_cast<double *>(query_vector), 1, _row_major_data_matrix.cols);
    vector<vector<int> > temp_nn_indices;
    vector<vector<double> > temp_nn_squared_distances;
    _index.knnSearch(query_row_vectors,
                     temp_nn_indices,
                     temp_nn_squared_distances,
                     num_nearest_neighbors,
                     _search_params);
    nn_indices = temp_nn_indices[0];
    nn_squared_distances = temp_nn_squared_distances[0];
  }

  void KDTree::knn_search(const MatrixXd& query_column_vectors,
                          size_t num_nearest_neighbors,
                          vector<vector<int> >& nn_indices,
                          vector<vector<double> >& nn_squared_distances)
  {
    if (query_column_vectors.rows() != _row_major_data_matrix.cols)
    {
      std::string error_msg("queries.rows() != data_.cols");
      throw std::runtime_error(error_msg);
    }
    flann::Matrix<double> query_row_vectors(
      const_cast<double *>(query_column_vectors.data()),
      query_column_vectors.cols(),
      query_column_vectors.rows());
    _index.knnSearch(query_row_vectors, nn_indices, nn_squared_distances,
                     num_nearest_neighbors, _search_params);
  }

  void KDTree::knn_search(size_t query_vector_index,
                          size_t num_nearest_neighbors,
                          vector<int>& nn_indices,
                          vector<double>& nn_squared_distances)
  {
    flann::Matrix<double> query_row_vectors(
      _row_major_data_matrix[query_vector_index],
      1, _row_major_data_matrix.cols);

    vector<vector<int> > temp_nn_indices;
    vector<vector<double> > temp_nn_squared_distances;

    _index.knnSearch(query_row_vectors,
                     temp_nn_indices,
                     temp_nn_squared_distances,
                     num_nearest_neighbors+1,
                     _search_params);

    nn_indices = temp_nn_indices[0];
    nn_squared_distances = temp_nn_squared_distances[0];

    nn_indices.erase(nn_indices.begin());
    nn_squared_distances.erase(nn_squared_distances.begin());
  }

  void KDTree::knn_search(const vector<size_t>& query_vector_indices,
                          size_t num_nearest_neighbors,
                          vector<vector<int> >& nn_indices,
                          vector<vector<double> >& nn_squared_distances)
  {
    MatrixXd query_column_vectors(_row_major_data_matrix.cols,
                                  query_vector_indices.size());
    for (size_t i = 0; i != query_vector_indices.size(); ++i)
      for (size_t j = 0; j != _row_major_data_matrix.cols; ++j)
        query_column_vectors(j,i) = _row_major_data_matrix[i][j];

    knn_search(query_column_vectors,
               num_nearest_neighbors+1,
               nn_indices,
               nn_squared_distances);

    for (size_t i = 0; i != query_vector_indices.size(); ++i)
    {
      nn_indices[i].erase(nn_indices[i].begin());
      nn_squared_distances[i].erase(nn_squared_distances[i].begin());
    }
  }


  // ======================================================================== //
  // Radius search methods.
  int KDTree::radius_search(const double *query_vector_data,
                            double squared_search_radius,
                            vector<int>& nn_indices,
                            vector<double>& nn_squared_distances,
                            size_t max_num_nearest_neighbors)
  {
    flann::Matrix<double> query_vector(
      const_cast<double *>(query_vector_data), 1, _row_major_data_matrix.cols);

    vector<vector<int> > temp_nn_indices;
    vector<vector<double> > temp_nn_squared_distances;

    // Store the initial maximum number of neighbors.
    size_t saved_max_neighbors = _search_params.max_neighbors;
    // Set the new value for the maximum number of neighbors.
    _search_params.max_neighbors = max_num_nearest_neighbors;

    // Search.
    _index.radiusSearch(query_vector,
                        temp_nn_indices,
                        temp_nn_squared_distances,
                        static_cast<float>(squared_search_radius),
                        _search_params);

    nn_indices = temp_nn_indices[0];
    nn_squared_distances = temp_nn_squared_distances[0];

    // Restore the initial maximum number of neighbors.
    _search_params.max_neighbors = saved_max_neighbors;

    return nn_indices.size();
  }

  int KDTree::radius_search(size_t query_vector_index,
                            double squared_search_radius,
                            vector<int>& nn_indices,
                            vector<double>& nn_squared_distances,
                            size_t max_num_nearest_neighbors)
  {
    if (max_num_nearest_neighbors != std::numeric_limits<size_t>::max())
      ++max_num_nearest_neighbors;
    radius_search(_row_major_data_matrix[query_vector_index],
                  squared_search_radius,
                  nn_indices,
                  nn_squared_distances,
                  max_num_nearest_neighbors);
    nn_indices.erase(nn_indices.begin());
    nn_squared_distances.erase(nn_squared_distances.begin());
    return nn_indices.size();
  }

  void KDTree::radius_search(const MatrixXd& queries,
                             double squared_search_radius,
                             vector<vector<int> >& nn_indices,
                             vector<vector<double> >& nn_squared_distances,
                             size_t max_num_nearest_neighbors)
  {
    if (queries.rows() != _row_major_data_matrix.cols)
    {
      std::string error_msg("queries.rows() != _row_major_data_matrix.cols");
      throw std::runtime_error(error_msg);
    }

    // Store the initial maximum number of neighbors.
    size_t saved_max_neighbors = _search_params.max_neighbors;
    // Set the new value for the maximum number of neighbors.
    _search_params.max_neighbors = max_num_nearest_neighbors;

    // Search.
    flann::Matrix<double> query_vectors(
      const_cast<double *>(queries.data()),
      queries.cols(), queries.rows());

    _index.radiusSearch(query_vectors,
                        nn_indices,
                        nn_squared_distances,
                        static_cast<float>(squared_search_radius),
                        _search_params);

    // Restore the initial maximum number of neighbors.
    _search_params.max_neighbors = saved_max_neighbors;

 }

  void KDTree::radius_search(const vector<size_t>& query_vector_indices,
                             double squared_search_radius,
                             vector<vector<int> >& nn_indices,
                             vector<vector<double> >& nn_squared_distances,
                             size_t max_num_nearest_neighbors)
  {
    if (max_num_nearest_neighbors != std::numeric_limits<size_t>::max())
      ++max_num_nearest_neighbors;

    MatrixXd query_column_vectors(_row_major_data_matrix.cols,
                                  query_vector_indices.size());
    for (size_t i = 0; i != query_vector_indices.size(); ++i)
      for (size_t j = 0; j != _row_major_data_matrix.cols; ++j)
        query_column_vectors(j,i) = _row_major_data_matrix[i][j];

    radius_search(query_column_vectors,
                  squared_search_radius,
                  nn_indices,
                  nn_squared_distances,
                  max_num_nearest_neighbors);

    // Don't include the first neighbor which is the query vector itself.
    for (size_t i = 0; i != nn_indices.size(); ++i)
    {
      nn_indices[i].erase(nn_indices[i].begin());
      nn_squared_distances[i].erase(nn_squared_distances[i].begin());
    }
  }

}