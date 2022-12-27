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

#pragma once

#if defined(_WIN32)
#  pragma warning(push)
#  pragma warning(disable : 4267)
#  pragma warning(disable : 4334)
#  pragma warning(disable : 4996)
#elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  if defined(__has_warning)  // clang
#    if __has_warning("-Wimplicit-int-conversion")
#      pragma GCC diagnostic ignored "-Wimplicit-int-conversion"
#    endif
#  endif
#endif
#include <flann/flann.hpp>
#if defined(_WIN32)
#  pragma warning(pop)
#elif defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO::Sara {

  //! @addtogroup KDTree
  //! @{

  /*!
   *  N.B.: MatrixXd uses a *** COLUMN-MAJOR *** storage in the core library.
   *  The matrix must be transposed before.
   *
   *  Therefore, data points are column vectors in MatrixXd !!
   */
  class DO_SARA_EXPORT KDTree
  {
  public:
    //! Constructor.
    KDTree(const MatrixXd& data_matrix,
           const flann::KDTreeIndexParams& index_params =
               flann::KDTreeIndexParams(1),
           const flann::SearchParams& search_params = flann::SearchParams(-1));

    //! k-NN search for a single query column vector.
    template <int N, int Options, int MaxRows, int MaxCols>
    void
    knn_search(const Matrix<double, N, 1, Options, MaxRows, MaxCols>& query,
               int num_nearest_neighbors, std::vector<int>& nn_indices,
               std::vector<double>& nn_squared_distances)
    {
      if (static_cast<int>(_row_major_matrix_view.cols) != query.size())
        throw std::runtime_error{"Dimension of query vector do not match "
                                 "dimension of input feature space!"};
      knn_search(query.data(), num_nearest_neighbors, nn_indices,
                 nn_squared_distances);
    }

    //! Batch k-NN search for a set of query column vectors.
    void knn_search(const MatrixXd& query_column_vectors,
                    int num_nearest_neighbors,
                    std::vector<std::vector<int>>& nn_indices,
                    std::vector<std::vector<double>>& nn_squared_distances);

    //! k-NN search for a query vector living in the data. In this case, the
    //! set of nearest neighbors does not include this query vector.
    void knn_search(size_t query_vector_index, int num_nearest_neighbors,
                    std::vector<int>& nn_indices,
                    std::vector<double>& nn_squared_distances);

    //! k-NN search for a set of query vectors living in the data. In this case,
    //! Each set of nearest neighbors does not include their corresponding
    //! query vector.
    void knn_search(const std::vector<size_t>& queries,
                    int num_nearest_neighbors,
                    std::vector<std::vector<int>>& nn_indices,
                    std::vector<std::vector<double>>& nn_squared_distances);

    //! Radius search for a single query column vector.
    template <int N, int Options, int MaxRows, int MaxCols>
    int
    radius_search(const Matrix<double, N, 1, Options, MaxRows, MaxCols>& query,
                  double squared_search_radius, std::vector<int>& nn_indices,
                  std::vector<double>& nn_squared_distances,
                  int max_num_nearest_neighbors = -1)
    {
      if (static_cast<int>(_row_major_matrix_view.cols) != query.size())
        throw std::runtime_error{"Dimension of query vector do not match "
                                 "dimension of input feature space !"};

      radius_search(query.data(), squared_search_radius, nn_indices,
                    nn_squared_distances, max_num_nearest_neighbors);

      return static_cast<int>(nn_indices.size());
    }

    //! Radius search for a set of of query column vectors.
    void radius_search(const MatrixXd& queries, double squared_search_radius,
                       std::vector<std::vector<int>>& nn_indices,
                       std::vector<std::vector<double>>& nn_squared_distances,
                       int max_num_nearest_neighbors = -1);

    //! Radius search for a query vector living in the data. In this case, the
    //! set of nearest neighbors does not include this query vector.
    int radius_search(size_t query_vector_index, double squared_search_radius,
                      std::vector<int>& nn_indices,
                      std::vector<double>& nn_squared_distances,
                      int max_num_nearest_neighbors = -1);

    //! Radius search for a set of query vectors living in the data. In this
    //! case,
    //! Each set of nearest neighbors does not include their corresponding
    //! query vector.
    void radius_search(const std::vector<size_t>& queries,
                       double squared_search_radius,
                       std::vector<std::vector<int>>& nn_indices,
                       std::vector<std::vector<double>>& nn_squared_distances,
                       int max_num_nearest_neighbors = -1);

  private:
    void knn_search(const double* query_vector, int num_nearest_neighbors,
                    std::vector<int>& nn_indices,
                    std::vector<double>& nn_squared_distances);

    int radius_search(const double* query_vector, double squared_search_radius,
                      std::vector<int>& nn_indices,
                      std::vector<double>& nn_squared_distances,
                      int max_num_nearest_neighbors);

  private:
    flann::Matrix<double> _row_major_matrix_view;
    flann::Index<flann::L2<double>> _index;
    flann::KDTreeIndexParams _index_params;
    flann::SearchParams _search_params;
  };

  //! @}

}  // namespace DO::Sara
