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

#ifndef DO_KDTREE_KDTREE_HPP
#define DO_KDTREE_KDTREE_HPP


#include <DO/Core.hpp>
#include <flann/flann.hpp>


namespace DO {

  /*!
   *  N.B.: MatrixXd uses a *** COLUMN-MAJOR *** storage in the core library.
   *  The matrix must be transposed before.
   *
   *  Therefore, data points are column vectors in MatrixXd !!
   */
  class KDTree
  {
  public:
    //! Constructor.
    KDTree(const MatrixXd& colMajorColStackedDataMatrix,
           const flann::KDTreeIndexParams& indexParams
             = flann::KDTreeIndexParams(1),
           const flann::SearchParams& searchParams
             = flann::SearchParams(-1));
    
    //! \brief Destructor
    ~KDTree();

    //! Basic k-NN search wrapped function.
    template <int N>
    void knn_search(const Matrix<double, N, 1>& query,
                    size_t k,
                    std::vector<int>& indices,
                    std::vector<double>& sqDists,
                    bool remove1NN = false)
    {
      if (data_.cols != query.size())
        throw std::runtime_error("Dimension of query vector do not match \
                                 dimension of input feature space!");
      if (remove1NN)
      {
        knn_search(query.data(), k+1, indices, sqDists);
        indices.erase(indices.begin());
        sqDists.erase(sqDists.begin());
      }
      else
        knn_search(query.data(), k, indices, sqDists);
    }

    //! Basic k-NN search wrapped function.
    template <int N, int Options, int MaxRows, int MaxCols>
    void knn_search(const Matrix<double, N, 1, Options, MaxRows, MaxCols>& query,
                    size_t k, std::vector<int>& indices,
                    bool remove1NN = false)
    {
      if (data_.cols != query.size())
        throw std::runtime_error("Dimension of query vector do not match \
                                 dimension of input feature space!");
      std::vector<double> sqDists;
      if (remove1NN)
      {
        knn_search(query.data(), k+1, indices, sqDists);
        indices.erase(indices.begin());
        sqDists.erase(sqDists.begin());
      }
      else
        knn_search(query.data(), k, indices, sqDists);
    }

    //! Basic k-NN search  wrapped function.
    void knn_search(const MatrixXd& queries, size_t k,
                   std::vector<std::vector<int> >& indices,
                   std::vector<std::vector<double> >& sqDists,
                   bool remove1NN = false);

    //! Basic k-NN search wrapped function.
    void knn_search(const MatrixXd& queries, size_t k,
                   std::vector<std::vector<int> >& indices,
                   bool remove1NN = false);

    //! In case the point query is a point in data, call this method preferably.
    void knn_search(size_t i, size_t k,
                    std::vector<int>& indices, std::vector<double>& sqDists);

    //! In case the point query is a point in data, call this method preferably.
    void knn_search(size_t i, size_t k, std::vector<int>& indices);

    //! In case the point query is a point in data, call this method preferably.
    void knn_search(const std::vector<size_t>& queries, size_t k,
                    std::vector<std::vector<int> >& indices,
                    std::vector<std::vector<double> >& sqDists);

    //! In case the point query is a point in data, call this method preferably.
    void knn_search(const std::vector<size_t>& queries, size_t k,
                    std::vector<std::vector<int> >& indices);
    
    //! Basic radius search wrapped function.
    template <int N>
    size_t radius_search(const Matrix<double, N, 1>& query,
                         double sqSearchRadius,
                         std::vector<int>& indices,
                         std::vector<double>& sqDists,
                         bool remove1NN = false)
    {
      if (data_.cols != query.size())
          throw std::runtime_error("Dimension of query vector do not match \
                                   dimension of input feature space!");
      radius_search(query.data(), sqSearchRadius, indices, sqDists);
      if (remove1NN)
      {
        indices.erase(indices.begin());
        sqDists.erase(sqDists.begin());
      }
      return indices.size(); 
    }

    //! Basic radius search wrapped function.
    template <int N, int Options, int MaxRows, int MaxCols>
    size_t radius_search(const Matrix<double, N, 1>& query,
                         double sqSearchRadius,
                         std::vector<int>& indices,
                         bool remove1NN = false)
    {
      if (data_.cols != query.size())
          throw std::runtime_error("Dimension of query vector do not match \
                                   dimension of input feature space!");
      std::vector<double> sqDists;
      radius_search(query.data(), sqSearchRadius, indices, sqDists);
      if (remove1NN)
      {
        indices.erase(indices.begin());
        sqDists.erase(sqDists.begin());
      }
      return indices.size();
    }

    //! Basic radius search wrapped function.
    void radius_search(const MatrixXd& queries, double sqSearchRadius,
                       std::vector<std::vector<int> >& indices,
                       std::vector<std::vector<double> >& sqDists,
                       bool remove1NN = false);

    //! Basic radius search wrapped function.
    void radius_search(const MatrixXd& queries, double sqSearchRadius,
                       std::vector<std::vector<int> >& indices,
                       bool remove1NN);

    //! In case the point query is a point in data, call this method preferably.
    int radius_search(size_t i, double sqSearchRadius,
                      std::vector<int>& indices, std::vector<double>& sqDists);

    //! In case the point query is a point in data, call this method preferably.
    int radius_search(size_t i, double sqSearchRadius,
                      std::vector<int>& indices);

    //! In case the point query is a point in data, call this method preferably.
    void radius_search(const std::vector<size_t>& queries, double sqSearchRadius,
                       std::vector<std::vector<int> >& indices,
                       std::vector<std::vector<double> >& sqDists);

    //! In case the point query is a point in data, call this method preferably.
    void radius_search(const std::vector<size_t>& queries,
                       double sqSearchRadius,
                       std::vector<std::vector<int> >& indices);

  private:
    void knn_search(const double *query, size_t k,
                    std::vector<int>& indices, std::vector<double>& sqDists);

    int radius_search(const double *query, double sqSearchRadius,
                      std::vector<int>& indices, std::vector<double>& sqDists);

  private:
    bool wrapped_data_;
    flann::Matrix<double> data_;
    flann::Index<flann::L2<double> > index_;
    flann::KDTreeIndexParams index_params_;
    flann::SearchParams search_params_;
  };

}


#endif /* DO_KDTREE_KDTREE_HPP */