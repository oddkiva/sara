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

// DISCLAIMER REGARDING FLANN 1.8.4:
//
// I have disabled compiler warnings for MSVC in the CMakeLists.txt of FLANN.
//
// There is still a disturbing compiler warning with MSVC:
// FLANN 1.8.4 will issue compiler warning because of a reimplemented
// operator new*(...) and the corresponding operator delete [] is not implemented.
// I seriously hope there won't be memory-leak as argued in the changelog...
// serialization functions in FLANN 1.8.4 do not compile as well...

#include <DO/Core.hpp>
#include <flann/flann.hpp>

// The data structure will likely evolve again in order to be fully "templated".
// Still, it is enough as it does what I need to do.
// See if the compile time of template instantiation is not a burden for a quick
// prototyping in practice.

namespace DO {

  /*! VERY IMPORTANT TECHNICAL DETAIL: MatrixXd uses a *** COLUMN-MAJOR *** 
   *  storage in the core library.
   *  The matrix must be transposed before.
   *  I say this because it seems common to stack data in a row major fashion.
   * 
   *  Therefore, data points are column vectors in MatrixXd !!
   *  However FLANN uses a row major storage.
   *  So, please listen to Michael Jackson: 
   *    DO THINK TWICE ! (... She told my baby that we danced 'til three...)
   *  And have a look on the DOKDTree.cpp, which tests the KDTree data-structure
   */
  class KDTree
  {
  public:
    KDTree(const MatrixXd& colMajorColStackedDataMatrix,
           const flann::KDTreeIndexParams& indexParams = flann::KDTreeIndexParams(1),
           const flann::SearchParams& searchParams = flann::SearchParams(-1));
    ~KDTree();

    //! Basic k-NN search wrapped function.
    template <int N, int Options, int MaxRows, int MaxCols>
    void knnSearch(const Matrix<double, N, 1, Options, MaxRows, MaxCols>& query,
                   size_t k,
                   std::vector<int>& indices, std::vector<double>& sqDists,
                   bool remove1NN = false)
    {
      if (data_.cols != query.size())
        throw std::runtime_error("Dimension of query vector do not match \
                                 dimension of input feature space!");
      if (remove1NN)
      {
        knnSearch(query.data(), k+1, indices, sqDists);
        indices.erase(indices.begin());
        sqDists.erase(sqDists.begin());
      }
      else
        knnSearch(query.data(), k, indices, sqDists);
    }
    //! Basic k-NN search wrapped function.
    template <int N, int Options, int MaxRows, int MaxCols>
    void knnSearch(const Matrix<double, N, 1, Options, MaxRows, MaxCols>& query,
                   size_t k, std::vector<int>& indices,
                   bool remove1NN = false)
    {
      if (data_.cols != query.size())
        throw std::runtime_error("Dimension of query vector do not match \
                                 dimension of input feature space!");
      std::vector<double> sqDists;
      if (remove1NN)
      {
        knnSearch(query.data(), k+1, indices, sqDists);
        indices.erase(indices.begin());
        sqDists.erase(sqDists.begin());
      }
      else
        knnSearch(query.data(), k, indices, sqDists);
    }
    //! Basic k-NN search  wrapped function.
    void knnSearch(const MatrixXd& queries, size_t k,
                   std::vector<std::vector<int> >& indices,
                   std::vector<std::vector<double> >& sqDists,
                   bool remove1NN = false);
    //! Basic k-NN search wrapped function.
    void knnSearch(const MatrixXd& queries, size_t k,
                   std::vector<std::vector<int> >& indices,
                   bool remove1NN = false);
    //! In case the point query is a point in data, call this method preferably.
    void knnSearch(size_t i, size_t k,
                   std::vector<int>& indices, std::vector<double>& sqDists);
    //! In case the point query is a point in data, call this method preferably.
    void knnSearch(size_t i, size_t k, std::vector<int>& indices);
    //! In case the point query is a point in data, call this method preferably.
    void knnSearch(const std::vector<size_t>& queries, size_t k,
                   std::vector<std::vector<int> >& indices,
                   std::vector<std::vector<double> >& sqDists);
    //! In case the point query is a point in data, call this method preferably.
    void knnSearch(const std::vector<size_t>& queries, size_t k,
                   std::vector<std::vector<int> >& indices);
    
    //! Basic radius search wrapped function.
    template <int N, int Options, int MaxRows, int MaxCols>
    size_t radiusSearch(const Matrix<double, N, 1, Options, MaxRows, MaxCols>& query,
                        double sqSearchRadius,
                        std::vector<int>& indices, std::vector<double>& sqDists,
                        bool remove1NN = false)
    {
      if (data_.cols != query.size())
          throw std::runtime_error("Dimension of query vector do not match \
                                   dimension of input feature space!");
      radiusSearch(query.data(), sqSearchRadius, indices, sqDists);
      if (remove1NN)
      {
        indices.erase(indices.begin());
        sqDists.erase(sqDists.begin());
      }
      return indices.size(); 
    }
    //! Basic radius search wrapped function.
    template <int N, int Options, int MaxRows, int MaxCols>
    size_t radiusSearch(const Matrix<double, N, 1, Options, MaxRows, MaxCols>& query,
                        double sqSearchRadius, std::vector<int>& indices,
                        bool remove1NN = false)
    {
      if (data_.cols != query.size())
          throw std::runtime_error("Dimension of query vector do not match \
                                   dimension of input feature space!");
      std::vector<double> sqDists;
      radiusSearch(query.data(), sqSearchRadius, indices, sqDists);
      if (remove1NN)
      {
        indices.erase(indices.begin());
        sqDists.erase(sqDists.begin());
      }
      return indices.size();
    }
    //! Basic radius search wrapped function.
    void radiusSearch(const MatrixXd& queries, double sqSearchRadius,
                      std::vector<std::vector<int> >& indices,
                      std::vector<std::vector<double> >& sqDists,
                      bool remove1NN = false);
    //! Basic radius search wrapped function.
    void radiusSearch(const MatrixXd& queries, double sqSearchRadius,
                      std::vector<std::vector<int> >& indices,
                      bool remove1NN);
    //! In case the point query is a point in data, call this method preferably.
    int radiusSearch(size_t i, double sqSearchRadius,
                      std::vector<int>& indices, std::vector<double>& sqDists);
    //! In case the point query is a point in data, call this method preferably.
    int radiusSearch(size_t i, double sqSearchRadius, std::vector<int>& indices);
    //! In case the point query is a point in data, call this method preferably.
    void radiusSearch(const std::vector<size_t>& queries, double sqSearchRadius,
                      std::vector<std::vector<int> >& indices,
                      std::vector<std::vector<double> >& sqDists);
    //! In case the point query is a point in data, call this method preferably.
    void radiusSearch(const std::vector<size_t>& queries, double sqSearchRadius,
                      std::vector<std::vector<int> >& indices);

  private:
    void knnSearch(const double *query, size_t k,
                  std::vector<int>& indices, std::vector<double>& sqDists);

    int radiusSearch(const double *query, double sqSearchRadius,
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