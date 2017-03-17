#pragma once

#include <iostream>
#include <set>
#include <vector>

#include <DO/Sara/Core/EigenExtension.hpp>


template <typename T>
inline std::set<T> to_std_set(const std::vector<T>& v)
{
  return std::set<T>(v.begin(), v.end());
}

template <typename T, int M, int N, int Opts, int MaxRows, int MaxCols>
inline
std::set<
  Eigen::Matrix<T, M, N, Opts, MaxRows, MaxCols>,
  DO::Sara::LexicographicalOrder
>
to_std_set(const std::vector<Eigen::Matrix<T, M, N, Opts, MaxRows, MaxCols> >& v)
{
  return std::set<
    Eigen::Matrix<T, M, N, Opts, MaxRows, MaxCols>,
    DO::Sara::LexicographicalOrder
  >(v.begin(), v.end());
}

// Define convenient macros like 'self.assertItemsEqual' in Python.
#define BOOST_CHECK_ITEMS_EQUAL(collection1, collection2)                      \
  BOOST_CHECK(to_std_set(collection1) == to_std_set(collection2))

#define BOOST_REQUIRE_ITEMS_EQUAL(collection1, collection2)                    \
  BOOST_REQUIRE(to_std_set(collection1) == to_std_set(collection2))

// Linear algebra.
#define BOOST_CHECK_SMALL_L2_DISTANCE(m1, m2, eps)                             \
  BOOST_CHECK_SMALL((m1 - m2).norm(), eps)

#define BOOST_REQUIRE_SMALL_L2_DISTANCE(m1, m2, eps)                           \
  BOOST_REQUIRE_SMALL((m1 - m2).norm(), eps)

#define BOOST_CHECK_CLOSE_L2_DISTANCE(m1, m2, eps)                             \
  BOOST_CHECK_SMALL((m1 - m2).norm() / m1.norm(), eps)

#define BOOST_REQUIRE_CLOSE_L2_DISTANCE(m1, m2, eps)                           \
  BOOST_REQUIRE_SMALL((m1 - m2).norm() / m1.norm(), eps)


struct CoutRedirect {
  CoutRedirect(std::streambuf * new_buffer)
    : old(std::cout.rdbuf(new_buffer))
  { }

  ~CoutRedirect() {
    std::cout.rdbuf(old);
  }

private:
  std::streambuf * old;
};
