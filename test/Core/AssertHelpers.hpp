#ifndef DO_ASSERT_HELPERS_HPP
#define DO_ASSERT_HELPERS_HPP


#include <Eigen/Core>

#include <gtest/gtest.h>


template <typename T, int M, int N>
::testing::AssertionResult assert_matrix_equal(const char* m_expr,
                                               const char* n_expr,
                                               const Eigen::Matrix<T, M, N>& m,
                                               const Eigen::Matrix<T, M, N>& n)
{
  if (m == n)
    return ::testing::AssertionSuccess();

  return ::testing::AssertionFailure()
    << "Expected matrix: " << m_expr << ":\n" << m << "\n and \n"
    << "actual matrix: " << n_expr << ":\n" << n << "\n"
    << "are not equal";
}


template <typename T, int M, int N>
::testing::AssertionResult assert_matrix_near(const char* m_expr,
                                              const char* n_expr,
                                              const char* abs_error_expr,
                                              const Eigen::Matrix<T, M, N>& m,
                                              const Eigen::Matrix<T, M, N>& n,
                                              double abs_error)
{
  T distance = (m-n).norm();
  if (distance < abs_error)
    return ::testing::AssertionSuccess();

  return ::testing::AssertionFailure()
    << "Expected matrix: " << m_expr << ":\n" << m << "\n and \n"
    << "actual matrix: " << n_expr << ":\n" << n << "\n"
    << "have actual Euclidean distance: " << distance << "\n"
    << "but expected Euclidean distance: " << abs_error_expr
    << " = " << abs_error;
}


#define ASSERT_MATRIX_EQ(m, n) ASSERT_PRED_FORMAT2(assert_matrix_equal, m, n)
#define EXPECT_MATRIX_EQ(m, n) EXPECT_PRED_FORMAT2(assert_matrix_equal, m, n)
#define ASSERT_MATRIX_NEAR(m, n, abs_error)\
  ASSERT_PRED_FORMAT3(assert_matrix_near, m, n, abs_error)
#define EXPECT_MATRIX_NEAR(m, n, abs_error)\
  EXPECT_PRED_FORMAT3(assert_matrix_near, m, n, abs_error)

#endif /* DO_ASSERT_HELPERS_HPP */