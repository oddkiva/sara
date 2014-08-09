#ifndef DO_ASSERT_HELPERS_HPP
#define DO_ASSERT_HELPERS_HPP


template <typename T, int M, int N>
::testing::AssertionResult assert_matrix_equal(const char* m_expr,
                                               const char* n_expr,
                                               const Eigen::Matrix<T, M, N>& m,
                                               const Eigen::Matrix<T, M, N>& n)
{
  if (m == n)
    return ::testing::AssertionSuccess();

  return ::testing::AssertionFailure()
    << "Value of " << m_expr << ":\n" << m << "\n and \n"
    << "Value of " << n_expr << ":\n" << n << "\n"
    << "are not equal";
}


#define ASSERT_MATRIX_EQ(m, n) ASSERT_PRED_FORMAT2(assert_matrix_equal, m, n)
#define EXPECT_MATRIX_EQ(m, n) EXPECT_PRED_FORMAT2(assert_matrix_equal, m, n)

#endif /* DO_ASSERT_HELPERS_HPP */