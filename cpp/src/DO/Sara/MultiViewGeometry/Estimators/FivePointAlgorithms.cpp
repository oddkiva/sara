#include <DO/Sara/Core/Math/JenkinsTraub.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/FivePointAlgorithms.hpp>


#define LOG_DEBUG std::cout << "[" << __FUNCTION__ << ":" << __LINE__ << "] "


using namespace std;


namespace DO { namespace Sara {

  auto
  NisterFivePointAlgorithm::extract_null_space(const Matrix<double, 3, 5>& p_left,
                                               const Matrix<double, 3, 5>& p_right)
      -> std::array<Matrix3d, 4>
  {
    Matrix<double, 5, 9> A;

    for (int i = 0; i < 5; ++i)
      A.row(i) <<                                     //
          p_right(i, 0) * p_left.col(i).transpose(),  //
          p_right(i, 1) * p_left.col(i).transpose(),  //
          p_right(i, 2) * p_left.col(i).transpose();

    LOG_DEBUG << "A = \n" << A << endl;

    LOG_DEBUG << "svd = " << A.bdcSvd(Eigen::ComputeFullV).singularValues() << endl;  // K as Ker.
    // Calculate the bases of the null-space.
    MatrixXd K =
        A.bdcSvd(Eigen::ComputeFullV).matrixV().rightCols(4);  // K as Ker.
    LOG_DEBUG << "K = \n" << K << endl;

    // The essential matrix lives in right null space K.
    const Matrix3d X = Map<Matrix<double, 3, 3, RowMajor>>{K.col(0).data()};
    const Matrix3d Y = Map<Matrix<double, 3, 3, RowMajor>>{K.col(1).data()};
    const Matrix3d Z = Map<Matrix<double, 3, 3, RowMajor>>{K.col(2).data()};
    const Matrix3d W = Map<Matrix<double, 3, 3, RowMajor>>{K.col(3).data()};

    return {X, Y, Z, W};
  }

  auto NisterFivePointAlgorithm::essential_matrix_expression(
      const std::array<Matrix3d, 4>& null_space_bases) -> Polynomial<Matrix3d>
  {
    const auto& [X, Y, Z, W] = null_space_bases;
    return x * X + y * Y + z * Z + one_ * W;
  }

  auto NisterFivePointAlgorithm::build_epipolar_constraints(
      const Polynomial<Matrix3d>& E)
    -> Matrix<double, 10, 20>
  {
    const auto EEt = E * E.t();
    auto P = EEt * E - 0.5 * trace(EEt) * E;

#ifdef DEBUG
    const auto P00 = P(0, 0);
    std::cout << "P00 has " << P00.coeffs.size() << " monomials" << std::endl;
    for (const auto& c : P00.coeffs)
      std::cout << "Monomial: " << c.first.to_string() << std::endl;
#endif

    auto Q = det(E);
#ifdef DEBUG
    std::cout << "det(E) = " << Q.to_string() << std::endl;
#endif

    // ===========================================================================
    // As per Nister paper.
    //
    Matrix<double, 10, 20> A;
    A.setZero();

    // Save Q in the matrix.
    for (int j = 0; j < 20; ++j)
    {
      auto coeff = Q.coeffs.find(monomials[j]);
      if (coeff == Q.coeffs.end())
        continue;
      A(0, j) = coeff->second;
    }

    // Save P in the matrix.
    for (int a = 0; a < 3; ++a)
    {
      for (int b = 0; b < 3; ++b)
      {
        const auto i = 3 * a + b;
        for (int j = 0; j < 20; ++j)
          A(i, j) = P(a, b).coeffs[monomials[j]];
      }
    }

    LOG_DEBUG << "Constraint matrix = \n" << A << endl;

    return A;
  }

  auto NisterFivePointAlgorithm::solve_epipolar_constraints(
      const Matrix<double, 10, 20>& A) -> std::vector<Vector3d>
  {
    // ===========================================================================
    // 1. Perform Gauss-Jordan elimination on A and stop four rows earlier.
    //    lower diagonal of A is zero (minus some block)
    Eigen::FullPivLU<Matrix<double, 10, 10>> lu(A.block<10, 10>(0, 0));

    // Calculate <n> = det(B)
    // 2. B is the right-bottom block after Gauss-Jordan elimination of A.
    Matrix<double, 10, 10> B = lu.solve(A.block<10, 10>(0, 10));
    LOG_DEBUG << "B = " << B << endl;

    auto to_poly = [this](const auto& row_vector) {
      auto p = Polynomial<double>{};
      for (int i = 0; i < row_vector.size(); ++i)
        p.coeffs[this->monomials[i + 10]] = row_vector[i];
      return p;
    };

    auto e = B.row(4 /* 'e' - 'a' */);
    auto f = B.row(5 /* 'f' - 'a' */);
    auto g = B.row(6 /* 'g' - 'a' */);
    auto h = B.row(7 /* 'h' - 'a' */);
    auto i = B.row(8 /* 'i' - 'a' */);
    auto j = B.row(9 /* 'j' - 'a' */);
    LOG_DEBUG << "e = " << to_poly(e).to_string() << endl;
    LOG_DEBUG << "f = " << to_poly(f).to_string() << endl;
    LOG_DEBUG << "g = " << to_poly(g).to_string() << endl;
    LOG_DEBUG << "h = " << to_poly(h).to_string() << endl;
    LOG_DEBUG << "i = " << to_poly(i).to_string() << endl;
    LOG_DEBUG << "j = " << to_poly(j).to_string() << endl;

    auto k = to_poly(e) - z * to_poly(f);
    auto l = to_poly(g) - z * to_poly(h);
    auto m = to_poly(i) - z * to_poly(j);
    LOG_DEBUG << "k = " << k.to_string() << endl;
    LOG_DEBUG << "l = " << l.to_string() << endl;
    LOG_DEBUG << "m = " << m.to_string() << endl;

    // 3. [x, y, 1]^T is a non-zero null vector in Null(B).
    using Univariate::UnivariatePolynomial;
    auto B00 = UnivariatePolynomial<double>{3};
    auto B01 = UnivariatePolynomial<double>{3};
    auto B02 = UnivariatePolynomial<double>{3};

    auto B10 = UnivariatePolynomial<double>{3};
    auto B11 = UnivariatePolynomial<double>{3};
    auto B12 = UnivariatePolynomial<double>{3};

    auto B20 = UnivariatePolynomial<double>{4};
    auto B21 = UnivariatePolynomial<double>{4};
    auto B22 = UnivariatePolynomial<double>{4};

    // 1st row.
    B00[0] = k.coeffs[x];
    B00[1] = k.coeffs[x * z];
    B00[2] = k.coeffs[x * z.pow(2)];

    B01[0] = k.coeffs[y];
    B01[1] = k.coeffs[y * z];
    B01[2] = k.coeffs[y * z.pow(2)];

    B02[0] = k.coeffs[one_];
    B02[1] = k.coeffs[z];
    B02[2] = k.coeffs[z.pow(2)];
    B02[3] = k.coeffs[z.pow(3)];

    // 2nd row.
    B10[0] = l.coeffs[x];
    B10[1] = l.coeffs[x * z];
    B10[2] = l.coeffs[x * z.pow(2)];

    B11[0] = l.coeffs[y];
    B11[1] = l.coeffs[y * z];
    B11[2] = l.coeffs[y * z.pow(2)];

    B12[0] = l.coeffs[one_];
    B12[1] = l.coeffs[z];
    B12[2] = l.coeffs[z.pow(2)];
    B12[3] = l.coeffs[z.pow(3)];

    // 3rd row.
    B20[0] = m.coeffs[x];
    B20[1] = m.coeffs[x * z];
    B20[2] = m.coeffs[x * z.pow(2)];

    B21[0] = m.coeffs[y];
    B21[1] = m.coeffs[y * z];
    B21[2] = m.coeffs[y * z.pow(2)];

    B22[0] = m.coeffs[one_];
    B22[1] = m.coeffs[z];
    B22[2] = m.coeffs[z.pow(2)];
    B22[3] = m.coeffs[z.pow(3)];

    // Follows paragraph "3.2.4 Step 4: Determinant Expansion" in Nister's
    // paper.
    const auto p0 = B01 * B12 - B02 * B11;
    const auto p1 = B02 * B11 - B00 * B12;
    const auto p2 = B00 * B11 - B01 * B10;

    const auto n = p0 * B20 + p1 * B21 + p2 * B22;
    LOG_DEBUG << "n = " << n << endl;

    auto roots = decltype(rpoly(n)){};
    try {
      roots = rpoly(n);
    }
    catch(exception& e)
    {
      LOG_DEBUG << "Polynomial solver failed: " << e.what() << endl;
      // And it's OK because it seems that some correspondences are so wrong
      // that the polynomial evaluation at the root estimate become very
      // unstable numerically.
    }
    LOG_DEBUG << "roots.size() = " << roots.size() << endl;

    auto xyzs = std::vector<Vector3d>{};
    for (const auto& z_complex : roots)
    {
      if (z_complex.imag() != 0)
        continue;

      const auto z = z_complex.real();
      LOG_DEBUG << "z = " << z << endl;

      const auto p0_z = p0(z);
      const auto p1_z = p1(z);
      const auto p2_z = p2(z);

      const auto x = p0_z / p2_z;
      const auto y = p1_z / p2_z;

      if (std::isnan(x) || std::isinf(x) ||
          std::isnan(y) || std::isnan(y))
        continue;

      xyzs.push_back({x, y, z});
    }

    return xyzs;
  }

  auto NisterFivePointAlgorithm::find_essential_matrices(
      const Matrix<double, 3, 5>& p, const Matrix<double, 3, 5>& q)
      -> std::vector<Matrix3d>
  {
    const auto null_space = extract_null_space(p, q);
    const auto& [X, Y, Z, W] = null_space;
    std::cout << "X =\n" << X << std::endl;
    std::cout << "Y =\n" << Y << std::endl;
    std::cout << "Z =\n" << Z << std::endl;
    std::cout << "W =\n" << W << std::endl;

    auto E_expr = essential_matrix_expression(null_space);

    auto A = build_epipolar_constraints(E_expr);

    auto xyzs = solve_epipolar_constraints(A);
    for (const auto& xyz : xyzs)
      LOG_DEBUG << "xyz = " << xyz.transpose() << std::endl;

    auto Es = std::vector<Matrix3d>{xyzs.size()};
    for (auto i = 0u; i < xyzs.size(); ++i)
    {
      const auto& xyz = xyzs[i];
      const auto& x = xyz[0];
      const auto& y = xyz[1];
      const auto& z = xyz[2];
      Es[i] = x * X + y * Y + z * Z + W;
      LOG_DEBUG << "E =\n" << Es[i] << endl;
    };

    return Es;
  }

} /* namespace Sara */
} /* namespace DO */
