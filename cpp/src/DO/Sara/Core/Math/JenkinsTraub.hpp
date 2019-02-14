#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>
#include <DO/Sara/Core/Math/UnivariatePolynomial.hpp>

#include <complex>
#include <ctime>
#include <memory>


namespace DO { namespace Sara {

  auto compute_moduli_lower_bound(const UnivariatePolynomial<double>& P)
      -> double;

  auto K0_(const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>;

  auto K1_stage1(const UnivariatePolynomial<double>& K0,
                 const UnivariatePolynomial<double>& P)
      -> UnivariatePolynomial<double>;

  auto K1_stage2(const UnivariatePolynomial<double>& K0,
                 const UnivariatePolynomial<double>& P,
                 const UnivariatePolynomial<double>& sigma,
                 const std::complex<double>& s1, const std::complex<double>& s2)
      -> UnivariatePolynomial<double>;

  auto sigma_(const std::complex<double>& s1)
    -> UnivariatePolynomial<double>;


} /* namespace Sara */
} /* namespace DO */
