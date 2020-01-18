#define BOOST_TEST_MODULE "Arithmetic Symbolic Calculus"
#include <drafts/Expression.hpp>
#include <drafts/Expression/Debug.hpp>
#include <drafts/Expression/UsualFunctions.hpp>
#include <drafts/Expression/Differential/Variable.hpp>
#include <drafts/Expression/Differential/Rules.hpp>

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <valarray>


using namespace sara::expression;


BOOST_AUTO_TEST_CASE(test_remove_ref_if_type_is_rvalue_ref_t)
{
  static_assert(std::is_same<                            //
                int,                                     //
                remove_ref_if_type_is_rvalue_ref_t<int>  //
                >::value);
  static_assert(std::is_same<                              //
                int,                                       //
                remove_ref_if_type_is_rvalue_ref_t<int&&>  //
                >::value);
  static_assert(std::is_same<                                   //
                const int&,                                     //
                remove_ref_if_type_is_rvalue_ref_t<const int&>  //
                >::value);
  static_assert(std::is_same<                             //
                int&,                                     //
                remove_ref_if_type_is_rvalue_ref_t<int&>  //
                >::value);
}

BOOST_AUTO_TEST_CASE(test_calculate_expr_type_t)
{
  static_assert(std::is_same<Terminal<int>,
                             calculate_expr_type_t<int>  //
                             >::value);
  static_assert(std::is_same<Terminal<int>,
                             calculate_expr_type_t<int&&>  //
                             >::value);
  static_assert(std::is_same<Terminal<const int&>,
                             calculate_expr_type_t<const int&>  //
                             >::value);
  static_assert(std::is_same<Terminal<int&>,
                             calculate_expr_type_t<int&>  //
                             >::value);

  static_assert(std::is_same<calculate_expr_type_t<Terminal<int>>,  //
                             Terminal<int>>::value);
  static_assert(std::is_same<calculate_expr_type_t<const Terminal<int>&>,  //
                             const Terminal<int>&>::value);
  static_assert(std::is_same<calculate_expr_type_t<Terminal<int>&>,  //
                             Terminal<int>&>::value);
  // Check this carefully.
  static_assert(std::is_same<calculate_expr_type_t<Terminal<int>&&>,  //
                             Terminal<int>>::value);
}


BOOST_AUTO_TEST_CASE(test_terminal_symbol)
{
  {
    auto value = 0;
    auto terminal = make_terminal(value);
    static_assert(std::is_same<decltype(terminal.value), int&>::value);

    value = 1;
    BOOST_CHECK_EQUAL(terminal.value, 1);
  }

  {
    auto terminal = make_terminal(0);
    static_assert(std::is_same<decltype(terminal.value), int>::value);
  }
}


BOOST_AUTO_TEST_CASE(test_add)
{
  {
    auto x_ = std::valarray<float>(2.f, 10);
    auto x = make_terminal(x_);
    auto sum = x + 1;
    static_assert(std::is_same<                        //
                  decltype(sum),                       //
                  AddXpr<decltype(x)&, Terminal<int>>  //
                  >::value);

    std::cout << type_name<decltype(sum)>() << std::endl;
  }

  {
    auto x_ = std::valarray<float>(2.f, 10);
    auto x = make_terminal(x_);
    auto y = make_terminal(std::valarray<float>(1.f, 10));
    auto sum = x + y;

    static_assert(std::is_same<         //
                  decltype(sum),        //
                  AddXpr<decltype(x)&,  //
                         decltype(y)&>  //
                  >::value);

    std::cout << type_name<decltype(sum)>() << std::endl;
  }

  {
    auto x_ = std::valarray<float>(2.f, 10);
    auto x = make_terminal(x_);

    const auto y_ = std::valarray<float>(1.f, 10);
    const auto y = make_terminal(y_);

    auto sum = x + y;

    static_assert(std::is_same<                                         //
                  decltype(sum),                                        //
                  AddXpr<decltype(x)&,                                  //
                         Terminal<std::valarray<float> const&> const&>  //
                  >::value);

    std::cout << type_name<decltype(sum)>() << std::endl;
  }

  {
    auto x_ = std::valarray<float>(2.f, 10);

    auto sum = make_terminal(x_) + make_terminal(1.f);
    static_assert(
        std::is_same<decltype(sum), AddXpr<Terminal<std::valarray<float>&>,
                                           Terminal<float>>>::value);
    std::cout << type_name<decltype(sum)>() << std::endl;
  }

  // Fancy testing.
  {
    auto x_ = std::valarray<float>(2.f, 10);
    auto x = make_terminal(x_);

    const auto y_ = std::valarray<float>(1.f, 10);
    const auto y = make_terminal(y_);

    auto sum = x + y + 1 + std::exp(2.) + 11.f;
    std::cout << type_name<decltype(sum)>() << std::endl;
  }

}

BOOST_AUTO_TEST_CASE(test_subscript)
{
  {
    auto x_ = std::valarray<float>(2.f, 10);
    auto x = make_terminal(x_);
    const auto i_ = 1;

    auto coeff = x[i_];
    static_assert(std::is_same<                                     //
                  decltype(coeff),                                  //
                  SubscriptXpr<decltype(x)&, Terminal<const int&>>  //
                  >::value);

    std::cout << type_name<decltype(coeff)>() << std::endl;
  }

  {
    auto x_ = std::valarray<float>(2.f, 10);
    auto x = make_terminal(x_);

    auto coeff = x[1];
    static_assert(std::is_same<                              //
                  decltype(coeff),                           //
                  SubscriptXpr<decltype(x)&, Terminal<int>>  //
                  >::value);

    std::cout << type_name<decltype(coeff)>() << std::endl;
  }

  {
    auto x_ = std::valarray<float>(2.f, 10);
    auto x = make_terminal(x_);

    auto coeff = x[make_terminal(1)];
    static_assert(std::is_same<                              //
                  decltype(coeff),                           //
                  SubscriptXpr<decltype(x)&, Terminal<int>>  //
                  >::value);

    std::cout << type_name<decltype(coeff)>() << std::endl;
  }

  {
    auto x_ = std::valarray<float>(2.f, 10);
    auto y_ = std::valarray<float>(2.f, 10);
    auto x = make_terminal(x_);
    auto y = make_terminal(y_);

    auto coeff = (x + y)[1];
    std::cout << type_name<decltype(coeff)>() << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(test_function_composition)
{
  auto sin_ = make_terminal<double (*)(double)>(std::sin);
  auto cos_ = make_terminal<double (*)(double)>(std::cos);
  auto log_ = make_terminal<double (*)(double)>(std::log);
  auto exp_ = make_terminal<double (*)(double)>(std::exp);

  auto sin_pi_2 = sin_(M_PI / 2.);
  std::cout << "sin_pi_2.eval() = " << sin_pi_2.eval() << std::endl;

  auto log_exp_sin_0 = log_.circle(exp_).circle(sin_)(0);
  std::cout << "log_exp_sin_0.eval() = " << log_exp_sin_0.eval() << std::endl;

  auto sum = sin_pi_2 + log_exp_sin_0;
  std::cout << "sum = " << sum.eval() << std::endl;

  auto sum_fn = sin_ + log_.circle(exp_).circle(sin_);
  std::cout << "sum_fn(0).eval() = " << sum_fn(0).eval() << std::endl;

  auto complex_fn =
      (sin_ + log_.circle(exp_).circle(sin_)) * cos_ - sin_ / exp_;
  std::cout << "complex_fn(0).eval() = " << complex_fn(0).eval() << std::endl;
}

BOOST_AUTO_TEST_CASE(test_negate_operation)
{
  auto minus_sin = negate_.circle(sin_);
  auto x = make_terminal(abc::x<double>{M_PI / 2.});
  std::cout << "x = " << type_name<decltype(x)>() << std::endl;
  std::cout << "minus_sin = " << type_name<decltype(minus_sin)>() << std::endl;

  auto minus_sin_pi_2 = minus_sin(x);
  std::cout << "minus_sin_pi_2 = " << type_name<decltype(minus_sin_pi_2)>() << std::endl;
  std::cout << "minus_sin_pi_2.eval() = " << minus_sin_pi_2.eval() << std::endl;
}

BOOST_AUTO_TEST_CASE(test_differential)
{
  auto x = abc::x<double>{M_PI / 2.};
  auto y = abc::y<double>{1.};

  std::cout << "sin_(x) = " << sin_(x).eval() << std::endl;

  auto x_ = make_terminal(x);
  auto y_ = make_terminal(y);
  auto one_ = make_terminal(One{});

  // Check that dx/dx = 1.
  static_assert(std::is_same_v<calculate_expr_type_2_t<decltype(x_)>,
                               Terminal<Variable<double, 'x'>>>);
  //std::cout << type_name<calculate_expr_type_2_t<decltype(x_)>>() << std::endl;
  static_assert(std::is_same_v<derivative_t<decltype(x_), decltype(x_)>,  //
                               One>);

  // Check the derivative of d(x + x)/dx = 2.
  std::cout << type_name<calculate_expr_type_2_t<decltype(x_ + x_)>>()
            << std::endl;
  std::cout << type_name<derivative_t<decltype(x_ + x_), decltype(x_)>>()
            << std::endl;
}

BOOST_AUTO_TEST_CASE(test_derive_chain)
{
  constexpr auto x = abc::x<double>{M_PI / 2.};
  const auto x_ = make_terminal(x);

  // auto d_sin_x = derivative(sin_(x_), x_);
  // std::cout << "d_sin_x = " << type_name<decltype(d_sin_x)>() << std::endl;
  // std::cout << "d_sin_x.eval() = " << d_sin_x.eval() << std::endl;

  //auto d_sinsinx = derivative(sin_(sin_(x_)), x_);
  //std::cout << "d_sinsinx = " << type_name<decltype(d_sinsinx)>() << std::endl;

  //auto d_cos_x = derivative(cos_(x_), x_);
  //std::cout << "d_cos_x = " << type_name<decltype(d_cos_x)>() << std::endl;

  //auto log_exp_sin_x = log_.circle(exp_).circle(sin_)(x);
  //std::cout << "log_exp_sin_x.eval() = " << log_exp_sin_x.eval() << std::endl;

  //auto sum = sin_x + log_exp_sin_x;
  //std::cout << "sum = " << sum.eval() << std::endl;

  //auto sum_fn = sin_ + log_.circle(exp_).circle(sin_);
  //std::cout << "sum_fn(x).eval() = " << sum_fn(x).eval() << std::endl;

  //auto complex_fn = (sin_ + log_.circle(exp_).circle(sin_)) * cos_ - sin_ / exp_;
  //std::cout << "complex_fn(x).eval() = " << complex_fn(x).eval() << std::endl;
}
