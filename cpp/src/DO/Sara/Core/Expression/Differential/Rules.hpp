#pragma once

#include <DO/Sara/Core/Expression/Arithmetics.hpp>
#include <DO/Sara/Core/Expression/Expression.hpp>
#include <DO/Sara/Core/Expression/TypeCalculation.hpp>

#include <DO/Sara/Core/Expression/Differential/Variable.hpp>
#include <DO/Sara/Core/Expression/UsualFunctions.hpp>


// Constants.
namespace sara::expression {

  struct One : Expression<One> {
    using decayed_type = One;
  };

  struct Zero : Expression<Zero> {
    using decayed_type = Zero;
  };

}  // namespace sara::expression


// Differentiation rules (Part 1).
namespace sara::expression {

  template <typename Expr, typename Var>
  struct Diff;


  template <typename Var>
  struct Diff<Var, Var> : Expression<Diff<Var, Var>>
  {
    using result_type = One;
  };

  template <typename T1, typename T2, int ID1, int ID2>
  struct Diff<Variable<T1, ID1>, Variable<T2, ID2>>
    : Expression<Diff<Variable<T1, ID1>, Variable<T2, ID2>>>
  {
    using result_type =
        std::conditional_t<std::is_same_v<T1, T2> && ID1 == ID2, One, Zero>;
  };


  template <typename Expr, typename Var>
  using derivative_t = typename Diff<Expr, Var>::result_type;

  template <typename Expr, typename Var>
  using df_t = typename Diff<Expr, Var>::df_type;

  template <typename Expr, typename Var>
  using dg_t = typename Diff<Expr, Var>::dg_type;

  template <typename Expr, typename Var>
  using f_t = typename Diff<Expr, Var>::f_type;

  template <typename Expr, typename Var>
  using g_t = typename Diff<Expr, Var>::g_type;


  template <typename Y, typename X>
  inline auto derivative(Y&& y, X&& x) noexcept
  {
    using y_type = typename std::decay_t<Y>::decayed_type;
    using x_type = typename std::decay_t<X>::decayed_type;
    using dy_dx_type = derivative_t<y_type, x_type>;

    using f_type = f_t<Y, X>;
    using g_type = g_t<Y, X>;

    using df_type = df_t<Y, X>;
    using dg_type = dg_t<Y, X>;

    return dy_dx_type{std::forward(y), std::forward(x)};
  }

}  // namespace sara::expression::function


// Differentiation of usual functions.
namespace sara::expression {

  template <typename Y, typename X>
  struct Diff<FunXpr<Sin, Y>, X> : Expression<Diff<FunXpr<Sin, Y>, X>>
  {
    using y_type = typename std::decay_t<Y>::decayed_type;
    using x_type = typename std::decay_t<X>::decayed_type;
    using dy_dx_type = derivative_t<y_type, x_type>;

    using f_type = FunXpr<Sin, y_type>;
    using g_type = y_type;

    using df_type = Cos;
    using dg_type = derivative_t<y_type, x_type>;

    // using result_type =
    //    std::conditional_t<std::is_same_v<y_type, x_type>,
    //                       decltype(cos_(y_type{})),
    //                       decltype(cos_(y_type{}) * dy_dx_type{})>;

    using result_type = decltype(cos_(y_type{}) * dy_dx_type{});
  };

  template <typename Y, typename X>
  inline auto derivative(FunXpr<decltype(sin_), Y> siny, X x) noexcept
  {
    using diff_t = Diff<FunXpr<Sin, Y>, X>;
    using dsiny_dx_t = typename diff_t::result_type;
    using dy_dx_t = typename diff_t::dy_dx_type;
    auto y = std::get<1>(siny.exprs);
    return cos_(y) * dy_dx_t{}(x);
  }

}  // namespace sara::expression


// Differentiation rules (Continued).
namespace sara::expression {

  template <typename F, typename G, typename X>
  struct Diff<AddXpr<F, G>, X> : Expression<Diff<AddXpr<F, G>, X>>
  {
    using f_type = typename std::decay_t<F>::decayed_type;
    using g_type = typename std::decay_t<G>::decayed_type;
    using x_type = typename std::decay_t<X>::decayed_type;
    using df_type = derivative_t<f_type, x_type>;
    using dg_type = derivative_t<g_type, x_type>;
    using result_type = decltype(df_type{} + dg_type{});
  };

  template <typename F, typename G, typename X>
  struct Diff<SubXpr<F, G>, X> : Expression<Diff<SubXpr<F, G>, X>>
  {
    using f_type = typename std::decay_t<F>::decayed_type;
    using g_type = typename std::decay_t<G>::decayed_type;
    using x_type = typename std::decay_t<X>::decayed_type;
    using df_type = derivative_t<f_type, x_type>;
    using dg_type = derivative_t<g_type, x_type>;
    using result_type = decltype(df_type{} - dg_type{});
  };

  template <typename F, typename G, typename X>
  struct Diff<MulXpr<F, G>, X> : Expression<Diff<MulXpr<F, G>, X>>
  {
    using f_type = typename std::decay_t<F>::decayed_type;
    using g_type = typename std::decay_t<G>::decayed_type;
    using x_type = typename std::decay_t<X>::decayed_type;
    using df_type = derivative_t<f_type, x_type>;
    using dg_type = derivative_t<g_type, x_type>;
    using result_type = decltype(df_type{} * g_type{} - f_type{} * dg_type{});
  };

  template <typename F, typename G, typename X>
  struct Diff<DivXpr<F, G>, X> : Expression<Diff<DivXpr<F, G>, X>>
  {
    using f_type = typename std::decay_t<F>::decayed_type;
    using g_type = typename std::decay_t<G>::decayed_type;
    using x_type = typename std::decay_t<X>::decayed_type;
    using df_type = derivative_t<f_type, x_type>;
    using dg_type = derivative_t<g_type, x_type>;
    using result_type = decltype((df_type{} * g_type{} - f_type{} * dg_type{}) /
                                 (g_type{} * g_type{}));
  };

  template <typename F, typename G, typename X>
  struct Diff<FunXpr<F, G>, X> : Expression<Diff<FunXpr<F, G>, X>>
  {
    using f_type = typename std::decay_t<F>::decayed_type;
    using g_type = typename std::decay_t<G>::decayed_type;
    using x_type = typename std::decay_t<X>::decayed_type;
    using df_type = derivative_t<f_type, g_type>;
    using dg_type = derivative_t<g_type, x_type>;
    using result_type = decltype(df_type{}(g_type{}) * dg_type{});
  };

} /* sara::expression */
