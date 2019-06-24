#pragma once

#include <DO/Sara/Core/Expression/Arithmetics.hpp>
#include <DO/Sara/Core/Expression/Expression.hpp>
#include <DO/Sara/Core/Expression/TypeCalculation.hpp>

#include <DO/Sara/Core/Expression/Differential/Variable.hpp>


namespace sara::expression {

  struct One : Expression<One> {
    using decayed_type = One;
  };

  struct Zero : Expression<Zero> {
    using decayed_type = Zero;
  };


  template <typename Expr, typename Var>
  struct Diff;


  template <typename Var>
  struct Diff<Var, Var> : Expression<Diff<Var, Var>>
  {
    using result_type = One;
  };


  //template <typename T, int ID>
  //using VarTerminal = Teminal<Variable<T, ID>>;


  //// Rule for dy/dx where (x, y) are variables.
  //template <typename T1, typename T2, int ID1, int ID2>
  //struct Diff<VarTerminal<T1, ID1>, VarTerminal<T2, ID2>>
  //  : Expression<Diff<VarTerminal<T1, ID1>, VarTerminal<T2, ID2>>>
  //{
  //  using result_type =
  //      std::conditional_t<std::is_same_v<T1, T2> && ID1 == ID2, One, Zero>;
  //};


  template <typename T1, typename T2, int ID1, int ID2>
  struct Diff<Variable<T1, ID1>, Variable<T2, ID2>>
    : Expression<Diff<Variable<T1, ID1>, Variable<T2, ID2>>>
  {
    using result_type =
        std::conditional_t<std::is_same_v<T1, T2> && ID1 == ID2, One, Zero>;
  };


  template <typename Expr, typename Var>
  using derivative_t = typename Diff<Expr, Var>::result_type;

  template <typename Y, typename X>
  inline auto derivative(Y&& y, X&& x) noexcept
  {
    using y_type = typename std::decay_t<Y>::decayed_type;
    using x_type = typename std::decay_t<X>::decayed_type;
    using dy_dx_type = derivative_t<y_type, x_type>;
    return dy_dx_type{std::forward(y), std::forward(x)};
  }


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

    //inline Diff(F&& f, G&& g, X&& x)
    //{
    //  auto dg_dx = dg_type{}(std::forward<x_type>(x));
    //  auto df_dg = df_type{}(std::forward<g_type>(g));
    //  return df_dg * dg_dx;
    //}
  };

} /* sara::expression */
