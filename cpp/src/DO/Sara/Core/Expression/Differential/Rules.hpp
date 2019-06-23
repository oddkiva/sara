#pragma once

#include <DO/Sara/Core/Expression/Expression.hpp>
#include <DO/Sara/Core/Expression/Arithmetics.hpp>
#include <DO/Sara/Core/Expression/TypeCalculation.hpp>


namespace sara::expression {

  struct One : Expression<One> {
    using decayed_type = One;
  };

  struct Zero : Expression<One> {
    using decayed_type = One;
  };


  template <typename Expr, typename Var>
  struct Diff;


  template <typename Var>
  struct Diff<Var, Var> : Expression<Diff<Var, Var>>
  {
    using result_type = One;
  };

  template <typename T1, typename T2, int ID1, int ID2>
  struct Diff<Variable<T1, ID1>, Variable<T2, ID2>> : Expression<?
  {
    //using decayed_type= ;
    using result_type = Zero<T2>;
  };


  template <typename Expr, typename Var>
  using derivative_t = typename Diff<Expr, Var>::result_type;


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

} /* sara::expression */
