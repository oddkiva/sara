#pragma once

#include <DO/Sara/Core/Expression/ForwardDeclarations.hpp>

#include <tuple>


namespace sara::expression {

  template <typename L, typename R>
  struct AddXpr : Expression<AddXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;
    using decayed_type = AddXpr<typename std::decay_t<lhs_type>::decayed_type,  //
                                typename std::decay_t<rhs_type>::decayed_type>;

    std::tuple<L, R> exprs;

    inline constexpr AddXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }

    inline auto eval() const
    {
      const auto& [a, b] = exprs;
      return a.eval() + b.eval();
    }

    //! Evaluation function at variable x.
    template <typename X>
    inline auto eval_at(X&& x) const
    {
      const auto& [f, g] = exprs;

      using x_expr_type = calculate_expr_type_t<decltype(x)>;
      const auto x_expr = std::forward<x_expr_type>(x);

      return f.eval_at(x_expr.eval()) + g.eval_at(x_expr.eval());
    }

    //! Get coefficient at index i.
    template <typename Index>
    inline auto at(Index&& i) const
    {
      using index_expr_type = calculate_expr_type_t<decltype(i)>;
      const auto i_expr = std::forward<index_expr_type>(i);

      const auto& [a, b] = exprs;
      return a.at(i_expr.eval()) + b.at(i_expr.eval());
    }
  };

  template <typename L, typename R>
  struct SubXpr : Expression<SubXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;
    using decayed_type =
        SubXpr<typename std::decay_t<lhs_type>::decayed_type,  //
               typename std::decay_t<rhs_type>::decayed_type>;

    std::tuple<L, R> exprs;

    inline constexpr SubXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }

    inline auto eval() const
    {
      const auto& [a, b] = exprs;
      return a.eval() - b.eval();
    }

    //! Evaluation function at variable x.
    template <typename X>
    inline auto eval_at(X&& x) const
    {
      const auto& [f, g] = exprs;

      using x_expr_type = calculate_expr_type_t<decltype(x)>;
      const auto x_expr = std::forward<x_expr_type>(x);

      return f.eval_at(x_expr.eval()) - g.eval_at(x_expr.eval());
    }

    //! Get coefficient at index i.
    template <typename Index>
    inline auto at(Index&& i) const
    {
      using index_expr_type = calculate_expr_type_t<decltype(i)>;
      const auto i_expr = std::forward<index_expr_type>(i);

      const auto& [a, b] = exprs;
      return a.at(i_expr.eval()) - b.at(i_expr.eval());
    }
  };

  template <typename L, typename R>
  struct MulXpr : Expression<MulXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;
    using decayed_type =
        MulXpr<typename std::decay_t<lhs_type>::decayed_type,  //
               typename std::decay_t<rhs_type>::decayed_type>;

    std::tuple<L, R> exprs;

    inline constexpr MulXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }

    inline auto eval() const
    {
      const auto& [a, b] = exprs;
      return a.eval() * b.eval();
    }

    template <typename X>
    inline auto eval_at(X&& x) const
    {
      const auto& [f, g] = exprs;

      using x_expr_type = calculate_expr_type_t<decltype(x)>;
      const auto x_expr = std::forward<x_expr_type>(x);

      return f.eval_at(x_expr.eval()) * g.eval_at(x_expr.eval());
    }

    //! Get coefficient at index i.
    template <typename Index>
    inline auto at(Index&& i) const
    {
      using index_expr_type = calculate_expr_type_t<decltype(i)>;
      const auto i_expr = std::forward<index_expr_type>(i);

      const auto& [a, b] = exprs;
      return a.at(i_expr.eval()) * b.at(i_expr.eval());
    }
  };

  template <typename L, typename R>
  struct DivXpr : Expression<DivXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;
    using decayed_type =
        DivXpr<typename std::decay_t<lhs_type>::decayed_type,  //
               typename std::decay_t<rhs_type>::decayed_type>;

    std::tuple<L, R> exprs;

    inline constexpr DivXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }

    inline auto eval() const
    {
      const auto& [a, b] = exprs;
      return a.eval() / b.eval();
    }

    template <typename X>
    inline auto eval_at(X&& x) const
    {
      const auto& [f, g] = exprs;

      using x_expr_type = calculate_expr_type_t<decltype(x)>;
      const auto x_expr = std::forward<x_expr_type>(x);

      return f.eval_at(x_expr.eval()) / g.eval_at(x_expr.eval());
    }

    //! Get coefficient at index i.
    template <typename Index>
    inline auto at(Index&& i) const
    {
      using index_expr_type = calculate_expr_type_t<decltype(i)>;
      auto&& i_expr = std::forward<index_expr_type>(i);

      const auto& [a, b] = exprs;
      return a.at(i_expr.eval()) / b.at(i_expr.eval());
    }
  };

  template <typename L, typename R>
  struct FunComposeXpr : Expression<FunComposeXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;

    using decayed_type =
        FunComposeXpr<typename std::decay_t<lhs_type>::decayed_type,  //
                      typename std::decay_t<rhs_type>::decayed_type>;

    std::tuple<L, R> exprs;

    inline constexpr FunComposeXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }

    template <typename X>
    inline auto eval_at(X&& x) const
    {
      const auto& [f, g] = exprs;

      using x_expr_type = calculate_expr_type_t<decltype(x)>;
      auto&& x_expr = std::forward<x_expr_type>(x);

      auto&& gx = g.eval_at(x_expr.eval());
      using gx_expr_type = calculate_expr_type_t<decltype(gx)>;
      auto&& gx_expr = std::forward<gx_expr_type>(gx);

      return f.eval_at(gx_expr.eval());
    }
  };


  template <typename L, typename R>
  struct FunXpr : Expression<FunXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;
    using decayed_type =
        FunXpr<typename std::decay_t<lhs_type>::decayed_type,  //
               typename std::decay_t<rhs_type>::decayed_type>;

    std::tuple<L, R> exprs;

    inline constexpr FunXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }

    inline auto eval() const
    {
      const auto& [f, x] = exprs;
      using x_expr_type = calculate_expr_type_t<decltype(x)>;

      auto&& x_expr = std::forward<x_expr_type>(x.eval());
      return f.eval_at(x_expr.eval());
    }

    //! Get coefficient at index i.
    template <typename Index>
    inline auto at(Index&& i) const
    {
      using index_expr_type = calculate_expr_type_t<decltype(i)>;
      auto&& i_expr = std::forward<index_expr_type>(i);

      const auto& [f, x] = exprs;
      using x_expr_type = calculate_expr_type_t<decltype(x)>;

      auto&& x_expr = std::forward<x_expr_type>(x.eval());
      return f.eval_at(x_expr[i_expr.eval()].eval());
    }
  };


  template <typename L, typename R>
  struct SubscriptXpr : Expression<SubscriptXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;

    std::tuple<L, R> exprs;

    inline constexpr SubscriptXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }

    inline auto eval() const
    {
      const auto& [x, i] = exprs;
      return x.at(i.eval());
    }
  };

  template <typename E>
  struct NegXpr : Expression<Expr<E>>
  {
    E expr;

    inline NegXpr(E&& e) noexcept
      : expr{std::forward<E>(e)}
    {
    }

    inline auto eval() const
    {
      return -e.eval();
    }
  };

} /* namespace sara::expression */
