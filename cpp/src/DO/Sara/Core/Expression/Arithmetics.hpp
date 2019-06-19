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

    std::tuple<L, R> exprs;

    inline constexpr AddXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }

    inline auto eval() const
    {
      return std::get<0>(exprs).eval() + std::get<1>(exprs).eval();
    }
  };

  template <typename L, typename R>
  struct SubXpr : Expression<SubXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;

    std::tuple<L, R> exprs;

    inline constexpr SubXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }
  };

  template <typename L, typename R>
  struct MulXpr : Expression<MulXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;

    std::tuple<L, R> exprs;

    inline constexpr MulXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }
  };

  template <typename L, typename R>
  struct DivXpr : Expression<DivXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;

    std::tuple<L, R> exprs;

    inline constexpr DivXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }
  };

  template <typename L, typename R>
  struct FunXpr : Expression<FunXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;

    std::tuple<L, R> exprs;

    inline constexpr FunXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }

    inline auto eval() const
    {
      const auto x = std::get<1>(exprs).eval();
      const auto& f = std::get<0>(exprs);
      return f.eval(x.eval());
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
  };

} /* namespace sara::expression */
