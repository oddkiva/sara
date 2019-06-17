#pragma once

#include <DO/Sara/Core/Expression/Expression.hpp>

#include <tuple>


namespace sara::expression {

  template <typename L, typename R>
  struct PlusXpr : Expression<PlusXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;

    std::tuple<L, R> exprs;

    inline constexpr PlusXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }
  };

  template <typename L, typename R>
  struct MinusXpr : Expression<MinusXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;

    std::tuple<L, R> exprs;

    inline constexpr MinusXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }
  };

  template <typename L, typename R>
  struct MultipliesXpr : Expression<MultipliesXpr<L, R>>
  {
    using tuple_type = std::tuple<L, R>;
    using lhs_type = typename std::tuple_element<0, tuple_type>::type;
    using rhs_type = typename std::tuple_element<1, tuple_type>::type;

    std::tuple<L, R> exprs;

    inline constexpr MultipliesXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }
  };

  template <typename L, typename R>
  struct DividesXpr : Expression<DividesXpr<L, R>>
  {
    std::tuple<L, R> exprs;

    inline constexpr DividesXpr(L&& l, R&& r) noexcept
      : exprs{std::forward_as_tuple(l, r)}
    {
    }
  };

} /* namespace sara::expression */
