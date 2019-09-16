#pragma once

#include <DO/Sara/Core/Expression/Differential/Rules.hpp>
#include <DO/Sara/Core/Expression/Terminal.hpp>

#include <cmath>
#include <functional>


namespace sara::expression {

struct sin_t
{
  template <typename T>
  inline auto operator()(T x) const
  {
    return std::sin(x);
  }
};

struct cos_t
{
  template <typename T>
  inline auto operator()(T x) const
  {
    return std::cos(x);
  }
};

struct negate_t
{
  template <typename T>
  inline auto operator()(T x) const
  {
    return -x;
  }
};

constexpr auto sin_ = make_terminal(sin_t{});
constexpr auto cos_ = make_terminal(cos_t{});
constexpr auto negate_ = make_terminal(negate_t{});

using Sin = Terminal<sin_t>;//decltype(sin_);
using Cos = Terminal<cos_t>;//decltype(cos_);


template <typename Y, typename X>
struct Diff<FunXpr<Sin, Y>, X> : Expression<Diff<FunXpr<Sin, Y>, X>>
{
  using y_type = typename std::decay_t<Y>::decayed_type;
  using x_type = typename std::decay_t<X>::decayed_type;
  using dy_dx_type = derivative_t<y_type, x_type>;
  using result_type = std::conditional_t<
      std::is_same_v<y_type, x_type>, decltype(cos_(y_type{})),
      decltype(cos_(y_type{}) * derivative_t<y_type, x_type>{})>;
};


template <typename Y, typename X>
struct Diff<FunXpr<Cos, Y>, X> : Expression<Diff<FunXpr<Cos, Y>, X>>
{
  using y_type = typename std::decay_t<Y>::decayed_type;
  using x_type = typename std::decay_t<X>::decayed_type;
  using dy_dx_type = derivative_t<y_type, x_type>;
  using result_type = std::conditional_t<
      std::is_same_v<y_type, x_type>, decltype(negate_.circle(sin_)(y_type{})),
      decltype(negate_.circle(sin_)(y_type{}) * derivative_t<y_type, x_type>{})>;
};

}  // namespace sara::expression::function
