#pragma once

#include <drafts/Expression/Terminal.hpp>

#include <cmath>
#include <functional>


// Usual functions.
namespace sara::expression {

  //! @addtogroup Expression
  //! @{

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

  using Sin = Terminal<sin_t>;  // decltype(sin_);
  using Cos = Terminal<cos_t>;  // decltype(cos_);

  //! @}

}  // namespace sara::expression
