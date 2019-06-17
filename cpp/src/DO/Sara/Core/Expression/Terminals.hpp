#pragma once

#include <DO/Sara/Core/Expression/Expression.hpp>

#include <string>


namespace sara::expression {

  template <typename T>
  struct Terminal : Expression<Terminal<T>>
  {
    inline constexpr Terminal(T v)
      : value{v}
    {
    }

    T value;
  };

  template <typename T>
  inline auto make_terminal(T&& t) -> Terminal<T> 
  {
    return {t};
  }

} /* namespace sara::expression */
