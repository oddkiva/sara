#pragma once

#include <drafts/Expression/Terminal.hpp>
#include <drafts/Expression/Differential/Variable.hpp>


namespace sara::expression {

  template <typename T>
  using Zero = Terminal<Variable<const T, 1>>;

  template <typename T>
  using One = Terminal<Variable<const T, 1>>;


  namespace constants {

    template <typename T>
    constexpr auto zero = Zero<T>{};

    template <typename T>
    constexpr auto one = Zero<T>{};

  } /* namespace constants */


} /* namespace sara::expression */
