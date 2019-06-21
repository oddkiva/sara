#pragma once


namespace sara::expression {

  template <typename Derived>
  struct Expression;

  template <typename T>
  struct Terminal;

  template <typename L, typename R>
  struct AddXpr;

  template <typename L, typename R>
  struct SubXpr;

  template <typename L, typename R>
  struct MulXpr;

  template <typename L, typename R>
  struct DivXpr;

  template <typename L, typename R>
  struct FunXpr;

  template <typename L, typename R>
  struct SubscriptXpr;

} /* namespace sara::expression */
