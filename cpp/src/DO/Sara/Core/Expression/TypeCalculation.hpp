#pragma once

#include <DO/Sara/Core/Expression/Terminals.hpp>

#include <type_traits>


namespace sara::expression {

  //! @{
  //! @brief Type calculations.
  template <typename E>
  struct is_expr
  {
    using expr_type = std::remove_cv_t<std::remove_reference_t<E>>;

    static constexpr bool value =
        std::is_base_of<Expression<expr_type>, expr_type>::value;
  };

  template <typename E>
  constexpr auto is_expr_v = is_expr<E>::value;
  //! @}


  //! @{
  //! @brief Calculate the expression type if the type T is not an expression.
  template <typename T>
  using remove_ref_if_type_is_rvalue_ref_t = std::conditional_t<  //
      std::is_rvalue_reference<T>::value,                         //
      std::remove_reference_t<T>,                                 //
      T                                                           //
      >;

  template <typename T>
  using wrap_as_terminal_t = Terminal<remove_ref_if_type_is_rvalue_ref_t<T>>;

  template <typename T>
  using calculate_expr_type_t = std::conditional_t<  //
      is_expr_v<T>,                                  //
      T,                                             //
      wrap_as_terminal_t<T>                          //
      >;
  //! @}

} /* namespace sara::expression */
