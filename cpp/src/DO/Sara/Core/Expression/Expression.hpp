#pragma once


namespace sara::expression {

  template <typename Derived>
  struct Expression
  {
    using derived_type = Derived;

    //! @{
    //! @brief derived object
    inline auto& derived() & noexcept
    {
      return static_cast<derived_type&>(*this);
    }

    inline const auto& derived() const& noexcept
    {
      return static_cast<const derived_type&>(*this);
    }

    inline auto&& derived() && noexcept
    {
      return static_cast<derived_type&&>(*this);
    }
    //! @}


    //! @{
    //! @brief Addition operation.
    template <typename R>
    inline auto operator+(R&& rhs) & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return plus_expr<lhs_type, rhs_type>{
          derived(),  //
          std::forward<R>(rhs)         //
      };
    }

    template <typename R>
    inline auto operator+(R&& rhs) const & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return plus_expr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator+(R&& rhs) && noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<R&&>;

      return plus_expr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),  //
          std::forward<R>(rhs)                //
      };
    }
    //! @}

    //! @{
    //! @brief Subtraction operation.
    template <typename R>
    inline auto operator-(R&& rhs) & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return minus_expr<lhs_type, rhs_type>{
          derived(),  //
          std::forward<R>(rhs)         //
      };
    }

    template <typename R>
    inline auto operator-(R&& rhs) const & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return minus_expr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator-(R&& rhs) && noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<R&&>;

      return minus_expr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),  //
          std::forward<R>(rhs)                //
      };
    }
    //! @}

    //! @{
    //! @brief Multiplication operation.
    template <typename R>
    inline auto operator*(R&& rhs) & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return multiplies_expr<lhs_type, rhs_type>{
          derived(),  //
          std::forward<R>(rhs)         //
      };
    }

    template <typename R>
    inline auto operator*(R&& rhs) const & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return multiplies_expr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator*(R&& rhs) && noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<R&&>;

      return multiplies_expr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),  //
          std::forward<R>(rhs)                //
      };
    }
    //! @}


    //! @{
    //! @brief Division operation.
    template <typename R>
    inline auto operator/(R&& rhs) & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return divides_expr<lhs_type, rhs_type>{
          derived(),  //
          std::forward<R>(rhs)         //
      };
    }

    template <typename R>
    inline auto operator/(R&& rhs) const & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return multiplies_expr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator/(R&& rhs) && noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<R&&>;

      return divides_expr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),  //
          std::forward<R>(rhs)                //
      };
    }
    //! @}
  };

} /* namespace sara::expression */
