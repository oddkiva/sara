// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_CORE_STATICASSERT_HPP
#define DO_CORE_STATICASSERT_HPP

//! @file
//! \brief Implementation from:
//! http://stackoverflow.com/questions/1980012/boost-static-assert-without-boost

//! Concatenation macro used for the implementation of DO_STATIC_ASSERT.
#define CAT(arg1, arg2)   CAT1(arg1, arg2)
#ifndef DOXYGEN_SHOULD_SKIP_THIS
#define CAT1(arg1, arg2)  CAT2(arg1, arg2)
#define CAT2(arg1, arg2)  arg1##arg2
#endif

/*!
  \ingroup Meta

  \brief Static assertion macro.

  \param expression a boolean expression
  \param message some error message

  Usage:

  DO_STATIC_ASSERT(expression, message); // **don't forget** the semi-colon!

  When the static assertion test fails, a compiler error message that somehow
  contains the "STATIC_ASSERTION_FAILED_AT_LINE_xxx_message" is generated.

  WARNING: message has to be a valid C++ identifier, that is to say it must not
  contain space characters, cannot start with a digit, etc.

  DO_STATIC_ASSERT(true, this_message_will_never_be_displayed);
 */
#define DO_STATIC_ASSERT(expression, message)                               \
struct CAT(__static_assertion_at_line_, __LINE__)                           \
{                                                                           \
  DO::Meta::StaticAssertion<static_cast<bool>((expression))>                \
    CAT(CAT(CAT(STATIC_ASSERTION_FAILED_AT_LINE_, __LINE__), _), message);  \
}

// Note that we wrap the non existing type inside a struct to avoid warning
// messages about unused variables when static assertions are used at function
// scope
// the use of sizeof makes sure the assertion error is not ignored by SFINAE

namespace DO { namespace Meta {

  //! Used for the implementation of DO_STATIC_ASSERT.
  template <bool> struct StaticAssertion;
  //! Used for the implementation of DO_STATIC_ASSERT.
  template <> struct StaticAssertion<true> {};
  //! Used for the implementation of DO_STATIC_ASSERT.
  template<int i> struct StaticAssertionTest {};

} /* namespace Meta */
} /* namespace DO */


#endif /* DO_CORE_STATICASSERT_HPP */
