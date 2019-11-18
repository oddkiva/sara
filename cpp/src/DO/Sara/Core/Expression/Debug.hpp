#pragma once

#include <boost/core/demangle.hpp>
#include <boost/algorithm/string.hpp>

#include <string>


template <class T>
std::string type_name()
{
  typedef typename std::remove_reference<T>::type TR;
  const auto name = typeid(TR).name();

  auto r = boost::core::demangle(name);
  boost::algorithm::replace_all(r, "sara::expression::", "");
  boost::algorithm::replace_all(r, "Variable<double, 120>", "X");
  boost::algorithm::replace_all(r, "Terminal", "T");
  boost::algorithm::replace_all(r, "Fun", "F");
  boost::algorithm::replace_all(r, "Xpr", "");

  boost::algorithm::replace_all(r, "&", "");
  boost::algorithm::replace_all(r, "const", "");

  if constexpr (std::is_const<TR>::value)
    r += " const";
  if constexpr (std::is_volatile<TR>::value)
    r += " volatile";
  if constexpr (std::is_lvalue_reference<T>::value)
    r += "&";
  else if constexpr (std::is_rvalue_reference<T>::value)
    r += "&&";
  return r;
}
