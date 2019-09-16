#pragma once

#include <boost/core/demangle.hpp>

#include <string>


template <class T>
std::string type_name()
{
  typedef typename std::remove_reference<T>::type TR;
  const char *name = typeid(TR).name();

  std::string r = boost::core::demangle(name);
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
