#ifndef DO_SARA_CORE_STRINGFORMAT_HPP
#define DO_SARA_CORE_STRINGFORMAT_HPP

#include <cstdarg>
#include <string>

#include <DO/Sara/Defines.hpp>


namespace DO { namespace Sara {

  DO_SARA_EXPORT
  std::string format(const char *fmt, ...);

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_STRINGFORMAT_HPP */
