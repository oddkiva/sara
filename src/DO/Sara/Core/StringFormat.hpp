#pragma once

#include <cstdarg>
#include <string>

#include <DO/Sara/Defines.hpp>


namespace DO { namespace Sara {

  DO_SARA_EXPORT
  std::string format(const char *fmt, ...);

} /* namespace Sara */
} /* namespace DO */
