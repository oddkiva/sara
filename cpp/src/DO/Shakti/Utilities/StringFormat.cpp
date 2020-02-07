// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Shakti/Utilities/StringFormat.hpp>

#include <cstdarg>
#include <cstdio>
#include <vector>

namespace DO { namespace Shakti {

  static std::string vformat(const char* format, va_list args)
  {
    size_t size = 1024;
    std::vector<char> buffer(size);

    while (true)
    {
      int formatted_string_length =
          vsnprintf(buffer.data(), size, format, args);

      if (formatted_string_length <= int(size) && formatted_string_length >= 0)
        return std::string(buffer.data(), size_t(formatted_string_length));

      size =
          formatted_string_length > 0 ? formatted_string_length + 1 : size * 2;
      buffer.resize(size);
    }
  }

  std::string format(const char* fmt, ...)
  {
    va_list args;
    va_start(args, fmt);
    auto formatted_message = vformat(fmt, args);
    va_end(args);
    return formatted_message;
  }

}}  // namespace DO::Shakti
