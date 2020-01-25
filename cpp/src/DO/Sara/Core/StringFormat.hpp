// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <cstdarg>
#include <string>

#include <DO/Sara/Defines.hpp>


namespace DO { namespace Sara {

  //! @addtogroup Utility
  //! @{

  DO_SARA_EXPORT
  std::string format(const char *fmt, ...);

  //! @}

} /* namespace Sara */
} /* namespace DO */
