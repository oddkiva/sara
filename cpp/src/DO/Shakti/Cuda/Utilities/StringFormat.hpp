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

#pragma once

#include <DO/Shakti/Defines.hpp>

#include <string>


namespace DO { namespace Shakti {

  //! @todo Find a better way than duplicating this function already in Sara and
  //! I have not managed to resolve this via CMake because CUDA does not support
  //! C++ 17 yet.
  DO_SHAKTI_EXPORT
  std::string format(const char* fmt, ...);

}}  // namespace DO::Shakti
