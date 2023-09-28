// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <memory>


namespace DO::Sara {

  struct IterativePnPMethod
  {
    struct Impl;
    struct ImplDeleter
    {
      auto operator()(const Impl* p) const -> void;
    };

    IterativePnPMethod();

    std::unique_ptr<Impl, ImplDeleter> _pimpl;
  };

}  // namespace DO::Sara
