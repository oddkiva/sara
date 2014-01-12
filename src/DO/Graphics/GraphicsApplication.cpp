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

#include <DO/Graphics.hpp>
#include "GraphicsUtilities.hpp"

namespace DO {

  GraphicsApplication::GraphicsApplication(int argc, char **argv)
    : pimpl_(new GraphicsApplication::Impl(argc, argv))
  {
    pimpl_->userThread.registerUserMain(__main);
    pimpl_->userThread.start();
  }

  GraphicsApplication::~GraphicsApplication()
  {
    if (pimpl_)
      delete pimpl_;
    pimpl_ = 0;
  }

  int GraphicsApplication::exec()
  {
    return pimpl_->exec();
  }

} /* namespace DO */