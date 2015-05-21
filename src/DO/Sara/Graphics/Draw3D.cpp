// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Graphics.hpp>
#include "GraphicsUtilities.hpp"

namespace DO {

  void display_mesh(const SimpleTriangleMesh3f& mesh)
  {
    QMetaObject::invokeMethod(active_window(), "setMesh",
                              Qt::QueuedConnection,
                              Q_ARG(const SimpleTriangleMesh3f&, mesh));
  }

} /* namespace DO */