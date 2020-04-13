// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#if defined(__APPLE__)
# include <OpenCL/cl_gl.h>
# include <OpenGL/gl3.h>
#else
# include <CL/cl_gl.h>
# include <GL/glew.h>
#endif
