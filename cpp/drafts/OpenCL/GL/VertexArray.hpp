// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/Tensor.hpp>

#if defined(__APPLE__)
# include <OpenCL/cl_gl.h>
# include <OpenGL/gl3.h>
#else
# include <CL/cl_gl.h>
# include <GL/glew.h>
#endif


namespace DO::Sara { namespace GL {

  //! @addtogroup OpenGL
  //! @{

  struct VertexArray
  {
    static void generate(VertexArray* begin, VertexArray* end)
    {
      glGenVertexArrays(int(end - begin), reinterpret_cast<GLuint*>(begin));
    }

    void generate()
    {
      VertexArray::generate(this, this + 1);
    }

    operator GLuint&()
    {
      return object;
    }

    operator GLuint() const
    {
      return object;
    }

    void destroy()
    {
      glDeleteVertexArrays(1, &object);
      object = 0;
    }

    GLuint object{0};
  };

  //! @}

} /* namespace GL */
} /* namespace DO::Sara */
