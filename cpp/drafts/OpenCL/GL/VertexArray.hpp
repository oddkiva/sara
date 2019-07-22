#include <DO/Sara/Core/Tensor.hpp>

#if defined(__APPLE__)
# include <OpenCL/cl_gl.h>
# include <OpenGL/gl3.h>
#else
# include <CL/cl_gl.h>
# include <GL/glew.h>
#endif


namespace DO::Sara { namespace GL {

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

} /* namespace GL */
} /* namespace DO::Sara */
