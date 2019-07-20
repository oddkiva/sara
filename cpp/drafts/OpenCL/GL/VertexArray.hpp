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

    void vertex_attrib_pointer(int position, TensorView_<float, 2>& va)
    {
      glVertexAttribPointer(0, va.size(1), GL_FLOAT, GL_FALSE,
                            va.stride(0) * sizeof(float), nullptr);
    }

    GLuint object{0};
  };

} /* namespace GL */
} /* namespace DO::Sara */
