#include <DO/Sara/Core/Tensor.hpp>

#if defined(__APPLE__)
# include <OpenCL/cl_gl.h>
# include <OpenGL/gl3.h>
#else
# include <CL/cl_gl.h>
# include <GL/glew.h>
#endif


namespace DO::Sara { namespace GL {

  struct Buffer
  {
    static void generate(Buffer* begin, Buffer* end)
    {
      glGenBuffers(int(end - begin), reinterpret_cast<GLuint*>(begin));
    }

    void generate()
    {
      Buffer::generate(this, this + 1);
    }

    void destroy()
    {
      glDeleteBuffers(1, &object);
      object = 0;
    }

    operator GLuint&()
    {
      return object;
    }
    operator GLuint() const
    {
      return object;
    }

    template <typename T>
    auto bind_data(const DO::Sara::TensorView_<T, 2>& data) const
    {
      glBindBuffer(GL_ARRAY_BUFFER, object);
      glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(T), data.data(),
                   GL_STATIC_DRAW);
    }

    GLuint object{0};
  };

} /* namespace GL */
} /* namespace DO::Sara */
