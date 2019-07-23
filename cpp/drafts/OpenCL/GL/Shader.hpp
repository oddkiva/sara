#if defined(__APPLE__)
# include <OpenCL/cl_gl.h>
# include <OpenGL/gl3.h>
#else
# include <CL/cl_gl.h>
# include <GL/glew.h>
#endif

#include <string>
#include <stdexcept>


namespace DO::Sara { namespace GL {

  struct Shader
  {
    operator GLuint() const
    {
      return shader_object;
    }

    void create_from_source(GLenum shader_type, const std::string& source);

    void create_from_file(GLenum shader_type, const std::string& filepath);

    void destroy();

    GLenum shader_type;
    GLuint shader_object{0};
  };


  struct ShaderProgram
  {
    inline ShaderProgram() = default;

    operator GLuint() const
    {
      return program_object;
    }

    void attach(const Shader& vertex_shader, const Shader& fragment_shader);

    void validate();

    void detach();

    void use(bool on = true);

    void create();

    void clear();

    template <typename T>
    void set_uniform_param(const char* param_name, const T& param_value)
    {
      auto param_location = glGetUniformLocation(program_object, param_name);
      if (GL_INVALID_VALUE == param_location ||
          GL_INVALID_OPERATION == param_location)
        throw std::runtime_error{"Invalid uniform parameter"};

      if constexpr(std::is_same_v<T, int>)
        glUniform1i(param_location, param_value);
      else if constexpr(std::is_same_v<T, float>)
        glUniform1f(param_location, param_value);
      else
        throw std::runtime_error{"Error: not implemented!"};
    }

    void set_uniform_matrix4f(const char* mat_name, const float* mat_coeffs);

    GLuint program_object{0};
    GLuint vertex_shader{0};
    GLuint fragment_shader{0};
  };

} /* namespace GL */
} /* namespace DO::Sara */
