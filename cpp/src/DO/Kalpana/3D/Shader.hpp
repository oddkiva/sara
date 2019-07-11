#pragma once

#include <string>

#include <QOpenGLFunctions_4_3_Core>


namespace DO { namespace Kalpana {

  class Shader : protected QOpenGLFunctions_4_3_Core
  {
  public:
    Shader() = default;

    ~Shader();

    operator GLuint() const { return _shader_object; }

    bool create_from_source(GLenum shader_type, const std::string& source);

    bool create_from_file(GLenum shader_type, const std::string& filepath);

  private:
    bool clear();

  private:
    GLenum _shader_type;
    GLuint _shader_object{ 0 };
  };

  class ShaderProgram : protected QOpenGLFunctions_4_3_Core
  {
  public:
    inline ShaderProgram() = default;

    ShaderProgram(const Shader& vertex_shader, const Shader& fragment_shader);

    ~ShaderProgram();

    operator GLuint() const { return _program_object; }

    bool attach(const Shader& vertex_shader, const Shader& fragment_shader);

    bool detach();

    bool use(bool on = true);

    bool set_uniform_matrix4f(const char *mat_name, const float* mat_coeffs);

  protected:
    bool create();

    bool clear();

  private:
    GLuint _program_object{ 0 };
    GLuint _vertex_shader{ 0 };
    GLuint _fragment_shader{ 0 };
  };

} /* namespace Kalpana */
} /* namespace DO */
