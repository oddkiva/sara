#include <fstream>
#include <iostream>
#include <stdexcept>

#include <DO/Sara/Core/StringFormat.hpp>

#include <DO/Kalpana/3D.hpp>


using namespace std;


namespace DO { namespace Kalpana {

  Shader::~Shader()
  {
    clear();
  }

  bool Shader::create_from_source(GLenum shader_type, const std::string& source)
  {
    initializeOpenGLFunctions();

    clear();

    _shader_type = shader_type;
    _shader_object = glCreateShader(shader_type);

    auto shader_src_data = reinterpret_cast<const GLchar *>(source.data());
    auto shader_src_sz = static_cast<GLint>(source.size());

    glShaderSource(_shader_object, 1, &shader_src_data, &shader_src_sz);
    glCompileShader(_shader_object);

    // Compilation sanity check.
    auto success = GLint{};
    auto log_max_sz = GLint{ 0 };
    auto log_sz = GLsizei{ 0 };
    auto log = string{};

    glGetShaderiv(_shader_object, GL_COMPILE_STATUS, &success);

    if (!success)
    {
      glGetShaderiv(_shader_object, GL_INFO_LOG_LENGTH, &log_max_sz);
      log.resize(log_max_sz);
      glGetShaderInfoLog(_shader_object, log_max_sz, &log_sz, &log[0]);
      log.resize(log_sz);

      std::cerr << Sara::format("Error: failed to create shader from source:\n"
                                "%s.\n"
                                "Compilation log:\n"
                                "%s",
                                source.c_str(), log.c_str()) << std::endl;
      return false;
    }

    return true;
  }

  bool Shader::create_from_file(GLenum shader_type, const std::string& filepath)
  {
    // Read source fle.
    std::ifstream file{ filepath.c_str() };
    if (!file.is_open())
    {
      std::cerr << Sara::format("Error: cannot open file: %s", filepath.c_str())
                << std::endl;
      return false;
    }

    auto source = std::string{};
    file.seekg(0, std::ios::end);
    source.reserve(file.tellg());

    file.seekg(0, std::ios::beg);
    source.assign(std::istreambuf_iterator<char>(file),
                  std::istreambuf_iterator<char>());

    return create_from_source(shader_type, source);
  }

  bool Shader::clear()
  {
    if (_shader_object)
    {
      glDeleteShader(_shader_object);

      auto success = GLint{};
      glGetShaderiv(_shader_object, GL_DELETE_STATUS, &success);
      if (!success)
      {
        std::cerr << Sara::format("Error: failed to delete shader: %d.",
                                  success) << std::endl;
        return false;
      }

      _shader_object = 0;
    }

    return true;
  }


  ShaderProgram::~ShaderProgram()
  {
    detach();
    clear();
  }

  bool ShaderProgram::attach(const Shader& vertex_shader,
                             const Shader& fragment_shader)
  {
    initializeOpenGLFunctions();

    if (!create())
      return false;

    glAttachShader(_program_object, vertex_shader);
    glAttachShader(_program_object, fragment_shader);

    glLinkProgram(_program_object);

    // Linking sanity check.
    auto success = GLint{};
    auto log_max_sz = GLint{ 0 };
    auto log_sz = GLsizei{ 0 };
    auto log = std::string{};

    glGetProgramiv(_program_object, GL_LINK_STATUS, &success);
    if (!success)
    {
      glGetProgramiv(_program_object, GL_INFO_LOG_LENGTH, &log_max_sz);

      log.resize(log_max_sz);
      glGetProgramInfoLog(_program_object, log_max_sz, &log_sz, &log[0]);
      log.resize(log_sz);

      std::cerr << Sara::format("Failed to link shader program: %d.\n"
                                "Linkage log:\n%s",
                                success, log.data()) << std::endl;
      return false;
    }

    glValidateProgram(_program_object);

    // Validation sanity check.
    glGetProgramiv(_program_object, GL_VALIDATE_STATUS, &success);
    if (!success)
    {
      glGetProgramiv(_program_object, GL_INFO_LOG_LENGTH, &log_max_sz);

      log.resize(log_max_sz);
      glGetProgramInfoLog(_program_object, log_max_sz, &log_sz, &log[0]);
      log.resize(log_sz);

      std::cerr << Sara::format("Failed to link shader program: %d\n"
                                "Linkage log:\n%s",
                                success, log.data()) << std::endl;
      return false;
    }

    return true;
  }

  bool ShaderProgram::detach()
  {
    if (_program_object)
    {
      glDetachShader(_program_object, _vertex_shader);
      glDetachShader(_program_object, _fragment_shader);
    }

    return true;
  }

  bool ShaderProgram::use(bool on)
  {
    if (on)
      glUseProgram(_program_object);
    else
      glUseProgram(0);
  }

  bool ShaderProgram::create()
  {
    initializeOpenGLFunctions();

    if (!_program_object)
      _program_object = glCreateProgram();

    if (!_program_object)
    {
      std::cerr << "Failed to create shader program!" << std::endl;
      return false;
    }

    return true;
  }

  bool ShaderProgram::clear()
  {
    if (!_program_object)
      return true;

    glDeleteProgram(_program_object);
    auto success = GLint{};
    glGetProgramiv(_program_object, GL_DELETE_STATUS, &success);

    if (!success)
    {
      glValidateProgram(_program_object);

      auto log_max_sz = GLint{ 0 };
      auto log_sz = GLsizei{ 0 };
      auto log = std::string{};

      glGetProgramiv(_program_object, GL_INFO_LOG_LENGTH, &log_max_sz);

      log.resize(log_max_sz);
      glGetProgramInfoLog(_program_object, log_max_sz, &log_sz, &log[0]);
      log.resize(log_sz);

      std::cerr << Sara::format("Failed to delete shader program: %d.\n"
                                "Deletion log:\n%s",
                                success, log.data()) << std::endl;

      return false;
    }

    _program_object = 0;

    return true;
  }

  bool ShaderProgram::set_uniform_matrix4f(const char *mat_name,
                                           const float *mat_coeffs)
  {
    auto mat_location = glGetUniformLocation(_program_object, mat_name);
    if (GL_INVALID_VALUE == mat_location ||
        GL_INVALID_OPERATION == mat_location)
    {
      std::cerr << "Invalid shader program" << std::endl;
      return false;
    }

    glUniformMatrix4fv(mat_location, 1, GL_FALSE, mat_coeffs);

    return true;
  }

} /* namespace Kalpana */
} /* namespace DO */
