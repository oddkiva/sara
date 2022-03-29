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

#include "Shader.hpp"

#include <DO/Sara/Core/StringFormat.hpp>

#include <iostream>
#include <fstream>


namespace DO::Sara { namespace GL {

  void Shader::create_from_source(GLenum shader_type_,
                                  const std::string& source)
  {
    destroy();

    shader_type = shader_type_;
    shader_object = glCreateShader(shader_type);

    auto shader_src_data = reinterpret_cast<const GLchar*>(source.data());
    auto shader_src_sz = static_cast<GLint>(source.size());

    glShaderSource(shader_object, 1, &shader_src_data, &shader_src_sz);
    glCompileShader(shader_object);

    // Compilation sanity check.
    auto success = GLint{};
    auto log_max_sz = GLint{0};
    auto log_sz = GLsizei{0};
    auto log = std::string{};

    glGetShaderiv(shader_object, GL_COMPILE_STATUS, &success);

    if (success)
      return;

    glGetShaderiv(shader_object, GL_INFO_LOG_LENGTH, &log_max_sz);
    log.resize(log_max_sz);
    glGetShaderInfoLog(shader_object, log_max_sz, &log_sz, &log[0]);
    log.resize(log_sz);

    throw std::runtime_error{
        DO::Sara::format("Error: failed to create shader from source:\n"
                         "%s.\n"
                         "Compilation log:\n"
                         "%s",
                         source.c_str(), log.c_str())};
  }

  void Shader::create_from_file(GLenum shader_type_,
                                const std::string& filepath)
  {
    // Read source fle.
    std::ifstream file{filepath.c_str()};
    if (!file.is_open())
      throw std::runtime_error{
          format("Error: cannot open file: %s", filepath.c_str())};

    auto source = std::string{};
    file.seekg(0, std::ios::end);
    source.reserve(file.tellg());

    file.seekg(0, std::ios::beg);
    source.assign(std::istreambuf_iterator<char>(file),
                  std::istreambuf_iterator<char>());

    create_from_source(shader_type_, source);
  }

  void Shader::destroy()
  {
    if (!shader_object)
      return;

    glDeleteShader(shader_object);

    auto success = GLint{};
    glGetShaderiv(shader_object, GL_DELETE_STATUS, &success);
    if (success == GL_FALSE)
      throw std::runtime_error{
          format("Error: failed to delete shader: %d.", success)};

    shader_object = 0;
  }


  void ShaderProgram::attach(const Shader& vertex_shader,
                             const Shader& fragment_shader)
  {
    create();

    glAttachShader(program_object, vertex_shader);
    glAttachShader(program_object, fragment_shader);

    glLinkProgram(program_object);

    // Linking sanity check.
    auto success = GLint{};
    auto log_max_sz = GLint{0};
    auto log_sz = GLsizei{0};
    auto log = std::string{};

    glGetProgramiv(program_object, GL_LINK_STATUS, &success);
    if (!success)
    {
      glGetProgramiv(program_object, GL_INFO_LOG_LENGTH, &log_max_sz);

      log.resize(log_max_sz);
      glGetProgramInfoLog(program_object, log_max_sz, &log_sz, &log[0]);
      log.resize(log_sz);

      throw std::runtime_error{format("Failed to link shader program: %d.\n"
                                      "Linkage log:\n%s",
                                      success, log.data())};
    }
  }

  void ShaderProgram::validate()
  {
    glValidateProgram(program_object);

    // Validation sanity check.
    auto success = GLint{};
    glGetProgramiv(program_object, GL_VALIDATE_STATUS, &success);
    if (success)
      return;

    auto log_max_sz = GLint{0};
    auto log_sz = GLsizei{0};
    auto log = std::string{};
    glGetProgramiv(program_object, GL_INFO_LOG_LENGTH, &log_max_sz);

    log.resize(log_max_sz);
    glGetProgramInfoLog(program_object, log_max_sz, &log_sz, &log[0]);
    log.resize(log_sz);

    throw std::runtime_error{format("Failed to link shader program: %d\n"
                                    "Linkage log:\n%s",
                                    success, log.data())};
  }

  void ShaderProgram::detach()
  {
    if (!program_object)
      return;

    glDetachShader(program_object, vertex_shader);
    glDetachShader(program_object, fragment_shader);
  }

  void ShaderProgram::use(bool on)
  {
    if (on)
      glUseProgram(program_object);
    else
      glUseProgram(0);
  }

  void ShaderProgram::create()
  {
    if (!program_object)
      program_object = glCreateProgram();
    if (!program_object)
      throw std::runtime_error{"Failed to create shader program!"};
  }

  void ShaderProgram::clear()
  {
    if (!program_object)
      return;

    glDeleteProgram(program_object);
    auto success = GLint{};
    glGetProgramiv(program_object, GL_DELETE_STATUS, &success);

    if (success)
      return;

    glValidateProgram(program_object);

    auto log_max_sz = GLint{0};
    auto log_sz = GLsizei{0};
    auto log = std::string{};

    glGetProgramiv(program_object, GL_INFO_LOG_LENGTH, &log_max_sz);

    log.resize(log_max_sz);
    glGetProgramInfoLog(program_object, log_max_sz, &log_sz, &log[0]);
    log.resize(log_sz);

    program_object = 0;

    throw std::runtime_error{
        DO::Sara::format("Failed to delete shader program: %d."
                         "Delete log:\n%s)log",
                         success, log.data())};
  }

  void ShaderProgram::set_uniform_matrix4f(const char* mat_name,
                                           const float* mat_coeffs)
  {
    auto mat_location = glGetUniformLocation(program_object, mat_name);
    if (GL_INVALID_VALUE == mat_location ||
        GL_INVALID_OPERATION == mat_location)
      throw std::runtime_error{"Invalid uniform parameter"};

    glUniformMatrix4fv(mat_location, 1, GL_FALSE, mat_coeffs);
  }

} /* namespace GL */
} /* namespace DO::Sara */
