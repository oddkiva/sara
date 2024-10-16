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

#include <DO/Kalpana/EasyGL/Shader.hpp>

#include <fmt/format.h>

#include <fstream>
#include <iostream>


namespace DO::Kalpana::GL {

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
        fmt::format("Error: failed to create shader from source:\n"
                    "{}.\n"
                    "Compilation log:\n"
                    "{}",
                    source, log)};
  }

  void Shader::create_from_file(GLenum shader_type_,
                                const std::string& filepath)
  {
    // Read source fle.
    std::ifstream file{filepath.c_str()};
    if (!file.is_open())
      throw std::runtime_error{
          fmt::format("Error: cannot open file: {}", filepath)};

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
          fmt::format("Error: failed to delete shader: {}.", success)};

    shader_object = 0;
  }


  void ShaderProgram::attach(const Shader& vertex_shader,
                             const Shader& fragment_shader)
  {
    create();

#if 0
    std::cout << "Attaching vertex shader " << vertex_shader
              << " to program object " << program_object << std::endl;
#endif
    glAttachShader(program_object, vertex_shader);
#if 0
    std::cout << "Attaching vertex shader " << fragment_shader
              << " to program object " << program_object << std::endl;
#endif
    glAttachShader(program_object, fragment_shader);

#if 0
    std::cout << "Link shader program " << program_object << std::endl;
#endif
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

      throw std::runtime_error{
          fmt::format("Failed to link shader program: {}.\n"
                      "Linkage log:\n"
                      "{}",
                      success, log)};
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

    throw std::runtime_error{fmt::format("Failed to link shader program: {}\n"
                                         "Linkage log:\n"
                                         "{}",
                                         success, log)};
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

#if 0
    std::cout << "Created shader program with ID: " << program_object
              << std::endl;
#endif
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

    std::cerr << fmt::format("Error: failed to delete shader program {}: {}."
                             "Delete log:\n{}",
                             program_object, success, log);
    program_object = 0;
  }

}  // namespace DO::Kalpana::GL
