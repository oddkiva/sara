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

#pragma once

#include <DO/Kalpana/EasyGL/OpenGL.hpp>

#include <stdexcept>
#include <string>


namespace DO::Kalpana::GL {

  //! @addtogroup EasyGL
  //! @{

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

    inline auto get_uniform_location(const char* param_name) const -> GLint
    {
      const auto param_location =
          glGetUniformLocation(program_object, param_name);
      if (GL_INVALID_VALUE == param_location ||
          GL_INVALID_OPERATION == param_location)
        throw std::runtime_error{"Invalid uniform parameter"};
      return param_location;
    }

    template <typename T>
    inline auto set_uniform_param(GLint param_location,
                                  const T& param_value) const -> void
    {
      if constexpr (std::is_same_v<T, int>)
        glUniform1i(param_location, param_value);
      else if constexpr (std::is_same_v<T, float>)
        glUniform1f(param_location, param_value);
      else
        throw std::runtime_error{"Error: not implemented!"};
    }


    template <typename T>
    inline auto set_uniform_param(const char* param_name,
                                  const T& param_value) const -> void
    {
      const auto param_location = get_uniform_location(param_name);
      set_uniform_param(param_location, param_value);
    }

    inline auto set_uniform_vector2f(GLint mat_location,
                                     const float* mat_coeffs) const -> void
    {
      glUniform2fv(mat_location, 1, mat_coeffs);
    }

    inline auto set_uniform_vector4f(GLint mat_location,
                                     const float* mat_coeffs) const -> void
    {
      glUniform4fv(mat_location, 1, mat_coeffs);
    }

    inline auto set_uniform_matrix3f(GLint mat_location,
                                     const float* mat_coeffs) const -> void
    {
      glUniformMatrix3fv(mat_location, 1, GL_FALSE, mat_coeffs);
    }

    inline auto set_uniform_matrix4f(GLint mat_location,
                                     const float* mat_coeffs) const -> void
    {
      glUniformMatrix4fv(mat_location, 1, GL_FALSE, mat_coeffs);
    }

    inline auto set_uniform_texture(GLint tex_location,
                                    GLuint texture_unit) const -> void
    {
      glUniform1i(tex_location, texture_unit);
    }

    auto set_uniform_vector2f(const char* mat_name,
                              const float* mat_coeffs) const -> void;

    auto set_uniform_vector4f(const char* mat_name,
                              const float* mat_coeffs) const -> void;

    auto set_uniform_matrix3f(const char* mat_name,
                              const float* mat_coeffs) const -> void;

    auto set_uniform_matrix4f(const char* mat_name,
                              const float* mat_coeffs) const -> void;

    auto set_uniform_texture(const char* texture_name,
                             GLuint texture_unit) const -> void;

    GLuint program_object{0};
    GLuint vertex_shader{0};
    GLuint fragment_shader{0};
  };

  //! @}

}  // namespace DO::Kalpana::GL
