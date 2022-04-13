// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#include "ImageDewarpRenderer.hpp"


namespace sara = DO::Sara;


std::unique_ptr<ImageDewarpRenderer> ImageDewarpRenderer::_instance = nullptr;


auto ImageDewarpRenderer::instance() -> ImageDewarpRenderer&
{
  if (_instance == nullptr)
    _instance.reset(new ImageDewarpRenderer{});
  return *_instance;
}

auto ImageDewarpRenderer::initialize() -> void
{
  // Create a vertex shader.
  static const std::map<std::string, int> arg_pos = {{"in_coords", 0},       //
                                                     {"in_tex_coords", 1}};  //

  static const auto vertex_shader_source = R"shader(#version 300 es
    layout (location = 0) in vec3 in_coords;
    layout (location = 1) in vec2 in_tex_coords;

    uniform mat4 projection;
    uniform mat4 model_view;
    uniform vec2 image_sizes;

    out vec2 out_tex_coords;

    void main()
    {
      // Change the aspect ratio of the square so that it conforms with the
      // image one.
      float image_ratio = image_sizes.x / image_sizes.y;
      vec2 final_coords = vec2(in_coords.x * image_ratio, in_coords.y);

      gl_Position = projection * model_view * vec4(final_coords, 0.0, 1.0);
      out_tex_coords = in_tex_coords;
    }
  )shader";
  _vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);

  // Create a fragment shader.
  static const auto fragment_shader_source = R"shader(#version 300 es
    #ifdef GL_ES
    precision highp float;
    #endif

    in vec2 out_tex_coords;
    out vec4 frag_color;

    uniform int dewarp_mode;

    // Relative rotation of the destination stereographic camera.
    uniform mat3 R;
    // Intrinsic parameters of the destination stereographic camera.
    uniform mat3 K_inverse;

    // Intrinsic parameters of the source omnidirectional camera.
    uniform vec2 image_sizes;
    uniform mat3 K;
    uniform vec2 k;
    uniform vec2 p;
    uniform float xi;

    uniform sampler2D image;

    vec2 to_image_coordinates(vec2 tex_coords)
    {
      vec2 pix = tex_coords;
      pix.y = 1. - pix.y;
      pix.x *= image_sizes.x;
      pix.y *= image_sizes.y;
      return pix;
    }

    vec2 to_texture_coordinates(vec2 pn)
    {
      pn.x /= image_sizes.x;
      pn.y /= image_sizes.y;
      pn.y = 1. - pn.y;
      return pn;
    }

    vec2 lens_distortion(vec2 m)
    {
      // Radial component (additive).
      float r2 = dot(m, m);
      float r4 = r2 * r2;
      vec2 radial_factor = (k[0] * r2 + k[1] * r4) * m;

      // Tangential component (additive).
      float tx = 2. * p[0] * m.x * m.y + p[1] * (r2 + 2. * p[0] * m.x);
      float ty = p[0] * (r2 + 2. * p[0] * m.y) + 2. * p[1] * m.x * m.y;

      // Apply the distortion.
      vec2 delta = radial_factor + vec2(tx, ty);

      return delta;
    }

    vec3 backproject_from_dst_stereographic_camera(vec2 pix_dst)
    {
      // Backproject the pixel from the destination camera plane.
      // const Eigen::Vector2f xy_dst = camera_dst.backproject(uv).head(2);
      vec3 p3 = K_inverse * vec3(pix_dst, 1.);

      if (dewarp_mode == 0)
        return p3;

      vec2 xy_dst = (p3 / p3.z).xy;

      // Solve a polynomial in y.
      float xy_dst_squared_norm = dot(xy_dst, xy_dst);
      float a = xy_dst_squared_norm + 4.;
      float b = -2. * xy_dst_squared_norm;
      float c = xy_dst_squared_norm - 4.;
      float delta = b * b - 4. * a * c;

      float y = (-b - sqrt(delta)) / (2. * a);
      float x = xy_dst.x * (1. - y) / 2.;
      float w = xy_dst.y * (1. - y) / 2.;

      // Retrieve the unit ray.
      // N.B.: the ray has already a norm equal to 1 by construction.
      vec3 ray = vec3(x, y, w);

      // Re-express the ray w.r.t. to the source camera frame.
      return ray;
    }

    vec3 project_to_src_omnidir_camera(vec3 ray)
    {
      // 3D ray in the camera frame.
      vec3 Xs = normalize(ray);

      // Change coordinates.
      vec3 Xe = Xs + vec3(0., 0., xi);

      // Project to the camera film plane.
      Xe /= Xe.z;
      vec2 m = Xe.xy;

      // Add distortion.
      vec2 m_distorted = m + lens_distortion(m);

      vec3 p = K * vec3(m_distorted, 1.);

      return p;
    }

    void main()
    {
      vec2 pix_dst = to_image_coordinates(out_tex_coords);
      vec3 ray_dst = backproject_from_dst_stereographic_camera(pix_dst);

      // Rotate the ray to get.
      ray_dst = R * ray_dst;

      vec3 pix_src3 = project_to_src_omnidir_camera(ray_dst);
      vec2 pix_src = (pix_src3 / pix_src3.z).xy;

      vec2 pix_tex_coords = to_texture_coordinates(pix_src);

      if (pix_tex_coords.x < 0. || pix_tex_coords.x > 1. ||
          pix_tex_coords.y < 0. || pix_tex_coords.y > 1.)
        frag_color = vec4(0., 0., 0., 1.);
      else
        frag_color = texture(image, pix_tex_coords);
    }
  )shader";
  _fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                      fragment_shader_source);
  _shader_program.create();
  _shader_program.attach(_vertex_shader, _fragment_shader);

#ifndef EMSCRIPTEN
  // Clearing the shaders after attaching them to the shader program does not
  // work on WebGL 2.0/OpenGL ES 3.0... I don't know why.
  _shader_program.use();
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();
#endif

  // Generate a single square geometry data. This data will be constantly reused
  // to display any image on the screen.
  //
  // Encode the vertex data of the square in a tensor.
  auto vertices = sara::Tensor_<float, 2>{{4, 5}};
  // clang-format off
  vertices.flat_array() <<
  // coords              texture coords
     0.5f, -0.5f, 0.0f,  1.0f, 0.0f,  // bottom-right
     0.5f,  0.5f, 0.0f,  1.0f, 1.0f,  // top-right
    -0.5f,  0.5f, 0.0f,  0.0f, 1.0f,  // top-left
    -0.5f, -0.5f, 0.0f,  0.0f, 0.0f;  // bottom-left
  // clang-format on

  auto triangles = sara::Tensor_<std::uint32_t, 2>{{2, 3}};
  // clang-format off
  triangles.flat_array() <<
    0, 1, 2,
    2, 3, 0;
  // clang-format on

  if (!_vao)
    _vao.generate();
  if (!_vbo)
    _vbo.generate();
  if (!_ebo)
    _ebo.generate();

  glBindVertexArray(_vao);

  // Copy vertex data.
  _vbo.bind_vertex_data(vertices);

  // Copy geometry data.
  _ebo.bind_triangles_data(triangles);

  static constexpr auto row_bytes =
      [](const sara::TensorView_<float, 2>& data) {
        return data.size(1) * sizeof(float);
      };
  static constexpr auto float_pointer = [](int offset) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  // Map the parameters to the argument position for the vertex shader.
  //
  // Vertex coordinates.
  glVertexAttribPointer(                           //
      arg_pos.at("in_coords"), 3 /* 3D points */,  //
      GL_FLOAT, GL_FALSE, row_bytes(vertices), float_pointer(0));
  glEnableVertexAttribArray(arg_pos.at("in_coords"));

  // Texture coordinates.
  glVertexAttribPointer(
      arg_pos.at("in_tex_coords"), 2 /* texture coordinates */,  //
      GL_FLOAT, GL_FALSE, row_bytes(vertices), float_pointer(3));
  glEnableVertexAttribArray(arg_pos.at("in_tex_coords"));
}

auto ImageDewarpRenderer::destroy_gl_objects() -> void
{
  _shader_program.detach();
  _vertex_shader.destroy();
  _fragment_shader.destroy();
  _vao.destroy();
  _vbo.destroy();
  _ebo.destroy();
}

auto ImageDewarpRenderer::render(const ImagePlaneRenderer::ImageTexture& image,
                                 const CameraParameters& camera,
                                 int dewarp_mode) -> void
{
  _shader_program.use(true);

  // Set the projection-model-view matrix uniforms.
  _shader_program.set_uniform_texture("image", image._texture_unit);
  _shader_program.set_uniform_vector2f("image_sizes",
                                       image._image_sizes.data());
  static const Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
  _shader_program.set_uniform_matrix4f("model_view", identity.data());
  _shader_program.set_uniform_matrix4f("projection", image._projection.data());

  // Destination stereographic camera.
  _shader_program.set_uniform_matrix3f("R", camera.R.data());
  _shader_program.set_uniform_matrix3f("K_inverse", camera.K_inverse.data());

  // Dewarp mode.
  _shader_program.set_uniform_param("dewarp_mode", dewarp_mode);

  // Source omnidirectional camera parameters.
  _shader_program.set_uniform_matrix3f(  //
      "K", camera._intrinsics.K.data());
  _shader_program.set_uniform_vector2f(
      "k", camera._intrinsics.radial_distortion_coefficients.data());
  _shader_program.set_uniform_vector2f(
      "p", camera._intrinsics.tangential_distortion_coefficients.data());
  _shader_program.set_uniform_param(  //
      "xi", camera._intrinsics.xi);

  glBindVertexArray(_vao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}
