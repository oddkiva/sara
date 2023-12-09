#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ModelViewProjection {
  mat4 model;
  mat4 view;
  mat4 projection;
} mvp;

// Vertex data.
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_tex_coords;

// Forwarded data to the fragment shader.
layout(location = 0) out vec2 tex_coords;

void main() {
  gl_Position = mvp.projection * mvp.view * mvp.model * vec4(in_position, 0.0, 1.0);
  tex_coords = in_tex_coords;
}
