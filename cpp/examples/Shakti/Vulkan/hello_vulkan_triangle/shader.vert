#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ModelViewProjection {
  mat4 model;
  mat4 view;
  mat4 projection;
} mvp;

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 frag_color;

void main() {
  gl_Position = mvp.projection * mvp.view * mvp.model * vec4(in_position, 0.0, 1.0);
  frag_color = in_color;
}
