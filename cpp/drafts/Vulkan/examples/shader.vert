#version 450
#extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) out vec3 frag_color;
//
// vec2 positions[3] = vec2[](
//   vec2(0.0, -0.5),
//   vec2(0.5,  0.5),
//   vec2(-0.5, +0.5)
// );
//
// vec3 colors[3] = vec3[](
//   vec3(1.0, 0.0, 0.0),
//   vec3(0.0, 1.0, 0.0),
//   vec3(0.0, 0.0, 1.0)
// );
//
// void main() {
//   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
//   frag_color = colors[gl_VertexIndex];
// }

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 frag_color;

void main() {
  gl_Position = vec4(in_position, 0.0, 1.0);
  frag_color = in_color;
}
