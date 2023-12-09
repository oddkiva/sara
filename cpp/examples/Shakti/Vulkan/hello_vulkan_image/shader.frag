#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D image;

layout(location = 0) in vec2 tex_coords;

layout(location = 0) out vec4 out_color;

void main() {
  out_color = texture(image, tex_coords);
}
