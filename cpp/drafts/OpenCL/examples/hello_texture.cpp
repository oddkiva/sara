#include <drafts/OpenCL/GL.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <map>


using namespace DO::Sara;
using namespace std;


inline auto init_gl_boilerplate()
{
  // Initialize the windows manager.
  if (!glfwInit())
    throw std::runtime_error{"Error: failed to initialize GLFW!"};

#ifndef __APPLE__
  // Initialize GLEW.
  auto err = glewInit();
  if (err != GLEW_OK)
  {
    std::cerr << format("Error: failed to initialize GLEW: %s",
                        glewGetErrorString(err))
              << std::endl;
    return EXIT_FAILURE;
  }
#endif

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
}


int main()
{
  init_gl_boilerplate();

  // Create a window.
  const auto width = 800;
  const auto height = 600;
  auto window =
      glfwCreateWindow(width, height, "Hello Triangle", nullptr, nullptr);
  glfwMakeContextCurrent(window);

  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1},   //
                                        {"in_tex_coords", 2},   //
                                        {"out_color", 0}};

  const auto vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec3 in_color;
  layout (location = 2) in vec2 in_tex_coords;

  out vec3 out_color;
  out vec2 out_tex_coords;

  void main()
  {
    gl_Position = vec4(in_coords, 1.0);
    gl_PointSize = 200.0;
    out_color = in_color;
    out_tex_coords = vec2(in_tex_coords.x, in_tex_coords.y);
  }
  )shader";
  auto vertex_shader = GL::Shader{};
  vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);


  const auto fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
  in vec2 out_tex_coords;
  out vec4 frag_color;

  uniform sampler2D texture1;

  void main()
  {
    frag_color = texture(texture1, out_tex_coords) * vec4(out_color, 1.0);
  }
  )shader";
  auto fragment_shader = GL::Shader{};
  fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                     fragment_shader_source);

  auto shader_program = GL::ShaderProgram{};
  shader_program.create();
  shader_program.attach(vertex_shader, fragment_shader);

  vertex_shader.destroy();
  fragment_shader.destroy();

  // Encode the vertex data in a tensor.
  auto vertices = Tensor_<float, 2>{{4, 8}};
  vertices.flat_array() << //
    // coords            color              texture coords
     0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,  // bottom-right
     0.5f,  0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 1.0f,  // top-right
    -0.5f,  0.5f, 0.0f,  1.0f, 1.0f, 0.0f,  0.0f, 1.0f,  // top-left
    -0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f;  // bottom-left

  auto triangles = Tensor_<unsigned int, 2>{{2, 3}};
  triangles.flat_array() <<
    0, 1, 2,
    2, 3, 0;

  const auto row_bytes = [](const TensorView_<float, 2>& data) {
    return data.size(1) * sizeof(float);
  };
  const auto float_pointer = [](int offset) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  auto vao = GL::VertexArray{};
  vao.generate();

  // Vertex attributes.
  auto vbo = GL::Buffer{};
  vbo.generate();

  // Triangles data.
  auto ebo = GL::Buffer{};
  ebo.generate();

  {
    glBindVertexArray(vao);

    // Copy vertex data.
    vbo.bind_vertex_data(vertices);

    // Copy geometry data.
    ebo.bind_triangles_data(triangles);

    // Map the parameters to the argument position for the vertex shader.
    //
    // Vertex coordinates.
    glVertexAttribPointer(arg_pos["in_coords"], 3 /* 3D points */, GL_FLOAT,
                          GL_FALSE, row_bytes(vertices), float_pointer(0));
    glEnableVertexAttribArray(arg_pos["in_coords"]);

    // Colors.
    glVertexAttribPointer(arg_pos["in_color"], 3 /* 3D colors */, GL_FLOAT,
                          GL_FALSE, row_bytes(vertices), float_pointer(3));
    glEnableVertexAttribArray(arg_pos["in_color"]);

    // Texture coordinates.
    glVertexAttribPointer(arg_pos["in_tex_coords"], 2 /* 3D colors */, GL_FLOAT,
                          GL_FALSE, row_bytes(vertices), float_pointer(6));
    glEnableVertexAttribArray(arg_pos["in_tex_coords"]);
  }

  // Texture data.
  auto texture = GL::Texture2D{};
  {
    // Read the image from the disk.
    auto image =
        imread<Rgb8>("/Users/david/GitLab/DO-CV/sara/data/ksmall.jpg");
    // Flip vertically so that the image data matches OpenGL image coordinate
    // system.
    flip_vertically(image);

    // Copy the image to the GPU texture.
    texture.setup_with_pretty_defaults(image, 0);
  }

  // Activate the shader program once and for all.
  shader_program.use(true);

  // Activate the texture 0 once for all.
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);

  // Display image.
  glfwSwapInterval(1);
  while (!glfwWindowShouldClose(window))
  {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw triangles.
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, triangles.size(), GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  vao.destroy();
  vbo.destroy();
  ebo.destroy();

  // Clean up resources.
  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
