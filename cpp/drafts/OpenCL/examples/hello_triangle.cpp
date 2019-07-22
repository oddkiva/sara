#include <drafts/OpenCL/GL.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>

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
}


Tensor_<float, 2> read_point_cloud(const std::string& h5_filepath)
{
  auto h5_file = H5File{h5_filepath, H5F_ACC_RDONLY};
  auto coords = Tensor_<float, 2>{};
  h5_file.read_dataset("points", coords);
  return coords;
}


int main()
{
  init_gl_boilerplate();

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  // Create a window.
  const auto width = 800;
  const auto height = 600;
  auto window =
      glfwCreateWindow(width, height, "Hello Triangle", nullptr, nullptr);
  glfwMakeContextCurrent(window);

  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1},   //
                                        {"out_color", 0}};

  const auto vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec3 in_color;

  out vec3 out_color;

  void main()
  {
    gl_Position = vec4(in_coords, 1.0);
    gl_PointSize = 200.0;
    out_color = in_color;
  }
  )shader";
  auto vertex_shader = GL::Shader{};
  vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);


  const auto fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
  out vec4 frag_color;

  void main()
  {
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;

    float dist = length(gl_PointCoord - vec2(0.5));

    if (dot(circCoord, circCoord) > 1.0)
        discard;
    float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
    frag_color = vec4(out_color, alpha);
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

  auto vertices = Tensor_<float, 2>{{3, 6}};
  vertices.flat_array() << //
    // coords            color
    -0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // left
     0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // right
     0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f;  // top
#if 0
  // Encode the vertex data in a tensor.
  auto coords = read_point_cloud("/Users/david/Desktop/geometry.h5");
  auto vertices = Tensor_<float, 2>{{coords.size(0), 6}};
  vertices.flat_array().fill(1.f);
  vertices.matrix().leftCols(3) = coords.matrix();
  vertices.matrix().col(2) *= -1.f;

  SARA_DEBUG << "coords sizes = " << coords.sizes().transpose() << std::endl;

  SARA_DEBUG << "coords =\n" << coords.matrix().topRows(10) << std::endl;
  SARA_DEBUG << "vertices =\n" << vertices.matrix().topRows(10) << std::endl;
#endif

  const auto row_bytes = [](const TensorView_<float, 2>& data) {
    return data.size(1) * sizeof(float);
  };
  const auto float_pointer = [](int offset) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  auto vao = GL::VertexArray{};
  vao.generate();

  auto vbo = GL::Buffer{};
  vbo.generate();

  // Specify the vertex attributes here.
  {
    glBindVertexArray(vao);

    // Copy the vertex data into the GPU buffer object.
    vbo.bind_vertex_data(vertices);

    // Specify that the vertex shader param 0 corresponds to the first 3 float
    // data of the buffer object.
    glVertexAttribPointer(arg_pos["in_coords"], 3 /* 3D points */, GL_FLOAT,
                          GL_FALSE, row_bytes(vertices), float_pointer(0));
    glEnableVertexAttribArray(arg_pos["in_coords"]);

    // Specify that the vertex shader param 1 corresponds to the first 3 float
    // data of the buffer object.
    glVertexAttribPointer(arg_pos["in_color"], 3 /* 3D colors */, GL_FLOAT,
                          GL_FALSE, row_bytes(vertices), float_pointer(3));
    glEnableVertexAttribArray(arg_pos["in_color"]);

    // Unbind the vbo to protect its data.
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
  }

  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_DEPTH_TEST);

  // Activate the shader program once and for all.
  shader_program.use(true);

  // Display image.
  glfwSwapInterval(1);
  while (!glfwWindowShouldClose(window))
  {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw triangles
    glBindVertexArray(vao); // geometry specified by the VAO.
    glDrawArrays(GL_POINTS, 0, vertices.size(0)); //

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  vao.destroy();
  vbo.destroy();

  // Clean up resources.
  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
